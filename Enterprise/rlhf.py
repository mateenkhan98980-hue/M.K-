from .deps import *
from .logger import StructuredLogger

class PreferenceDataset:
    """
    Stores human preference pairs:
      (prompt, chosen_response, rejected_response)
    Used for both reward-model training and DPO.
    """
    def __init__(self):
        self.pairs: List[Dict] = []

    def add(
        self,
        prompt:     str,
        chosen:     str,
        rejected:   str,
        score_diff: float = 1.0,
    ):
        self.pairs.append({
            "prompt":     prompt,
            "chosen":     chosen,
            "rejected":   rejected,
            "score_diff": score_diff,
        })

    def to_hf_dataset(self) -> "Dataset":
        return Dataset.from_list(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)


class ComparisonRewardModel(nn.Module):
    """
    Bradley-Terry reward model for RLHF Stage 2.

    Architecture:
      base_model hidden states  →  linear value head  →  scalar reward

    Loss (preference loss):
      L = -log σ(r_chosen - r_rejected)
      Trained to assign higher scores to preferred responses.
    """
    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.base       = base_model
        self.tokenizer  = tokenizer
        hidden_size     = base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False)

    def forward_reward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return scalar reward per sequence (last-token pooling)."""
        outputs     = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]   # (B, H)
        return self.value_head(last_hidden).squeeze(-1)      # (B,)

    def preference_loss(
        self,
        chosen_ids:    torch.Tensor,
        rejected_ids:  torch.Tensor,
        chosen_mask:   torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bradley-Terry loss:
          L = -log σ(r_chosen - r_rejected)
        Returns: (loss, mean_chosen_reward, mean_rejected_reward)
        """
        r_chosen   = self.forward_reward(chosen_ids,   chosen_mask)
        r_rejected = self.forward_reward(rejected_ids, rejected_mask)
        loss       = -F.logsigmoid(r_chosen - r_rejected).mean()
        return loss, r_chosen.mean(), r_rejected.mean()


class SFTTrainer_:
    """
    RLHF Stage 1 — Supervised Fine-Tuning.

    Trains the base model on curated (prompt, response) pairs
    BEFORE reward modelling or RL.  Uses LoRA r=64 (higher than
    plain fine-tuning) to capture conversational style well.

    Data format expected:
      [{"prompt": "...", "response": "..."}, ...]
    """
    def __init__(
        self,
        model_name: str,
        logger:     StructuredLogger,
        lora_r:     int = 64,
    ):
        self.logger    = logger
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype      = (torch.bfloat16
                      if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                      else torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )

        # All projection layers targeted (better quality than q/v only)
        cfg = LoraConfig(
            r              = lora_r,
            lora_alpha     = lora_r * 2,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            lora_dropout   = 0.05,
            bias           = "none",
            task_type      = TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, cfg)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        self.logger.info(
            f"SFT LoRA r={lora_r}: "
            f"{trainable/1e6:.1f}M / {total/1e6:.1f}M params trainable "
            f"({100*trainable/total:.2f}%)"
        )

    def prepare_sft_dataset(self, conversations: List[Dict]) -> "Dataset":
        """Format conversations and tokenise with prompt masking."""
        texts = [
            f"<|user|>\n{c['prompt']}\n<|assistant|>\n"
            f"{c['response']}{self.tokenizer.eos_token}"
            for c in conversations
        ]
        ds = Dataset.from_dict({"text": texts})

        def tok(ex):
            enc          = self.tokenizer(
                ex["text"], truncation=True, max_length=1024, padding=False
            )
            enc["labels"] = enc["input_ids"].copy()
            return enc

        return ds.map(tok, batched=True, remove_columns=["text"])

    def train(
        self,
        conversations: List[Dict],
        output_dir:    str   = "./sft_model",
        epochs:        int   = 3,
        lr:            float = 2e-4,
    ) -> Tuple:
        dataset  = self.prepare_sft_dataset(conversations)
        split    = dataset.train_test_split(test_size=0.05)
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        args = TrainingArguments(
            output_dir                  = output_dir,
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 8,
            num_train_epochs            = epochs,
            learning_rate               = lr,
            lr_scheduler_type           = "cosine",
            warmup_ratio                = 0.03,
            weight_decay                = 0.01,
            bf16                        = use_bf16,
            fp16                        = not use_bf16 and torch.cuda.is_available(),
            gradient_checkpointing      = True,
            logging_steps               = 10,
            save_steps                  = 100,
            eval_strategy               = "steps",
            eval_steps                  = 100,
            load_best_model_at_end      = True,
            report_to                   = "none",
        )
        trainer = Trainer(
            model         = self.model,
            args          = args,
            train_dataset = split["train"],
            eval_dataset  = split["test"],
            data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.logger.info(f"SFT complete → {output_dir}")
        return self.model, self.tokenizer


class PPOTrainer_:
    """
    RLHF Stage 3a — Proximal Policy Optimization.

    PPO objective (clipped surrogate):
      L = E[ min( r_t·A_t,  clip(r_t, 1-ε, 1+ε)·A_t ) ]
              - β · KL(π_θ ‖ π_ref)
    where:
      r_t  = π_θ(a|s) / π_ref(a|s)   (importance ratio)
      A_t  = reward − value_baseline   (advantage estimate)
      β    = KL coefficient (default 0.1)
      ε    = clip range (default 0.2)

    A frozen copy of the SFT model acts as the reference policy
    to prevent the model from drifting too far during RL.
    """
    def __init__(
        self,
        model,
        ref_model,
        reward_model: ComparisonRewardModel,
        tokenizer,
        logger:       StructuredLogger,
        kl_coeff:     float = 0.1,
        clip_eps:     float = 0.2,
        lr:           float = 1e-5,
        vf_coeff:     float = 0.1,
    ):
        self.model        = model
        self.ref_model    = ref_model
        self.reward_model = reward_model
        self.tokenizer    = tokenizer
        self.logger       = logger
        self.kl_coeff     = kl_coeff
        self.clip_eps     = clip_eps
        self.vf_coeff     = vf_coeff

        device            = next(model.parameters()).device
        hidden_size       = model.config.hidden_size
        self.value_head   = nn.Linear(hidden_size, 1).to(device)
        self.optimizer    = torch.optim.Adam(
            list(model.parameters()) + list(self.value_head.parameters()),
            lr=lr,
        )

    def _logprobs(self, model, input_ids, attention_mask) -> torch.Tensor:
        """Per-token log-probabilities for a sequence."""
        cm = torch.no_grad() if model is self.ref_model else torch.enable_grad()
        with cm:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits   = out.logits[:, :-1, :]
        labels   = input_ids[:, 1:]
        lp       = F.log_softmax(logits, dim=-1)
        return lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    def _reward(self, input_ids, attention_mask) -> torch.Tensor:
        with torch.no_grad():
            return self.reward_model.forward_reward(input_ids, attention_mask)

    def ppo_step(self, prompt: str) -> Dict:
        """
        One PPO update step:
          generate → reward → KL penalty → advantage → clip loss → backprop
        """
        device = next(self.model.parameters()).device
        enc    = self.tokenizer(prompt, return_tensors="pt").to(device)

        # ── Generate ─────────────────────────────────────────
        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                max_new_tokens=128, do_sample=True, temperature=0.9,
                output_scores=True, return_dict_in_generate=True,
            )
        full_ids  = gen.sequences
        full_mask = (full_ids != self.tokenizer.pad_token_id).long()
        response  = self.tokenizer.decode(
            full_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # ── Reward + KL ───────────────────────────────────────
        reward   = self._reward(full_ids, full_mask)
        lp_pi    = self._logprobs(self.model,     full_ids, full_mask)
        lp_ref   = self._logprobs(self.ref_model, full_ids, full_mask)
        kl       = (lp_pi - lp_ref).mean()

        # ── Advantage ─────────────────────────────────────────
        advantage = (reward - self.kl_coeff * kl).detach()

        # ── PPO clip loss ─────────────────────────────────────
        ratio      = torch.exp(lp_pi - lp_ref.detach())
        clip_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        policy_loss = -torch.min(ratio * advantage, clip_ratio * advantage).mean()

        # ── Value loss ────────────────────────────────────────
        hidden    = self.model(
            full_ids, attention_mask=full_mask, output_hidden_states=True
        )
        value     = self.value_head(hidden.hidden_states[-1][:, -1, :]).squeeze(-1)
        value_loss = F.mse_loss(value, reward.detach())

        total_loss = policy_loss + self.vf_coeff * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        result = {
            "reward":      reward.item(),
            "kl":          kl.item(),
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item(),
            "total_loss":  total_loss.item(),
            "response":    response,
        }
        self.logger.info(
            f"PPO | reward={result['reward']:.4f}  kl={result['kl']:.4f}  "
            f"loss={result['total_loss']:.4f}"
        )
        return result

    def train_ppo(self, prompts: List[str], epochs: int = 1) -> List[Dict]:
        """Full PPO training loop over a prompt list."""
        results = []
        for ep in range(epochs):
            self.logger.info(f"PPO Epoch {ep+1}/{epochs}")
            for i, prompt in enumerate(prompts):
                try:
                    results.append(self.ppo_step(prompt))
                except Exception as e:
                    self.logger.error(f"PPO step {i} error: {e}")
        return results


class DPOTrainer_:
    """
    RLHF Stage 3b — Direct Preference Optimization.

    DPO replaces the reward model + PPO with a single closed-form loss:
      L = -log σ( β·[log π(y_w|x) − log π_ref(y_w|x)]
                  − β·[log π(y_l|x) − log π_ref(y_l|x)] )
    where:
      y_w = chosen response
      y_l = rejected response
      β   = temperature controlling deviation from reference (default 0.1)

    Advantages over PPO:
      • No reward model needed (removes Stage 2)
      • No online generation during training
      • Simpler, more stable training
      • Often competitive quality with PPO
    """
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        logger:     StructuredLogger,
        beta:       float = 0.1,
        lr:         float = 5e-7,
        max_length: int   = 1024,
    ):
        self.model      = model
        self.ref_model  = ref_model
        self.tokenizer  = tokenizer
        self.logger     = logger
        self.beta       = beta
        self.max_length = max_length
        self.optimizer  = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01
        )

    def _sum_logprobs(
        self,
        model,
        input_ids: torch.Tensor,
        labels:    torch.Tensor,
    ) -> torch.Tensor:
        """Sum log-probs over response (non-masked) tokens only."""
        cm = torch.no_grad() if model is self.ref_model else torch.enable_grad()
        with cm:
            out = model(input_ids=input_ids)
        logits   = out.logits[:, :-1, :]
        lp       = F.log_softmax(logits, dim=-1)
        tgt      = input_ids[:, 1:].clone()
        mask     = (labels[:, 1:] != -100)
        per_tok  = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return (per_tok * mask).sum(dim=-1)

    def _encode_pair(
        self,
        prompt:   str,
        response: str,
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode (prompt + response) and mask prompt tokens in labels."""
        p_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        r_ids = self.tokenizer.encode(
            response + self.tokenizer.eos_token, add_special_tokens=False
        )
        full   = (p_ids + r_ids)[: self.max_length]
        labels = ([-100] * len(p_ids) + r_ids)[: self.max_length]
        return (
            torch.tensor([full],   dtype=torch.long, device=device),
            torch.tensor([labels], dtype=torch.long, device=device),
        )

    def dpo_step(self, prompt: str, chosen: str, rejected: str) -> Dict:
        """Single DPO update on one preference pair."""
        device = next(self.model.parameters()).device

        c_ids, c_lab = self._encode_pair(prompt, chosen,   device)
        r_ids, r_lab = self._encode_pair(prompt, rejected, device)

        pi_chosen   = self._sum_logprobs(self.model,     c_ids, c_lab)
        pi_rejected = self._sum_logprobs(self.model,     r_ids, r_lab)
        ref_chosen  = self._sum_logprobs(self.ref_model, c_ids, c_lab)
        ref_rejected = self._sum_logprobs(self.ref_model, r_ids, r_lab)

        logits = self.beta * (
            (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
        )
        loss   = -F.logsigmoid(logits).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        result = {
            "loss":          loss.item(),
            "chosen_reward": (pi_chosen  - ref_chosen).mean().item(),
            "reject_reward": (pi_rejected - ref_rejected).mean().item(),
            "margin":        ((pi_chosen - ref_chosen) -
                              (pi_rejected - ref_rejected)).mean().item(),
        }
        self.logger.info(
            f"DPO | loss={result['loss']:.4f}  margin={result['margin']:.4f}"
        )
        return result

    def train_dpo(
        self,
        preference_data: "PreferenceDataset",
        epochs: int = 1,
    ) -> List[Dict]:
        """Full DPO training loop."""
        results = []
        for ep in range(epochs):
            self.logger.info(f"DPO Epoch {ep+1}/{epochs}")
            for pair in preference_data.pairs:
                try:
                    results.append(
                        self.dpo_step(pair["prompt"], pair["chosen"], pair["rejected"])
                    )
                except Exception as e:
                    self.logger.error(f"DPO step failed: {e}")

        if results:
            avg_loss = sum(r["loss"] for r in results) / len(results)
            self.logger.info(
                f"DPO complete | avg_loss={avg_loss:.4f}  steps={len(results)}"
            )
        return results


class RLHFPipeline:
    """
    End-to-end RLHF orchestrator.

    Two paths available:
      ┌─────────────────────────────────────────────────────┐
      │  PPO path  (classic RLHF, InstructGPT-style)        │
      │    Stage 1 → SFTTrainer_          (fine-tune)        │
      │    Stage 2 → ComparisonRewardModel (train RM)        │
      │    Stage 3 → PPOTrainer_           (RL optimise)     │
      ├─────────────────────────────────────────────────────┤
      │  DPO path  (simpler, no RM needed)                   │
      │    Stage 1 → SFTTrainer_          (fine-tune)        │
      │    Stage 2 → DPOTrainer_           (direct prefs)    │
      └─────────────────────────────────────────────────────┘

    Quick start:
        rlhf = RLHFPipeline("meta-llama/Llama all model", logger)

        # Collect data
        convos = [{"prompt": "What is LoRA?", "response": "LoRA is …"}, …]
        prefs  = PreferenceDataset()
        prefs.add("Explain AI", good_answer, bad_answer)

        # Run (choose one)
        rlhf.full_pipeline_ppo(convos, prefs, ppo_prompts=["Tell me …"])
        rlhf.full_pipeline_dpo(convos, prefs)
    """
    def __init__(self, model_name: str, logger: StructuredLogger):
        self.model_name   = model_name
        self.logger       = logger
        self.sft_model    = None
        self.sft_tokenizer = None
        self.reward_model  = None
        self._stage        = 0

    # ── Stage 1 ──────────────────────────────────────────────
    def run_sft(
        self,
        conversations: List[Dict],
        output_dir:    str = "./rlhf_sft",
        epochs:        int = 3,
        lora_r:        int = 64,
    ):
        self.logger.info("═══ RLHF Stage 1 / 3 : SFT ═══")
        sft = SFTTrainer_(self.model_name, self.logger, lora_r=lora_r)
        self.sft_model, self.sft_tokenizer = sft.train(
            conversations, output_dir=output_dir, epochs=epochs
        )
        self._stage = 1
        self.logger.info("Stage 1 (SFT) ✓")
        return self.sft_model

    # ── Stage 2 ──────────────────────────────────────────────
    def train_reward_model(
        self,
        preference_data: PreferenceDataset,
        output_dir:      str   = "./rlhf_rm",
        epochs:          int   = 2,
        lr:              float = 1e-5,
    ):
        self.logger.info("═══ RLHF Stage 2 / 3 : Reward Model ═══")
        assert self._stage >= 1, "Run run_sft() first."

        self.reward_model = ComparisonRewardModel(self.sft_model, self.sft_tokenizer)
        rm_opt = torch.optim.Adam(self.reward_model.parameters(), lr=lr)
        device = next(self.sft_model.parameters()).device

        for ep in range(epochs):
            epoch_loss = 0.0
            for pair in preference_data.pairs:
                c_enc = self.sft_tokenizer(
                    pair["chosen"],   return_tensors="pt",
                    truncation=True,  max_length=512
                ).to(device)
                r_enc = self.sft_tokenizer(
                    pair["rejected"], return_tensors="pt",
                    truncation=True,  max_length=512
                ).to(device)
                loss, r_c, r_r = self.reward_model.preference_loss(
                    c_enc["input_ids"],   r_enc["input_ids"],
                    c_enc["attention_mask"], r_enc["attention_mask"],
                )
                rm_opt.zero_grad()
                loss.backward()
                rm_opt.step()
                epoch_loss += loss.item()

            avg = epoch_loss / max(len(preference_data), 1)
            self.logger.info(
                f"RM Epoch {ep+1}/{epochs} | avg_loss={avg:.4f}"
            )

        Path(output_dir).mkdir(exist_ok=True)
        torch.save(self.reward_model.state_dict(), f"{output_dir}/reward_model.pt")
        self._stage = 2
        self.logger.info("Stage 2 (Reward Model) ✓")
        return self.reward_model

    # ── Stage 3a ───────────────────────────────────────
    def run_ppo(
        self,
        prompts:    List[str],
        output_dir: str   = "./rlhf_ppo",
        epochs:     int   = 1,
        kl_coeff:   float = 0.1,
    ) -> List[Dict]:
        self.logger.info("═══ RLHF Stage 3a / 3 : PPO ═══")
        assert self._stage >= 2, "Train reward model (run train_reward_model()) first."

        import copy
        ref = copy.deepcopy(self.sft_model)
        for p in ref.parameters():
            p.requires_grad_(False)

        ppo = PPOTrainer_(
            model=self.sft_model, ref_model=ref,
            reward_model=self.reward_model,
            tokenizer=self.sft_tokenizer,
            logger=self.logger, kl_coeff=kl_coeff,
        )
        results = ppo.train_ppo(prompts, epochs=epochs)

        self.sft_model.save_pretrained(output_dir)
        self.sft_tokenizer.save_pretrained(output_dir)

        avg_r = sum(r["reward"] for r in results) / max(len(results), 1)
        self.logger.info(f"Stage 3a (PPO) ✓ | avg_reward={avg_r:.4f}")
        return results

    # ── Stage 3b ─────────────────────────────────────────────
    def run_dpo(
        self,
        preference_data: PreferenceDataset,
        output_dir:      str   = "./rlhf_dpo",
        epochs:          int   = 1,
        beta:            float = 0.1,
    ) -> List[Dict]:
        self.logger.info("═══ RLHF Stage 2b / 2 : DPO ═══")
        assert self._stage >= 1, "Run run_sft() first."

        import copy
        ref = copy.deepcopy(self.sft_model)
        for p in ref.parameters():
            p.requires_grad_(False)

        dpo = DPOTrainer_(
            model=self.sft_model, ref_model=ref,
            tokenizer=self.sft_tokenizer,
            logger=self.logger, beta=beta,
        )
        results = dpo.train_dpo(preference_data, epochs=epochs)

        self.sft_model.save_pretrained(output_dir)
        self.sft_tokenizer.save_pretrained(output_dir)

        avg_l = sum(r["loss"] for r in results) / max(len(results), 1)
        self.logger.info(f"Stage 3b (DPO) ✓ | avg_loss={avg_l:.4f}")
        return results

    # ── Full pipelines ────────────────────────────────────────
    def full_pipeline_ppo(
        self,
        conversations:   List[Dict],
        preference_data: PreferenceDataset,
        ppo_prompts:     List[str],
        base_dir:        str = "./rlhf_output",
    ) -> Dict:
        """
        SFT → Reward Model → PPO  (classic RLHF, 3 stages)
        """
        Path(base_dir).mkdir(exist_ok=True)
        self.logger.info("╔══════════════════════════════════╗")
        self.logger.info("║  RLHF FULL PIPELINE  (PPO path)  ║")
        self.logger.info("╚══════════════════════════════════╝")
        self.run_sft(conversations,     output_dir=f"{base_dir}/stage1_sft")
        self.train_reward_model(preference_data, output_dir=f"{base_dir}/stage2_rm")
        ppo_results = self.run_ppo(ppo_prompts,  output_dir=f"{base_dir}/stage3_ppo")
        avg_r = sum(r["reward"] for r in ppo_results) / max(len(ppo_results), 1)
        self.logger.info(f"PPO Pipeline complete | avg_reward={avg_r:.4f}")
        return {"path": "PPO", "avg_reward": avg_r, "results": ppo_results}

    def full_pipeline_dpo(
        self,
        conversations:   List[Dict],
        preference_data: PreferenceDataset,
        base_dir:        str = "./rlhf_output",
    ) -> Dict:
        """
        SFT → DPO  (simpler 2-stage pipeline, no reward model needed)
        """
        Path(base_dir).mkdir(exist_ok=True)
        self.logger.info("╔══════════════════════════════════╗")
        self.logger.info("║  RLHF FULL PIPELINE  (DPO path)  ║")
        self.logger.info("╚══════════════════════════════════╝")
        self.run_sft(conversations,  output_dir=f"{base_dir}/stage1_sft")
        dpo_results = self.run_dpo(preference_data, output_dir=f"{base_dir}/stage2_dpo")
        avg_l = sum(r["loss"] for r in dpo_results) / max(len(dpo_results), 1)
        self.logger.info(f"DPO Pipeline complete | avg_loss={avg_l:.4f}")
        return {"path": "DPO", "avg_loss": avg_l, "results": dpo_results}
