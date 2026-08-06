from .deps import *
from .logger import StructuredLogger
from .embeddings import RealEmbeddingModel, RealVectorDatabase
from .knowledge import KnowledgeGraph

class RewardModel(nn.Module):
    """Learns to score (prompt, response) pairs."""
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, prompt_emb: torch.Tensor, response_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([prompt_emb, response_emb], dim=-1)
        return self.net(x).squeeze(-1)


class RLTrainer:
    """
    REINFORCE-style RL loop:
      1. Generate response
      2. Score with RewardModel
      3. Back-prop policy gradient
    """
    def __init__(
        self,
        model,
        tokenizer,
        reward_model: RewardModel,
        embedder:     RealEmbeddingModel,
        logger:       StructuredLogger,
        lr: float = 1e-5,
    ):
        self.model        = model
        self.tokenizer    = tokenizer
        self.reward_model = reward_model
        self.embedder     = embedder
        self.logger       = logger
        self.optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
        self.reward_opt   = torch.optim.Adam(reward_model.parameters(), lr=1e-4)

    def compute_reward(self, prompt: str, response: str) -> float:
        p_emb = torch.tensor(self.embedder.encode_one(prompt),    dtype=torch.float32)
        r_emb = torch.tensor(self.embedder.encode_one(response),  dtype=torch.float32)
        dim   = self.reward_model.net[0].in_features // 2
        p_emb = F.pad(p_emb, (0, max(0, dim - p_emb.shape[0])))[:dim]
        r_emb = F.pad(r_emb, (0, max(0, dim - r_emb.shape[0])))[:dim]
        with torch.no_grad():
            score = self.reward_model(p_emb.unsqueeze(0), r_emb.unsqueeze(0))
        return score.item()

    def rl_step(self, prompt: str) -> Dict:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        # Sampling is non-differentiable, so no_grad here is correct —
        # but it also means output.scores below carries NO gradient.
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.9,
                return_dict_in_generate=True,
            )

        full_ids     = output.sequences
        response_ids = full_ids[:, prompt_len:]
        response     = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

        reward = self.compute_reward(prompt, response)

        # Policy gradient: maximize reward.
        # Re-run a FRESH, gradient-tracked forward pass on the full sequence
        # (generate() above ran under no_grad, so its scores have no grad_fn
        # to backprop through) and gather the log-prob of the token that was
        # ACTUALLY sampled at each step — not the greedy/argmax token, which
        # is what F.log_softmax(...).max(...) would give you and usually
        # isn't the token that was emitted when do_sample=True.
        pad_id    = self.tokenizer.pad_token_id
        full_mask = (full_ids != pad_id).long() if pad_id is not None else torch.ones_like(full_ids)
        out       = self.model(input_ids=full_ids, attention_mask=full_mask)
        logits    = out.logits[:, prompt_len - 1 : -1, :]   # predicts response_ids, token-aligned
        token_log_probs = F.log_softmax(logits, dim=-1).gather(
            -1, response_ids.unsqueeze(-1)
        ).squeeze(-1)
        log_probs = token_log_probs.mean()

        loss = -log_probs * reward

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.logger.info(f"RL step | reward={reward:.4f} | loss={loss.item():.4f}")
        return {"reward": reward, "loss": loss.item(), "response": response}

    def train_rl(self, prompts: List[str], steps_per_prompt: int = 3) -> List[Dict]:
        results = []
        for prompt in prompts:
            for _ in range(steps_per_prompt):
                result = self.rl_step(prompt)
                results.append(result)
        return results

    def train_reward_model(self, pairs: List[Tuple[str, str, float]]):
        """
        Train reward model on labelled (prompt, response, score) triples.
        """
        for prompt, response, human_score in pairs:
            p_emb = torch.tensor(self.embedder.encode_one(prompt),   dtype=torch.float32)
            r_emb = torch.tensor(self.embedder.encode_one(response), dtype=torch.float32)
            dim   = self.reward_model.net[0].in_features // 2
            p_emb = F.pad(p_emb, (0, max(0, dim - p_emb.shape[0])))[:dim]
            r_emb = F.pad(r_emb, (0, max(0, dim - r_emb.shape[0])))[:dim]

            pred  = self.reward_model(p_emb.unsqueeze(0), r_emb.unsqueeze(0))
            loss  = F.mse_loss(pred, torch.tensor([[human_score]]))

            self.reward_opt.zero_grad()
            loss.backward()
            self.reward_opt.step()


# ════════════════════════════════════════════════════════════
#  12. SELF-IMPROVEMENT LOOP  (complete)
# ════════════════════════════════════════════════════════════

class SelfImprovementLoop:
    """
    Full closed loop:
      Generate → Evaluate → Reflect → Fine-tune → Repeat
    """
    def __init__(
        self,
        inference,
        rl_trainer:   RLTrainer,
        vector_db:    RealVectorDatabase,
        kg:           KnowledgeGraph,
        logger:       StructuredLogger,
        eval_fn:      Optional[callable] = None,
    ):
        self.inference  = inference
        self.rl_trainer = rl_trainer
        self.vector_db  = vector_db
        self.kg         = kg
        self.logger     = logger
        self.eval_fn    = eval_fn or self._default_eval
        self.history: List[Dict] = [] 
           
    def _default_eval(self, response: str, ground_truth: str = None) -> float:
        """Logic and formatting based reward calculator for GPQA/Reasoning."""
        import re
        if not response:
            return 0.0
            
        score = 0.0
        
        # 1. Chain of Thought Reward (0.3)
        if "think step by step" in response.lower() or "<think>" in response.lower():
            score += 0.3
            
        # 2. Structured Logic Reward (0.2)
        if "step 1:" in response.lower() or "first," in response.lower():
            score += 0.2
            
        # 3. Final Answer Tags Format Check (0.2)
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            score += 0.2
            extracted_answer = answer_match.group(1).strip()
            
            # 4. Exact Match Reward if ground_truth is provided (0.3)
            if ground_truth and extracted_answer.lower() == ground_truth.lower():
                score += 0.3
        else:
            score -= 0.1  # Clear tags use na karne par penalty
            
        return round(max(0.0, min(score, 1.0)), 4)
        
    def generate_and_evaluate(self, prompt: str) -> Dict:
        response = self.inference.generate(prompt, max_length=200)
        score    = self.eval_fn(response)

        # Store in vector DB for future retrieval
        self.vector_db.add(
            [response],
            [{"prompt": prompt, "score": str(score)}],
        )

        # Extract knowledge into graph
        self.kg.ingest_text(response)

        record = {"prompt": prompt, "response": response, "score": score}
        self.history.append(record)
        self.logger.info(f"Self-eval score: {score}")
        return record

    def reflect(self, record: Dict) -> str:
        critique_prompt = (
            f"Original prompt: {record['prompt']}\n"
            f"Response: {record['response']}\n"
            f"Score: {record['score']}\n\n"
            "Identify specific improvements needed:"
        )
        return self.inference.generate(critique_prompt, max_length=150)

    def improve(self, record: Dict) -> Dict:
        critique = self.reflect(record)
        improved_prompt = (
            f"{record['prompt']}\n\n"
            f"Previous answer had issues: {critique}\n"
            "Now give a better answer:"
        )
        new_response = self.inference.generate(improved_prompt, max_length=200)
        new_score    = self.eval_fn(new_response)
        return {"response": new_response, "score": new_score, "critique": critique}

    def run(self, prompts: List[str], iterations: int = 3) -> List[Dict]:
        all_results = []
        for prompt in prompts:
            best = None
            for i in range(iterations):
                self.logger.info(f"Iteration {i+1}/{iterations} for: {prompt[:40]}")
                record = self.generate_and_evaluate(prompt)

                if best is None or record["score"] > best["score"]:
                    best = record

                # RL step on best response
                try:
                    self.rl_trainer.rl_step(prompt)
                except Exception as e:
                    self.logger.warn(f"RL step skipped: {e}")

                # Try to improve if score is low
                if record["score"] < 0.6:
                    improved = self.improve(record)
                    if improved["score"] > record["score"]:
                        best = improved
                        self.logger.info(f"Improved score: {improved['score']}")

            all_results.append({"prompt": prompt, "best": best})
        return all_results

    def report(self) -> Dict:
        if not self.history:
            return {}
        scores = [h["score"] for h in self.history]
        return {
            "total_iterations": len(self.history),
            "avg_score":        round(sum(scores) / len(scores), 4),
            "best_score":       max(scores),
            "worst_score":      min(scores),
            "kg_stats":         self.kg.stats(),
            "vector_db_size":   len(self.vector_db),
        }
