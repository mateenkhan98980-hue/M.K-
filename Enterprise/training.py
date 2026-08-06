from .deps import *
from .logger import StructuredLogger

class GatingNetworkTopK(nn.Module):
    def __init__(self, dim, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate  = nn.Sequential(nn.Linear(dim, dim//2), nn.ReLU(), nn.Linear(dim//2, num_experts))
    def forward(self, x):
        logits = self.gate(x)
        top_logits, top_indices = torch.topk(logits, k=self.top_k, dim=-1)
        return F.softmax(top_logits, dim=-1), top_indices


class SparseMoE(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()
        self.top_k = top_k; self.num_experts = num_experts
        self.router  = GatingNetworkTopK(dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim))
            for _ in range(num_experts)
        ])
    def forward(self, x):
        B, T, C = x.shape; flat_x = x.reshape(-1, C)
        weights, indices = self.router(flat_x)
        output = torch.zeros_like(flat_x)
        for i in range(flat_x.shape[0]):
            token = flat_x[i:i+1]
            for k in range(self.top_k):
                idx = indices[i,k].item()
                output[i] += self.experts[idx](token)[0] * weights[i,k]
        balance_loss = torch.bincount(indices.flatten(), minlength=self.num_experts).float().std()
        return output.view(B,T,C), balance_loss


class MoEFFNWrapper(nn.Module):
    def __init__(self, original_mlp, hidden_size):
        super().__init__()
        self.original_mlp = original_mlp
        self.moe = SparseMoE(dim=hidden_size, num_experts=8, top_k=2)
        self.balance_loss = torch.tensor(0.0)
    def forward(self, x):
        out, bl = self.moe(x); self.balance_loss = bl; return out


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs  = model(**inputs); lm_loss = outputs.loss
        router_loss = torch.tensor(0.0, device=lm_loss.device)
        for layer in getattr(model.model, "layers", []):
            bl = getattr(layer.mlp, "balance_loss", None)
            if bl is not None: router_loss = router_loss + bl
        total = lm_loss + 0.01 * router_loss
        return (total, outputs) if return_outputs else total


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds         = np.argmax(logits, axis=-1)
    shift_preds   = preds[:,:-1].flatten()
    shift_labels  = labels[:,1:].flatten()
    mask = shift_labels != -100
    acc  = (shift_preds[mask] == shift_labels[mask]).mean() if mask.any() else 0.0
    return {"accuracy": float(acc)}


from transformers import TrainerCallback as _TC
class TrainingCallbacks(_TC):
    def __init__(self, logger): self.logger = logger
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs: self.logger.info(str(logs))


class InferencePipeline:
    def __init__(self, model, tokenizer):
        self.model = model; self.tokenizer = tokenizer
    def generate(self, prompt, max_length=200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_length=max_length,
                do_sample=True, temperature=0.7, top_p=0.9
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


class EnterpriseTrainer:
    def __init__(self, model_name, logger):
        self.logger    = logger
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def apply_lora(self):
        cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"],
                         lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
        self.model = get_peft_model(self.model, cfg)
        self.logger.info("LoRA applied.")

    def inject_moe(self):
        for i, layer in enumerate(getattr(self.model.model,"layers",[])):
            mlp  = getattr(layer,"mlp",None)
            gate = getattr(mlp,"gate_proj",None)
            if gate is None: continue
            layer.mlp = MoEFFNWrapper(mlp, gate.in_features)
        self.logger.info("MoE injected.")

    def prepare_dataset(self, texts):
        ds = Dataset.from_dict({"text": texts})
        def tok(ex):
            t = self.tokenizer(ex["text"], truncation=True, padding=False, max_length=512)
            t["labels"] = t["input_ids"].copy()
            return t
        ds = ds.map(tok, batched=True, remove_columns=["text"])
        return ds.train_test_split(test_size=0.1)

    def train(self, dataset, output_dir="./model", num_epochs=3):
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            num_train_epochs=num_epochs, learning_rate=2e-4, logging_steps=10,
            save_steps=100, fp16=torch.cuda.is_available(),
            gradient_checkpointing=True, save_strategy="steps",
            eval_strategy="steps", eval_steps=100, load_best_model_at_end=True,
        )
        trainer = CustomTrainer(
            model=self.model, args=args,
            train_dataset=dataset["train"], eval_dataset=dataset["test"],
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            callbacks=[TrainingCallbacks(self.logger)],
            compute_metrics=compute_metrics,
        )
        try:
            trainer.train()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); self.logger.error("CUDA OOM.")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
