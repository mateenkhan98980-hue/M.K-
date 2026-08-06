from .deps import *
from .logger import StructuredLogger
from .quality import DatasetQualityControl
from .training import EnterpriseTrainer, InferencePipeline

class SelfImprovementLoopWithCoT:
    """
    CoT-Enhanced Self-Improvement Loop
    Generates reasoning traces before evaluating responses.
    """
    def __init__(self, inference, rl_trainer, vector_db, kg, logger, eval_fn=None):
        self.inference   = inference
        self.rl_trainer  = rl_trainer
        self.vector_db   = vector_db
        self.kg          = kg
        self.logger      = logger
        self.eval_fn     = eval_fn or self._enhanced_eval_with_cot
        self.history     = []
    
    def _enhanced_eval_with_cot(self, response: str, ground_truth: str = None) -> float:
        """
        Evaluates CoT-structured responses.
        Expects: <think>reasoning</think><answer>answer</answer>
        """
        score = 0.0
        
        # Extract thinking block
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            # Reward structured reasoning
            score += 0.2
            
            # Reward step-by-step logic
            steps = len(re.findall(r"step|because|therefore|thus|so|hence", thinking.lower()))
            if steps >= 3:
                score += 0.2
        
        # Extract answer block  
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            score += 0.2
            
            # Exact match with ground truth
            if ground_truth and answer.lower() == ground_truth.lower():
                score += 0.4
        else:
            score -= 0.1  # Penalty for missing answer tag
        
        return round(max(0.0, min(score, 1.0)), 4)
    
    def generate_with_cot(self, prompt: str) -> Dict:
        """Generate response with explicit CoT reasoning."""
        cot_prompt = (
            f"Please think step by step, then provide your answer.\n"
            f"Format your response as:\n"
            f"<think>Your step-by-step reasoning here</think>\n"
            f"<answer>Your final answer here</answer>\n\n"
            f"Question: {prompt}"
        )
        response = self.inference.generate(cot_prompt, max_length=300)
        score = self.eval_fn(response)
        
        return {
            "prompt": prompt,
            "response": response,
            "score": score,
            "has_thinking": "<think>" in response,
            "has_answer": "<answer>" in response,
        }
    
    def run_cot_loop(self, prompts: List[str], iterations: int = 3) -> List[Dict]:
        """Full CoT-enhanced self-improvement loop."""
        results = []
        for prompt in prompts:
            best = None
            for i in range(iterations):
                self.logger.info(f"CoT iteration {i+1}/{iterations}: {prompt[:40]}")
                result = self.generate_with_cot(prompt)
                
                if best is None or result["score"] > best["score"]:
                    best = result
                
                # Store in vector DB for retrieval
                self.vector_db.add([result["response"]], [{"prompt": prompt, "score": str(result["score"])}])
                
                self.history.append(result)
            
            results.append({"prompt": prompt, "best": best})
        
        return results
#============================================================================
#. SYNTHETIC REASONING DATA GENERATION
#=============================
class SyntheticCoTDataGenerator:
    """
    Generates high-quality synthetic CoT training data.
    Uses templates + variation to create reasoning traces.
    """
    def __init__(self, logger, inference_model=None):
        self.logger = logger
        self.inference = inference_model
    
    def generate_math_cot(self, difficulty: str = "medium") -> Dict:
        """Generate synthetic math reasoning examples."""
        templates = {
            "easy": [
                ("What is 15 + 23?", "15 + 23 = 38"),
                ("What is 50 - 17?", "50 - 17 = 33"),
            ],
            "medium": [
                ("What is 45 * 12?", "45 * 12 = 45 * 10 + 45 * 2 = 450 + 90 = 540"),
                ("Solve: 2x + 5 = 13", "2x + 5 = 13, so 2x = 8, thus x = 4"),
            ],
            "hard": [
                ("Solve: x² + 2x - 8 = 0", "(x + 4)(x - 2) = 0, so x = -4 or x = 2"),
            ],
        }
        
        template = random.choice(templates.get(difficulty, templates["medium"]))
        prompt, reasoning = template
        
        return {
            "prompt": prompt,
            "cot_response": f"<think>Let me work through this step by step: {reasoning}</think>\n"
                           f"<answer>{reasoning.split('=')[-1].strip()}</answer>",
            "difficulty": difficulty,
            "domain": "mathematics",
        }
    
    def generate_logic_cot(self) -> Dict:
        """Generate synthetic logic/reasoning examples."""
        scenarios = [
            {
                "prompt": "If all birds have wings, and penguins are birds, do penguins have wings?",
                "reasoning": "1. All birds have wings (given). 2. Penguins are birds (given). "
                            "3. Therefore, penguins have wings (by logical deduction).",
                "answer": "Yes",
            },
            {
                "prompt": "If A implies B, and B implies C, does A imply C?",
                "reasoning": "1. A → B (given). 2. B → C (given). "
                            "3. By transitivity of implication, A → C.",
                "answer": "Yes",
            },
        ]
        
        scenario = random.choice(scenarios)
        return {
            "prompt": scenario["prompt"],
            "cot_response": f"<think>Let me reason through this: {scenario['reasoning']}</think>\n"
                           f"<answer>{scenario['answer']}</answer>",
            "difficulty": "medium",
            "domain": "logic",
        }
    
    def generate_commonsense_cot(self) -> Dict:
        """Generate synthetic commonsense reasoning."""
        scenarios = [
            {
                "prompt": "Why do we need umbrellas when it rains?",
                "reasoning": "1. Rain is water falling from the sky. 2. Water makes things wet. "
                            "3. We want to stay dry. 4. Umbrellas shield us from water. "
                            "5. Therefore, umbrellas help us stay dry in rain.",
                "answer": "To stay dry and protect ourselves from water",
            },
        ]
        
        scenario = random.choice(scenarios)
        return {
            "prompt": scenario["prompt"],
            "cot_response": f"<think>{scenario['reasoning']}</think>\n"
                           f"<answer>{scenario['answer']}</answer>",
            "difficulty": "easy",
            "domain": "commonsense",
        }
    
    def batch_generate(self, count: int = 100) -> List[Dict]:
        """Generate a batch of synthetic CoT data."""
        dataset = []
        generators = [
            (self.generate_math_cot, 0.4),      # 40% math
            (self.generate_logic_cot, 0.3),     # 30% logic
            (self.generate_commonsense_cot, 0.3), # 30% commonsense
        ]
        
        for _ in range(count):
            gen_fn, weight = random.choices(
                [(fn, w) for fn, w in generators],
                weights=[w for _, w in generators],
                k=1
            )[0]
            dataset.append(gen_fn())
        
        self.logger.info(f"Generated {count} synthetic CoT examples")
        return dataset
#===========================================================================
#3. HIGH-QUALITY / SMART DATA PIPELINE
#============================
class SmartDataQualityControl(DatasetQualityControl):
    """
    Enhanced QC that rewards reasoning structure and synthetic data quality.
    """
    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)
        self.reasoning_patterns = {
            "cot_structure": re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL),
            "step_markers": re.compile(r"(?:step|first|second|next|then|therefore|thus|so)\s*[0-9:.]?"),
            "logical_connectors": re.compile(
                r"(?:because|since|as|if|then|implies|leads to|results in|thus|hence|so)"
            ),
        }
    
    def score_reasoning_quality(self, text: str) -> float:
        """Score how well-structured the reasoning is (0-1)."""
        score = 0.0
        
        # Check for CoT structure
        if self.reasoning_patterns["cot_structure"].search(text):
            score += 0.3
        
        # Check for step markers
        steps = len(self.reasoning_patterns["step_markers"].findall(text))
        if steps >= 3:
            score += 0.3
        elif steps >= 1:
            score += 0.1
        
        # Check for logical connectors
        connectors = len(self.reasoning_patterns["logical_connectors"].findall(text))
        if connectors >= 2:
            score += 0.2
        
        # Check for answer structure
        if "<answer>" in text and "</answer>" in text:
            score += 0.2
        
        return min(score, 1.0)
    
    def validate_smart(self, raw_text: str) -> Dict:
        """
        Validate text with emphasis on reasoning quality for synthetic data.
        """
        result = self.validate(raw_text)  # Parent validation
        
        if result["ok"]:
            reasoning_score = self.score_reasoning_quality(raw_text)
            result["reasoning_quality"] = reasoning_score
            result["is_high_quality"] = reasoning_score > 0.5
        
        return result
    
    def batch_validate_smart(self, texts: List[str]) -> List[Dict]:
        """Validate and score a batch of texts."""
        results = []
        for text in texts:
            result = self.validate_smart(text)
            results.append(result)
        return results
#============================================================================
#4. INTEGRATED TRAINING PIPELINE
#=================================================
class HighQualityCoTTrainingPipeline:
    """
    Full pipeline: Generate → Filter → Train → Evaluate → Iterate
    """
    def __init__(self, model_name: str, logger: StructuredLogger):
        self.logger = logger
        self.trainer = EnterpriseTrainer(model_name, logger)
        self.cot_gen = SyntheticCoTDataGenerator(logger)
        self.qc = SmartDataQualityControl(logger)
        self.rl_trainer = None  # Will be initialized
    
    def generate_and_filter_synthetic_data(self, count: int = 500) -> List[str]:
        """
        Step 1: Generate synthetic CoT data and filter for quality.
        """
        self.logger.info(f"Generating {count} synthetic CoT examples...")
        raw_data = self.cot_gen.batch_generate(count)
        
        self.logger.info("Filtering for quality...")
        formatted = [
            d["cot_response"] for d in raw_data
        ]
        
        validated = self.qc.batch_validate_smart(formatted)
        high_quality = [
            v["text"] for v in validated 
            if v["ok"] and v.get("is_high_quality", False)
        ]
        
        quality_rate = len(high_quality) / len(validated)
        self.logger.info(
            f"Quality filter: {len(high_quality)}/{len(validated)} "
            f"({quality_rate*100:.1f}%) passed"
        )
        
        return high_quality
    
    def train_on_synthetic_cot(
        self,
        num_synthetic: int = 500,
        epochs: int = 3,
    ):
        """
        Step 2: Train on high-quality synthetic CoT data.
        """
        self.logger.info("╔════════════════════════════════════╗")
        self.logger.info("║  Training on Synthetic CoT Data    ║")
        self.logger.info("╚════════════════════════════════════╝")
        
        synthetic_texts = self.generate_and_filter_synthetic_data(num_synthetic)
        
        if not synthetic_texts:
            self.logger.error("No high-quality synthetic data generated!")
            return None
        
        dataset = self.trainer.prepare_dataset(synthetic_texts)
        self.trainer.train(dataset, output_dir="./model_synthetic_cot", num_epochs=epochs)
        
        return self.trainer.model
    
    def evaluate_cot_quality(self, test_prompts: List[str]) -> Dict:
        """
        Step 3: Evaluate reasoning quality of generated responses.
        """
        self.logger.info("Evaluating CoT quality...")
        inference = InferencePipeline(self.trainer.model, self.trainer.tokenizer)
        
        results = []
        for prompt in test_prompts:
            cot_prompt = (
                f"Think step by step:\n{prompt}\n\n"
                f"<think>"
            )
            response = inference.generate(cot_prompt, max_length=250)
            response = "<think>" + response + "</think>"
            
            quality = self.qc.score_reasoning_quality(response)
            results.append({
                "prompt": prompt,
                "response": response,
                "reasoning_quality": quality,
            })
        
        avg_quality = sum(r["reasoning_quality"] for r in results) / len(results)
        self.logger.info(f"Average reasoning quality: {avg_quality:.3f}")
        
        return {"results": results, "avg_quality": avg_quality}
    
    def run_full_pipeline(
        self,
        num_synthetic: int = 500,
        train_epochs: int = 3,
        test_prompts: Optional[List[str]] = None,
    ) -> Dict:
        """
        Full end-to-end pipeline.
        """
        if test_prompts is None:
            test_prompts = [
                "What is the capital of France?",
                "Explain photosynthesis.",
                "Solve: 2x + 10 = 30",
            ]
        
        # Step 1: Train on synthetic CoT
        self.train_on_synthetic_cot(num_synthetic, train_epochs)
        
        # Step 2: Evaluate quality
        eval_results = self.evaluate_cot_quality(test_prompts)
        
        return {
            "synthetic_data_count": num_synthetic,
            "eval_quality": eval_results,
            "model_path": "./model_synthetic_cot",
        }
