"""M.K SETUP - Enterprise AI (modular package)."""
from .deps import *
from .logger import StructuredLogger
from .embeddings import RealEmbeddingModel, RealVectorDatabase
from .knowledge import KnowledgeGraph
from .vision import VisionAgent
from .speech import SpeechIO
from .browser import BrowserAgent
from .gui import GUIController
from .computer_use import ComputerUseAgent
from .sandbox import SecureSandbox
from .agents import (
    AgentMessage, MessageBus, BaseAgent, ResearchAgent, CodingAgent,
    ReasoningAgent, VisionAgentWorker, MultiAgentOrchestrator,
)
from .quality import DatasetQualityControl
from .training import (
    GatingNetworkTopK, SparseMoE, MoEFFNWrapper, CustomTrainer,
    compute_metrics, TrainingCallbacks, InferencePipeline, EnterpriseTrainer,
)
from .pretraining import PDFCorpusBuilder, WebCorpusBuilder, FullPreTrainer
from .rl import RewardModel, RLTrainer, SelfImprovementLoop
from .cot import (
    SmartDataQualityControl,
    SelfImprovementLoopWithCoT, SyntheticCoTDataGenerator,
    HighQualityCoTTrainingPipeline,
)
from .rlhf import (
    PreferenceDataset, ComparisonRewardModel, SFTTrainer_,
    PPOTrainer_, DPOTrainer_, RLHFPipeline,
)
from .tools import calc_tool, python_tool, web_search_tool, TOOLS

__all__ = [
    "StructuredLogger",
    "RealEmbeddingModel", "RealVectorDatabase",
    "KnowledgeGraph",
    "VisionAgent", "SpeechIO", "BrowserAgent", "GUIController",
    "ComputerUseAgent", "SecureSandbox",
    "AgentMessage", "MessageBus", "BaseAgent", "ResearchAgent", "CodingAgent",
    "ReasoningAgent", "VisionAgentWorker", "MultiAgentOrchestrator",
    "DatasetQualityControl", "SmartDataQualityControl",
    "GatingNetworkTopK", "SparseMoE", "MoEFFNWrapper", "CustomTrainer",
    "compute_metrics", "TrainingCallbacks", "InferencePipeline", "EnterpriseTrainer",
    "PDFCorpusBuilder", "WebCorpusBuilder", "FullPreTrainer",
    "RewardModel", "RLTrainer", "SelfImprovementLoop",
    "SelfImprovementLoopWithCoT", "SyntheticCoTDataGenerator",
    "HighQualityCoTTrainingPipeline",
    "PreferenceDataset", "ComparisonRewardModel", "SFTTrainer_",
    "PPOTrainer_", "DPOTrainer_", "RLHFPipeline",
    "calc_tool", "python_tool", "web_search_tool", "TOOLS",
]
