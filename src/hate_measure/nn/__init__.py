"""Neural network modules for hate speech scoring."""

from .config import HateSpeechScorerConfig
from .scorer import HateSpeechScorer

__all__ = ["HateSpeechScorerConfig", "HateSpeechScorer"]
