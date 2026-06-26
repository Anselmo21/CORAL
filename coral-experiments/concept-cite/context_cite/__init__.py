from .context_citer import ContextCiter
from .concept_partitioner import (
    ConceptCriteriaPartitioner,
    CONCEPT_CRITERIA_PROMPT_TEMPLATE,
)

try:
    from .vllm_backend import VLLMContextCiter
    _VLLM_AVAILABLE = True
except ImportError:
    VLLMContextCiter = None  # type: ignore[assignment,misc]
    _VLLM_AVAILABLE = False

__version__ = "0.0.1"
VERSION = __version__

__all__ = [
    "ContextCiter",
    "VLLMContextCiter",
    "ConceptCriteriaPartitioner",
    "CONCEPT_CRITERIA_PROMPT_TEMPLATE",
]
