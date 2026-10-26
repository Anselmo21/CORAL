"""ContextCite adapted to remove concepts instead of source text"""

from __future__ import annotations

from coral.surrogate.concept_prompt import ConceptPrompt
from coral.surrogate.context_citer import ContextCiter
from coral.surrogate.lasso_surrogate import LassoSurrogate

__all__ = [
    'ConceptPrompt',
    'ContextCiter',
    'LassoSurrogate',
]
