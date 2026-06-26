import json
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .context_partitioner import BaseContextPartitioner

# Use with ContextCiter(prompt_template=..., query="") so the entire prompt is
# built by the partitioner (recommended).
CONCEPT_CRITERIA_PROMPT_TEMPLATE = "{context}"


class ConceptCriteriaPartitioner(BaseContextPartitioner):
    """
    Partition the *criteria* (concepts) for ContextCite-style ablation.

    Each concept is one mask coordinate. Sources are never ablated: the
    ``sources_block`` is included verbatim in every ``get_context`` result.
    Present concepts are joined with ``\", \"``; omitted concepts are left out
    entirely (empty criteria section when no concept is kept).
    """

    _INTRO = (
        "You are solving a multiple-choice question using only the provided sources as context.\n"
        "Choose exactly one option from ANSWER OPTIONS based on the provided criteria.\n"
        "The criteria and answers do not follow any particular order of importance.\n"
        "Weight the criteria equally when forming your answer.\n"
        "Output must be the exact option text only (no numbering, no punctuation changes).\n"
        "Do not include any explanation, reasoning, or extra words.\n\n"
    )

    def __init__(
        self,
        question: str,
        concepts: List[str],
        sources_block: str,
        answers: Optional[List[str]] = None,
    ) -> None:
        q = question.strip()
        cc = [c.strip() for c in concepts]
        aa = [a.strip() for a in (answers or [])]
        self.question = q
        self._concepts = cc
        self.sources_block = sources_block
        self.answers = aa
        super().__init__(self.canonical_context(q, cc, sources_block, aa))

    @staticmethod
    def canonical_context(
        question: str,
        concepts: List[str],
        sources_block: str,
        answers: Optional[List[str]] = None,
    ) -> str:
        """
        Stable string to pass as ``ContextCiter(context=..., ...)``; must match
        the partitioner's ``context`` attribute.
        """
        payload = {
            "question": question.strip(),
            "concepts": [c.strip() for c in concepts],
            "sources_block": sources_block,
            "answers": [a.strip() for a in (answers or [])],
        }
        return "__context_cite_concepts__:" + json.dumps(
            payload, sort_keys=True, ensure_ascii=False
        )

    def split_context(self) -> None:
        return None

    @property
    def num_sources(self) -> int:
        return len(self._concepts)

    def get_source(self, index: int) -> str:
        return self._concepts[index]

    def get_context(self, mask: Optional[NDArray] = None) -> str:
        if mask is None:
            mask = np.ones(self.num_sources, dtype=bool)
        kept = [self._concepts[i] for i in range(self.num_sources) if mask[i]]
        criteria_line = ", ".join(kept)
        answers_lines = (
            "\n".join(f"{i + 1}. {answer}" for i, answer in enumerate(self.answers))
            if self.answers
            else "N/A"
        )
        return (
            f"{self._INTRO}"
            f"QUESTION\n{self.question}\n\n"
            f"ANSWER OPTIONS\n{answers_lines}\n\n"
            f"CRITERIA\n{criteria_line}\n\n"
            f"SOURCES\n{self.sources_block}"
        )
