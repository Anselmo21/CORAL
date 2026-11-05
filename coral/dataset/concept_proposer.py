"""Asks an LLM for the candidate answers and the concepts of a user's question"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coral.llm.language_model import LanguageModel

logger = logging.getLogger(__name__)

MAXIMUM_CONCEPTS = 10

SYSTEM_PROMPT = """You propose the candidate answers and the criteria for explaining how an LLM answers an open-ended question.

Given the user's question, reply with a single JSON object with this exact schema:

{
  "plausible_answers": [
    <string, format "Answer X (SHORT_LABEL): <2-3 sentence answer>">,
    ... 3 to 5 answers
  ],
  "criteria": [
    <string, a short phrase usable in "based on X" context, e.g. "low upfront cost", "clinical evidence">,
    ... exactly 10 criteria
  ]
}

Rules for plausible_answers:
- 3 to 5 answers with short descriptive labels (e.g. "Answer A (Debt-First Conservative)").
- Answers should represent genuinely different strategies or recommendations.
- No single answer should be obviously best without knowing the active criteria.

Rules for criteria:
- Exactly 10, short and usable as "based on X".
- Each criterion is an assumption an answer could depend on that the question does not state.
- Mix kinds of considerations (practical, financial, values-based, scope, temporal) where they fit the question.
- Must be semantically meaningful for the specific question.

Respond ONLY with a single valid JSON object. No markdown fences, no preamble."""

FENCE = re.compile(r'```(?:json)?\s*(.*?)```', re.DOTALL)


def parse_proposal(response: str) -> tuple[list[str], list[str]] | None:
    """Read the answers and criteria out of a reply or return None

    Only the first ten criteria are kept

    Args:
        response: The raw reply text

    Returns:
        The answers and the criteria, or None if the reply is not the expected JSON
    """
    text = (response or '').strip()
    fenced = FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end <= start:
        return None
    try:
        proposal = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    answers = proposal.get('plausible_answers') if isinstance(proposal, dict) else None
    criteria = proposal.get('criteria') if isinstance(proposal, dict) else None
    if not isinstance(answers, list) or not isinstance(criteria, list):
        return None
    answers = list(dict.fromkeys(str(answer).strip() for answer in answers if str(answer).strip()))
    criteria = list(dict.fromkeys(str(entry).strip() for entry in criteria if str(entry).strip()))
    if len(answers) < 2 or not criteria:
        return None
    return answers, criteria[:MAXIMUM_CONCEPTS]


class ConceptProposer:
    """Proposes the candidate answers and the criteria of one question with any LLM backend

    A reply that is not the expected JSON gets asked again up to the attempt limit
    """

    def __init__(
        self,
        model: LanguageModel,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_attempts: int = 3,
    ) -> None:
        """Store the LLM and its generation settings

        Args:
            model: The LLM backend
            temperature: The sampling temperature
            max_tokens: The token cap per response
            max_attempts: The most responses to request
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts

    def propose(self, question: str) -> dict[str, Any]:
        """Return the question with its proposed answers and concepts

        The pinned list starts empty for the user to fill in before the search

        Args:
            question: The user's question

        Returns:
            The question, its answers, its concepts, and its pinned concepts
        """
        conversation = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f'Question: {question}'},
        ]
        for attempt in range(1, self.max_attempts + 1):
            reply = self.model.complete([conversation], self.temperature, self.max_tokens)[0]
            proposal = parse_proposal(reply.text)
            if proposal is not None:
                answers, concepts = proposal
                return {
                    'question': question,
                    'answers': answers,
                    'concepts': concepts,
                    'pinned': [],
                }
            logger.warning('reply %d of %d did not parse', attempt, self.max_attempts)
        raise RuntimeError(f'no parseable proposal after {self.max_attempts} replies')
