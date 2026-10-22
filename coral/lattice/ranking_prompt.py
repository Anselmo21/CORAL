"""Builds the prompts that ask the LLM to rank the answers and parses its replies"""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import permutations

SYSTEM_MESSAGE = (
    'You are a helpful assistant that ranks items as JSON arrays. '
    'Follow formatting instructions exactly.'
)


"""Prompts"""


def criteria_prompt(
    question: str, criteria: Sequence[str], sources: str, answers: Sequence[str]
) -> str:
    """Build the prompt that ranks the answers under the given criteria

    An empty subset still gets a prompt, with the criteria marked as none specified

    Args:
        question: The query text
        criteria: The criteria active at this node
        sources: The retrieved sources
        answers: The candidate answers in the order of the indices the LLM returns

    Returns:
        The prompt text
    """
    criteria_text = f'{", ".join(criteria)}.' if criteria else 'None specified.'
    n = len(answers)
    options_block = '\n'.join(f'{index}. {answer}' for index, answer in enumerate(answers))
    return f"""
    You are an answering machine.
    Rank the options based on the following question using the following sources and question criteria, if provided.
    Weigh all criteria equally when ranking the options.
    The options are not provided in any order of importance, consider them all when answering the question.
    The answer ranked first is the best answer to the question and should be placed at the beginning of the list.

    Return a JSON array of exactly {n} integers — the 0-based indices of the options from OPTIONS, ordered best answer first.
    The array must be a permutation of the indices 0 to {n - 1}: include every index exactly once, with no duplicates and none missing.
    Output only the index numbers, not the option text.
    Do not include any explanation or additional text, just the JSON array.

    QUESTION:\n
    {question}

    SOURCES:\n
    {sources}

    CRITERIA:\n
    {criteria_text}

    OPTIONS (rank these {n} options by their index number):\n
    {options_block}

    ANSWER (JSON array of exactly {n} integers, a permutation of 0 to {n - 1}, best answer first):\n"""


def pragma_prompt(
    question: str, pragmas: Sequence[str], sources: str, answers: Sequence[str]
) -> str:
    """Build the prompt that ranks the answers under the given pragmas

    The wording favors sources that match because pragmas filter the sources instead of weighing
    the answers

    Args:
        question: The query text
        pragmas: The pragmas active at this node
        sources: The retrieved sources
        answers: The candidate answers in the order of the indices the LLM returns

    Returns:
        The prompt text
    """
    directives = ' AND '.join(pragmas) if pragmas else 'NO SOURCE CRITERIA'
    n = len(answers)
    options_block = '\n'.join(f'{index}. {answer}' for index, answer in enumerate(answers))
    return f"""
    You are an answering machine.
    Rank the options to the following question using only the following sources.
    Heavily prioritize sources that match the provided source directives when answering the question.
    Source directives can be temporal (e.g., specific year or date range), geographic (e.g., a location or region), trustworthy (e.g., official, expert, user-generated), or perspective (e.g., neutral, opinionated).
    If no source directives are provided, all sources will be considered equally.
    The options are not provided in any order of importance, consider them all when answering the question.
    The answer ranked first is the best answer to the question and should be placed at the beginning of the list.

    Return a JSON array of exactly {n} integers — the 0-based indices of the options from OPTIONS, ordered best answer first.
    The array must be a permutation of the indices 0 to {n - 1}: include every index exactly once, with no duplicates and none missing.
    Output only the index numbers, not the option text.
    Do not include any explanation or additional text, just the JSON array.

    QUESTION:\n
    {question}

    SOURCES:\n
    {sources}

    SOURCE DIRECTIVES:\n
    {directives}

    OPTIONS (rank these {n} options by their index number):\n
    {options_block}

    ANSWER (JSON array of exactly {n} integers, a permutation of 0 to {n - 1}, best answer first):\n"""


def ranking_prompt(
    concept_set: str, question: str, concepts: Sequence[str], sources: str, answers: Sequence[str]
) -> str:
    """Build the ranking prompt for a concept set

    Anything other than 'pragmas' falls through to the criteria prompt

    Args:
        concept_set: Either 'criteria' or 'pragmas'
        question: The query text
        concepts: The concepts active at this node
        sources: The retrieved sources
        answers: The candidate answers

    Returns:
        The prompt text
    """
    if concept_set == 'pragmas':
        return pragma_prompt(question, concepts, sources, answers)
    return criteria_prompt(question, concepts, sources, answers)


def correction_prompt(prompt: str, previous_answer: str) -> str:
    """Repeat a prompt after an unparsable answer

    Args:
        prompt: The original prompt
        previous_answer: The answer that failed to parse

    Returns:
        The original prompt followed by the correction request
    """
    return (
        f'{prompt}\n\n'
        'Your previous answer was invalid.\n'
        'Reason: Not a strict JSON array permutation of the option indices 0..N-1.\n'
        'Previous answer:\n'
        f'{(previous_answer or "").strip()}\n\n'
        'Now return a corrected answer that strictly follows all output rules.\n'
        'Return ONLY the JSON array.'
    )


"""Answer parsing"""


def permutation_regex(count: int) -> str:
    """Return a regex that only matches a JSON permutation of 0 to count - 1

    The regex allows spaces and newlines because forcing a single line can change the top answer

    Args:
        count: The number of answers

    Returns:
        An alternation with one branch per ordering
    """
    gap = r'[ \n]{0,4}'
    return '|'.join(
        r'\[' + gap + (gap + ',' + gap).join(map(str, order)) + gap + r'\]'
        for order in permutations(range(count))
    )


def parse_ranking(response: str, answers: Sequence[str]) -> list[str] | None:
    """Read a ranking from a reply or return None if it is not a permutation

    Replies can use indices or the answer text inside code fences or parentheses

    Args:
        response: The raw reply text
        answers: The candidate answers in prompt order

    Returns:
        The answers best first, or None if the reply does not use each one exactly once
    """
    text = (response or '').strip()
    if text.startswith('```'):
        lines = text.split('\n')
        text = '\n'.join(lines[1:-1]).strip() if len(lines) > 2 else text
        if text.startswith('json'):
            text = text[4:].strip()
    text = text.strip()
    if text.startswith('(') and text.endswith(')') and text[1:-1].strip().startswith('['):
        text = text[1:-1].strip()
    if text.startswith('[') and not text.endswith(']'):
        text += ']'
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None

    def index(item: object) -> int | None:
        """Read an item as an index and reject booleans

        Booleans have to be caught first because Python treats True as the integer 1

        Args:
            item: One element of the decoded array

        Returns:
            The integer it stands for, or None if it is not an index
        """
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            return item
        if isinstance(item, str) and item.strip().lstrip('-').isdigit():
            return int(item.strip())
        return None

    indices = [index(item) for item in items]
    if all(position is not None for position in indices):
        if len(indices) == len(answers) and sorted(indices) == list(range(len(answers))):
            return [answers[position] for position in indices]
        return None
    ranked = [str(item).strip() for item in items]
    if (
        len(ranked) != len(answers)
        or len(set(ranked)) != len(ranked)
        or set(ranked) != set(answers)
    ):
        return None
    return ranked
