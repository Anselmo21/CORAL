"""Generates the curated dataset with Claude

Each of the 60 entries in TOPICS becomes one query with five sources, three to five answers, ten
criteria, and ten pragmas
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

import anthropic

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

MODEL = 'claude-sonnet-4-6'

SYSTEM_PROMPT = """You are a dataset generator for an AI evaluation pipeline that evaluates counterfactual explanation algorithms on open-ended questions answered by LLMs.

Generate a single data point as a JSON object with this exact schema:

{
  "id": <integer>,
  "user_query": <string — a realistic open-ended question a real person would type into an LLM seeking advice>,
  "retrieved_sources": [
    <string — format: "Source N (YEAR, OUTLET, LOCATION): <excerpt that directly addresses the query>">,
    ... exactly 5 sources
  ],
  "plausible_answers": [
    <string — format: "Answer X (SHORT_LABEL): <2-3 sentence answer>">,
    ... 3 to 5 answers
  ],
  "criteria": [
    <string — a short phrase usable in "based on X" context, e.g. "QS rankings", "low upfront cost", "clinical evidence">,
    ... exactly 10 criteria
  ],
  "pragmas": [
    <string — format: "P{N} (CATEGORY): <value>", where CATEGORY is one of: Temporal, Geographic, Perspective, Trustworthiness>,
    ... exactly 10 pragmas
  ]
}

Rules for retrieved_sources:
- Each source string must include: year, outlet name (use real outlets appropriate to domain), geographic location, and a direct excerpt or paraphrase that addresses the query.
- Finance outlets: Investopedia, NerdWallet, Morningstar, WSJ, Reddit r/PersonalFinance, MoneySense, etc.
- Healthcare outlets: Mayo Clinic, WebMD, NEJM, JAMA, CDC, NHS, Reddit r/AskDocs, etc.
- Education outlets: US News, Maclean's, Reddit r/cscareerquestions, 80000Hours, university sites, etc.
- Product outlets: Wirecutter, RTINGS, Consumer Reports, The Verge, Ars Technica, Reddit r/BuyItForLife, etc.
- Sources must have varied stances — not all agreeing with each other.

Rules for plausible_answers:
- 3 to 5 answers with short descriptive labels (e.g. "Answer A (Debt-First Conservative)").
- Answers should represent genuinely different strategies or recommendations.
- No single answer should be obviously best without knowing the active criteria.

Rules for criteria:
- Exactly 10, short and usable as "based on X".
- Mix types: financial (e.g. "low fees"), clinical (e.g. "clinical trial evidence"), practical (e.g. "ease of use"), values-based (e.g. "environmental impact"), scope (e.g. "suitable for beginners"), temporal (e.g. "long-term horizon").
- Must be semantically meaningful for the specific query.

Rules for pragmas:
- Exactly 10, labeled P1–P10.
- Use all 4 categories across the 10 pragmas (Temporal, Geographic, Perspective, Trustworthiness).
- Each pragma should loosely correspond to one or more of the 5 sources — i.e. activating the pragma filters the source pool meaningfully.
- Values should be specific and diverse (e.g. not all "Global" for geographic).

Respond ONLY with a single valid JSON object. No array, no markdown fences, no preamble."""

# The query id, domain, topic, and question type of each query to generate
TOPICS = [
    (
        0,
        'personal finance',
        'index funds vs actively managed funds for a 30-year-old',
        'comparative',
    ),
    (1, 'personal finance', 'paying off student loans vs investing', 'trade-off'),
    (2, 'personal finance', 'whole life vs term life insurance', 'comparative'),
    (3, 'personal finance', 'buying vs renting a home in an expensive city', 'trade-off'),
    (4, 'personal finance', 'Roth IRA vs traditional IRA', 'comparative'),
    (5, 'personal finance', 'how much emergency fund to keep', 'recommendation'),
    (6, 'personal finance', 'paying off mortgage early vs investing the difference', 'trade-off'),
    (7, 'personal finance', 'best high-yield savings account strategy', 'recommendation'),
    (8, 'personal finance', '401k contribution when employer match is low', 'recommendation'),
    (9, 'personal finance', 'consolidating credit card debt', 'recommendation'),
    (10, 'personal finance', 'diversifying into real estate vs stock market', 'comparative'),
    (11, 'personal finance', 'should I use a financial advisor or DIY invest', 'trade-off'),
    (12, 'personal finance', "best strategy for saving for a child's college", 'recommendation'),
    (13, 'personal finance', 'how to invest a $10k windfall', 'recommendation'),
    (14, 'personal finance', 'when to start taking Social Security', 'trade-off'),
    (15, 'healthcare', 'SSRIs vs therapy for mild depression', 'comparative'),
    (16, 'healthcare', 'managing type 2 diabetes through diet vs medication', 'trade-off'),
    (17, 'healthcare', 'knee pain: physical therapy vs surgery', 'comparative'),
    (18, 'healthcare', 'best approach to chronic lower back pain', 'recommendation'),
    (19, 'healthcare', 'melatonin vs prescription sleep aids for insomnia', 'comparative'),
    (20, 'healthcare', 'intermittent fasting for weight loss', 'evaluative'),
    (21, 'healthcare', 'statins for borderline high cholesterol', 'trade-off'),
    (22, 'healthcare', 'managing anxiety without medication', 'recommendation'),
    (23, 'healthcare', 'HPV vaccine for adults over 26', 'evaluative'),
    (
        24,
        'healthcare',
        'treating ADHD in adults: medication vs behavioral strategies',
        'comparative',
    ),
    (25, 'healthcare', 'best treatment for seasonal allergies', 'recommendation'),
    (26, 'healthcare', 'should I get a standing desk for back pain', 'trade-off'),
    (27, 'healthcare', 'managing migraines: preventive vs acute treatment', 'comparative'),
    (28, 'healthcare', 'best diet for lowering blood pressure', 'recommendation'),
    (29, 'healthcare', 'probiotics for gut health: are they worth it', 'evaluative'),
    (30, 'education', 'CS degree vs coding bootcamp for a career change', 'comparative'),
    (31, 'education', 'best MBA program for entrepreneurship', 'recommendation'),
    (32, 'education', 'online vs in-person university for working adults', 'comparative'),
    (33, 'education', 'community college then transfer vs 4-year university', 'trade-off'),
    (34, 'education', 'best undergraduate program for machine learning', 'recommendation'),
    (35, 'education', 'law school vs paralegal certification', 'comparative'),
    (36, 'education', 'self-taught vs formal degree for UX design', 'comparative'),
    (37, 'education', 'best country to study medicine abroad', 'recommendation'),
    (38, 'education', 'part-time vs full-time graduate school', 'trade-off'),
    (39, 'education', 'liberal arts vs vocational training for job prospects', 'trade-off'),
    (40, 'education', 'is a PhD in data science worth it', 'evaluative'),
    (41, 'education', 'best online platform for learning programming', 'recommendation'),
    (
        42,
        'education',
        'choosing between two universities with different rankings and costs',
        'trade-off',
    ),
    (43, 'education', 'should I do a gap year before university', 'trade-off'),
    (44, 'education', 'best certification for a career in cybersecurity', 'recommendation'),
    (45, 'product purchasing', 'best laptop for a college student under $1000', 'recommendation'),
    (46, 'product purchasing', 'MacBook vs Windows laptop for software development', 'comparative'),
    (47, 'product purchasing', 'best budget Android phone in 2024', 'recommendation'),
    (48, 'product purchasing', 'buying a new vs used car', 'trade-off'),
    (49, 'product purchasing', 'best robot vacuum for pet hair', 'recommendation'),
    (50, 'product purchasing', 'air fryer vs convection oven for a small kitchen', 'comparative'),
    (51, 'product purchasing', 'best mattress for back pain', 'recommendation'),
    (52, 'product purchasing', '4K OLED vs QLED TV for a bright living room', 'comparative'),
    (53, 'product purchasing', 'best mechanical keyboard for programming', 'recommendation'),
    (54, 'product purchasing', 'best electric vehicle for a first-time EV buyer', 'recommendation'),
    (
        55,
        'product purchasing',
        'best noise-cancelling headphones for working from home',
        'recommendation',
    ),
    (56, 'product purchasing', 'standing desk: motorized vs manual crank', 'comparative'),
    (57, 'product purchasing', 'best home espresso machine under $500', 'recommendation'),
    (58, 'product purchasing', 'NAS vs cloud storage for home backup', 'comparative'),
    (59, 'product purchasing', 'best smartwatch for fitness tracking', 'recommendation'),
]

PRAGMA_PATTERN = re.compile(r'^P\d+\s*\(([^)]+)\):\s*(.*)$')

PRAGMA_CATEGORIES = ('temporal', 'geographic', 'perspective', 'trustworthiness')


class CuratedDatasetGenerator:
    """Asks Claude for one query at a time and saves the file after each one"""

    def __init__(self, model: str = MODEL, pause: float = 0.5) -> None:
        """Create the Anthropic client and store the request settings

        The client reads ANTHROPIC_API_KEY from the environment

        Args:
            model: The Claude model id
            pause: Seconds to wait between requests
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.pause = pause

    def generate(self, path: Path) -> list[dict[str, Any]]:
        """Generate the queries missing from the file and return every row

        A reply that gets cut off or has a malformed pragma stops the run

        Args:
            path: The dataset file to create or extend

        Returns:
            Every row in the file, old and new
        """
        rows = json.loads(path.read_text(encoding='utf-8'))['rows'] if path.exists() else []
        done = {row['query_id'] for row in rows}
        for query_id, domain, topic, archetype in TOPICS:
            if query_id in done:
                continue
            logger.info('query %d of %d, %s: %s', query_id + 1, len(TOPICS), domain, topic)
            prompt = f"""Generate a single data point.

ID: {query_id}
Domain: {domain}
Topic: {topic}
Archetype: {archetype}

Phrase the query as a real person would ask it — conversational and specific.
Respond ONLY with a single valid JSON object."""
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt}],
            )
            if message.stop_reason != 'end_turn':
                raise RuntimeError(f'query {query_id} stopped with {message.stop_reason!r}')
            text = next(block.text for block in message.content if block.type == 'text').strip()
            if text.startswith('```'):
                text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
            point = json.loads(text)
            pragmas: dict[str, list[str]] = {category: [] for category in PRAGMA_CATEGORIES}
            for pragma in point['pragmas']:
                match = PRAGMA_PATTERN.match(pragma)
                if match is None or match.group(1).strip().lower() not in pragmas:
                    raise ValueError(f'query {query_id} has a malformed pragma {pragma!r}')
                pragmas[match.group(1).strip().lower()].append(match.group(2).strip())
            rows.append(
                {
                    'query_id': query_id,
                    'query': point['user_query'],
                    'sources': '\n\n'.join(point['retrieved_sources']),
                    'criteria': list(point['criteria']),
                    'pragmas': pragmas,
                    'plausible_answers': list(point['plausible_answers']),
                }
            )
            document = {
                'version': 1,
                'description': (
                    'Synthetic curated dataset; queries, sources, criteria, pragmas, '
                    'and plausible answers all generated by Claude Sonnet '
                    '(one API call per data point).'
                ),
                'pragmas_model': f'anthropic:{self.model}',
                'criteria_model': f'anthropic:{self.model}',
                'rows': rows,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(document, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
            )
            time.sleep(self.pause)
        return rows
