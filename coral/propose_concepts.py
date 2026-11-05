"""Asks an LLM for the candidate answers and concepts of a question and saves them for editing"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from coral.dataset.concept_proposer import ConceptProposer
from coral.utilities.arguments import add_model_arguments, create_model
from coral.utilities.files import write_json


def main() -> None:
    """Parse the command line, propose the answers and concepts, and write the file

    An existing file at the output path gets replaced by the new proposal
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-7s  %(message)s')
    parser = argparse.ArgumentParser(
        description='Propose the candidate answers and concepts of a question'
    )
    parser.add_argument('question', help='the question to explain')
    parser.add_argument('--output', type=Path, required=True, help='JSON file to write')
    add_model_arguments(parser)
    arguments = parser.parse_args()

    proposer = ConceptProposer(
        create_model(arguments), arguments.temperature, arguments.max_tokens, arguments.max_attempts
    )
    proposal = proposer.propose(arguments.question)
    write_json(arguments.output, proposal)
    print(f'Answers in {arguments.output}')
    for answer in proposal['answers']:
        print(f'  {answer}')
    print('Concepts')
    for concept in proposal['concepts']:
        print(f'  {concept}')


if __name__ == '__main__':
    main()
