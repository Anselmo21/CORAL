"""ContextCite from the Madry Lab code adapted to ablate concepts"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForCausalLM, AutoTokenizer

from coral.surrogate.concept_prompt import ConceptPrompt
from coral.surrogate.lasso_surrogate import LassoSurrogate

logger = logging.getLogger(__name__)


def create_mask(count: int, keep_probability: float, seed: int) -> NDArray[np.bool_]:
    """Draw which concepts one ablation keeps

    The seed is the ablation's index

    Args:
        count: The number of concepts
        keep_probability: The chance each concept is kept
        seed: The seed of this mask

    Returns:
        The boolean mask over the concepts
    """
    return np.random.RandomState(seed).choice(
        [False, True], size=count, p=[1 - keep_probability, keep_probability]
    )


def aggregate_logit_probabilities(token_logit_probabilities: NDArray) -> NDArray:
    """Combine per-token logit probabilities into one per response

    A logit probability is log(p / (1 - p)) and a response's p is the product of its tokens' p

    Args:
        token_logit_probabilities: The logit probabilities, one row per response and one column per
            token

    Returns:
        One logit probability per response
    """
    values = torch.tensor(token_logit_probabilities)
    log_probabilities = torch.nn.functional.logsigmoid(values).sum(dim=1)
    return (log_probabilities - torch.log1p(-torch.exp(log_probabilities))).numpy()


class ContextCiter:
    """Fits a linear surrogate of the response on random concept ablations

    The full prompt's response is appended to every ablated prompt to score each response token
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompt: ConceptPrompt,
        ablations: int,
        keep_probability: float = 0.5,
        batch_size: int = 1,
        max_new_tokens: int = 512,
    ) -> None:
        """Store the model and the ablation settings

        A tokenizer with no pad token borrows its end token because batched scoring has to pad

        Args:
            model: The loaded LLM
            tokenizer: The LLM's tokenizer
            prompt: The prompt to ablate
            ablations: The random ablations to score
            keep_probability: The chance an ablation keeps each concept
            batch_size: The ablations per forward pass
            max_new_tokens: The token cap of the response
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.ablations = ablations
        self.keep_probability = keep_probability
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.usage = {'prompt_tokens': 0, 'output_tokens': 0, 'llm_calls': 0}
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def load_model(model_id: str) -> tuple[Any, Any]:
        """Load a Hugging Face model in half precision

        The tokenizer pads on the left so each response ends its row in a batch

        Args:
            model_id: The Hugging Face model id

        Returns:
            The model and its tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map='auto'
        )
        tokenizer.padding_side = 'left'
        return model, tokenizer

    def prompt_ids(self, mask: NDArray[np.bool_] | None = None) -> list[int]:
        """Return the token ids of the prompt with the mask's concepts

        No mask means the full prompt

        Args:
            mask: Which concepts to keep

        Returns:
            The prompt's token ids
        """
        return self._chat_prompt(mask)[0]

    def fit(self) -> LassoSurrogate:
        """Generate the response, score it under every ablation, and fit the surrogate

        Each score has the score of the prompt with no concepts subtracted

        Returns:
            The fitted surrogate
        """
        prompt_ids, prompt_text = self._chat_prompt()
        output = prompt_text + self._generate(prompt_ids)
        output_tokens = self.tokenizer(output, add_special_tokens=False)
        response_ids = output_tokens['input_ids'][len(prompt_ids) :]

        def token_at(character: int) -> int:
            """Return the token that holds a character of the output

            Only start offsets get compared because a token's span can be empty

            Args:
                character: The character's position in the output

            Returns:
                The token's index
            """
            for index in range(len(output_tokens['input_ids']) - 1):
                if character < output_tokens.token_to_chars(index + 1).start:
                    return index
            return len(output_tokens['input_ids']) - 1

        response_offset = output_tokens.token_to_chars(len(prompt_ids)).start
        if len(output) <= response_offset:
            raise ValueError('the LLM gave an empty response to attribute')
        start = token_at(response_offset) - len(prompt_ids)
        end = token_at(len(output) - 1) + 1 - len(prompt_ids)

        concept_count = len(self.prompt.concepts)
        masks = np.zeros((self.ablations, concept_count), dtype=bool)
        for seed in range(self.ablations):
            masks[seed] = create_mask(concept_count, self.keep_probability, seed)
        scores = self._score(masks, response_ids)
        reference = self._score(np.zeros((1, concept_count), dtype=bool), response_ids)
        outputs = aggregate_logit_probabilities(scores[:, start:end])
        outputs = outputs - aggregate_logit_probabilities(reference[:, start:end])
        surrogate = LassoSurrogate()
        surrogate.fit(masks, outputs, end - start)
        return surrogate

    def _chat_prompt(self, mask: NDArray[np.bool_] | None = None) -> tuple[list[int], str]:
        """Return the prompt's token ids and text in the chat template

        The text is encoded without special tokens because the template already added them

        Args:
            mask: Which concepts to keep

        Returns:
            The token ids and the text
        """
        messages = [{'role': 'user', 'content': self.prompt.render(mask)}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer.encode(text, add_special_tokens=False), text

    def _generate(self, prompt_ids: list[int]) -> str:
        """Generate the response to the full prompt

        Sampling is off so the response stays fixed while the ablations get scored

        Args:
            prompt_ids: The prompt's token ids

        Returns:
            The response text
        """
        input_ids = torch.tensor([prompt_ids], device=self.model.device)
        output_ids = self.model.generate(
            input_ids, max_new_tokens=self.max_new_tokens, do_sample=False
        )[0]
        self.usage['prompt_tokens'] += len(prompt_ids)
        self.usage['output_tokens'] += len(output_ids) - len(prompt_ids)
        self.usage['llm_calls'] += 1
        # Decoding can alter the prompt text so cut at the decoded length
        decoded_prompt = self.tokenizer.decode(prompt_ids)
        return self.tokenizer.decode(output_ids)[len(decoded_prompt) :]

    def _score(self, masks: NDArray[np.bool_], response_ids: list[int]) -> NDArray:
        """Return each response token's logit probability under each mask

        Args:
            masks: One mask per ablation
            response_ids: The response's token ids

        Returns:
            The logit probabilities, one row per mask and one column per response token
        """
        length = len(response_ids)
        rows = []
        for first in range(0, len(masks), self.batch_size):
            sequences = [
                self.prompt_ids(mask) + response_ids
                for mask in masks[first : first + self.batch_size]
            ]
            width = max(len(sequence) for sequence in sequences)
            # Left padding keeps every response at the end of its row
            input_ids = torch.tensor(
                [
                    [self.tokenizer.pad_token_id] * (width - len(sequence)) + sequence
                    for sequence in sequences
                ],
                device=self.model.device,
            )
            attention_mask = torch.tensor(
                [[0] * (width - len(sequence)) + [1] * len(sequence) for sequence in sequences],
                device=self.model.device,
            )
            device_type = self.model.device.type
            with torch.no_grad(), torch.autocast(device_type, enabled=device_type == 'cuda'):
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits = logits[:, -(length + 1) : -1].reshape(-1, logits.shape[-1])
            labels = torch.tensor(response_ids, device=logits.device).repeat(len(sequences))
            chosen = logits.gather(-1, labels[:, None])[:, 0]
            others = logits.clone().scatter_(-1, labels[:, None], -torch.inf).logsumexp(dim=-1)
            rows.append((chosen - others).reshape(len(sequences), length).float().cpu().numpy())
            self.usage['prompt_tokens'] += sum(len(sequence) for sequence in sequences)
            self.usage['llm_calls'] += len(sequences)
        return np.concatenate(rows).astype(np.float32)
