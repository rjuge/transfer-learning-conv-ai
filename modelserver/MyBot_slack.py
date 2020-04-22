# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
from itertools import chain
import warnings
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parents[1]
sys.path.insert(0,str(BASE_DIR))

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from modelserver.train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from modelserver.utils import download_pretrained_model

from slack.slack_api import getMessages,send_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


class MyBot:
    def __init__(
        self,
        personality,
        min_length=1,
        max_length=20,
        max_history=2,
        temperature=0.7,
        top_k=0,
        top_p=0.9,
        device="cpu",
        slack_token="",
        channelId=""
    ):

        self.min_length = min_length
        self.max_length = max_length
        self.max_history = max_history
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.device = device

        self.slack_token=slack_token
        self.channelId=channelId
        
        self.history = []
        self.load_model_and_tokenizer()
        self.personality = [
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(p)) for p in personality
        ]
        logger.info("Selected personality: %s", self.tokenizer.decode(chain(*self.personality)))

    def _top_filtering(self, logits, threshold=-float("Inf"), filter_value=-float("Inf")):
        """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
                top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                    whose total probability mass is greater than or equal to the threshold top_p.
                    In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                    the threshold top_p.
                threshold: a minimal threshold to keep logits
        """
        assert (
            logits.dim() == 1
        )  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
        top_k = min(self.top_k, logits.size(-1))
        if top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if self.top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > self.top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value

        return logits

    def _sample_sequence(self, current_output=None):
        special_tokens_ids = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        if current_output is None:
            current_output = []

        for i in range(self.max_length):
            instance = build_input_from_segments(
                self.personality, self.history, current_output, self.tokenizer, with_eos=False
            )

            input_ids = torch.tensor(instance["input_ids"], device=self.device).unsqueeze(0)
            token_type_ids = torch.tensor(
                instance["token_type_ids"], device=self.device
            ).unsqueeze(0)

            logits = self.model(input_ids, token_type_ids=token_type_ids)
            if isinstance(logits, tuple):  # for gpt2 and maybe others
                logits = logits[0]
            logits = logits[0, -1, :] / self.temperature
            logits = self._top_filtering(logits)
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1]
            if i < self.min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1:
                        warnings.warn(
                            "Warning: model generating special token with probability 1."
                        )
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)

            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())

        return current_output

    def load_model_and_tokenizer(self):
        pretrained_model_checkpoint = download_pretrained_model()

        logger.info("Get pretrained model and tokenizer")
        tokenizer_class, model_class = (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_model_checkpoint)
        self.model = model_class.from_pretrained(pretrained_model_checkpoint)
        self.model.to(self.device)
        add_special_tokens_(self.model, self.tokenizer)

    def answer(self, text: str):
        self.history.append(self.tokenizer.encode(text))
        with torch.no_grad():
            out_ids = self._sample_sequence()
        out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        return out_text

    def chat(self):
        while True:
            raw_text = getMessages(slack_token=self.slack_token,
                                   channelId=self.channelId)
            while not raw_text:
                raw_text = getMessages(slack_token=self.slack_token,
                                       channelId=self.channelId)
            self.history.append(self.tokenizer.encode(raw_text))
            with torch.no_grad():
                out_ids = self._sample_sequence()
            self.history.append(out_ids)
            self.history = self.history[-(2 * self.max_history + 1) :]
            out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            send_message(out_text,
                         slack_token=self.slack_token,
                         channelId=self.channelId)
            print(out_text)


