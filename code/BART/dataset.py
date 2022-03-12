import itertools
import random
from dataclasses import dataclass, field
from typing import *

import ipdb
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import jsonlines


@dataclass
class MWPExample:
    """
    Each sample in MWP datasets - Question / Equation (both with numbers abstracted to "number0", "number1", ...)
    """
    question: str
    equation: str
    numbers: List[float]
    answer: float
    positive_list: List[str]
    negative_list: List[str]

    @staticmethod
    def from_json(json_data: dict, with_contrastive: bool = False):
        question = json_data['question']
        equation = json_data['equation']
        numbers = json_data['numbers']
        answer = json_data['answer']

        if with_contrastive:
            positive_list = json_data['positive_list']
            positive_list.insert(0, equation)
            positive_list = list(OrderedDict.fromkeys(positive_list))[:5]
            negative_list = json_data['negative_list'][:5]
            negative_list = negative_list + [negative_list[-1]] * (5 - len(negative_list))  # ensure at least 5 E_n
        else:
            positive_list = None
            negative_list = None

        return MWPExample(
            question=question,
            equation=equation,
            numbers=numbers,
            answer=answer,
            positive_list=positive_list,
            negative_list=negative_list
        )

    def get_question_and_equation(self) -> tuple:
        return self.question, self.equation

    def prepare_negative_pairs(self):
        return [(self.question, negative_equation) for negative_equation in self.negative_list]


class MWPDataset(Dataset):
    tokenizer: PreTrainedTokenizerFast = None

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            example_list: List[MWPExample],
    ):
        MWPDataset.tokenizer = tokenizer

        self.example_list = example_list

    def __len__(self):
        return len(self.example_list)

    def __iter__(self):
        return iter(self.example_list)

    def __getitem__(self, idx):
        return self.example_list[idx]

    @staticmethod
    def from_file(
            tokenizer: PreTrainedTokenizerFast,
            filename: str,
            **kwargs,
    ):
        with jsonlines.open(filename, 'r') as f:
            example_list = [MWPExample.from_json(data) for data in f]

        return MWPDataset(tokenizer, example_list)

    @staticmethod
    def collate_fn(batched_examples: List[MWPExample]):
        batch_question_and_equation = [example.get_question_and_equation() for example in batched_examples]

        batch_question = [sample[0] for sample in batch_question_and_equation]
        question_encoding = MWPDataset.tokenizer(batch_question, return_tensors='pt', padding=True)

        batch_equation = [sample[1] for sample in batch_question_and_equation]
        equation_encoding = MWPDataset.tokenizer(batch_equation, return_tensors='pt', padding=True)

        batch_numbers = [example.numbers for example in batched_examples]

        batch_answer = [example.answer for example in batched_examples]

        return {
            'Q_input_ids': question_encoding.input_ids,
            'Q_attention_mask': question_encoding.attention_mask,
            'E_input_ids': equation_encoding.input_ids,
            'numbers': batch_numbers,
            'answer': batch_answer,
        }
