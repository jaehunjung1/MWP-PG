from argparse import Namespace

import ipdb
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import *
from transformers import AutoTokenizer, AutoConfig, BartForConditionalGeneration


def load_tokenizer_and_model(model_class: nn.Module, args: Namespace):
    if model_class == BartForConditionalGeneration:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    else:
        raise NotImplementedError

    return tokenizer, model
