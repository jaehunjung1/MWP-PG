import math
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime

import ipdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, BartForConditionalGeneration, PreTrainedTokenizerFast

from model import load_tokenizer_and_model
from config import CONFIGS
from util import *


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    args: Namespace, epoch: int, total_length: int, batch_global_count: int):
    model.train()

    total_loss = 0.
    total_generation_loss = 0.
    total_contrastive_loss = 0.
    total_count = 0
    batch_epoch_count = 0

    with tqdm(dataloader, desc=f"Train Ep {epoch}", total=total_length) as tq:
        for batch in tq:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            batch_epoch_count += 1

            generation_labels = batch['E_input_ids'].masked_fill(
                batch['E_input_ids'] == model.config.pad_token_id,
                -100
            )

            output = model(
                input_ids=batch['Q_input_ids'],
                attention_mask=batch['Q_attention_mask'],
                labels=generation_labels,
            )
            loss = output.loss

            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if (batch_global_count + batch_epoch_count) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            loss = loss.item()
            total_loss += loss * torch.sum(batch['E_input_ids'].ne(model.config.pad_token_id))
            total_count += torch.sum(batch['E_input_ids'].ne(model.config.pad_token_id))

            tq.set_postfix({"Loss": f"{(total_loss / total_count):.3}"})

            if loss != loss:
                print(f"Nan came up at epoch {epoch}.")
                ipdb.set_trace()

    return batch_epoch_count


def evaluate(model: nn.Module, dataloader: DataLoader, args: Namespace, total_length: int,
             tokenizer: PreTrainedTokenizerFast = None):
    model.eval()

    total_num_correct_syntax = 0.
    total_num_correct_equation_equivalence = 0.
    total_count = 0

    with tqdm(dataloader, desc='Dev', total=total_length) as tq:
        for batch in tq:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            output = model.generate(
                input_ids=batch['Q_input_ids'],
                attention_mask=batch['Q_attention_mask'],
                max_length=batch['Q_input_ids'].size(0) + args.max_new_tokens,
                num_beams=args.num_beams,
                early_stopping=True,
            )
            output_str_list = tokenizer.batch_decode(output, skip_special_tokens=True)
            gold_str_list = tokenizer.batch_decode(batch['E_input_ids'], skip_special_tokens=True)

            inferred_equations = [normalize_equation(equation) for equation in output_str_list]
            gold_equations = [normalize_equation(equation) for equation in gold_str_list]

            num_correct_syntax = 0
            num_correct_equation_equivalence = 0
            num_correct_answer = 0
            for inferred_equation, gold_equation, numbers in zip(inferred_equations, gold_equations,
                                                                 batch['numbers']):
                if gold_equation is None:
                    num_correct_syntax += 1
                    num_correct_equation_equivalence += 1
                    num_correct_answer += 1
                else:
                    if inferred_equation is not None:
                        num_correct_syntax += 1
                        num_correct_equation_equivalence += equation_equivalence(inferred_equation, gold_equation)

            total_num_correct_syntax += num_correct_syntax
            total_num_correct_equation_equivalence += num_correct_equation_equivalence
            total_count += batch['E_input_ids'].size(0)

            tq.set_postfix({"Acc": f"{(total_num_correct_equation_equivalence / total_count):.3}"})

    print("Syntax: ", total_num_correct_syntax / total_count)
    print("Equation Equivalence: ", total_num_correct_equation_equivalence / total_count)

    return (total_num_correct_syntax / total_count,
            total_num_correct_equation_equivalence / total_count)


def main(args):
    set_seed(args.seed)

    best_equation_equivalence_list = []
    for file_idx, (train_filename, dev_filename) in enumerate(zip(args.train_filename, args.dev_filename)):

        tokenizer, model = load_tokenizer_and_model(args.model_class, args)
        model = model.to(args.device)

        train_dataset = args.dataset_class.from_file(tokenizer, train_filename, is_train=True)
        dev_dataset = args.dataset_class.from_file(tokenizer, dev_filename, is_train=False)
        train_collate_fn = lambda x: train_dataset.collate_fn(x)
        dev_collate_fn = lambda x: dev_dataset.collate_fn(x)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                      collate_fn=train_collate_fn, num_workers=4)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False,
                                    collate_fn=dev_collate_fn, num_workers=4)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)

        if args.test:
            load_checkpoint(args.ckpt, model, optimizer)
            syntax_acc, equation_equivalence = evaluate(
                model=model,
                dataloader=dev_dataloader,
                args=args,
                total_length=math.ceil(len(dev_dataset) / float(args.dev_batch_size)),
                tokenizer=tokenizer,
            )
            print(f"Result for {dev_filename}")
            print(f"Syntax Acc: {syntax_acc:.5} / Equation Equivalence: {equation_equivalence:.5}")
        else:
            os.makedirs(args.log_dir, exist_ok=True)
            if args.save_ckpt:
                os.makedirs(args.ckpt_dir, exist_ok=True)

            best_equation_equivalence = 0
            batch_global_count = 0

            if file_idx == 0:
                log_and_save_checkpoint(args)

            for epoch in range(1, args.num_epochs + 1):
                batch_epoch_count = train_one_epoch(
                    model=model,
                    dataloader=train_dataloader,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    total_length=math.ceil(len(train_dataset) / float(args.train_batch_size)),
                    batch_global_count=batch_global_count,
                )
                batch_global_count += batch_epoch_count

                syntax_acc, equation_equivalence = evaluate(
                    model=model,
                    dataloader=dev_dataloader,
                    args=args,
                    total_length=math.ceil(len(dev_dataset) / float(args.dev_batch_size)),
                    tokenizer=tokenizer,
                )

                log_and_save_checkpoint(args, model, optimizer, epoch,
                                        best_equation_equivalence, equation_equivalence, dev_filename)

                if equation_equivalence > best_equation_equivalence:
                    best_equation_equivalence = equation_equivalence

                print(f"Best Dev Equation Equivalence: {best_equation_equivalence:.5}")

            best_equation_equivalence_list.append(best_equation_equivalence)

    if not args.test:
        print(f"Final Equation Equivalence List: {best_equation_equivalence_list}")
        print(f"Mean Equation Equivalence: {sum(best_equation_equivalence_list) / len(best_equation_equivalence_list)}")


def parse_args():
    args = ArgumentParser(description="MWP solution generation")

    # Experiments
    args.add_argument('--results_dir', default='results', type=str, help="Results directory.")
    args.add_argument('--run', default=datetime.now().strftime("%y%m%d_%H%M"), type=str, help="Results directory.")
    args.add_argument('--device_id', type=int, help="Device to use (0 ~ 7).")
    args.add_argument('--seed', default=999, type=int, help="Random seed.")
    args.add_argument('--test', action='store_true', help="Inference mode.")
    args.add_argument('--ckpt', default=None, type=str, help="Checkpoint file name.")
    args.add_argument('--save_ckpt', action='store_true', help="Save best checkpoint or not.")

    # Dataset
    args.add_argument('--dataset', type=str, help="Dataset config name.")

    # Hyperparameters
    args.add_argument('--learning_rate', default=1e-5, type=float, help="Learning rate.")
    args.add_argument('--train_batch_size', default=4, type=int, help="Train batch size.")
    args.add_argument('--dev_batch_size', default=32, type=int, help="Dev batch size.")
    args.add_argument('--accumulation_size', default=4, type=int, help="Gradient accumulation size.")
    args.add_argument('--num_epochs', default=100, type=int, help="Number of train epochs.")
    args.add_argument('--max_grad_norm', default=1.0, type=float, help="Max gradient norm.")

    # Generation configs
    args.add_argument('--max_new_tokens', default=20, type=int, help="Max number of new tokens to generate.")
    args.add_argument('--num_beams', default=20, type=float, help="Generation number of beams.")
    args.add_argument('--temperature', default=0.5, type=float, help="Generation temperature.")

    args = args.parse_args()

    # Dataset specific configs
    for arg_name, arg_value in CONFIGS[args.dataset].items():
        args.__dict__[arg_name] = arg_value

    # Experiment directories
    args.log_dir = os.path.join(args.results_dir, 'log', args.dataset, args.run)
    if args.save_ckpt:
        args.ckpt_dir = os.path.join(args.results_dir, 'ckpt', args.dataset, args.run)

    # CUDA device
    args.device = torch.device(f"cuda:{args.device_id}")

    # Gradient accumulation steps
    if args.train_batch_size >= args.accumulation_size:
        args.gradient_accumulation_steps = 1
    else:
        args.gradient_accumulation_steps = int(args.accumulation_size / args.train_batch_size)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
