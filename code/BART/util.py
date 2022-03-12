import math
import os
import pprint
import random
from collections import deque
import re
from typing import Optional

import numpy as np
import torch
from sympy.parsing.sympy_parser import parse_expr


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"


def load_checkpoint(ckpt_path, model, optimizer=None):
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint[_MODEL_STATE_DICT])

    if optimizer:
        optimizer.load_state_dict(checkpoint[_OPTIMIZER_STATE_DICT])


def save_checkpoint(ckpt_path, model, optimizer, epoch_id, metric):
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optimizer.state_dict(),
    }, os.path.join(ckpt_path, f"{metric:.5}.{epoch_id}.tar"))


def log_and_save_checkpoint(args, model=None, optimizer=None, epoch_id=None,
                            best_equation_equivalence=None, equation_equivalence=None, dev_filename=None):
    if epoch_id is None:
        with open(os.path.join(args.log_dir, "log.txt"), "w") as f:
            f.write("Arguments\n")
            f.write(f"{pprint.pformat(vars(args), sort_dicts=False, indent=2)}\n\n")

    else:
        with open(os.path.join(args.log_dir, "log.txt"), "a") as f:
            f.write(f"Dev filename: {dev_filename}")
            f.write(f"Epoch {epoch_id}\n")
            f.write(f"Equation Equivalence: {equation_equivalence:.5}\n")
            f.write(f"Current best Equation Equivalence: "
                    f"{equation_equivalence if equation_equivalence > best_equation_equivalence else best_equation_equivalence}\n")
            f.write("\n\n")

        if args.save_ckpt and equation_equivalence > best_equation_equivalence:
            save_checkpoint(args.ckpt_dir, model, optimizer, epoch_id, equation_equivalence)
            print(f"Saved checkpoint for epoch {epoch_id}.")


def parse_polish_notation(equation: str):
    def _parse(tokens):
        token = tokens.popleft()
        if token == '+':
            return f"({_parse(tokens)} + {_parse(tokens)})"
        elif token == '-':
            return f"({_parse(tokens)} - {_parse(tokens)})"
        elif token == '*':
            return f"({_parse(tokens)} * {_parse(tokens)})"
        elif token == '/':
            return f"({_parse(tokens)} / {_parse(tokens)})"
        else:
            # must be just a number
            return token

    expression = deque(equation.split())
    return _parse(expression)


def solve_equation(equation: str) -> float:
    try:
        return float(parse_expr(equation))
    except Exception:
        return math.inf  # todo 0?


def normalize_equation(equation: str) -> Optional[str]:
    """Check if equation matches polish notation, and modify if possible."""
    atom_list = equation.strip().split()
    atom_list = [atom.strip() for atom in atom_list]

    operation_list = []
    number_list = []
    mask_list = []

    for atom in atom_list:
        if atom in ["+", "-", "*", "/"]:
            operation_list.append(atom)
            mask_list.append("0")
        else:
            number_list.append(atom)
            mask_list.append("1")

    mask_str = "".join(mask_list)

    if len(operation_list) + 1 > len(number_list):
        for _ in range(len(operation_list) + 1 - len(number_list)):
            if "0" in mask_str:
                last_operation_idx = mask_str.rindex("0")
                mask_list[last_operation_idx] = "x"
                mask_str = "".join(mask_list)

    atom_list = [atom for idx, atom in enumerate(atom_list) if mask_list[idx] != "x"]

    return " ".join(atom_list) if len(atom_list) > 0 else "0"


def metric_equation_EM(inferred_equation: str, gold_equation: str) -> int:
    """Check if equation exactly matches."""
    return int(inferred_equation.strip() == gold_equation.strip())


def check_polish_notation_syntax(equation: str) -> bool:
    atom_list = equation.strip().split()
    atom_list = [atom.strip() for atom in atom_list]

    operation_list = []
    number_list = []
    for atom in atom_list:
        if atom in ["+", "-", "*", "/"]:
            operation_list.append(atom)
        elif re.match(r"^number\d$", atom):
            number_list.append(atom)
        else:
            try:
                atom = float(atom)
                number_list.append(atom)
            except ValueError:
                continue

    if len(operation_list) + len(number_list) < len(atom_list):
        return False
    elif len(operation_list) != len(number_list) - 1:
        return False
    else:
        return True


def mark_perturbation(inferred_equation: str, gold_equation: str) -> bool:
    if not check_polish_notation_syntax(inferred_equation):
        return False
    else:
        inferred_equation = parse_polish_notation(inferred_equation)
        gold_equation = parse_polish_notation(gold_equation)
        return parse_expr(f"{inferred_equation} - {gold_equation}").simplify() == 0


def equation_equivalence(inferred_equation: str, gold_equation: str) -> int:
    inferred_equation = parse_polish_notation(inferred_equation)
    gold_equation = parse_polish_notation(gold_equation)

    equivalence = f"{inferred_equation} - {gold_equation}"

    try:
        simplified = parse_expr(equivalence).simplify()
    except Exception:
        simplified = 0

    return int(simplified == 0)

