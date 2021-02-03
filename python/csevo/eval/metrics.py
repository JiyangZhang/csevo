from typing import *

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def bleu(ref: List[str], output: List[str]) -> float:
    """
    Computes BLEU score, using smoothing function #2.
    Range 0~100.
    """
    if len(ref) == 0 or len(output) == 0:
        return 0

    return sentence_bleu(
        [ref],
        output,
        smoothing_function=SmoothingFunction().method2,
        auto_reweigh=True,
    ) * 100


def f1(ref: List[str], output: List[str]) -> float:
    """
    Computes f1 score.
    Range 0~100
    """
    prec = precision(ref, output)
    recl = recall(ref, output)
    if prec + recl > 0:
        f1_score = 2 * prec * recl / (prec + recl)
    else:
        f1_score = 0
    return f1_score


def precision(ref: List[str], output: List[str]) -> float:
    """
    Computes precision score.
    Range 0~100
    """
    true_positive, false_positive, false_negative = 0, 0, 0
    if ''.join(ref) == ''.join(output):
        true_positive += len(ref)
        return 100
    for subtok in output:
        if subtok in ref:
            true_positive += 1
        else:
            false_positive += 1
    for subtok in ref:
        if not subtok in output:
            false_negative += 1
    if true_positive + false_positive > 0:
        prec = true_positive / (true_positive + false_positive) * 100
    else:
        prec = 0

    return prec


def recall(ref: List[str], output: List[str]) -> float:
    """
    Computes recall score.
    Range 0~100
    """
    true_positive, false_positive, false_negative = 0, 0, 0
    if ''.join(ref) == ''.join(output):
        true_positive += len(ref)
        return 100
    for subtok in output:
        if subtok in ref:
            true_positive += 1
        else:
            false_positive += 1
    for subtok in ref:
        if not subtok in output:
            false_negative += 1
    if true_positive + false_negative > 0:
        recl = true_positive / (true_positive + false_negative) * 100
    else:
        recl = 0
    return recl


def xmatch(ref: List[str], output: List[str]) -> float:
    """
    Computes exact match metric.
    Range 0~100
    """
    if len(ref) == len(output) and all(r == o for r, o in zip(ref, output)):
        return 100
    else:
        return 0
