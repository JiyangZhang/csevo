from pathlib import Path
import sys
from seutil import IOUtils
SOS = '<S>'
EOS = '</S>'
PAD = '<PAD>'
UNK = '<UNK>'


def legal_method_names_checker(name):
    return not name in [UNK, PAD, EOS]


def filter_impossible_names(top_words):
    result = list(filter(legal_method_names_checker, top_words))
    return result


def calculate_results(true_positive, false_positive, false_negative):
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1


def get_eval_stats(pred_file: Path, ref_file: Path, result_dir: Path):
    true_positive, false_positive, false_negative = 0, 0, 0
    with open(pred_file, "r") as pf, open(ref_file, "r") as rf:
        pred_lines = pf.readlines()
        ref_lines = rf.readlines()
    true_positive, false_positive, false_negative = update_per_subtoken_statistics(zip(ref_lines, pred_lines),
                                                                                       true_positive, false_positive,
                                                                                       false_negative)
    precision, recall, f1 = calculate_results(true_positive, false_positive, false_negative)
    test_result = {
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    IOUtils.dump(result_dir, test_result, IOUtils.Format.jsonPretty)

def update_per_subtoken_statistics(results, true_positive, false_positive, false_negative):
    """Used in evaluating method name generation task."""
    for original_name, predicted in results:
        filtered_predicted_names = filter_impossible_names(predicted.strip().split())
        filtered_original_subtokens = filter_impossible_names(original_name.strip().split())

        if ''.join(filtered_original_subtokens) == ''.join(filtered_predicted_names):
            true_positive += len(filtered_original_subtokens)
            continue
        for subtok in filtered_predicted_names:
            if subtok in filtered_original_subtokens:
                true_positive += 1
            else:
                false_positive += 1
        for subtok in filtered_original_subtokens:
            if not subtok in filtered_predicted_names:
                false_negative += 1
    return true_positive, false_positive, false_negative

if __name__ == "__main__":
    pred_file = sys.argv[1]
    ref_file = sys.argv[2]
    output_file = sys.argv[3]
    get_eval_stats(pred_file, ref_file, output_file)