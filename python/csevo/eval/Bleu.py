from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sys
from seutil import IOUtils

class Bleu:

    @classmethod
    def compute_bleu(cls, references:str, hypotheses: str, test_result_file: str) -> int:
        with open(references, 'r') as fr, open(hypotheses, 'r') as fh:
            refs = fr.readlines()
            hyps = fh.readlines()
        bleu_4_sentence_scores = []
        for ref, hyp in zip(refs, hyps):
            if len(hyp.strip().split()) < 2:
                bleu_4_sentence_scores.append(0)
            else:
                bleu_4_sentence_scores.append(sentence_bleu([ref.strip().split()], hyp.strip().split(),
                                                        smoothing_function=SmoothingFunction().method2,
                                                        auto_reweigh=True))
        score = 100 * sum(bleu_4_sentence_scores) / float(len(bleu_4_sentence_scores))

        result = {"bleu": score}
        IOUtils.dump(test_result_file, result)
        return score

if __name__ == "__main__":
    ref_file = sys.argv[1]
    hyp_file = sys.argv[2]
    test_result_file = sys.argv[3]
    Bleu.compute_bleu(ref_file, hyp_file, test_result_file)
