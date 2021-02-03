import sys
from seutil import *

def fix(input_file):
    num_data_valid = 2384
    results = IOUtils.load(input_file)

    for exp, exp_results in results.items():
        for test_set, set_results in exp_results.items():
            # Only use test_common set
            if test_set != "test_common":
                continue

            for metric, trials_results in set_results.items():

                # Merge the results from all trials
                for i in range(len(trials_results)):
                    if trials_results[i] is not None:
                        trials_results[i] = trials_results[i][-num_data_valid:]

    IOUtils.dump(input_file, results, IOUtils.Format.jsonPretty)

if __name__ == "__main__":
    fix(sys.argv[1])