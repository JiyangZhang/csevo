#!/bin/bash

# This script documents the exact procedures we use to get the
# dataset, get the models running, and collect the results.

# This script is for ASE'20

# Each function is a group of commands and a later function usually
# requires the execution of all the proceeding functions.

# The commands within each function should always be executed one
# after one sequentially unless only partial functionality is wanted.


_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
readonly DATASET_PATH=${_DIR}/../csevo-dataset
readonly RESULTS_PATH=${_DIR}/../csevo-results


function setup_env() {
        echo "Environment needed:"
        echo "Python 3.7+, with the packages specified in requirements.txt."
        # echo "MongoDB server"
}

function parse_projects_deepcom() {
        python -m csevo.main parse_projects_deepcom
}

function dataset_get_urls() {
        python -m csevo.main get_github_top_repos
}

function dataset_collect() {
        python -m csevo.main collect_data
}

function dataset_filter() {
        python -m csevo.main filter_data --which=alpha
}

function dataset_split() {
        python -m csevo.main split_project_data --task=com-gen
}

function project_split() {
        python -m csevo.main split_projects --random_seed=7
}

function load_data() {
        python -m csevo.main load_data --data_type=latest\
	       --task=MN
}

function cut_data() {
        python -m csevo.main cut_data
}

function clean_data() {
        python -m csevo.main clean_comgen_data --config=DataFilter.yaml
}


# Shared preparation (for ComGen and MetNam)

function prepare_shared_method_naming() {
        python -m csevo.main process_data_shared\
               --eval_settings=evo --eval_settings=crossproj --eval_settings=mixedproj --eval_settings=crossproj-evo\
               --task=MN\
               --years=2020
}

function prepare_shared_comment_generation() {
        python -m csevo.main process_data_shared\
               --eval_settings=evo --eval_settings=crossproj --eval_settings=mixedproj --eval_settings=crossproj-evo\
               --task=CG\
               --years=2020
}

# DeepCom-Hybrid

function prepare_deepcom() {
        for eval_setting in evo crossproj mixedproj crossproj-evo; do
                python -m csevo.main process_data --model=DeepCom\
	               --task=CG\
	               --eval_setting=$eval_setting\
                       --year=2020
        done
}

function prepare_deepcom_tacc() {
        for eval_setting in evo crossproj mixedproj; do
                python -m csevo.main prepare_model --model=DeepCom\
                       --eval_setting=$eval_setting\
                       --year=2020
        done
}

function train_deepcom_tacc() {
        python -m csevo.main run_models --mode=train\
               --models=DeepCom\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=24
}

function test_common_deepcom_tacc() {
        python -m csevo.main run_models --mode=test_common\
               --models=DeepCom\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=24
}

function test_standard_deepcom_tacc() {
        python -m csevo.main run_models --mode=test_standard\
               --models=DeepCom\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=24
}

function collect_deepcom_results() {
        python -m csevo.main collect_model_results --model=DeepCom --task=com-gen --re_eval
}

# DeepCom-SBT

function prepare_deepcom_sbt_tacc() {
       python -m csevo.main prepare_model --model=DeepCom-SBT\
              --use_latest
}

function run_deepcom_sbt_tacc() {
       python -m csevo.main run_models --mode=val\
              --models=DeepCom-SBT\
              --trials=0 --trials=1 --trials=2\
              --timeout=24\
              --use_latest
}

function collect_deepcom_sbt_results() {
        python -m csevo.main collect_model_results --model=DeepCom-SBT --task=com-gen --re_eval
}


# DeepCom-Preorder

function prepare_deepcom_preorder_tacc() {
       python -m csevo.main prepare_model --model=DeepCom-Preorder
}

function run_deepcom_preorder_tacc() {
       python -m csevo.main run_models --mode=train\
              --models=DeepCom-Preorder\
              --trials=0\
              --timeout=24
}


# Seq2seq

# process_data already done in prepare_deepcom

function prepare_seq2seq_tacc() {
        for eval_setting in evo crossproj mixedproj; do
                python -m csevo.main prepare_model --model=Seq2seq\
                       --eval_setting=$eval_setting\
                       --year=2020
        done
}

function train_seq2seq_tacc() {
        python -m csevo.main run_models --mode=train\
               --models=Seq2seq\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=24
}

function test_common_seq2seq_tacc() {
        python -m csevo.main run_models --mode=test_common\
               --models=Seq2seq\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=24
}

function test_standard_seq2seq_tacc() {
        python -m csevo.main run_models --mode=test_standard\
               --models=Seq2seq\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=24
}

function collect_seq2seq_results() {
        python -m csevo.main collect_model_results --model=Seq2seq --task=com-gen --re_eval
}


# Seq2seqAtt

# process_data already done in prepare_deepcom

function prepare_seq2seq_att_tacc() {
        for eval_setting in evo crossproj mixedproj; do
                python -m csevo.main prepare_model --model=Seq2seqAtt\
                       --eval_setting=$eval_setting\
                       --year=2020
        done
}

function train_seq2seq_att_tacc() {
        python -m csevo.main run_models --mode=train\
               --models=Seq2seqAtt\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=24
}

function test_common_seq2seq_att_tacc() {
        python -m csevo.main run_models --mode=test_common\
               --models=Seq2seqAtt\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=24
}

function test_standard_seq2seq_att_tacc() {
        python -m csevo.main run_models --mode=test_standard\
               --models=Seq2seqAtt\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=24
}

function collect_seq2seq_att_results() {
        python -m csevo.main collect_model_results --model=Seq2seqAtt --task=com-gen --re_eval
}


# All comment generation models

function prepare_comment_generation_tacc() {
        prepare_deepcom_tacc
        prepare_seq2seq_tacc
        prepare_seq2seq_att_tacc
}

function train_comment_generation_tacc_0_16() {
        python -m csevo.main run_models --mode=train\
               --models=DeepCom --models=Seq2seq --models=Seq2seqAtt\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0 --trials=1 --trials=2\
               --timeout=24\
               --beg=0 --cnt=16
}

function train_comment_generation_tacc_16_27() {
        python -m csevo.main run_models --mode=train\
               --models=DeepCom --models=Seq2seq --models=Seq2seqAtt\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0 --trials=1 --trials=2\
               --timeout=24\
               --beg=16 --cnt=11
}

function test_comment_generation_tacc() {
        python -m csevo.main run_models --mode=test_common\
               --models=DeepCom --models=Seq2seq --models=Seq2seqAtt\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0 --trials=1 --trials=2\
               --timeout=1 --local
        # python -m csevo.main run_models --mode=test_standard\
        #        --models=DeepCom --models=Seq2seq --models=Seq2seqAtt\
        #        --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
        #        --trials=0 --trials=1 --trials=2\
        #        --timeout=2 --local
}

function collect_comment_generation_results() {
        collect_deepcom_results
        collect_seq2seq_results
        collect_seq2seq_att_results
}


# All method naming models

function prepare_method_naming_tacc() {
        prepare_code2seq_tacc
        prepare_biLSTM_tacc
        prepare_nosplit_biLSTM_tacc
}

function train_method_naming_tacc_0_16() {
        python -m csevo.main run_models --mode=train\
               --models=Code2Seq --models=Bi-LSTM --models=no-split-Bi-LSTM\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0 --trials=1 --trials=2\
               --timeout=24\
               --beg=8 --cnt=16
}

function train_method_naming_tacc_16_27() {
        python -m csevo.main run_models --mode=train\
               --models=Code2Seq --models=Bi-LSTM --models=no-split-Bi-LSTM\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0 --trials=1 --trials=2\
               --timeout=24\
               --beg=24 --cnt=3
}

function test_method_naming_tacc() {
        python -m csevo.main run_models --mode=test_common\
               --models=Code2Seq --models=Bi-LSTM --models=no-split-Bi-LSTM\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0 --trials=1 --trials=2\
               --timeout=12
        python -m csevo.main run_models --mode=test_standard\
               --models=Code2Seq --models=Bi-LSTM --models=no-split-Bi-LSTM\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0 --trials=1 --trials=2\
               --timeout=12
}

function collect_method_naming_results() {
        collect_code2seq_results
        collect_biLSTM_results
        collect_nosplit_biLSTM_results
}


# Code2Seq

function prepare_code2seq() {
        for eval_setting in evo crossproj mixedproj crossproj-evo; do
                      python -m csevo.main process_data --model=Code2Seq\
                       --task=MN\
                       --eval_setting=$eval_setting\
                             --year=2020
        done
}

function prepare_code2seq_tacc() {
         for eval_setting in evo crossproj mixedproj; do
                python -m csevo.main prepare_model --model=Code2Seq\
                       --eval_setting=$eval_setting\
                       --year=2020
        done
}

function train_code2seq_tacc() {
        python -m csevo.main run_models --mode=train\
               --models=Code2Seq\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=24
}

function test_common_code2seq_tacc() {
        python -m csevo.main run_models --mode=test_common\
               --models=Code2Seq\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=2
}

function test_standard_code2seq_tacc() {
        python -m csevo.main run_models --mode=test_standard\
               --models=Code2Seq\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=24
}

function run_code2seq_tacc() {
        python -m csevo.main run_models --mode=train\
               --models=Code2Seq\
               --trials=0\
               --timeout=5
}

function collect_code2seq_results() {
        python -m csevo.main collect_model_results --model=Code2Seq --task=methd-name --re_eval
}

# 2-layer BiLSTM
function prepare_biLSTM() {
        for eval_setting in evo crossproj mixedproj crossproj-evo; do
                          python -m csevo.main process_data --model=Bi-LSTM\
                           --task=MN\
                           --eval_setting=$eval_setting\
                                 --year=2020
        done
}

function prepare_biLSTM_tacc() {
         for eval_setting in evo crossproj mixedproj; do
                python -m csevo.main prepare_model --model=Bi-LSTM\
                       --eval_setting=$eval_setting\
                       --year=2020
        done
}

function train_biLSTM_tacc() {
        python -m csevo.main run_models --mode=train\
                   --models=Bi-LSTM\
                   --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
                   --trials=0\
                   --timeout=24
}

function test_common_biLSTM_tacc() {
        python -m csevo.main run_models --mode=test_common\
               --models=Bi-LSTM\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=2
}

function test_standard_biLSTM_tacc() {
        python -m csevo.main run_models --mode=test_standard\
               --models=Bi-LSTM\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=24
}

function collect_biLSTM_results() {
        python -m csevo.main collect_model_results --model=Bi-LSTM --task=methd-name --re_eval
}

# No split Bi-LSTM
function prepare_nosplit_biLSTM() {
         for eval_setting in evo crossproj mixedproj crossproj-evo; do
                          python -m csevo.main process_data --model=no-split-Bi-LSTM\
                           --task=MN\
                           --eval_setting=$eval_setting\
                                 --year=2020
        done
}

function prepare_nosplit_biLSTM_tacc() {
         for eval_setting in evo crossproj mixedproj evo-crossproj; do
                python -m csevo.main prepare_model --model=no-split-Bi-LSTM\
                       --eval_setting=$eval_setting\
                       --year=2020
        done
}

function train_nosplit_biLSTM_tacc() {
        python -m csevo.main run_models --mode=train\
                       --models=no-split-Bi-LSTM\
                       --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
                       --trials=0\
                       --timeout=24
}

function test_common_nosplit_biLSTM_tacc() {
        python -m csevo.main run_models --mode=test_common\
               --models=no-split-Bi-LSTM\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=2
}


function test_standard_nosplit_biLSTM_tacc() {
        python -m csevo.main run_models --mode=test_standard\
               --models=no-split-Bi-LSTM\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0\
               --timeout=24
}

function collect_nosplit_biLSTM_results() {
        python -m csevo.main collect_model_results --model=no-split-Bi-LSTM --task=methd-name --re_eval
}


#------------------------------------------------------

function prepare_transformer() {
        for eval_setting in evo crossproj mixedproj; do
                            python -m csevo.main process_data --model=Transformer\
                             --task=CG\
                             --eval_setting=$eval_setting\
                                   --year=2020
              done
}

function prepare_transformer_tacc() {
        for eval_setting in evo crossproj mixedproj; do
                python -m csevo.main prepare_model --model=Transformer\
                       --eval_setting=$eval_setting\
                       --year=2020
        done
}

function train_transformer_tacc() {
        python -m csevo.main run_models --mode=train\
               --models=Transformer\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0 --trials=1 --trials=2\
               --timeout=24
}

function test_common_transformer_tacc() {
        python -m csevo.main run_models --mode=test_common\
               --models=Transformer\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0 --trials=1 --trials=2\
               --timeout=5
}

function test_standard_transformer_tacc() {
        python -m csevo.main run_models --mode=test_standard\
               --models=Transformer\
               --exps=evo-2020 --exps=crossproj-2020 --exps=mixedproj-2020\
               --trials=0 --trials=1 --trials=2\
               --timeout=12
}

function collect_results_code2seq() {
        python -m csevo.main make_tables --which=draft-model-results --results-path=../csevo-results/test-results/Code2Seq-small --output-name=Code2Seq-small
        python -m csevo.main make_plots --which=draft-learning-curve --training-log-path=../csevo-results/training-logs/Code2Seq-small --output-name=Code2Seq-small
}

# AST-AttendGRU

function exp_ast_attendgru() {
        # TODO: modify in future according to model / data path
        python3 train.py --gpu 0 --modeltype ast-attendgru --data ./data

}


# Process results

function process_comment_generation_results() {
        local models="DeepCom Seq2seq Seq2seqAtt"
        local models_arg="--models=DeepCom --models=Seq2seq --models=Seq2seqAtt"
        local metrics_arg="--metrics=bleu --metrics=xmatch"

        # Stats of the results
        # Ignore the stderr output regarding "test_standard"
        for model in $models; do
                python -m csevo.main collect_metrics --which=model-stat-results --model=$model
        done

        # Stat significance tests
        python -m csevo.main collect_metrics --which=stat-sign-test\
               --output=ComGen\
               --exps=mixedproj-2020 --exps=crossproj-2020 --exps=evo-2020\
               $models_arg\
               $metrics_arg\
               --test_set=test_common

        # Make numbers and tables
        for model in $models; do
                python -m csevo.main make_tables --which=models-numbers --model=$model
        done

        python -m csevo.main make_tables --which=models-results --task=ComGen

        # Make plots
        python -m csevo.main make_plots --which=models-results-metrics-dist --task=ComGen
        python -m csevo.main make_plots --which=models-results-variance-dist --task=ComGen
}

function process_method_naming_results() {
        local models="Code2Seq Bi-LSTM no-split-Bi-LSTM"
        local models_arg="--models=Code2Seq --models=Bi-LSTM --models=no-split-Bi-LSTM"
        local metrics_arg="--metrics=f1 --metrics=precision --metrics=recall --metrics=xmatch"
        
        # Stats of the results
        # Ignore the stderr output regarding "test_standard"
        for model in $models; do
                python -m csevo.main collect_metrics --which=model-stat-results --model=$model
        done

        # Stat significance tests
        python -m csevo.main collect_metrics --which=stat-sign-test\
               --output=MethNam\
               --exps=mixedproj-2020 --exps=crossproj-2020 --exps=evo-2020\
               $models_arg\
               $metrics_arg\
               --test_set=test_common

        # Make numbers and tables
        for model in $models; do
                python -m csevo.main make_tables --which=models-numbers --model=$model
        done

        python -m csevo.main make_tables --which=models-results --task=MethNam

        # Make plots
        python -m csevo.main make_plots --which=models-results-metrics-dist --task=MethNam
        python -m csevo.main make_plots --which=models-results-variance-dist --task=MethNam
}

function collect_dataset_metrics() {
        python -m csevo.main collect_metrics --which=dataset
        python -m csevo.main collect_metrics --which=raw-dataset
}

function make_dataset_tables_plots() {
        python -m csevo.main make_tables --which=dataset-metrics
}


# function dataset_paper() {
#         python -m csevo.main make_tables --which=time-wise-filtered-dataset-metrics\
#                --dataset=debug\
#                --filter=beta
# }

# function model_number() {
#         python -m csevo.main make_numbers --model=$1\
#                --use_latest
# }

# function tables_models_results() {
#         python -m csevo.main make_tables --which=models-results\
#                --task=Comment-generation
# }

# function dataset_statistics() {
#         python -m csevo.main collect_metrics_time_wise --which=filtered\
#                --dataset=debug\
#                --filter=beta
# }


# ==========
# Main function -- program entry point
# This script can be executed as ./run.sh the_function_to_run

function main() {
        local action=${1:?Need Argument}; shift

        ( cd ${_DIR}
          $action "$@"
        )
}

main "$@"


# ==========
# Some notes of useful Bash commands

# Export Anaconda environment
# conda env export --from-history > env.yml

# Load Anaconda envrionment
# conda env create -n NAME -f env.yml
