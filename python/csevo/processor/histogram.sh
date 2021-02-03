#!/usr/bin/env bash
###########################################################
# Build histogram for creating vocabulary
DATASET_PATH=$1

# We have got raw.txt
TRAIN_DATA_FILE=${DATASET_PATH}/train.raw.txt

TARGET_HISTOGRAM_FILE=${DATASET_PATH}/histo.tgt.c2s
SOURCE_SUBTOKEN_HISTOGRAM=${DATASET_PATH}/histo.ori.c2s
NODE_HISTOGRAM_FILE=${DATASET_PATH}/histo.node.c2s

echo "Creating histograms from the training data"
cat ${TRAIN_DATA_FILE} | cut -d' ' -f1 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${SOURCE_SUBTOKEN_HISTOGRAM}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${NODE_HISTOGRAM_FILE}
