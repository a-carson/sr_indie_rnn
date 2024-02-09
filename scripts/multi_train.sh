#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python3 main.py fit --config configs/baseline_rnn.yaml  \
--data.config configs/data/ht1.json --custom.experiment_name ht1 --custom.name ht1_gru64

python3 main.py fit --config configs/baseline_rnn.yaml  \
--data.config configs/data/ht1-low-gain.json --custom.experiment_name ht1-low-gain --custom.name ht1-low-gain_gru64

python3 main.py fit --config configs/baseline_rnn.yaml \
--data.config configs/data/gypsy.json --custom.experiment_name gypsy --custom.name gypsy_gru64

python3 main.py fit --config configs/baseline_rnn.yaml  \
--data.config configs/data/gypsy-extreme.json --custom.experiment_name gypsy-extreme --custom.name gypsy-extreme_gru64

python3 main.py fit --config configs/baseline_rnn.yaml  \
--data.config configs/data/bender.json --custom.experiment_name bender --custom.name bender_gru64

python3 main.py fit --config configs/baseline_rnn.yaml \
--data.config configs/data/bender-extreme.json --custom.experiment_name bender-extreme --custom.name bender-extreme_gru64