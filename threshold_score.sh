#!/usr/bin/env bash
python3.6 evaluate.py $1 ./predict1.json ./predict1.json $2
python3.6 evaluate.py $1 ./predict3.json ./predict3.json $2
python3.6 evaluate.py $1 ./predict5.json ./predict5.json $2
python3.6 evaluate.py $1 ./predict7.json ./predict7.json $2
python3.6 evaluate.py $1 ./predict9.json ./predict9.json $2
python3.6 plot_answer_threshold.py