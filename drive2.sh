#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "ev3way_train_run.py.*v18i" >/dev/null; do sleep 120; done
EVAL_WORKERS=8 venv/bin/python -u eval_candidates.py v17:ev3way_w1_rob_v7.npy:ev3way_w2_rob_v7.npy:0:1 v18s3:ev3way_w1_v18s3.npy:ev3way_w2_v18s3.npy:1:0 v18i1:ev3way_w1_v18i1.npy:ev3way_w2_v18i1.npy:1:0:1 v18i2:ev3way_w1_v18i2.npy:ev3way_w2_v18i2.npy:1:0:1 > eval_round2.txt 2>&1
