#!/bin/bash
set -eux
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
	    python behavioral_cloning.py expert_data/$e.pkl $e --epochs 100 --batch_size 128 --activation tf.nn.relu
    done
