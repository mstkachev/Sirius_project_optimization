#!/bin/bash
python3 local_spider_homo.py --dataset mushrooms --num_workers 20 --num_local_steps 10 --batch_size 20 --continue 0 --max_it 100000 --max_num_comm 1000                       --launch_number 1 --tol 1e-12
