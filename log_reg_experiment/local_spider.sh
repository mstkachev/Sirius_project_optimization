#!/bin/bash
python3 local_spider_homo.py --dataset mushrooms --num_workers 20 --num_local_steps 10 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 20 --num_local_steps 20 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 20 --num_local_steps 30 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 20 --num_local_steps 50 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 20 --num_local_steps 100 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 50 --num_local_steps 10 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 50 --num_local_steps 20 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 50 --num_local_steps 30 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 50 --num_local_steps 50 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 50 --num_local_steps 100 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 100 --num_local_steps 10 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 100 --num_local_steps 20 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 100 --num_local_steps 30 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 100 --num_local_steps 50 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
python3 local_spider_homo.py --dataset mushrooms --num_workers 100 --num_local_steps 100 --batch_size 10 --continue 0 --max_it 100 --launch_number 1 --tol 1e-12
