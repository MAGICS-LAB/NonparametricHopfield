CUDA_VISIBLE_DEVICES=4 python3 real_world_mil_main.py --mode sparsemax --dataset ucsb
CUDA_VISIBLE_DEVICES=4 python3 real_world_mil_main.py --mode favor --dataset ucsb --kernel_fn relu