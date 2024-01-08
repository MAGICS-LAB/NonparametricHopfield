for dataset in "fox" "elephant" "tiger" "ucsb" 
do
    CUDA_VISIBLE_DEVICES=6 python3 real_world_mil_main.py --mode linear --dataset $dataset --kernel_fn elu
done

# CUDA_VISIBLE_DEVICES=5 python3 real_world_mil_main.py --mode softmax --dataset ucsb
# CUDA_VISIBLE_DEVICES=4 python3 real_world_mil_main.py --mode sparsemax --dataset ucsb
# CUDA_VISIBLE_DEVICES=4 python3 real_world_mil_main.py --mode favor --dataset ucsb --kernel_fn relu