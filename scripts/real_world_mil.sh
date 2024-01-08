method="half"
d_model=64
data="mnist"
scale=0.01

for dataset in "fox" "elephant" "tiger" "ucsb" 
do
    python3 real_world_mil_main.py --mode softmax --dataset $dataset
done

sleep 2m 30s

for dataset in "fox" "elephant" "tiger" "ucsb" 
do
    python3 real_world_mil_main.py --mode sparsemax --dataset $dataset
done

sleep 2m 30s

for dataset in "fox" "elephant" "tiger" "ucsb" 
do
    python3 real_world_mil_main.py --mode favor --dataset $dataset
done

sleep 2m 30s

for dataset in "fox" "elephant" "tiger" "ucsb" 
do
    python3 real_world_mil_main.py --mode linear --dataset $dataset --kernel_fn elu
done

sleep 2m 30s
 

for dataset in "fox" "elephant" "tiger" "ucsb" 
do
    for prob in 0.2 0.5 0.8 
    do
        python3 real_world_mil_main.py --mode topk --dataset $dataset --prob $prob
    done 
done

sleep 2m 30s
 

for dataset in "fox" "elephant" "tiger" "ucsb" 
do
    for prob in 0.2 0.5 0.8 
    do
        python3 real_world_mil_main.py --mode rand --dataset $dataset --prob $prob
    done 
done