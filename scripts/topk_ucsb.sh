for dataset in "ucsb" 
do
    for prob in 0.2 0.5 0.8 
    do
        python3 real_world_mil_main.py --mode topk --dataset $dataset --prob $prob
    done 
done

sleep 30s
 
for dataset in "ucsb" 
do
    for prob in 0.2 0.5 0.8 
    do
        python3 real_world_mil_main.py --mode rand --dataset $dataset --prob $prob
    done 
done