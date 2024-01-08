method=="half"
d_model=512
data="mnist"
scale=0.1

for bag_size in 5 10 20 30 50 80 100
do
    python3 mnist_mil_main.py --mode softmax --bag_size $bag_size
done

sleep 2m 30s
 

for bag_size in 5 10 20 30 50 80 100
do
    python3 mnist_mil_main.py --mode sparsemax --bag_size $bag_size
done

sleep 2m 30s
 

for bag_size in 5 10 20 30 50 80 100
do
    python3 mnist_mil_main.py --mode favor --bag_size $bag_size
done

sleep 2m 30s
 

for bag_size in 5 10 20 30 50 80 100 
do
    python3 mnist_mil_main.py --mode linear --bag_size $bag_size
done

sleep 2m 30s
 

for bag_size in 5 10 20 30 50 80 100 
do
    for prob in 0.2 0.5 0.8 
    do
        python3 mnist_mil_main.py --mode topk --bag_size $bag_size --prob $prob
    done 
done

sleep 2m 30s
 

for bag_size in 5 10 20 30 50 80 100
do
    for prob in 0.2 0.5 0.8 
    do
        python3 mnist_mil_main.py --mode rand --bag_size $bag_size --prob $prob
    done 
done