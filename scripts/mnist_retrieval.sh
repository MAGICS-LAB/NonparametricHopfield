method="half"
d_model=784
data="mnist"
thr=0.1
scale=0.01


# python3 retrieval_main.py --mode softmax --method half --d_model 784 --data mnist --thr 0.1 --mem_size 10 --scale 0.01       

# for mem_size in 10
# do 
#     python3 retrieval_main.py --mode softmax --method half --d_model $d_model --data $data --thr $thr --mem_size $mem_size --scale $scale       
# done


for mem_size in 10 20 30 40 50 60 70 80 90 100 
do 
    python3 retrieval_main.py --mode softmax --method half --d_model $d_model --data $data --thr $thr --mem_size $mem_size --scale $scale       
done

sleep 2m 30s

for mem_size in 10 20 30 40 50 60 70 80 90 100 
do
    python3 retrieval_main.py --mode sparsemax --method half --d_model $d_model --data $data --thr $thr --mem_size $mem_size --scale $scale
done

sleep 2m 30s


for mem_size in 10 20 30 40 50 60 70 80 90 100 
do
    python3 retrieval_main.py --mode favor --method half --d_model $d_model --data $data --thr $thr --mem_size $mem_size --kernel_fn relu --scale $scale
done

sleep 2m 30s

for mem_size in 10 20 30 40 50 60 70 80 90 100 
do
    python3 retrieval_main.py --mode linear --method half --d_model $d_model --data $data --thr $thr --mem_size $mem_size --kernel_fn elu --scale $scale
done

sleep 2m 30s
 

for mem_size in 10 20 30 40 50 60 70 80 90 100 0 
do
    for prob in 0.2 0.5 0.8 
    do
        python3 retrieval_main.py --mode topk --method half --d_model $d_model --data $data --thr $thr --mem_size $mem_size --prob $prob --scale $scale
    done 
done

sleep 2m 30s
 

for mem_size in 10 20 30 40 50 60 70 80 90 100 
do
    for prob in 0.2 0.5 0.8 
    do
        python3 retrieval_main.py --mode rand --method half --d_model $d_model --data $data --thr $thr --mem_size $mem_size --prob $prob --scale $scale
    done 
done

sleep 2m 30s
 
## Noisy Recovery

mem_size=20

for noise in 0.05 0.1 0.2 0.3 0.5 0.8 1.5 
do
    python3 retrieval_main.py --mode softmax --method noise --noise $noise --d_model $d_model --data $data --thr $thr --mem_size $mem_size --scale $scale
done

sleep 2m 30s
 

for noise in 0.05 0.1 0.2 0.3 0.5 0.8 1.5 
do
    python3 retrieval_main.py --mode sparsemax --method noise --noise $noise --d_model $d_model --data $data --thr $thr --mem_size $mem_size --scale $scale
done

sleep 2m 30s
 

for noise in 0.05 0.1 0.2 0.3 0.5 0.8 1.5 
do
    python3 retrieval_main.py --mode favor --method noise --noise $noise --d_model $d_model --data $data --thr $thr --mem_size $mem_size --kernel_fn relu --scale $scale
done

sleep 2m 30s
 

for noise in 0.05 0.1 0.2 0.3 0.5 0.8 1.5 
do
    python3 retrieval_main.py --mode linear --method noise --noise $noise --d_model $d_model --data $data --thr $thr --mem_size $mem_size --kernel_fn elu --scale $scale
done

sleep 2m 30s
 


for noise in 0.05 0.1 0.2 0.3 0.5 0.8 1.5 
do
    for prob in 0.2 0.5 0.8 
    do
        python3 retrieval_main.py --mode topk --method noise --noise $noise --d_model $d_model --data $data --thr $thr --mem_size $mem_size --prob $prob --scale $scale
    done 
done

sleep 2m 30s
 

for noise in 0.05 0.1 0.2 0.3 0.5 0.8 1.5 
do
    for prob in 0.2 0.5 0.8 
    do
        python3 retrieval_main.py --mode rand --method noise --noise $noise --d_model $d_model --data $data --thr $thr --mem_size $mem_size --prob $prob --scale $scale
    done 
done