for data in "ETTh1"
do
    python3 time_series_main.py --mode window --in_len 96 --data $data --win_size 8
    python3 time_series_main.py --mode window --in_len 192 --data $data --win_size 12
    python3 time_series_main.py --mode window --in_len 336 --data $data --win_size 14
    python3 time_series_main.py --mode window --in_len 720 --data $data --win_size 18
    # python3 time_series_main.py --mode window --in_len 1440 --data $data --win_size 30
    # python3 time_series_main.py --mode window --in_len 2880 --data $data --win_size 48

done

for data in "ETTh1"
do
    for in_len in 96 192 336 720
    do
        python3 time_series_main.py --mode sparsemax --in_len $in_len --data $data
    done
done


for data in "ETTh1"
do
    for in_len in 96 192 336 720
    do
        python3 time_series_main.py --mode softmax --in_len $in_len --data $data
    done
done

  
for data in "ETTh1"
do
    for in_len in 96 192 336 720
    do
        python3 time_series_main.py --mode linear --in_len $in_len --data $data --kernel_fn elu
    done
done


for data in "ETTh1"
do
    for in_len in 96 192 336 720
    do
        python3 time_series_main.py --mode favor --in_len $in_len --data $data --kernel_fn relu
    done
done
 

for data in "ETTh1"
do
    for in_len in 96 192 336 720
    do
        for prob in 0.2 0.5 0.8 
        do
            python3 time_series_main.py --mode topk --in_len $in_len --data $data --prob $prob
        done
    done
done