for data in "ETTh1" "ETTm1" "Traffic" "WTH" "ECL"
do
    for in_len in 96 192 336 720
    do
        for prob in 0.2 0.5 0.8 
        do
            python3 time_series_main.py --mode topk --in_len $in_len --data $data --prob $prob
        done
    done
done