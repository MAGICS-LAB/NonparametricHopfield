for data in "ETTh1" "ETTm1" "Traffic" "WTH" "ECL"
do
    for in_len in 96 192 336 720
    do
        python3 time_series_main.py --mode softmax --in_len $in_len --data $data
    done
done