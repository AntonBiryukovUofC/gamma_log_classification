FOLD=$1
GPU=$2

for BATCH in 8 4 2
do
    python Main_ud.py --start_fold $FOLD --gpu $GPU --batch $BATCH --add_trend
done
