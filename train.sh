FOLD=$1
GPU=$2

for BATCH in 48 32 24 16 8
do
    python Main.py --start_fold $FOLD --gpu $GPU --batch $BATCH --add_trend
done
