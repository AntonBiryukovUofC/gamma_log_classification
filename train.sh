FOLD=$1
GPU=$2

for BATCH in 32 48 64
do
    python Main.py --start_fold $FOLD --gpu $GPU --batch $BATCH --add_trend
done
