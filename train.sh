FOLD=$1
GPU=$2

for BATCH in 80 64 48 32 24
do
    python Main.py --start_fold $FOLD --gpu $GPU --batch $BATCH 
    python Main_ud.py --start_fold $FOLD --gpu $GPU --batch $BATCH 
done
