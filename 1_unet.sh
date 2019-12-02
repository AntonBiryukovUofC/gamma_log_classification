for FOLD in 0 1 2 3 4
do
	python src/models/train_model_UNET_batch.py --epochs 350 --fold $FOLD --gpu 1 --dropout 0.1 --batch_size 64 --kernel_size 3 --init_power 7 --mode regular 
done
