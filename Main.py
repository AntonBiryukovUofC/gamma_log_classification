#import modules
from DataGenerator import *
from Pipeline import *
from config import *
import sys
import logging as log
import click
import os

@click.command()
@click.option('--start_fold', default=0, help='fold to train')
@click.option('--gpu', default=0, help='gpu to train on')
def main(start_fold,gpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    GetData = DataGenerator()
    CV = Pipeline(GetData, DL_model,start_fold)
    score = CV.train()
    log.log(f'Model accuracy = {score}')

if __name__ == "__main__":
    main()