import os
import sys
import numpy as np

import torch
from easydict import EasyDict as edict

from utils.transforms import *

# Dataset path
PATH = edict()

PATH.DB_ROOT = '../data/kaist-rgbt'
PATH.JSON_GT_FILE = os.path.join('kaist_annotations_test20.json')

# train
train = edict()

train.day = "all"
train.img_set = f"train-all-02.txt"

# Load chekpoint
train.checkpoint = None

# Set train arguments
train.batch_size = 8       # batch size
train.start_epoch = 0      # start at this epoch
train.epochs = 40          # number of epochs to run

train.lr = 1e-4            # learning rate
train.momentum = 0.9       # momentum
train.weight_decay = 5e-4  # weight decay

train.print_freq = 100

# test & eval
test = edict()

test.result_path = './result'

test.day = "all" # note that day and night phases are evaluated in 'all'
test.img_set = f"test-{test.day}-20.txt"

test.input_size = [512, 640]

# test model 
# eval.py (evaluate with pretrained weight)
test.checkpoint = "../pretrained/pretrained.pth.tar"
test.batch_size = 1

# train_eval.py
test.eval_batch_size = 1

# train images' mean & standard variation for normalize
RGB_MEAN, LWIR_MEAN = [0.3465, 0.3219, 0.2842], [0.1598]
RGB_STD,  LWIR_STD  = [0.2358, 0.2265, 0.2274], [0.0813]

# dataset
dataset = edict()
dataset.workers = 8
dataset.OBJ_LOAD_CONDITIONS = {    
                                  'train': {'hRng': (12, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                                  'test': {'hRng': (-np.inf, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                              }

# main
args = edict(path=PATH,
             train=train,
             test=test,
             dataset=dataset)

args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

args.exp_time = None
args.exp_name = str(sys.argv[sys.argv.index("--exp") + 1]) if "--exp" in sys.argv else 'Evaluation'

args.n_classes = 3

args.exps_dir = './exps/'

args.augmentation = ["TT_RandomHorizontalShift",
                     "TT_RandomHorizontalFlip",
                     "TT_RandomResizedCrop"]

## Train dataset transform                             
args["train"].img_transform = Compose([ ColorJitter(0.3, 0.3, 0.3), 
                                        ColorJitterLWIR(contrast=0.3) ])
args["train"].co_transform = Compose([ TT_RandomHorizontalShift(p=0.3, x=-20),
                                       TT_RandomHorizontalFlip(p=0.5, flip=0.5), 
                                       TT_RandomResizedCrop([512,640], scale=(0.25, 4.0), ratio=(0.8, 1.2)), 
                                       ToTensor(), \
                                       Normalize(RGB_MEAN,  RGB_STD, 'R'), \
                                       Normalize(LWIR_MEAN, LWIR_STD, 'T') ], \
                                       args=args)

## Test dataset transform
args["test"].img_transform = Compose([ ])    
args["test"].co_transform = Compose([ Resize(test.input_size), \
                                      ToTensor(), \
                                      Normalize(RGB_MEAN,  RGB_STD, 'R'), \
                                      Normalize(LWIR_MEAN, LWIR_STD, 'T')                        
                                    ])
