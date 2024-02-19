import os
import sys
import time
import wandb
import argparse
import numpy as np
from typing import Dict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
import logging
from datasets import KAISTPed
from inference import val_epoch, save_results

from model import INSANet, MultiBoxLoss

from utils import utils
from utils.evaluation_script import evaluate

# Fix random seed for reproduction
utils.set_seed(seed=9)

# Parser (shell script)
parser = argparse.ArgumentParser(description='PyTorch INSA Train & Test')
parser.add_argument('--wandb_enable', default=False, type=bool, help='wandb enabled?')
parser.add_argument('--exp', type=str, help='set experiments name. see ./exps/')

args, config_args = parser.parse_known_args()
sys.argv[1:] = config_args

arg = parser.parse_args()


def main():
    """ Train and validate a model """
    args = config.args
    train_conf = config.train
    checkpoint = train_conf.checkpoint
    start_epoch = train_conf.start_epoch
    epochs = train_conf.epochs
    phase = "Multispectral"

    if arg.wandb_enable:
        wandb.run.log_code("./", include_fn=lambda path: path.endswith(".py"))

    # Initialize model
    if checkpoint is None:
        model = INSANet(n_classes=args.n_classes)
        # Initialize the optimizer
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * train_conf.lr},
                                            {'params': not_biases}],
                                    lr=train_conf.lr,
                                    momentum=train_conf.momentum,
                                    weight_decay=train_conf.weight_decay,
                                    nesterov=False)

        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                               milestones=[int(epochs * 0.5), int(epochs * 0.9)],
                                                               gamma=0.1)

    # Load model from checkpoint
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        train_loss = checkpoint['loss']
        print('\nLoaded checkpoint from epoch %d. Current loss is %.3f.\n' % (start_epoch, train_loss))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.5)], gamma=0.1)
    
    # Move to default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = nn.DataParallel(model)
    criterion = MultiBoxLoss(priors_cxcy=model.module.priors_cxcy).to(device)

    train_dataset = KAISTPed(args, condition='train')
    train_loader = DataLoader(train_dataset, batch_size=train_conf.batch_size, shuffle=True,
                              num_workers=config.dataset.workers,
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)

    test_dataset = KAISTPed(args, condition='test')
    test_batch_size = args["test"].eval_batch_size * torch.cuda.device_count()
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                             num_workers=config.dataset.workers,
                             collate_fn=test_dataset.collate_fn,
                             pin_memory=True)
  
    # Set experiments setting
    if args.exp_time is None:
        args.exp_time = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
 
    exps_dir = os.path.join('exps', args.exp_time + '_' + args.exp_name)
    os.makedirs(exps_dir, exist_ok=True)
    args.exps_dir = exps_dir
    print(f'exp_name: {args.exp_name}')

    # Make logger
    logger = utils.make_logger(args)

    # Epochs
    kwargs = {'print_freq': args['train'].print_freq}
    for epoch in range(start_epoch, epochs):
        # Model training
        if arg.wandb_enable:
            wandb.log({"Epoch": epoch})
        logger.info('#' * 20 + f' << Epoch {epoch:3d} >> ' + '#' * 20)
        train_loss = train_epoch(epoch=epoch,
                                 model=model,
                                 dataloader=train_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 logger=logger,
                                 **kwargs)

        optim_scheduler.step()

        # Save checkpoint
        # At early training epoch, you might see OOM in validation phase 
        if epoch >= 5:
            result_filename = os.path.join(jobs_dir, f'Epoch{epoch:03d}_test_det.txt')

            results = val_epoch(model, test_loader, config.test.input_size, min_score=0.1, epoch=epoch)
            
            save_results(results, result_filename)
            
            evaluate(config.PATH.JSON_GT_FILE, result_filename, phase) 


def train_epoch(epoch: int,
                model: INSANet,
                dataloader: torch.utils.data.DataLoader,
                criterion: MultiBoxLoss,
                optimizer: torch.optim.Optimizer,
                logger: logging.Logger,
                **kwargs: Dict) -> float:
    """
    Train the model during an epoch
    
    :param model: INSA network for multispectral pedestrian detection defined by src/model.py
    :param dataloader: Dataloader instance to feed training data (images, labels, etc) for KAISTPed dataset
    :param criterion: Compute multibox loss for single-shot detection
    :param optimizer: Pytorch optimizer(e.g. SGD, Adam, etc)
    :param logger: Logger instance
    :param kwargs: Other parameters to control print_freq
    :return: A single sclara value for averaged loss
    """

    device = next(model.parameters()).device
    model.train()

    batch_time = utils.AverageMeter()  # forward prop. + back prop. time
    data_time = utils.AverageMeter()   # data loading time
    losses_sum = utils.AverageMeter()  # loss_sum
    
    start = time.time()

    # Batches
    for batch_i, (image_vis, image_lwir, boxes, labels, _ ) in enumerate(dataloader):
        data_time.update(time.time() - start)

        # Move to default device
        image_vis = image_vis.to(device)
        image_lwir = image_lwir.to(device)
        
        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]
        
        # Forward prop.
        predicted_locs, predicted_scores = model(image_vis, image_lwir)  # (N, 41760, 4), (N, 41760, n_classes)

        # Loss
        loss, cls_loss, loc_loss, n_positives = criterion(predicted_locs, predicted_scores, boxes, labels) # (scalar,)
        
        if arg.wandb_enable:
            wandb.log({"loss":loss, "cls_loss":cls_loss, "loc_loss":loc_loss})

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Update model
        optimizer.step()

        losses_sum.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if batch_i % kwargs.get('print_freq', 10) == 0:
            logger.info('Iteration: [{0}/{1}]\t'
                        'Batch Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data Time: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Num of Positive: {positive}'.format(batch_i, len(dataloader),
                                                             batch_time=batch_time,
                                                             data_time=data_time,
                                                             loss=losses_sum,
                                                             positive=n_positives))
        
    return losses_sum.avg


if __name__ == '__main__':
    main()
