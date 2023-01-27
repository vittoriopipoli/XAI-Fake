import sys
import os
import argparse
import logging
import logging.config
import shutil
import yaml
import random
import time

import numpy as np
import torch
from torchvision import transforms

from hashlib import shake_256
from munch import munchify, unmunchify
import wandb

from model.model_manager import ModelManager
from dataloader.Caltech import Caltech
from torch.utils.data import Subset, DataLoader

# used to generate random names that will be appended to the
# experiment name
def timehash():
    t = time.time()
    t = str(t).encode()
    h = shake_256(t)
    h = h.hexdigest(5) # output len: 2*5=10
    return h.upper()

def setup(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="config.yaml", help="the config file to be used to run the experiment", required=True)
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")
    arg_parser.add_argument("--debug", action='store_true', help="debug, no wandb")
    args = arg_parser.parse_args()

    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit

    # Munchify the dict to access entries with both dot notation and ['name']
    logging.info(f'Loading the config file...')
    config = yaml.load(open(args.config, "r"), yaml.FullLoader)
    config = munchify(config)

    # Setup to be deterministic
    logging.info(f'setup to be deterministic')
    setup(config.seed)

    if args.debug:
        os.environ['WANDB_DISABLED'] = 'true'

    # start wandb
    wandb.init(
        project="XAI-Fake", 
        entity="xai_fake",
        config=unmunchify(config),
        mode=config.wandb.mode
    )

    # Check if project_dir exists
    if not os.path.exists(config.project_dir):
        logging.error("Project_dir does not exist: {}".format(config.project_dir))
        raise SystemExit

    # check if preprocessing is set and file exists
    logging.info(f'loading preprocessing')
    if config.data_loader.preprocessing is None:
        preprocessing = []
    elif not os.path.exists(config.data_loader.preprocessing):
        logging.error("Preprocessing file does not exist: {}".format(config.data_loader.preprocessing))
        preprocessing = []
    else:
        with open(config.data_loader.preprocessing, 'r') as preprocessing_file:
            preprocessing = yaml.load(preprocessing_file, yaml.FullLoader)

    # check if augmentation is set and file exists
    logging.info(f'loading augmentation')
    if config.data_loader.augmentation is None:
        augmentation = []
    elif not os.path.exists(config.data_loader.augmentation):
        logging.error("augmentation file does not exist: {}".format(config.data_loader.augmentation))
        augmentation = []
    else:
        with open(config.data_loader.augmentation, 'r') as augmentation_file:
            augmentation = yaml.load(augmentation_file, yaml.FullLoader)            

    # make title unique to avoid overriding
    config.title = f'{config.title}_{timehash()}'
    os.makedirs(config.title, exist_ok=True)
    logging.info(f'project directory: {config.title}')

    # Setup logger's handlers
    file_handler = logging.FileHandler(os.path.join(config.title, 'output.log'))
    log_format = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    if args.verbose:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_format)
        logger.addHandler(stdout_handler)

    # Copy config file to project_dir, to be able to reproduce the experiment
    copy_config_path = os.path.join(config.title, 'config.yaml')
    shutil.copy(args.config, copy_config_path)

    mm = ModelManager(config)

    # THE FOLLOWING TRANSFORMATIONS MUST BE CREATED ACCORDINGLY TO THE DATALOADER/TRANSFORMS.PY, PREPROCESSING, AUGMENTATIONS AND CONFIG(DATALOADER.NORMALIZE) YAML FILES
    train_transform = transforms.Compose([  transforms.Resize(256),      
                                            transforms.CenterCrop(224),                           
                                            transforms.ToTensor(), 
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    eval_transform = transforms.Compose([   transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                    
    ])
    # THE FOLLOWING IS A TOY DATASET 
    # MOST OF THE FOLLOWING INSTRUCTIONS MUST BE WRAPPED IN A DATALOADER CLASS
    train_dataset = Caltech(split='train', transform=train_transform)
    test_dataset = Caltech(split='test', transform=eval_transform)
    # SPLIT TRAIN IN TRAIN AND VAL
    train_indexes, val_indexes = train_dataset.retrieveTrainVal()
    val_dataset = Subset(train_dataset, val_indexes)
    train_dataset = Subset(train_dataset, train_indexes)
    # CREATE DATALOADERS
    train_dataloader = DataLoader(train_dataset, batch_size=config.data_loader.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.data_loader.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=config.data_loader.batch_size, shuffle=False, num_workers=4)

    if config.trainer.reload and not os.path.exists(config.trainer.checkpoint):
        logging.error(f'Checkpoint file does not exist: {config.trainer.checkpoint}')
        raise SystemExit        

    # Train the model
    if config.trainer.do_train:
        logging.info('Training...')
        mm.train(train_dataloader, val_dataloader)

    # Test the model
    if config.trainer.do_test:
        logging.info('Testing the model...')
        mm.evaluate(test_dataloader)



