import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch
from torch import nn
import torch.optim as optim
from torch.backends import cudnn
from torchvision.models import resnet18, ResNet18_Weights
import wandb
import logging

DEBUG_BATCHES = 3


class ModelManager():
    def __init__(self,
                 config
                 ):
        self.config = config
        self.out_classes = config.model.kwargs.num_classes
        self.device = config.model.device

        self.net = self.__build__()
        if config.trainer.reload:
            self.__reload_net__(config.trainer.checkpoint)
        self.net.to(self.device)

        self.loss_function = nn.CrossEntropyLoss()
        self.parameters_to_optimize = self.net.parameters()
        self.optimizer = self.__getOptimizer__()
        self.scheduler = self.__getScheduler__()
        wandb.watch(self.net, log_freq=1000)

        # logging.info(f"\nParameters:\n")
        # for k, v in vars(self).items():
        #     pad = ' '.join(['' for _ in range(25-len(k))])
        #     logging.info(k, f" :{pad}", v)

    def __build__(self):
        if self.config.model.name == "resnet18":
            model = resnet18(weights=ResNet18_Weights)
            fc_in = model.fc.in_features
            model.fc = torch.nn.Linear(fc_in, self.config.model.kwargs.num_classes)  # check initialization
            return model
        else:
            raise Exception(f"{self.config.model.name} is not supported!")

    def __getOptimizer__(self):
        if self.config.optimizer.name == "Adam":
            return optim.Adam(self.parameters_to_optimize,
                              lr=self.config.optimizer.learning_rate,
                              weight_decay=self.config.optimizer.weight_decay)
        else:
            raise Exception(f"{self.config.optimizer.name} is not supported!")

    def __getScheduler__(self):
        if self.config.scheduler.name == "OneCycleLR":
            self.kwargs = {
                'max_lr': self.config.optimizer.learning_rate,
                'steps_per_epoch': 1,  
                'epochs': self.config.trainer.epochs,
            }
            return torch.optim.lr_scheduler.OneCycleLR(self.optimizer, **self.kwargs)
        else:
            raise Exception(f"{self.config.scheduler.name} is not supported!")

    def __reload_net__(self, path):
        logging.info(f'\nRestoring model weigths from: {path}')
        self.net = torch.load(path)

    def train(self, train_dataloader, eval_dataloader, checkpoint="example/nn_model.pt", DEVICE="cuda", debug=False):
        cudnn.benchmark = False
        running_corrects = 0
        trainLoss = []
        validationLoss = []
        lowest_val_loss = np.Inf
        STOP = False


        for epoch in range(self.config.trainer.epochs):
            if STOP:
                logging.info(f'\nSTOPPED at epoch {epoch}')
                break
            logging.info('\nStarting epoch {}/{}, LR = {}'.format(epoch + 1, self.config.trainer.epochs,
                                                                  self.scheduler.get_last_lr()))
            tloss = []
            running_corrects = 0
            running_elements = 0
            batch_numb = 0
            for images, labels in tqdm(train_dataloader):
                if debug and batch_numb == DEBUG_BATCHES:
                    break
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                self.net.train()
                self.optimizer.zero_grad()
                outputs = self.net(images)

                loss = self.loss_function(outputs, labels)
                tloss.append(loss.item())

                loss.backward()
                self.optimizer.step()
                self.net.train(False)

                outputs = self.net(images)
                _, preds = torch.max(outputs.data, 1)

                running_corrects += torch.sum(preds == labels.data).data.item()
                running_elements += preds.shape[0]
                batch_numb += 1
            tloss = np.array(tloss)
            tloss = np.mean(tloss)
            trainLoss.append(tloss)
            train_accuracy = running_corrects / float(running_elements)

            self.net.train(False)
            vloss = []
            vlossWeights = []
            running_corrects = 0
            running_elements = 0
            batch_numb = 0
            for i, (images, labels) in enumerate(eval_dataloader):
                if debug and batch_numb == DEBUG_BATCHES:
                    break
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = self.net(images)
                loss = self.loss_function(outputs, labels)
                vloss.append(loss.item())
                vlossWeights.append(images.size(dim=0))
                _, preds = torch.max(outputs.data, 1)
                running_corrects += torch.sum(preds == labels.data).data.item()
                running_elements += preds.shape[0]
                batch_numb += 1
            val_accuracy = running_corrects / float(running_elements)
            vloss = np.array(vloss)
            vloss = np.average(vloss, weights=vlossWeights)
            validationLoss.append(vloss)

            # logging.info(
            #         'train_loss: %.5f'    %tloss,
            #         '\tval_loss: %.5f'    %vloss,
            #         '\tval_acc: %d'       %running_corrects,
            #         '/ %d'                %running_elements,
            #         '= %.3f'              %val_accuracy
            #      )

            wandb.log({
                f'Epoch': epoch,
                f'LR': self.optimizer.param_groups[0]['lr'],
                f'Train/Loss': tloss,
                f'Train/Accuracy': train_accuracy,
                f'Valid/Loss': vloss,
                f'Valid/Accuracy': val_accuracy,
            })

            self.scheduler.step()
            # Early stopping
            if vloss < lowest_val_loss:
                lowest_val_loss = vloss
                patience_counter = 0
                lowest_val_loss_epoch = epoch + 1
                logging.info(
                    "############################################ New lowest_val_loss reached #########################")
                torch.save(self.net, checkpoint)
                wandb.run.summary["Lowest_Validation_Loss/Epoch"] = lowest_val_loss_epoch
                wandb.run.summary["Lowest_Validation_Loss/Validation_Loss"] = lowest_val_loss
                wandb.run.summary["Lowest_Validation_Loss/Validation_Accuracy"] = val_accuracy
            elif patience_counter == self.config.trainer.patience:
                logging.info(f"End of training phase - Patience threshold reached",
                             f'\nWeights Restored from Lowest val_loss epoch: {lowest_val_loss_epoch}',
                             f'\nlowest_val_loss: {lowest_val_loss}')
                STOP = True
            else:
                patience_counter += 1

    def evaluate(self, test_dataloader, checkpoint=None, DEVICE="cuda", best=False):
        cudnn.benchmark=False
        logging.info("test")
        if best:
            net = torch.load(checkpoint)
            logging.info("\n Evalate best model")
        else:
            net = self.net
            logging.info("\n Evalate last model")
        net = net.to(DEVICE)
        net.train(False)
        running_corrects = 0
        running_elements = 0
        tloss = []
        tlossWeights = []
        for images, labels in test_dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = net(images)
            loss = self.loss_function(outputs, labels)
            tloss.append(loss.item())
            tlossWeights.append(images.size(dim=0))
            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data).data.item()
            running_elements += preds.shape[0]

        accuracy = running_corrects / float(running_elements)
        tloss = np.array(tloss)
        tloss = np.average(tloss, weights=tlossWeights)

        if best:
            wandb.run.summary["Lowest_Validation_Loss/Test_Loss"] = tloss
            wandb.run.summary["Lowest_Validation_Loss/Test_Accuracy"] = accuracy
