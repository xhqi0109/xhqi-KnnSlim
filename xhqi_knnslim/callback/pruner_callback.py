import os
import copy

import torch
import matplotlib.pyplot as plt

import sys
from xhqi_knnslim.utils import enums, pruner_utils

class ADMMBaseCallback:
    def __init__(self, admm_pruner, ckpt_dir='checkpoints', stage='pretrain'):
        """
        Initializes the ADMMBaseCallback.

        Args:
            admm_pruner (object): The ADMM pruner object.
            ckpt_dir (str): Directory to save checkpoints. Defaults to 'checkpoints'.
            stage (str): Stage of ADMM training. Defaults to 'pretrain'.
        """
        self.admm_pruner = admm_pruner
        self.ckpt_dir = admm_pruner.ckpt_dir
        
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        self.stage = stage

    def on_train_begin(self, optimizer):
        """ 
        Executed before the training starts.
        """
        if self.admm_pruner.resume:
            pruner_utils.load_breakpoint(self.admm_pruner)
        elif self.admm_pruner.load_model:
            pruner_utils.load_pretrained_model(self.admm_pruner)
        else:
            print("No need to load checkpoint")
        return self.admm_pruner.model, optimizer, self.admm_pruner.start_epoch

    def on_epoch_begin(self, epoch, optimizer=None):
        """ 
        Executed before each epoch of training.
        """
        self.admm_pruner.logger.info(f'Epoch: {epoch}')

    def on_batch_begin(self):
        """ 
        Executed before each batch of training.
        """
        pass

    def on_batch_end(self):
        """ 
        Executed after each batch of training.
        """
        pass

    def on_epoch_end(self, epoch, optimizer, is_best=False):
        """ 
        Executed after each epoch of training.
        """
        ckpt_path = f'./{self.ckpt_dir}/pid_{self.admm_pruner.pid}_{self.stage}_latest.pt'
        pruner_utils.save_checkpoint(self.admm_pruner, epoch, optimizer, ckpt_path)
        if is_best:
            ckpt_path = f'./{self.ckpt_dir}/pid_{self.admm_pruner.pid}_{self.stage}_best.pt'
            pruner_utils.publish(self.admm_pruner, ckpt_path)

    def on_train_end(self):
        """ 
        Executed after the training ends.
        """
        ckpt_path = f'./{self.ckpt_dir}/pid_{self.admm_pruner.pid}_{self.stage}.pt'
        pruner_utils.publish(self.admm_pruner, ckpt_path)

    def on_admm_loss(self):
        """
        Returns the ADMM loss.
        """
        return torch.zeros([], device=self.admm_pruner.device)


class ADMMPretrainCallback(ADMMBaseCallback):
    def __init__(self, admm_pruner):
        super().__init__(admm_pruner, stage='pretrain')


class ADMMPruneCallback(ADMMBaseCallback):
    def __init__(self, admm_pruner):
        super().__init__(admm_pruner, stage='prune')
        self.epochs = []
        self.admm_losses = []
        self.result_dir = './admm_loss_figure'
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir, exist_ok=True)

    def on_train_begin(self, optimizer):
        super().on_train_begin(optimizer)
        self.admm_pruner.logger.info('Initializing ADMM variables: Z and U')
        self.admm_pruner.initialize_z_u()
        return self.admm_pruner.model, optimizer, self.admm_pruner.start_epoch

    def on_epoch_begin(self, epoch, optimizer=None):
        super().on_epoch_begin(epoch)
        self.admm_pruner.update_z_u(epoch)
        lr = self.admm_pruner.adjust_learning_rate(optimizer, epoch)
        self.admm_pruner.logger.info(f'Current learning rate: {lr}')

    def on_epoch_end(self, epoch, optimizer, is_best=False):
        super().on_epoch_end(epoch, optimizer, is_best)
        admm_loss = self.admm_pruner.get_admm_loss().item()
        self.admm_pruner.logger.info(f'ADMM loss: {admm_loss}')
        self.epochs.append(epoch)
        self.admm_losses.append(admm_loss)

    def on_train_end(self):
        fig, ax = plt.subplots()
        ax.plot(self.epochs, self.admm_losses,
                label=f'rho_{self.admm_pruner.rho}_admm_epoch_{self.admm_pruner.admm_epoch}')
        ax.set_ylabel('ADMM loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        plt.savefig(f'./{self.result_dir}/pid_{self.admm_pruner.pid}_{self.stage}_admm_loss.jpg')

        super().on_train_end()

    def on_admm_loss(self):
        return self.admm_pruner.get_admm_loss()


class ADMMRetrainCallback(ADMMBaseCallback):
    def __init__(self, admm_pruner):
        super().__init__(admm_pruner, stage='retrain')

    def on_train_begin(self, optimizer):
        """
        Executed before the training starts.
        """
        super().on_train_begin(optimizer)
        self.admm_pruner.step()

        optimizer.__init__(self.admm_pruner.model.parameters(), **optimizer.defaults)
        self.admm_pruner.optimizer = optimizer

        return self.admm_pruner.model, self.admm_pruner.optimizer, self.admm_pruner.start_epoch
