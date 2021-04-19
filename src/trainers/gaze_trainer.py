import os
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
from torch.utils.tensorboard import SummaryWriter

from src.models.gaze_fc import GazeFC
from src.datasources.unityeyes import UnityEyesDataset


class GazeTrainer(object):
    def __init__(self,
                 model: GazeFC,
                 train_dataset: UnityEyesDataset,
                 val_dataset: UnityEyesDataset,
                 epochs=100,
                 batch_size=32,
                 initial_learning_rate=0.0001,
                 start_epoch=1,
                 version='v0.1',
                 tensorboard_dir='./logs'):
        super(GazeTrainer, self).__init__()
        self.version = version
        self.model = model

        self.train_dataloader = D.DataLoader(train_dataset, batch_size=batch_size)
        self.val_dataloader = D.DataLoader(val_dataset, batch_size=batch_size)

        self.epochs = epochs
        self.batch_size = batch_size
        self.current_learning_rate = initial_learning_rate
        self.start_epoch = start_epoch

        self.loss_obj = nn.MSELoss(reduction='mean')
        self.optimizer = None
        if not os.path.exists(os.path.join(tensorboard_dir)):
            os.makedirs(os.path.join(tensorboard_dir))
        self.summary_writer_train = SummaryWriter(tensorboard_dir + f'/train-{self.version}')
        self.summary_writer_val = SummaryWriter(tensorboard_dir + f'/val-{self.version}')

        self.patience_count = 0
        self.max_patience = 5
        self.last_val_loss = math.inf
        self.lowest_val_loss = math.inf
        self.best_model = None

    def lr_decay(self):
        """
        This effectively simulate ReduceOnPlateau learning rate schedule. Learning rate
        will be reduced by a factor of 5 if there's no improvement over [max_patience] epochs
        """
        if self.patience_count >= self.max_patience:
            self.current_learning_rate /= 10.0
            self.patience_count = 0
        elif self.last_val_loss == self.lowest_val_loss:
            self.patience_count = 0
        self.patience_count += 1
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.current_learning_rate, weight_decay=1e-4)

    def compute_coord_loss(self, predict, label):
        loss = self.loss_obj(predict, label)
        return loss

    def compute_angular_loss(self, predict, label):
        """Pytorch method to calculate angular loss (via cosine similarity)"""
        def angle_to_unit_vectors(y):
            sin = torch.sin(y)
            cos = torch.cos(y)
            return torch.stack([
                cos[:, 0] * sin[:, 1],
                sin[:, 0],
                cos[:, 0] * cos[:, 1],
            ], dim=1)

        a = angle_to_unit_vectors(predict)
        b = angle_to_unit_vectors(label)
        ab = torch.sum(a*b, dim=1)
        a_norm = torch.sqrt(torch.sum(torch.square(a), dim=1))
        b_norm = torch.sqrt(torch.sum(torch.square(b), dim=1))
        cos_sim = ab / (a_norm * b_norm)
        cos_sim = torch.clip(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        ang = torch.acos(cos_sim) * 180. / math.pi
        return torch.mean(ang)

    def train_step(self, inputs):
        ldmks_input = inputs['landmarks'].cuda()
        gaze_label = inputs['gaze'].cuda()

        gaze_predict = self.model(ldmks_input)
        loss_gaze = self.compute_angular_loss(gaze_predict, gaze_label)

        self.model.zero_grad()
        loss_gaze.backward()
        self.optimizer.step()
        return loss_gaze.item()

    def val_step(self, inputs):
        ldmks_input = inputs['landmarks'].cuda()
        gaze_label = inputs['gaze'].cuda()

        gaze_predict = self.model(ldmks_input)
        loss_gaze = self.compute_angular_loss(gaze_predict, gaze_label)
        return loss_gaze.item()

    def run(self):
        def train_epoch(dataset):
            print('Start training...')
            total_loss_gaze = 0.0
            num_train_batches = 0.0

            # start_time = time.clock()

            for one_batch in dataset:

                # print(time.clock() - start_time)
                # start_time = time.clock()

                batch_loss_gaze = self.train_step(one_batch)
                total_loss_gaze += batch_loss_gaze
                num_train_batches += 1
                if num_train_batches % 500 == 0:
                    print('Trained batch:', num_train_batches,
                          'Batch loss:', batch_loss_gaze,
                          'Epoch total loss:', total_loss_gaze)
                if num_train_batches >= 200000 / self.batch_size:
                    break

                # print(time.clock() - start_time)
                # start_time = time.clock()

            return total_loss_gaze / num_train_batches

        def val_epoch(dataset):
            print('Start validating...')
            total_loss_gaze = 0.0
            num_val_batches = 0.0
            for one_batch in dataset:
                batch_loss_gaze = self.val_step(one_batch)
                total_loss_gaze += batch_loss_gaze
                num_val_batches += 1
                if num_val_batches % 500 == 0:
                    print('Validated batch:', num_val_batches,
                          'Batch loss:', batch_loss_gaze,
                          'Epoch total loss:', total_loss_gaze)
                if num_val_batches >= 40000 / self.batch_size:
                    break
            return total_loss_gaze / num_val_batches

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.lr_decay()
            self.summary_writer_train.add_scalar('epoch learning rate', self.current_learning_rate, epoch)

            print('Start epoch {} with learning rate {}'.format(epoch, self.current_learning_rate))

            train_loss_gaze = train_epoch(self.train_dataloader)
            print('Epoch {} train loss {}'.format(epoch, train_loss_gaze))
            self.summary_writer_train.add_scalar('epoch loss gaze', train_loss_gaze, epoch)

            val_loss_gaze = val_epoch(self.val_dataloader)
            print('Epoch {} val loss {}'.format(epoch, val_loss_gaze))
            self.summary_writer_val.add_scalar('epoch loss gaze', val_loss_gaze, epoch)

            # save model when reach a new lowest validation loss
            if val_loss_gaze < self.lowest_val_loss:
                if not os.path.exists(os.path.join('./models')):
                    os.makedirs(os.path.join('./models'))
                model_name = './models/model-Gaze-{}-epoch-{}-loss-{:.4f}.pth'.format(self.version, epoch, val_loss_gaze)
                torch.save(self.model, model_name)
                print(f'Save model at: {model_name}')
                self.best_model = model_name
                self.lowest_val_loss = val_loss_gaze
            self.last_val_loss = val_loss_gaze

        return self.best_model
