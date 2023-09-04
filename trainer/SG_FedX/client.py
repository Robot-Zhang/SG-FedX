import copy
from trainer.FedAvg.client import Client as BaseClient
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from utils.base_train import train_model, evaluate_model


class Client(BaseClient):
    """
    Client for SG_FedX

    """

    def __init__(self, hidden_alpha=1.0, **config):
        super(Client, self).__init__(**config)
        self.neighbor = None
        self.hidden_alpha = hidden_alpha

        # hidden of local data samples
        self.g_representation = None
        self.loader_cal_hid = DataLoader(self.train_dataset, batch_size=1)

    def update_neighbor(self, adj_matrix):
        self.neighbor = []
        for j in range(adj_matrix.shape[0]):
            if adj_matrix[self.id][j] > 0:
                self.neighbor.append(j)

    def update_g_representation(self):
        self.model.eval()
        # self.model.to('cpu')
        self.g_representation = 0.
        with torch.no_grad():
            for x, _ in self.loader_cal_hid:
                x = x.to(self.device)
                # calculate loss
                h = self.model(x)
                self.g_representation += h
            self.g_representation = self.g_representation / len(self.train_dataset)
        self.g_representation = self.g_representation * self.hidden_alpha

    def update(self, epochs=1, verbose=0):
        # step 1. model init
        self.model.to(self.device)
        self.model.train()
        # step 2. train loop
        loss_metric = []  # to record avg loss
        for epoch in range(epochs):
            # init loss value
            loss_value, num_samples = 0, 0
            # one epoch train
            for i, (x, y) in enumerate(self.train_loader):
                # put tensor into same device
                x, y = x.to(self.device), y.to(self.device)
                # calculate loss
                h_k = self.model(x)
                y_ = h_k + torch.tile(self.g_representation.to(self.device).detach(),
                                      (h_k.size()[0], 1))
                # if self.glob_iter > 10:
                #     y_ = h_k + torch.tile(self.g_representation.to(self.device).detach(),
                #                           (h_k.size()[0], 1))
                # else:
                #     y_ = h_k
                y_ = F.log_softmax(y_, dim=-1)
                loss = self.loss_fn(y_, y)
                # backward & step optim
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # get loss valur of current bath
                loss_value += loss.item()
                num_samples += y.size(0)

            # Use mean loss value of each epoch as metric
            # Just a reference value, not precise. If you want precise, dataloader should set `drop_last = True`.
            loss_value = loss_value / num_samples
            # if verbose, print training metrics.
            if verbose == 1:
                if self.test_loader is not None and (epoch + 1) % 10 == 0:
                    accuracy, loss_value = evaluate_model(self.model, self.test_loader, self.loss_fn,
                                                          device=self.device, release=False)
                    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_value:.4f}, Accuracy: {accuracy:.3f}')
                    self.model.to(self.device)
                    self.model.train()
                else:
                    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_value:.4f}')
            loss_metric.append(loss_value)
        # step 3. release gpu resource
        # self.model.to('cpu')
        torch.cuda.empty_cache()

        avg_loss = sum(loss_metric) / len(loss_metric)
        return avg_loss