import numpy as np

from trainer.FedAvg.server import Server as Base_Server

import torch
from utils.graph import normalize_adj, sd_matrixing


class Server(Base_Server):
    def __init__(self, propagation_hops: int = 2, sfl_alpha=1.0, **config):
        super(Server, self).__init__(**config)
        self.algorithm_name = "SFL"
        self.propagation_hops = propagation_hops
        self.alpha = sfl_alpha

        # update adj matrix
        self.adj_matrix = self.init_adj()

        self.personalized_models = None

    def init_adj(self):
        """ref to (SFL github project/data_util.py/def split_equal_noniid)"""
        # get adj matrix without self loop
        # adj_matrix = fl_adj_by_labels(self.clients, self.num_classes, self_loop=True)
        adj_matrix = np.ones((self.num_clients, self.num_clients)).astype(float)
        # normalize adj matrix
        return torch.tensor(normalize_adj(adj_matrix), dtype=torch.float32)

    """Server working process of SG_FedX"""

    def aggregate(self):
        keys, key_shapes = [], []
        for key, param in self.clients[0].model.state_dict().items():
            keys.append(key)
            key_shapes.append(list(param.data.shape))

        models_dic = [client.model.state_dict() for client in self.clients]
        param_metrix = [sd_matrixing(model_dic).clone().detach() for model_dic in models_dic]
        param_metrix = torch.stack(param_metrix)

        aggregated_param = torch.mm(self.adj_matrix, param_metrix)
        for i in range(self.propagation_hops - 1):
            aggregated_param = torch.mm(self.adj_matrix, aggregated_param)

        new_param_matrix = (self.alpha * aggregated_param) + ((1 - self.alpha) * param_metrix)

        for i in range(len(models_dic)):
            pointer = 0
            for k in range(len(keys)):
                num_p = 1
                for n in key_shapes[k]:
                    num_p *= n
                models_dic[i][keys[k]] = new_param_matrix[i][pointer:pointer + num_p].reshape(key_shapes[k])
                pointer += num_p

        # update clients' personalized models
        for i, client in enumerate(self.clients): client.model.load_state_dict(models_dic[i])
        # read out server's model
        self.read_out()

    def read_out(self):
        Base_Server.aggregate(self)


