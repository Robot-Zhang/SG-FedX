import copy
import os
from typing import List
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import spectral_clustering

from trainer.FedAvg.server import Server as Base_Server
from utils.graph import normalize_adj, sd_matrixing, cal_hat_A
import math
import torch

from .community_tree import CommunityTree

import os

SG_FedX_Variants = ['Cluster', 'Adj', 'X', 'GCN', 'APPNP']


class Server(Base_Server):
    def __init__(self, propagation_hops=2, gfl_alpha=0.9, fed_x='X', **config):
        super(Server, self).__init__(**config)
        self.algorithm_name = f"SG_Fed{fed_x}"
        assert fed_x in SG_FedX_Variants, f"SG_Fed{fed_x} not implemented yet!"
        self.fed_x = fed_x
        self.propagation_hops = propagation_hops

        self.alpha = gfl_alpha

        # metrics
        self.init_se_metric()

    """Server working process of SG_FedX"""

    def init_se_metric(self):
        self.se_metric = {'knn': [], 'se-knn': [], 'se-tree': [],
                          'op_se1': [], 'op_seh': [], 'round': []}
        # dir for save sim matrix
        self.log_dir_sim = os.path.join(self.log_dir, 'sim_matrix')
        os.makedirs(self.log_dir_sim, exist_ok=True)
        # dir for save knn adj (with self loops)
        self.log_dir_knn = os.path.join(self.log_dir, 'knn_adj')
        os.makedirs(self.log_dir_knn, exist_ok=True)
        # dir for save com tree
        self.log_dir_tree = os.path.join(self.log_dir, 'com_tree')
        os.makedirs(self.log_dir_tree, exist_ok=True)
        # dir for save calibrated_adj
        self.log_dir_calibrate = os.path.join(self.log_dir, 'calibrated_adj')
        os.makedirs(self.log_dir_calibrate, exist_ok=True)

    def distribute_model(self):
        if self.glob_iter > 1 and self.fed_x == 'Cluster':
            return None
        Base_Server.distribute_model(self)

    def local_update(self, epochs):
        # extract representation with graph representations
        for client in self.selected_clients:
            client.update_g_representation()
        Base_Server.local_update(self, epochs)

    def aggregate(self):
        self.se_metric['round'].append(self.glob_iter + 1)

        # step 1. update graph and community tree
        # 1 - get adj matrix (diag is 1)
        grads = self.get_grads()
        adj_matrix = self.get_sim(grads)  # note that all adj matrix here with a self loop
        # 2 - get optimal knn adj
        se_knn_op, k_op, knn_adj = self.opt_knn(adj_matrix)
        # 3 - get optimal tree
        se_tree_op, tree = self.build_tree(adj_matrix)
        hie_communities = [tree.get_communities(d + 1)
                           for d in range(tree.tree.depth() - 1)]
        # 4 - sample based adj tune
        calibrated_adj, op_se1, op_seh = self.calibrate_graph(
            adj_matrix, se_knn_op, knn_adj, se_tree_op, hie_communities, tree)

        # step 2. aggregate by different variants
        if self.fed_x == 'X':
            self._aggregate_X(calibrated_adj)
        elif self.fed_x == 'Adj':
            self._aggregate_neighbor(calibrated_adj)
        elif self.fed_x == 'Cluster':
            # here, we use the highest cluster to aggregate
            self._aggregate_cluster(hie_communities[0])
        else:
            self._aggregate_graph(adj_matrix)

        # step 3. readout as server's model
        if self.fed_x != 'X':
            Base_Server.aggregate(self)

    def save_metric(self):
        Base_Server.save_metric(self)
        pd.DataFrame(self.se_metric).to_csv(
            os.path.join(self.log_dir, "se_metric.csv"), index=False)

    def get_grads(self):  # -> np.ndarray:
        new_params = [torch.cat([param.view(-1) for param in list(client.model.parameters())],
                                dim=0).view(1, -1)
                      for client in self.clients]
        old_param = torch.cat([param.view(-1) for param in list(self.model.parameters())],
                              dim=0).view(1, -1)  # .to('cpu')
        grads = [new_params[i] - old_param if i in self.selected_clients_ids else 0.
                 for i in range(self.num_clients)]
        grads = torch.squeeze(torch.stack(grads, dim=0))
        return grads.detach().cpu().numpy()

    def get_sim(self, grads):
        """update sim and by clients' gradients"""
        # sim_matrix = np.array([[0] * self.num_clients] * self.num_clients).astype(np.float)
        # for i in range(self.num_clients):
        #     for j in range(self.num_clients):
        #         if (i in self.selected_clients_ids) and (j in self.selected_clients_ids):
        #             sim_matrix[i][j] = F.cosine_similarity(grads[i], grads[j])
        sim_matrix = cosine_similarity(grads)
        # save sim matrix
        path = os.path.join(self.log_dir_sim, f'{self.glob_iter + 1}.txt')
        np.savetxt(path, sim_matrix)

        # get degree matrix
        adj_matrix = (sim_matrix > 0) * sim_matrix  # degree should be larger than 0
        return adj_matrix

    def opt_knn(self, adj_matrix):
        """get the optimal knn graph from the perspective of structural entropy"""
        h_op, k_op, knn_adj = 0, -1, None
        # adj_matrix without self loop
        adj_no_sl = adj_matrix * (1 - np.eye(self.num_clients))
        for k in range(1, len(self.selected_clients)):
            adj_k = get_knn(adj_no_sl, k)
            h_k = cal_one_dim_se(adj_k + np.eye(self.num_clients))
            if h_k > h_op: h_op, k_op, knn_adj = h_k, k, adj_k
        # add self loop
        knn_adj = knn_adj + np.eye(self.num_clients)
        # save knn_adj
        path = os.path.join(self.log_dir_knn, f'{self.glob_iter + 1}.txt')
        np.savetxt(path, knn_adj)
        # record SE metrics
        self.se_metric['knn'].append(k_op)
        self.se_metric['se-knn'].append(h_op)
        return h_op, k_op, knn_adj

    def build_tree(self, adj_matrix):
        """build tree by hierarchical spectral cluster"""
        # step 1. get k-partitions of clients by spectral cluster
        # partitions： [( num_partitions, ndarray([c_1,c_2,c_1] )), ... ]
        partitions = [(1, np.zeros(self.num_clients))]
        for i in range(2, self.num_clients):
            part = spectral_clustering(adj_matrix, n_clusters=i)
            partitions.append((i, part))

        # step 2. build init candidate trees by partitions
        trees = [CommunityTree(self.clients, [partitions[i]])
                 for i in range(self.num_clients - 1)]

        # step 3. iter search op tree
        h_op, tree_op = np.inf, None
        while len(trees) > 0:
            # 3-1 find if higher tree better
            flag = True  # flag for check if this height entropy decrease
            # find the best tree in this level
            for tree in trees:
                h_tree = cal_h_dim_se(adj_matrix, tree)
                if h_tree < h_op: h_op, tree_op, flag = h_tree, tree, False
            if flag: break  # if higher level don't have better one, break

            # update candidate trees in (level + 1)
            trees = []
            num_part, part = tree_op.hiera_part[-1]
            for i in range(1, num_part):
                num_p, p = partitions[i]
                p_set_list = []
                for j in range(num_p):
                    idx = np.argwhere(p == j)
                    p_set = set(part[idx.squeeze()]) if idx.shape[0] > 1 else set([part[idx.squeeze()]])
                    p_set_list.append(p_set)
                # check if higher level condition is satisfied.
                union_set = set.union(*p_set_list)
                inter_set_list = []
                for j in range(num_p):
                    for k in range(num_p):
                        if j != k: inter_set_list.append(p_set_list[j] & p_set_list[k])
                union_inter_set = set.union(*inter_set_list)
                if union_set == set(range(num_part)) and len(union_inter_set) == 0:
                    trees.append(CommunityTree(self.clients, tree_op.hiera_part + [partitions[i]]))
        # save tree
        f_name = os.path.join(self.log_dir_tree, f'{self.glob_iter + 1}.dot')
        tree_op.tree.to_graphviz(filename=f_name)
        # record metric
        self.se_metric['se-tree'].append(h_op)
        return h_op, tree_op

    def calibrate_graph(self, adj_matrix, se_knn_op, knn_adj, se_tree_op, hie_communities, tree):
        """calibrate the knn graph by hie_communities"""
        calibrated_adj = copy.deepcopy(knn_adj)
        op_se1, op_seh = 0, 0
        # 1. add edges in same communities, traversal in top-1 level is enough
        top_com = hie_communities[0]
        for com in top_com:
            for i in range(len(com)):
                for j in range(i + 1, len(com)):
                    # add edges
                    hat_adj = copy.deepcopy(calibrated_adj)
                    hat_adj[i][j] = adj_matrix[i][j]
                    hat_adj[j][i] = adj_matrix[i][j]
                    se_1, se_h = cal_one_dim_se(hat_adj), cal_h_dim_se(hat_adj, tree)
                    if se_1 > se_knn_op and se_h < se_tree_op:
                        calibrated_adj, op_se1, op_seh = hat_adj, se_1, se_h

        # 2. prune edges cross communities
        for com_level in hie_communities:
            for start_c in range(len(com_level)):
                for end_c in range(start_c + 1, len(com_level)):
                    for i in com_level[start_c]:
                        for j in com_level[end_c]:
                            # prune edges
                            hat_adj = copy.deepcopy(calibrated_adj)
                            hat_adj[i][j] = 0
                            hat_adj[j][i] = 0
                            se_1, se_h = cal_one_dim_se(hat_adj), cal_h_dim_se(hat_adj, tree)
                            if se_1 > se_knn_op and se_h < se_tree_op:
                                calibrated_adj, op_se1, op_seh = hat_adj, se_1, se_h
        # save calibrated_adj
        path = os.path.join(self.log_dir_calibrate, f'{self.glob_iter + 1}.txt')
        np.savetxt(path, calibrated_adj)
        # record SE metrics
        self.se_metric['op_se1'].append(op_se1)
        self.se_metric['op_seh'].append(op_seh)
        return calibrated_adj, op_se1, op_seh

    def _aggregate_neighbor(self, adj_matrix):
        """For each selected client, aggregate neighbor message """
        for client in self.selected_clients: client.update_neighbor(adj_matrix)

        # get samples and weights
        n = [client.num_samples if client.id in self.selected_clients_ids else 0
             for client in self.clients]
        w = [client.model.state_dict() for client in self.clients]

        w_agg = copy.deepcopy(w)
        for i in range(self.num_clients):
            for key in w_agg[i].keys():
                w_agg[i][key] = 0.

        # aggregated weights
        for client in self.selected_clients:
            n_sum = sum([n[i] for i in client.neighbor])
            for k in client.neighbor:
                for key in w[k].keys():
                    w_agg[client.id][key] += (n[k] / n_sum) * w[k][key]

        for client in self.selected_clients: client.model.load_state_dict(w_agg[client.id])

    def _aggregate_cluster(self, communities):
        w_coms = [copy.deepcopy(self.model.state_dict()) for _ in communities]

        # aggregate model of communities
        for i in range(len(communities)):
            community_clients_ids = communities[i]
            msg_list = []
            for j in community_clients_ids:
                if j in self.selected_clients_ids:
                    msg_list.append((self.clients[j].num_samples, self.clients[j].model.state_dict()))
            w_coms[i] = self.avg_weights(msg_list)

            # distribute community models to clients
            for j in community_clients_ids:
                if j in self.selected_clients_ids:
                    self.clients[j].model.load_state_dict(w_coms[i])

    def _aggregate_graph(self, adj_matrix):
        keys, key_shapes = [], []
        for key, param in self.clients[0].model.state_dict().items():
            keys.append(key)
            key_shapes.append(list(param.data.shape))

        models_dic = [client.model.state_dict() for client in self.clients]
        param_metrix = [sd_matrixing(model_dic).clone().detach() for model_dic in models_dic]
        param_metrix = torch.stack(param_metrix)

        hat_adj = torch.tensor(cal_hat_A(adj_matrix), dtype=torch.float32)
        if self.fed_x == 'GCN':
            aggregated_param = torch.mm(hat_adj, param_metrix)
            for i in range(self.propagation_hops - 1):
                aggregated_param = torch.mm(hat_adj, aggregated_param)
        elif self.fed_x == 'APPNP':
            H = torch.mm(hat_adj, param_metrix)
            aggregated_param = torch.mm(hat_adj, param_metrix)
            for i in range(self.propagation_hops):
                aggregated_param = (1 - self.alpha) * torch.mm(hat_adj, aggregated_param) + self.alpha * H
        else:
            raise NotImplemented
        new_param_matrix = aggregated_param

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

    def _aggregate_X(self, adj_matrix):
        adj_matrix = (adj_matrix > 0).astype(float)
        adj_matrix = torch.tensor(normalize_adj(adj_matrix), dtype=torch.float32)
        keys, key_shapes = [], []
        for key, param in self.clients[0].model.state_dict().items():
            keys.append(key)
            key_shapes.append(list(param.data.shape))

        models_dic = [client.model.state_dict() for client in self.clients]
        param_metrix = [sd_matrixing(model_dic).clone().detach() for model_dic in models_dic]
        param_metrix = torch.stack(param_metrix)

        aggregated_param = torch.mm(adj_matrix, param_metrix)
        for i in range(self.propagation_hops - 1):
            aggregated_param = torch.mm(adj_matrix, aggregated_param)

        new_param_matrix = (self.alpha * aggregated_param) + ((1 - self.alpha) * param_metrix)

        for i in range(len(models_dic)):
            pointer = 0
            for k in range(len(keys)):
                num_p = 1
                for n in key_shapes[k]:
                    num_p *= n
                models_dic[i][keys[k]] = new_param_matrix[i][pointer:pointer + num_p].reshape(key_shapes[k])
                pointer += num_p

        msg_list = [(client.num_samples, models_dic[client.id])
                    for client in self.selected_clients]
        w_dict = self.avg_weights(msg_list)
        self.model.load_state_dict(w_dict)


"""Funcs for calculating SE and getting KNN"""


def get_knn(adj_matrix, k):
    """
    Get a k-NN graph of the fully connected one
    :param adj_matrix: adjacent matrix
    :param k: number of nearset neighbors
    """
    mask = []
    # for each node keep the most nearst neighbor
    for i in range(adj_matrix.shape[0]):
        max_ks = list(np.argsort(adj_matrix[i])[-k:])
        mask.append([j in max_ks for j in range(adj_matrix.shape[0])])
    mask = np.array(mask)
    # FL is an undirected graph.
    # Here, we ensure that all nodes have at least k neighbors.
    mask = mask | mask.T
    return mask * adj_matrix


def cal_one_dim_se(adj_matrix):
    """calculate 1-dim structure entropy of graph"""
    # cal degree and volume
    d = [0] * adj_matrix.shape[0]
    for k in range(len(d)):
        d[k] = sum(adj_matrix[k])
    vol_g = sum(d)

    h = 0
    for k in range(len(d)):
        h = h - ((d[k] / vol_g) * math.log2(d[k] / vol_g))
    return h


def cal_h_dim_se(adj_matrix, tree: CommunityTree):
    d = [0] * adj_matrix.shape[0]
    for k in range(len(d)):
        d[k] = sum(adj_matrix[k])
    vol_g, h = sum(d), 0

    for alpha in tree.get_alpha_nodes():
        v_alpha, v_alpha_, g_alpha = 0, 0, 0
        # get $V_\alpha$ and $g_\alpha$
        idx_alpha = [n.identifier for n in tree.get_leaves(alpha)]
        for i in range(len(d)):
            if i in idx_alpha:
                v_alpha += sum([adj_matrix[i][j] for j in idx_alpha])
                g_alpha += sum([adj_matrix[i][j] if j not in idx_alpha else 0
                                for j in range(len(d))])
        # get $V_\alpha_{-}$
        idx_alpha_ = [n.identifier for n in tree.get_leaves(tree.get_parent(alpha))]
        for i in range(len(d)):
            if i in idx_alpha_: v_alpha_ += sum([adj_matrix[i][j] for j in idx_alpha_])

        h = h - ((g_alpha / vol_g) * math.log2(v_alpha / v_alpha_))
    return h
