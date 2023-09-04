"""Utils for GNNs"""
import numpy as np
import torch
import scipy.sparse as sp


def cal_hat_A(adj_matrix: np.matrix) -> np.matrix:
    """cal the hat_A of GCN"""
    # num_nodes = adj_matrix.shape[0]
    # A = adj_matrix + np.eye(num_nodes)
    A = adj_matrix
    D_vec = np.sum(A, axis=1)
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = np.diag(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    row_sum = np.array(mx.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sd_matrixing(state_dic):
    """
    Turn state dic into a vector
    :param state_dic:
    :return:
    """
    keys = []
    param_vector = None
    for key, param in state_dic.items():
        keys.append(key)
        if param_vector is None:
            param_vector = param.clone().detach().flatten().cpu()
        else:
            if len(list(param.size())) == 0:
                param_vector = torch.cat((param_vector, param.clone().detach().view(1).cpu().type(torch.float32)), 0)
            else:
                param_vector = torch.cat((param_vector, param.clone().detach().flatten().cpu()), 0)
    return param_vector

def calc_ppr_exact(adj_matrix: np.matrix, alpha: float) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    hat_A = cal_hat_A(adj_matrix)
    A_inner = np.eye(num_nodes) - (1 - alpha) * hat_A
    return alpha * np.linalg.inv(A_inner.toarray())
