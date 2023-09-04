import os
import json
from collections import defaultdict
from omegaconf import OmegaConf
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset
from .my_transforms import get_transform

DEFAULT_CONF_PATH = "configs/dataset_default.yaml"
SPLIT_SET = ["MNIST", "FashionMNIST", "EMNIST", "CIFAR10", "CIFAR100", "SVHN"]
LEAF_SET = ["femnist", "celeba", "reddit", "sent140", "shakespeare"]
GEN_SET = ["synthetic"]
NUM_CLASS = {"MNIST": 10, "FashionMNIST": 10, "EMNIST": 62, "CIFAR10": 10, "CIFAR100": 100, "femnist": 62, "celeba": 2,
             'sent140': 2, 'shakespeare': 80, "SVHN": 10}

def read_as_torch_dataset(conf_path, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    """read leaf data by conf_path and convert it as torch.util.data.Dataset

    Args:
        conf_path: path of config file, e.g. './template_conf/dirichlet.yaml'
        transform: function/transform for data
        target_transform: function/transform for targets

    Return:
        clients: list of client names
        train_datasets: clients' train data
        test_datasets: clients' test data
        all_train_dataset: sum all clients' train data
        all_test_dataset: sum all clients' test data
    """
    # read data as list
    clients, clients_train, clients_test = read_leaf_by_conf(conf_path)

    # get all set
    all_train_x, all_train_y = [], []
    all_test_x, all_test_y = [], []
    for c in clients:
        all_train_x += clients_train[c]['x']
        all_train_y += clients_train[c]['y']
        all_test_x += clients_test[c]['x']
        all_test_y += clients_test[c]['y']

    # shuffle list
    # train_ids, test_ids = [i for i in range(len(all_train_y))], [i for i in range(len(all_test_y))]
    # random.shuffle(train_ids)
    # random.shuffle(test_ids)
    # all_train_x, all_test_x = np.array(all_train_x)[train_ids].tolist(), np.array(all_test_x)[test_ids].tolist()
    # all_train_y, all_test_y = np.array(all_train_y)[train_ids].tolist(), np.array(all_test_y)[test_ids].tolist()

    # get transform for torch dataset
    conf = read_data_conf(conf_path)
    if transform is None and target_transform is None:
        transform, target_transform = get_transform(conf.dataset)

    # warp as torch.util.data.Dataset
    train_datasets = [WrappedDataset(clients_train[c]['x'], clients_train[c]['y'],
                                     transform, target_transform) for c in clients]
    test_datasets = [WrappedDataset(clients_test[c]['x'], clients_test[c]['y'],
                                    transform, target_transform) for c in clients]
    all_train_dataset = WrappedDataset(all_train_x, all_train_y, transform, target_transform)
    all_test_dataset = WrappedDataset(all_test_x, all_test_y, transform, target_transform)
    return clients, train_datasets, test_datasets, all_train_dataset, all_test_dataset


def read_leaf_by_conf(conf_path):
    """read leaf data by yaml conf path

    Return:
        clients: list of client names
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    save_dir, f_name, _ = get_f_by_conf(conf_path)
    train_clients, train_data, test_data = read_leaf_data(save_dir, f_name)
    return train_clients, train_data, test_data


def read_data_conf(conf_path):
    """read data conf by path"""
    # load conf
    default_conf = OmegaConf.load(DEFAULT_CONF_PATH)
    conf = OmegaConf.load(conf_path)
    conf = OmegaConf.merge(default_conf, conf)
    # check conf
    conf = check_conf(conf)
    return conf


def check_conf(conf):
    """check kwargs and correct part of it."""
    # check legality of args
    assert 0 < conf.train_frac < 1
    assert conf.mean_samples > 0
    assert conf.max_samples > conf.mean_samples > conf.min_samples
    if conf.dataset in GEN_SET:
        assert conf.sub_classes is None, f"sub_classes should be set as None when dataset is {conf.dataset}"

    # get implied args
    if conf.sub_classes is None:
        if conf.dataset in SPLIT_SET: conf.num_classes = NUM_CLASS[conf.dataset]
        conf.sub_classes = list(range(conf.num_classes))
    else:
        conf.num_classes = len(conf.sub_classes)
    if conf.iid: conf.split = 'shard'
    conf.dataset_name = conf.dataset
    return conf


def get_f_by_conf(conf_path):
    """get save dir, file name, and format by yaml config path"""
    conf = read_data_conf(conf_path)
    f_name = get_f_name(**conf)
    return conf.save_dir, f_name, conf.format


def get_f_name(dataset_name: str, num_clients: int, num_classes: int, alpha: float, sigma: float,
               mean_samples: int, min_samples: int, max_samples: int, train_frac: float, split: str,
               iid: bool, shard_size: int, dim: int, **kwargs):
    """get file name according to process args"""
    if iid:
        return f"{dataset_name}_iid_clt={num_clients}_s={sigma}" \
               f"_mu={mean_samples}_r=[{min_samples},{max_samples}]_tf={train_frac}"
    if split == 'leaf':
        return f"{dataset_name}_{split}_clt={num_clients}_r=[{min_samples},{max_samples}]"
    elif split == 'dirichlet':
        f_name = f"{dataset_name}_{split}_clt={num_clients}_cls={num_classes}" \
                 f"_a={alpha}_s={sigma}_mu={mean_samples}_r=[{min_samples},{max_samples}]" \
                 f"_tf={train_frac}"
        if dataset_name == 'synthetic':
            f_name = f_name + f'_dim={dim}'
        return f_name
    elif split == 'shard':
        return f"{dataset_name}_{split}_clt={num_clients}_cls={num_classes}" \
               f"_sz={shard_size}_s={sigma}_mu={mean_samples}_r=[{min_samples},{max_samples}]" \
               f"_tf={train_frac}"


def read_leaf_data(data_dir, f_name):
    """parses train and test of given file name.
    Note the file name can be obtained from 'get_f_name'.

    Return:
        clients: list of client ids
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    train_path = os.path.join(data_dir, f"train-{f_name}.json")
    test_path = os.path.join(data_dir, f"test-{f_name}.json")

    # print(os.path.abspath(train_path))

    train_clients, train_data = read_leaf_file(train_path)
    test_clients, test_data = read_leaf_file(test_path)

    assert train_clients == test_clients

    return train_clients, train_data, test_data


def read_leaf_file(file):
    """read leaf data from file"""
    assert file.endswith('.json'), "only '.json' data can be read."
    data = defaultdict(lambda: None)
    with open(file, 'r') as inf:
        cdata = json.load(inf)
    data.update(cdata['user_data'])
    clients = list(sorted(data.keys()))
    return clients, data


class WrappedDataset(Dataset):
    """Wraps list into a pytorch dataset


    Args:
        data (list): feature data list.
        targets (list): target list.
        transform (callable, optional): A function/transform that takes in a data sample
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, data: list,
                 targets: list = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):

        if targets is not None: assert len(targets) == len(data)

        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        feature = self.data[idx]
        target = self.targets[idx] if self.targets is not None \
            else self.targets

        if self.transform is not None:
            feature = self.transform(feature)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return feature, target
