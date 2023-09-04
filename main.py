import argparse
import os
import shutil
from importlib import import_module
import sys
import torch
import numpy as np
import random
from omegaconf import OmegaConf

from utils.read_dataset import read_as_torch_dataset, read_data_conf
from utils.my_transforms import MNIST_LIKE_DATASET, CIFAR_LIKE_DATASET, \
    mnist_data_transform, cifar_data_transform, sent140_pub_transform
from utils.nets import create_model

IMPLEMENTED_ALGORITHMS = ["Local", "Center",  # the most base benchmark
                          "FedAvg",  # aggregate by weighted average
                          "FedProx",  # control client-side drift
                          "FedSR",  # domain generalization by simple representation
                          "IFCA",  # aggregate by clusters
                          "FedGen",  # knowledge distillation
                          "FedRoD",  # multi-task
                          "FedFTG",  # fine-tune global model
                          "FedDF",  # model ensemble
                          "SFL",  # SOTA graph FL by weight aggregation.
                          "GFL_APPNP",  # SOTA graph FL by hidden sharing.
                          "SG_FedX",  # Ours
                          ]
NEED_PUBLIC_DATASET_ALGORITHMS = ["FedDF"]


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="FedAvg", choices=IMPLEMENTED_ALGORITHMS,
                        help=f"the implemented algorithms, choices include: {IMPLEMENTED_ALGORITHMS}")
    parser.add_argument("--num_clients", type=int, default=None,
                        help="num clients, if not specific, use the value in data_conf.yaml")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="exp name, sub dir for save log, if not specific, use the value in exp_conf.yaml")
    parser.add_argument("--exp_conf", type=str, default=f"./configs/example/mnist/exp.yaml",
                        help="experiment config yaml files")
    parser.add_argument("--data_conf", type=str, default=f"./configs/example/mnist/data.yaml",
                        help="dataset config yaml files")
    parser.add_argument("--public_conf", type=str, default=f"./configs/example/mnist/public.yaml",
                        help="public dataset config yaml files, default is None."
                             "Here, we use the synthetic data as public dataset")
    parser.add_argument("--model_conf", type=str, default=f"./configs/example/mnist/model.yaml",
                        help="model config yaml files")
    parser.add_argument("--device", type=str, default="cuda:0", help="run device (cpu | cuda:x, x:int > 0)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--save_model", type=bool, default=False, help="if save model")
    return parser.parse_args()


def exp_run(args, exp_conf):
    """ Run experiments with args and conf.yaml
    :return: log_dir
    """
    # 1. set random seed
    set_seed(args.seed)

    # 2. prepare dataset and public dataset (if specific)
    # a. read clients' dataset
    _, c_train_datasets, c_test_datasets, all_train_dataset, all_test_dataset = read_as_torch_dataset(args.data_conf)
    num_clients = len(c_train_datasets) if args.num_clients is None else args.num_clients
    # b. get num of classes
    data_conf = read_data_conf(args.data_conf)
    num_classes = data_conf.num_classes
    dataset_name = data_conf.dataset

    # c. get public dataset
    if (args.public_conf is not None) and (args.algorithm in NEED_PUBLIC_DATASET_ALGORITHMS):
        public_dataset = read_as_torch_dataset(args.public_conf)[3]
        if data_conf.dataset in MNIST_LIKE_DATASET:
            public_dataset.transform = mnist_data_transform
        elif data_conf.dataset in CIFAR_LIKE_DATASET:
            public_dataset.transform = cifar_data_transform
        elif data_conf.dataset == 'sent140':
            public_dataset.transform = sent140_pub_transform
    else:
        public_dataset = None

    # 3. init log dir, at ./log/exp_name/dataset_name/num_clients={num_clients}/algorithm/seed={seed}
    exp_name = exp_conf.exp_name if args.exp_name is None else args.exp_name
    log_dir = os.path.join(exp_name, dataset_name, f'num_clients={num_clients}', args.algorithm)

    if args.seed is not None:
        log_dir = os.path.join(log_dir, f'seed={args.seed}')

    # 4. create models
    clients_models, server_model = create_model(args.model_conf, num_clients)

    # record init models
    if args.save_model:
        os.makedirs('./ckpt/init', exist_ok=True)
        torch.save(server_model.state_dict(), './ckpt/init/server.pth')
        for i, cm in enumerate(clients_models):
            torch.save(cm.state_dict(), f'./ckpt/init/client_{i}.pth')

    # 5. create clients
    Client = getattr(import_module("trainer.%s.client" % args.algorithm), 'Client')
    clients = [Client(node_id=i,  # take i as clients' id
                      dataset_name=dataset_name,
                      model=clients_models[i],
                      train_dataset=c_train_datasets[i],
                      test_dataset=c_test_datasets[i],
                      num_classes=num_classes,
                      device=args.device,
                      log_dir=log_dir,
                      **exp_conf,
                      ) for i in range(num_clients)]

    # 6. create server
    Server = getattr(import_module("trainer.%s.server" % args.algorithm), 'Server')
    server = Server(
        # all clients are registered as default
        node_id=0,
        dataset_name=dataset_name,
        clients=clients,
        model=server_model,
        train_dataset=all_train_dataset,
        test_dataset=all_test_dataset,
        public_dataset=public_dataset,
        num_classes=num_classes,
        metric_type='accuracy',
        device=args.device,
        log_dir=log_dir,
        **exp_conf,
    )

    # 7. start training
    server.run(exp_conf.rounds, exp_conf.epochs, exp_conf.test_interval,
               verbose=0, save_ckpt=args.save_model)

    # 8. save config file in log dir
    shutil.copy(args.exp_conf, os.path.join("log", exp_conf.exp_name))


def set_seed(seed):
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def main(verbose=0):
    # parser args, read config yaml file
    args = parser_args()
    exp_conf = OmegaConf.load(args.exp_conf)

    if verbose > 0:
        print()
        print("=" * 30 + "{:^20}".format("Args") + "=" * 30)
        print(f" algorithms: {args.algorithm}\n"
              f"     device: {args.device}\n"
              f"       seed: {args.seed}\n"
              f"   exp_conf: {args.exp_conf}\n"
              f"  data_conf: {args.data_conf}\n"
              f"public_conf: {args.public_conf}\n"
              f" model_conf: {args.model_conf}\n")
        print("=" * 30 + "{:^20}".format("Exp Configs") + "=" * 30)
        print(OmegaConf.to_yaml(exp_conf))
        print("=" * 80)

    # check werther gpu available, if not use cpu
    if verbose > 0: print(f'check device {args.device}')
    if args.device != "cpu" and not torch.cuda.is_available():
        print("==== NOTE: CUDA not available, use cpu? (N for stop.)")
        choice = input()
        if choice in ['N', 'n']: sys.exit()
        args.device = "cpu"

    # run experiment with given args & conf
    exp_run(args, exp_conf)
    if verbose > 0: print("=" * 20 + "Finish all." + "=" * 20)


if __name__ == "__main__":
    main()
