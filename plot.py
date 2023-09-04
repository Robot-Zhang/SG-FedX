from typing import List
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt



def read_exp_metrics(exp_name, log_dir="./log", save=False):
    """read all metric files of the given exp_name, return a big metric data
    """
    # read log path
    path = os.path.abspath(os.path.join(log_dir, exp_name))
    metric_pd = pd.DataFrame()

    for dataset in os.listdir(path):
        if dataset.endswith('.yaml'):
            continue
        dataset_path = os.path.join(path, dataset)
        for num_str in os.listdir(dataset_path):
            assert num_str.startswith('num_clients=')
            num_clients = int(num_str.split("=")[1])
            num_clients_path = os.path.join(dataset_path, num_str)
            # loop all algorithms
            for algorithm in os.listdir(num_clients_path):
                if algorithm.endswith('.yaml'):
                    continue
                algorithm_path = os.path.join(num_clients_path, algorithm)
                # check if seed given, get metric files
                seed_dirs = []
                for p in os.listdir(algorithm_path):
                    if p.startswith('seed='):
                        seed_dirs.append(p)
                if len(seed_dirs) > 0:
                    metric_files = [os.path.join(algorithm_path, seed_dir, "metric.csv")
                                    for seed_dir in seed_dirs]
                    seed = [int(seed_dir.split("=")[1]) for seed_dir in seed_dirs]
                else:
                    metric_files = [os.path.join(algorithm_path, "metric.csv")]
                    seed = [None]

                # read all metric
                for i, f in enumerate(metric_files):
                    if os.path.exists(f):
                        metric_frame = pd.read_csv(f)
                        # add keys: seed, algorithm, device
                        frame_length = len(metric_frame["round"])
                        if seed[i] is not None:
                            metric_frame["seed"] = [seed[i]] * frame_length
                        metric_frame["algorithm"] = [f"{algorithm}"] * frame_length
                        metric_frame["dataset"] = [f"{dataset}"] * frame_length
                        metric_frame["num_clients"] = [num_clients] * frame_length
                        # concat metric
                        metric_pd = pd.concat([metric_pd, metric_frame])
                    else:
                        print(f"【Warning】 ---- metric file {f} not exist!")
    if save:
        metric_pd.to_csv(f"sum_all_metric_{exp_name}.csv", index=False)
    return metric_pd


def get_data_by_condition(data, **kwargs):
    """get pd data by condition,

    Args:
        data: pandas dataframe
        kwargs: conditions, format like round=1, or round=[1,2]

        e.g.
        data = get_data_by_condition(raw_data, round=[1, 2])
        data = get_data_by_condition(raw_data, round=1)
    """
    for k, v in kwargs.items():
        if v is not None:
            # or condition
            if isinstance(v, List):
                assert len(v) > 1
                cond = (data[k] == v[0])
                for i in range(len(v) - 1):
                    cond = cond | (data[k] == v[i + 1])
                data = data[cond]
            # and condition
            else:
                data = data[data[k] == v]
    return data


def print_data(raw_data, datasets, algorithms, is_per, seed=None):
    m = 'private' if is_per else 'general'
    device = 'client' if is_per else 'server'

    for _ in range(3): print('=' * 30 + m + '=' * 30)
    # print table head
    head_str = '\t Dataset \t|'
    for dataset in datasets:
        head_str += f'\t {dataset} \t|'
    print(head_str)

    for alg in algorithms:
        acc_str = ''
        for dataset in datasets:
            if seed is None:
                alg_data = get_data_by_condition(raw_data, algorithm=alg, dataset=dataset, device=device)
            else:
                alg_data = get_data_by_condition(raw_data, seed=seed, algorithm=alg, dataset=dataset, device=device)
            r_range = list(alg_data['round'].unique()[-1:])
            if len(r_range) > 0:
                final_state = get_data_by_condition(
                    alg_data,
                    round=r_range[0])  # use the -1 round for statics
                acc_mean = final_state[f'{m}_accuracy'].mean()
                acc_std = final_state[f'{m}_accuracy'].std()
                acc_str += f'\t {acc_mean * 100:.2f}' + '±' + f'{acc_std * 100:.2f} \t|'
            else:
                acc_str += '\t  \t|'
        if alg == 'SG_FedX': alg = alg + ' (Ours)'
        acc_str = '\t %s \t|%s' % (alg, acc_str)
        print(acc_str)
    for _ in range(3): print('=' * 65)


def plot_curves(raw_data, datasets, algorithms, is_per, metric='accuracy',
                save_dir='figs', use_base=False, show=True, seed=None,
                is_cross_data=False):
    os.makedirs(f'{save_dir}', exist_ok=True)
    for dataset in datasets:
        m = 'private' if is_per else 'general'
        if seed is None:
            data = get_data_by_condition(raw_data, dataset=dataset, algorithm=algorithms, device='server')
        else:
            data = get_data_by_condition(raw_data, seed=seed, dataset=dataset, algorithm=algorithms, device='server')
        if is_per:
            data = get_data_by_condition(raw_data, dataset=dataset,
                                         algorithm=algorithms, device='client')
        if not use_base:
            data = data[(data['algorithm'] != 'Center') & (data['algorithm'] != 'Local')]

        data.rename(columns={f"{m}_{metric}": metric}, inplace=True)
        sns.lineplot(x="round",
                     y=metric,
                     hue="algorithm",
                     style="algorithm",
                     data=data)
        if dataset == 'sent140': dataset = 'Twitter'
        if dataset in ['MNIST', 'SVHN', 'CIFAR10']: dataset = 'LT ' + dataset

        if is_cross_data: dataset = 'Cross Datasets'
        if is_per:
            plt.title(f' learning curves of {dataset} (personalization)')
            plt.savefig(f'{save_dir}/{dataset}-{m}-per.png')
        else:
            plt.title(f' learning curves of {dataset} (generalization)')
            plt.savefig(f'{save_dir}/{dataset}-{m}-gen.png')
        if show:
            plt.show()
        plt.close()


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    os.makedirs('figs', exist_ok=True)

    # 1. read log data
    raw_data = read_exp_metrics('example')

    # datasets = ['MNIST', 'femnist', 'SVHN', 'CIFAR10', 'synthetic', 'sent140']
    # algorithms = ['Center', 'Local', 'FedAvg', 'FedProx', 'FedSR', 'IFCA',
    #               'FedGen', 'FedRoD', 'FedFTG', 'FedDF', 'SFL', 'GFL_APPNP', 'SG_FedX']
    datasets = ['MNIST']
    algorithms = ['Center', 'Local', 'FedAvg', 'FedProx', 'FedSR', 'SFL', 'SG_FedX']
    # 2. print statics metric (can be used in latex )
    # 2-1 generalization
    print_data(raw_data, datasets, algorithms, False)
    # 2-1 personalization
    print_data(raw_data, datasets, algorithms, True)

    # 3. plot learning curves of each dataset
    # overall performance
    plot_curves(raw_data, datasets, algorithms, False, metric='accuracy', use_base=True, show=True)
    # personalization
    plot_curves(raw_data, datasets, algorithms, True, metric='accuracy', use_base=True, show=True)
