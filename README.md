# SG_FedX

Source code of "Structured Graph Federated Learning: Towards Exploiting High-Dimensional Information of Heterogeneity"

<!-- TODO: add paper link and author link after pub -->

## Declaration

This paper is currently under blind review, and we have ensured that **all contents** in the project **that may reveal the authors' information are removed**.

## 1. Requirements

A suitable [conda](https://conda.io/) environment named `SG-FedX` can be created
and activated with:

```bash
$ conda create -n SG-FedX python=3
$ conda activate SG-FedX
$ conda install pytorch torchvision -c pytorch
$ pip3 install -r requirements.txt
```

## 2. Getting Started

After completing the configuration, you can run as follows.

```bash
$ python main.py --algorithm <alg_name> --exp_conf <exp_conf.yaml> --data_conf <data_conf.yaml> --model_conf <model_conf.yaml> --seed <seed> --device <seed>
```

For example, run SG_FedX on MNIST with `yaml` settings:

```bash
$ python main.py --algorithm SG_FedX --exp_conf ./configs/example/mnist/exp.yaml --data_conf ./configs/example/mnist/data.yaml --model_conf ./configs/example/mnist/model.yaml --seed 15698 --device cuda:0
```

We also provide example of `run.sh` and yaml files:
```bash
$ bash run.sh
```
Here, the figures will be drawn automatically, 
and you can analyze the data recorded in the `log` according to your own needs.

## 3. Usage

### 3.1 Arguments

In this project, main.py takes the following arguments:

+ `--algorithm`: name of the implemented algorithms.
+ `--num_clients`: number clients, if not specific, use the value in `data_conf.yaml`
+ `--exp_name`: exp name, sub dir for save log, if not specific, use the value in `exp_conf.yaml`
+ `--exp_conf`: experiment config yaml files
+ `--data_conf`: dataset config yaml files
+ `--public_conf`: public dataset config yaml files, default is None. For FedMD and Kt-pFL. 
+ `--model_conf`: model config yaml files
+ `--device`:  run device (cpu | cuda:x, x:int > 0)
+ `--seed`: random seed
+ `--save_model`: bool, if save model at each test interval.

### 3.2 EXP Config YAML

This is a typic yaml file:

```yaml
# 1. Settings
exp_name: "example" # name of experiment
# 2. Basic args for FL
rounds: 100 # communication rounds
epochs: 5 # epochs of local update
loss: 'CrossEntropyLoss' # loss fn name in torch.nn.*
opt: 'Adam'  # optimizer name in torch.optim.*, e.g. Adam, SGD
optim_kwargs: # args for optimizer
  lr: 1e-6 # learning rate of local update
batch_size: 32 # batch_size of local update
sample_frac: 1.0 # select fraction of clients
test_interval: 1 # test each round
# 3. Optional args for FL algorithms
# ----3.1 Args for Center
center_update_samples: # if not None, means the used samples in each update epoch, recommend as None
# ---- 3.2 Args for FedProx
mu: 0.01   # coefficient for controlling client drift
# ---- 3.3 Args for FedDF
ensemble_epoch: 5
ensemble_lr: 1e-8 # lr for ensemble, suggest lower than lr,
distill_temperature: 20 # temperature for distillation
# ---- 3.4 Args for FedGen
#       note: distill_temperature s same as 3.3
generative_alpha: 10.0  # hyperparameters for clients' local update
generative_beta: 1.0  # hyperparameters for clients' local update
gen_epochs: 10  # epochs for updating generator
gen_lr: 1e-4  # lr for updating generator
# ---- 3.5 Args for FedFTG
#       note: gen_epochs, gen_lr is same as 3.4
#             ensemble_epoch, ensemble_lr, distill_temperature is same as 3.3
finetune_epochs: 1
lambda_cls: 1. # hype-parameters of updating generator
lambda_dis: 1. # hype-parameters of updating generator
# ---- 3.6 Args for FedSR
alpha_l2r: 0.01
alpha_cmi: 0.001
# ---- 3.7 Args for SFL
propagation_hops: 2
sfl_alpha: 1.0
# ---- 3.8 Args for SG_FedX
#       note: propagation_hops, sfl_alpha is same as 3.7
hidden_alpha: 1.0
# ---- 3.9 Args for IFCA
k: 3  # number of clusters
# ---- 3.10 Args for GFL-APPNP
#       note: propagation_hops, alpha is same as 3.7
```

## 4. Introduction to SG-FedX 

### 4.1 Outline

The SG-FedX is a type of Graph Federated Learning (GFL) algorithm that can capture the intricate hierarchical relationships between clients based on their model parameters. In contrast to other GFL methods, it optimizes a Federated Learning (FL) graph to improve learning efficiency and shares the graph's representation without revealing any additional information.

### 4.2 Background and Comparison

The Graph Federated Learning (GFL) paradigm, recently proposed, has showcased promising 
performance in effectively tackling the challenges associated with heterogeneity by 
systematically capturing the intricate relationships among clients. However, existing GFL 
methods suffer from two limitations. Firstly, the current methods using fixed or fully 
connected graphs fail to accurately depict the associations between clients, thereby 
compromising the performance of GFL. Secondly, these methods may disclose additional 
information when sharing client-side hidden representations. In order to address these limitations, we propose SG-FedX to explore a new pathway to establish a reasonable FL graph,
explore the hierarchical community structure of the client, and use the representation of the graph 
to improve the learning performance.



![](.\docs\comp-scheme.png)
This figure provides a comparison of information representation capability across heterogeneity. 
Here, the relationship among clients is described in a graph structure. 
(a) Vanilla FL, using star structure with fixed graph, and all clients are only connected to the server. (b) Existing GFL, the clients are treated as a node in a plane graph, and edges are fixed. (c) Our SGFL employs a hierarchical structure for representing high-dimensional heterogeneity, enabling the characterization of a broader range of information, and all edges are adjustable.
   


## 5. Contribution Navigation

To add new FL algorithms, just inherit `FedAvg` class directly, and then modify the corresponding function in the protocol. Include:

```python
"""Protocol of algorithm, in most kind of FL, this is same."""

def sample_clients(self):
    pass

def distribute_model(self):
    pass

def local_update(self, epochs):
    pass

def aggregate(self):
    pass

def test_performance(self, r, is_final_r=False, verbose=0, save_ckpt=False):
    pass
```

For more detailed information, see`./trainer/FedAvg`.



## 6. BibTeX

It will be given after publication.

<!-- TODO: refresh bib after publication-->