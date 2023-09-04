#!/bin/sh

#================  Examples  ================#
device="cuda:0"  # device ID
postfix=".yaml" # postfix of config file
config_dir="./configs/example" # config dir
data=(mnist) # test datasets
algorithms=(Local FedAvg FedProx FedSR SFL SG_FedX) # the algorithms
times=1 # times for running
seed=0 # seeds

echo "Examples is running now."

echo "===================================================="
echo "====================Run Center======================"
echo "===================================================="
for ((t=0; t<times; t+=1)); do
    for d in ${data[@]}; do
      let s=$seed+t
      python main.py --algorithm Center \
      --exp_conf ${config_dir}/${d}/exp${postfix} \
      --data_conf ${config_dir}/${d}/data${postfix} \
      --public_conf ${config_dir}/${d}/public${postfix} \
      --model_conf ${config_dir}/${d}/model${postfix} \
      --seed $s --device $device
    done
done

echo "===================================================="
echo "====================Run Comparison=================="
echo "===================================================="

for ((t=0; t<times; t+=1)); do
  for alg in ${algorithms[@]}; do
    for d in ${data[@]}; do
      let s=$seed+t
      python main.py --algorithm $alg  \
      --exp_conf ${config_dir}/${d}/exp${postfix} \
      --data_conf ${config_dir}/${d}/data${postfix} \
      --public_conf ${config_dir}/${d}/public${postfix} \
      --model_conf ${config_dir}/${d}/model${postfix} \
      --seed $s --device $device
    done
  done
done

echo "===================================================="
echo "====================Plot Figs=================="
echo "===================================================="
python plot.py

echo "Finish all algorithms, please see log dir."