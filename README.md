# Discrimination-aware Channel Pruning for Deep Neural Networks （NIPS 2018)

Pytorch implementation for "Discrimination-aware Channel Pruning for Deep Neural Networks"

## Architecture of Discrimination-aware Channel Pruning (DCP)

![Architecture of DCP](./imgs/supervised_pruning_framework_v12.png)

## Training Algorithm
![Algorithm](./imgs/algorithm.png)

## Requirements
* python 3.6
* pytorch1.0
* tensorflow
* pyhocon

## Usage Examples

1. Download pre-trained mdoel.
* [resnet-56 BaiduDrive](https://pan.baidu.com/s/1HFXzHNHFDa57RlVk2W71Aw).

2. Add DCP into PYTHONPATH.
```Shell
# This is my path of DCP. You need to change to your path of DCP.
export PYTHONPATH=/home/liujing/Codes/Discrimination-aware-Channel-Pruning-for-Deep-Neural-Networks:$PYTHONPATH
```

1. Set configuration for channel pruning.
Before pruning, you need to set `log_path`, `data_path`, `experiment_id` and the `retrain` in `dcp/channel_pruning/cifar10_resnet.hocon`.

```Shell
cd dcp/channel_pruning/
vim dcp/channel_pruning/cifar10_resnet.hocon
```

4. Run Discrimination-aware Channel Pruning.
```Shell
python channel_pruning.py cifar10_resnet.hocon
```

5. Set configuration for fine-tuning.
Before fine-tuning, you need to set `retrain` to the path of `model_004.pth` in `check_point` folder
```Shell
vim cifar10_resnet.hocon
```

1. Fine-tune the pruned model.
```Shell
python fine_tuning.py cifar10_resnet_ft.hocon
```

## Citation
If you find *DCP* useful in your research, please consider to cite the following related papers:
```
@article{zhuang2018discrimination,
title={Discrimination-aware Channel Pruning for Deep Neural Networks},
author={Zhuangwei Zhuang and Mingkui Tan and Bohan Zhuang and Jing Liu and Yong Guo and Qingyao Wu and Junzhou Huang and Jinhui Zhu},
journal={arXiv:1801.07698},
year={2018}
}
```