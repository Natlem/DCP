# Discrimination-aware Channel Pruning for Deep Neural Networks （NIPS 2018)

## Architecture of Discrimination-aware Channel Pruning (DCP)

![Architecture of DCP](./imgs/supervised_pruning_framework_v12.png)

## Training Algorithm

![Algorithm](./imgs/algorithm.png)

## Requirements

* python 2.7
* pytorch 0.4
* tensorflow
* pyhocon

## Testing

1. Download the pre-trained pruned model.
* [resnet-18-pruned0.3 BaiduDrive](https://pan.baidu.com/s/1V-kI0k8KGGEBuC4vukabMA), [resnet-18-pruned0.3 GoogleDrive](https://drive.google.com/open?id=1qWGi8wA60Ism6IhcEHjcYqmWyX5Rg1vK)
* [resnet-18-pruned0.5 BaiduDrive](https://pan.baidu.com/s/1KsHTmKwljbS-Y9C4iOX37w), [resnet-18-pruned0.5 GoogleDrive](https://drive.google.com/open?id=1cmDdi6y9MCEw3OmbDZpQPsnE0mRQIH8W)
* [resnet-18-pruned0.7 BaiduDrive](https://pan.baidu.com/s/1BOEfGxeH_9MGS7TT--h8cQ), [resnet-18-pruned0.7 GoogleDrive](https://drive.google.com/open?id=1jEMginHmPjPEJK9TzuGnGPI4NdFKVKJN)
* [resnet-50-pruned0.3 BaiduDrive](https://pan.baidu.com/s/1u4Vz5910F6ibH_J-wSnfqg), [resnet-50-pruned0.3 GoogleDrive](https://drive.google.com/file/d/185s4qod1ts813rLHwMIciB47KiSTxQrZ/view)
* [resnet-50-pruned0.5 BaiduDrive](https://pan.baidu.com/s/186x0KWe4jzhBqap7oMqbFA), [resnet-50-pruned0.5 GoogleDrive](https://drive.google.com/file/d/1uv8ACOmFzSDRWpW1T1qu5Psu46MB7WUt/view)
* [resnet-50-pruned0.7 BaiduDrive](https://pan.baidu.com/s/1-O0xuzDtPK8iZJDBe_m81g), [resnet-50-pruned0.7 GoogleDrive](https://drive.google.com/open?id=1gdS3IfTCWzF8TcVaUcN_M5ENe_AIOYN3)

2. Add DCP into PYTHONPATH.
```Shell
# This is my path of DCP. You need to change to your path of DCP.
export PYTHONPATH=/home/liujing/Codes/Discrimination-aware-Channel-Pruning-for-Deep-Neural-Networks:$PYTHONPATH
```

3. Set configuration for testing.
You need to set `data_path`, `pruning_rate`, `depth` and the `retrain` in `dcp/channel_pruning/test.hocon`.

```Shell
cd dcp/channel_pruning/
vim dcp/channel_pruning/test.hocon
```

4. Run testing.
```Shell
python test.py test.hocon
```

## Channel Pruning Examples

1. Download pre-trained mdoel.
* [resnet-56 BaiduDrive](https://pan.baidu.com/s/1HFXzHNHFDa57RlVk2W71Aw), [resnet-56 GoogleDrive](https://drive.google.com/open?id=1nCIqcSkFdErtmgNUfwW2RDN6EUlFfTfa).

2. Add DCP into PYTHONPATH.
```Shell
# This is my path of DCP. You need to change to your path of DCP.
export PYTHONPATH=/home/liujing/Codes/Discrimination-aware-Channel-Pruning-for-Deep-Neural-Networks:$PYTHONPATH
```

3. Set configuration for channel pruning.
Before pruning, you need to set `save_path`, `data_path`, `experiment_id` and the `retrain` in `dcp/channel_pruning/cifar10_resnet.hocon`.

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

6. Fine-tune the pruned model.
```Shell
python fine_tuning.py cifar10_resnet.hocon
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