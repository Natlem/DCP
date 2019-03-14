import argparse
import copy
import datetime
import logging
import math
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from option import Option
from trainer import SegmentWiseTrainer, NetworkWiseTrainer
from checkpoint import CheckPoint

import dcp.models as md
import dcp.utils as utils
from dcp.mask_conv import MaskConv2d
from dcp.models.preresnet import PreBasicBlock
from dcp.models.resnet import BasicBlock, Bottleneck
from dcp.utils.tensorboard_logger import TensorboardLogger
from visdom_logger.logger import VisdomLogger

block_num = {'vgg': 16, 'preresnet56': 27, 'resnet18': 8, 'resnet50': 16}
from visdom_logger.logger import VisdomLogger

class Experiment(object):
    """
    run experiments with pre-defined pipeline
    """

    def __init__(self, options=None, conf_path=None):
        self.settings = options or Option(conf_path)
        self.checkpoint = None
        self.train_loader = None
        self.val_loader = None
        self.ori_model = None
        self.pruned_model = None
        self.segment_wise_trainer = None

        self.aux_fc_state = None
        self.aux_fc_opt_state = None
        self.seg_opt_state = None
        self.current_pivot_index = None
        self.is_segment_wise_finetune = False
        self.is_channel_selection = False

        self.epoch = 0

        self.feature_cache_origin = {}
        self.feature_cache_pruned = {}

        self.settings.set_save_path()
        self.write_settings()
        self.logger = self.set_logger()
        self.v_logger = VisdomLogger(port=10999)

        self.prepare()

    def write_settings(self):
        """
        save experimental settings to a file
        """

        with open(os.path.join(self.settings.save_path, "settings.log"), "w") as f:
            for k, v in self.settings.__dict__.items():
                f.write(str(k) + ": " + str(v) + "\n")

    def set_logger(self):
        """
        initialize logger
        """
        logger = logging.getLogger('channel_selection')
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        console_formatter = logging.Formatter('%(message)s')
        # file log
        file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
        file_handler.setFormatter(file_formatter)

        # console log
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)
        return logger

    def prepare(self):
        """
        preparing experiments
        """

        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._cal_pivot()
        self._set_checkpoint()
        self._set_trainier()

    def _set_gpu(self):
        """
        initialize the seed of random number generator
        """

        # set torch seed
        # init random seed
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        torch.cuda.set_device(int(os.environ['CUDA_VISIBLE_DEVICES']))
        cudnn.benchmark = True

    def _set_dataloader(self):
        """
        create train loader and validation loader for channel pruning
        """

        if self.settings.dataset == 'cifar10':
            data_root = os.path.join(self.settings.data_path, "cifar")

            norm_mean = [0.49139968, 0.48215827, 0.44653124]
            norm_std = [0.24703233, 0.24348505, 0.26158768]
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

            train_dataset = datasets.CIFAR10(root=data_root,
                                             train=True,
                                             transform=train_transform,
                                             download=True)
            val_dataset = datasets.CIFAR10(root=data_root,
                                           train=False,
                                           transform=val_transform)

            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                            batch_size=self.settings.batch_size,
                                                            shuffle=True,
                                                            pin_memory=True,
                                                            num_workers=self.settings.n_threads)
            self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                          batch_size=self.settings.batch_size,
                                                          shuffle=False,
                                                          pin_memory=True,
                                                          num_workers=self.settings.n_threads)
        elif self.settings.dataset == 'imagenet':
            dataset_path = os.path.join(self.settings.data_path, "imagenet")
            traindir = os.path.join(dataset_path, "train")
            valdir = os.path.join(dataset_path, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            self.train_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(
                    traindir,
                    transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ])),
                batch_size=self.settings.batch_size,
                shuffle=True,
                num_workers=self.settings.n_threads,
                pin_memory=True)

            self.val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=self.settings.batch_size,
                shuffle=False,
                num_workers=self.settings.n_threads,
                pin_memory=True)

    def _set_trainier(self):
        """
        initialize segment-wise trainer trainer
        """

        # initialize segment-wise trainer
        self.segment_wise_trainer = SegmentWiseTrainer(ori_model=self.ori_model,
                                                       pruned_model=self.pruned_model,
                                                       train_loader=self.train_loader,
                                                       val_loader=self.val_loader,
                                                       settings=self.settings,
                                                       logger=self.logger,
                                                       v_logger=self.v_logger)
        if self.aux_fc_state is not None:
            self.segment_wise_trainer.update_aux_fc(self.aux_fc_state, self.aux_fc_opt_state, self.seg_opt_state)

    def _set_model(self):
        """
        get model
        """

        if self.settings.dataset in ["cifar10", "cifar100"]:
            if self.settings.net_type == "preresnet":
                self.ori_model = md.PreResNet(depth=self.settings.depth, num_classes=self.settings.n_classes)
                self.pruned_model = md.PreResNet(depth=self.settings.depth, num_classes=self.settings.n_classes)
            else:
                assert False, "use {} data while network is {}".format(self.settings.dataset, self.settings.net_type)

        elif self.settings.dataset in ["imagenet"]:
            if self.settings.net_type == "resnet":
                self.ori_model = md.ResNet(depth=self.settings.depth, num_classes=self.settings.n_classes)
                self.pruned_model = md.ResNet(depth=self.settings.depth, num_classes=self.settings.n_classes)
            else:
                assert False, "use {} data while network is {}".format(self.settings.dataset, self.settings.net_type)

        else:
            assert False, "unsupported data set: {}".format(self.settings.dataset)

    def _set_checkpoint(self):
        """
        load pre-trained model or resume checkpoint
        """

        assert self.ori_model is not None and self.pruned_model is not None, "please create model first"

        self.checkpoint = CheckPoint(self.settings.save_path, self.logger)
        self._load_retrain()
        self._load_resume()

    def _load_retrain(self):
        """
        load pre-trained model
        """

        if self.settings.retrain is not None:
            check_point_params = torch.load(self.settings.retrain)
            if "ori_model" not in check_point_params:
                model_state = check_point_params
                self.ori_model = self.checkpoint.load_state(self.ori_model, model_state)
                self.pruned_model = self.checkpoint.load_state(self.pruned_model, model_state)
                self.logger.info("|===>load restrain file: {}".format(self.settings.retrain))
            else:
                ori_model_state = check_point_params["ori_model"]
                pruned_model_state = check_point_params["pruned_model"]
                # self.current_block_count = check_point_params["current_pivot"]
                self.aux_fc_state = check_point_params["aux_fc"]
                # self.replace_layer()
                self.ori_model = self.checkpoint.load_state(self.ori_model, ori_model_state)
                self.pruned_model = self.checkpoint.load_state(self.pruned_model, pruned_model_state)
                self.logger.info("|===>load pre-trained model: {}".format(self.settings.retrain))

    def _load_resume(self):
        """
        load resume checkpoint
        """

        if self.settings.resume is not None:
            check_point_params = torch.load(self.settings.resume)
            ori_model_state = check_point_params["ori_model"]
            pruned_model_state = check_point_params["pruned_model"]
            self.aux_fc_state = check_point_params["aux_fc"]
            self.aux_fc_opt_state = check_point_params["aux_fc_opt"]
            self.seg_opt_state = check_point_params["seg_opt"]
            self.current_pivot_index = check_point_params["current_pivot"]
            self.is_segment_wise_finetune = check_point_params["segment_wise_finetune"]
            self.is_channel_selection = check_point_params["channel_selection"]
            self.epoch = check_point_params["epoch"]
            self.epoch = self.settings.segment_wise_n_epochs
            self.current_block_count = check_point_params["current_block_count"]

            if self.is_channel_selection or \
                    (self.is_segment_wise_finetune and self.current_pivot_index > self.settings.pivot_set[0]):
                self.replace_layer()
            self.ori_model = self.checkpoint.load_state(self.ori_model, ori_model_state)
            self.pruned_model = self.checkpoint.load_state(self.pruned_model, pruned_model_state)
            self.logger.info("|===>load resume file: {}".format(self.settings.resume))

    def _cal_pivot(self):
        """
        calculate the inserted layer for additional loss
        """

        self.num_segments = self.settings.n_losses + 1
        num_block_per_segment = (block_num[self.settings.net_type + str(self.settings.depth)] // self.num_segments) + 1
        pivot_set = []
        for i in range(self.num_segments - 1):
            pivot_set.append(num_block_per_segment * (i + 1))
        self.settings.pivot_set = pivot_set
        self.logger.info("pivot set: {}".format(pivot_set))

    def segment_wise_fine_tune(self, index):
        """
        conduct segment-wise fine-tuning
        :param index: segment index
        """

        best_top1 = 100
        best_top5 = 100

        start_epoch = 0
        if self.is_segment_wise_finetune and self.epoch != 0:
            start_epoch = self.epoch + 1
            self.epoch = 0
        for epoch in range(start_epoch, self.settings.segment_wise_n_epochs):
            train_error, train_loss, train5_error = self.segment_wise_trainer.train(epoch, index)
            val_error, val_loss, val5_error = self.segment_wise_trainer.val(epoch)

            # write and print result
            if isinstance(train_error, list):
                best_flag = False
                if best_top1 >= val_error[-1]:
                    best_top1 = val_error[-1]
                    best_top5 = val5_error[-1]
                    best_flag = True

            else:
                best_flag = False
                if best_top1 >= val_error:
                    best_top1 = val_error
                    best_top5 = val5_error
                    best_flag = True

            if best_flag:
                self.checkpoint.save_model(ori_model=self.ori_model, pruned_model=self.pruned_model,
                                           aux_fc=self.segment_wise_trainer.aux_fc,
                                           current_pivot=self.current_pivot_index,
                                           segment_wise_finetune=True, index=index)

            self.logger.info("|===>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}\n".format(best_top1, best_top5))
            self.logger.info("|===>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}\n".format(100 - best_top1,
                                                                                                      100 - best_top5))

            if self.settings.dataset in ["imagenet"]:
                self.checkpoint.save_checkpoint(ori_model=self.ori_model,
                                                pruned_model=self.pruned_model,
                                                aux_fc=self.segment_wise_trainer.aux_fc,
                                                aux_fc_opt=self.segment_wise_trainer.fc_optimizer,
                                                seg_opt=self.segment_wise_trainer.seg_optimizer,
                                                current_pivot=self.current_pivot_index,
                                                segment_wise_finetune=True, index=index, epoch=epoch)
            else:
                self.checkpoint.save_checkpoint(ori_model=self.ori_model,
                                                pruned_model=self.pruned_model,
                                                aux_fc=self.segment_wise_trainer.aux_fc,
                                                aux_fc_opt=self.segment_wise_trainer.fc_optimizer,
                                                seg_opt=self.segment_wise_trainer.seg_optimizer,
                                                current_pivot=self.current_pivot_index,
                                                segment_wise_finetune=True, index=index)

    def replace_layer(self):
        """
        Replace the convolutional layer to mask convolutional layer
        """

        block_count = 0
        if self.settings.net_type in ["preresnet", "resnet"]:
            for module in self.pruned_model.modules():
                if isinstance(module, (PreBasicBlock, BasicBlock, Bottleneck)):
                    block_count += 1
                    layer = module.conv2
                    if block_count <= self.current_block_count and not isinstance(layer, MaskConv2d):
                        temp_conv = MaskConv2d(
                            in_channels=layer.in_channels,
                            out_channels=layer.out_channels,
                            kernel_size=layer.kernel_size,
                            stride=layer.stride,
                            padding=layer.padding,
                            bias=(layer.bias is not None))
                        temp_conv.weight.data.copy_(layer.weight.data)

                        if layer.bias is not None:
                            temp_conv.bias.data.copy_(layer.bias.data)
                        module.conv2 = temp_conv

                    if isinstance(module, Bottleneck):
                        layer = module.conv3
                        if block_count <= self.current_block_count and not isinstance(layer, MaskConv2d):
                            temp_conv = MaskConv2d(
                                in_channels=layer.in_channels,
                                out_channels=layer.out_channels,
                                kernel_size=layer.kernel_size,
                                stride=layer.stride,
                                padding=layer.padding,
                                bias=(layer.bias is not None))
                            temp_conv.weight.data.copy_(layer.weight.data)

                            if layer.bias is not None:
                                temp_conv.bias.data.copy_(layer.bias.data)
                            module.conv3 = temp_conv

    def channel_selection(self):
        """
        conduct channel selection
        """

        # get testing error
        self.segment_wise_trainer.val(0)
        time_start = time.time()

        restart_index = None
        # find restart segment index
        if self.current_pivot_index:
            if self.current_pivot_index in self.settings.pivot_set:
                restart_index = self.settings.pivot_set.index(self.current_pivot_index)
            else:
                restart_index = len(self.settings.pivot_set)

        for index in range(self.num_segments):
            if restart_index is not None:
                if index < restart_index:
                    continue
                elif index == restart_index:
                    if self.is_channel_selection and self.current_block_count == self.current_pivot_index:
                        self.is_channel_selection = False
                        continue

            if index == self.num_segments - 1:
                self.current_pivot_index = self.segment_wise_trainer.final_block_count
            else:
                self.current_pivot_index = self.settings.pivot_set[index]

            # fine tune the network with additional loss and final loss
            if (not self.is_segment_wise_finetune and not self.is_channel_selection) or \
                    (self.is_segment_wise_finetune and self.epoch != self.settings.segment_wise_n_epochs - 1):
                self.segment_wise_fine_tune(index)
            else:
                self.is_segment_wise_finetune = False

            # load best model
            best_model_path = os.path.join(self.checkpoint.save_path, 'model_{:0>3d}_swft.pth'.format(index))
            check_point_params = torch.load(best_model_path)
            ori_model_state = check_point_params["ori_model"]
            pruned_model_state = check_point_params["pruned_model"]
            aux_fc_state = check_point_params["aux_fc"]
            self.ori_model = self.checkpoint.load_state(self.ori_model, ori_model_state)
            self.pruned_model = self.checkpoint.load_state(self.pruned_model, pruned_model_state)
            self.segment_wise_trainer.update_model(self.ori_model, self.pruned_model, aux_fc_state)

            # replace the baseline model
            if index == 0:
                if self.settings.net_type in ['preresnet']:
                    self.ori_model.conv = copy.deepcopy(self.pruned_model.conv)
                    for ori_module, pruned_module in zip(self.ori_model.modules(), self.pruned_model.modules()):
                        if isinstance(ori_module, PreBasicBlock):
                            ori_module.bn1 = copy.deepcopy(pruned_module.bn1)
                            ori_module.bn2 = copy.deepcopy(pruned_module.bn2)
                            ori_module.conv1 = copy.deepcopy(pruned_module.conv1)
                            ori_module.conv2 = copy.deepcopy(pruned_module.conv2)
                            if ori_module.downsample is not None:
                                ori_module.downsample = copy.deepcopy(pruned_module.downsample)
                    self.ori_model.bn = copy.deepcopy(self.pruned_model.bn)
                    self.ori_model.fc = copy.deepcopy(self.pruned_model.fc)
                elif self.settings.net_type in ['resnet']:
                    self.ori_model.conv1 = copy.deepcopy(self.pruned_model.conv)
                    self.ori_model.bn1 = copy.deepcopy(self.pruned_model.bn1)
                    for ori_module, pruned_module in zip(self.ori_model.modules(), self.pruned_model.modules()):
                        if isinstance(ori_module, BasicBlock):
                            ori_module.conv1 = copy.deepcopy(pruned_module.conv1)
                            ori_module.conv2 = copy.deepcopy(pruned_module.conv2)
                            ori_module.bn1 = copy.deepcopy(pruned_module.bn1)
                            ori_module.bn2 = copy.deepcopy(pruned_module.bn2)
                            if ori_module.downsample is not None:
                                ori_module.downsample = copy.deepcopy(pruned_module.downsample)
                        if isinstance(ori_module, Bottleneck):
                            ori_module.conv1 = copy.deepcopy(pruned_module.conv1)
                            ori_module.conv2 = copy.deepcopy(pruned_module.conv2)
                            ori_module.conv3 = copy.deepcopy(pruned_module.conv3)
                            ori_module.bn1 = copy.deepcopy(pruned_module.bn1)
                            ori_module.bn2 = copy.deepcopy(pruned_module.bn2)
                            ori_module.bn3 = copy.deepcopy(pruned_module.bn3)
                            if ori_module.downsample is not None:
                                ori_module.downsample = copy.deepcopy(pruned_module.downsample)
                    self.ori_model.fc = copy.deepcopy(self.pruned_model.fc)

                aux_fc_state = []
                for i in range(len(self.segment_wise_trainer.aux_fc)):
                    if isinstance(self.segment_wise_trainer.aux_fc[i], nn.DataParallel):
                        temp_state = self.segment_wise_trainer.aux_fc[i].module.state_dict()
                    else:
                        temp_state = self.segment_wise_trainer.aux_fc[i].state_dict()
                    aux_fc_state.append(temp_state)
                self.segment_wise_trainer.update_model(self.ori_model, self.pruned_model, aux_fc_state)
            self.segment_wise_trainer.val(0)

            # conduct channel selection
            # contains [0:index] segments
            net_origin_list = []
            net_pruned_list = []
            for j in range(index + 1):
                net_origin_list += utils.model2list(self.segment_wise_trainer.ori_segments[j])
                net_pruned_list += utils.model2list(self.segment_wise_trainer.pruned_segments[j])

            net_origin = nn.Sequential(*net_origin_list)
            net_pruned = nn.Sequential(*net_pruned_list)

            self._seg_channel_selection(
                net_origin=net_origin,
                net_pruned=net_pruned,
                aux_fc=self.segment_wise_trainer.aux_fc[index],
                pivot_index=self.current_pivot_index,
                index=index)

            # update optimizer
            aux_fc_state = []
            for i in range(len(self.segment_wise_trainer.aux_fc)):
                if isinstance(self.segment_wise_trainer.aux_fc[i], nn.DataParallel):
                    temp_state = self.segment_wise_trainer.aux_fc[i].module.state_dict()
                else:
                    temp_state = self.segment_wise_trainer.aux_fc[i].state_dict()
                aux_fc_state.append(temp_state)

            self.segment_wise_trainer.update_model(self.ori_model, self.pruned_model, aux_fc_state)

            self.checkpoint.save_checkpoint(self.ori_model, self.pruned_model,
                                            self.segment_wise_trainer.aux_fc,
                                            self.segment_wise_trainer.fc_optimizer,
                                            self.segment_wise_trainer.seg_optimizer,
                                            self.current_pivot_index,
                                            channel_selection=True,
                                            index=index,
                                            block_count=self.current_pivot_index)

            self.logger.info(self.ori_model)
            self.logger.info(self.pruned_model)
            self.segment_wise_trainer.val(0)
            self.current_pivot_index = None

        self.checkpoint.save_model(self.ori_model, self.pruned_model,
                                   self.segment_wise_trainer.aux_fc,
                                   self.segment_wise_trainer.final_block_count,
                                   index=self.num_segments)
        time_interval = time.time() - time_start
        log_str = "cost time: {}".format(str(datetime.timedelta(seconds=time_interval)))
        self.logger.info(log_str)

    def _hook_origin_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        self.feature_cache_origin[gpu_id] = output

    def _hook_pruned_feature(self, module, input, output):
        gpu_id = str(output.get_device())
        self.feature_cache_pruned[gpu_id] = output

    @staticmethod
    def _concat_gpu_data(data):
        data_cat = data["0"]
        for i in range(1, len(data)):
            data_cat = torch.cat((data_cat, data[str(i)].cuda(0)))
        return data_cat

    def _layer_channel_selection(self, net_origin, net_pruned,
                                 aux_fc, module, block_count, layer_name="conv2"):
        """
        conduct channel selection for module
        :param net_origin: original network segments
        :param net_pruned: pruned network segments
        :param aux_fc: auxiliary fully-connected layer
        :param module: the module need to be pruned
        :param block_count: current block no.
        :param layer_name: the name of layer need to be pruned
        """

        self.logger.info("|===>layer-wise channel selection: block-{}-{}".format(block_count, layer_name))
        # layer-wise channel selection
        if layer_name == "conv2":
            layer = module.conv2
        elif layer_name == "conv3":
            layer = module.conv3
        else:
            assert False, "unsupport layer: {}".format(layer_name)

        if not isinstance(layer, MaskConv2d):
            temp_conv = MaskConv2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                bias=(layer.bias is not None))
            temp_conv.weight.data.copy_(layer.weight.data)

            if layer.bias is not None:
                temp_conv.bias.data.copy_(layer.bias.data)
            temp_conv.pruned_weight.data.fill_(0)
            temp_conv.d.fill_(0)

            if layer_name == "conv2":
                module.conv2 = temp_conv
            elif layer_name == "conv3":
                module.conv3 = temp_conv
            layer = temp_conv

        # define criterion
        criterion_mse = nn.MSELoss().cuda()
        criterion_softmax = nn.CrossEntropyLoss().cuda()

        # register hook
        if layer_name == "conv2":
            hook_origin = net_origin[block_count].conv2.register_forward_hook(self._hook_origin_feature)
            hook_pruned = module.conv2.register_forward_hook(self._hook_pruned_feature)
        elif layer_name == "conv3":
            hook_origin = net_origin[block_count].conv3.register_forward_hook(self._hook_origin_feature)
            hook_pruned = module.conv3.register_forward_hook(self._hook_pruned_feature)

        net_origin_parallel = utils.data_parallel(net_origin, self.settings.n_gpus)
        net_pruned_parallel = utils.data_parallel(net_pruned, self.settings.n_gpus)

        # avoid computing the gradient
        for params in net_origin_parallel.parameters():
            params.requires_grad = False
        for params in net_pruned_parallel.parameters():
            params.requires_grad = False

        net_origin_parallel.eval()
        net_pruned_parallel.eval()

        layer.pruned_weight.requires_grad = True
        aux_fc.cuda()
        logger_counter = 0
        record_time = utils.AverageMeter()

        for channel in range(layer.in_channels):
            if layer.d.eq(0).sum() <= math.floor(layer.in_channels * self.settings.pruning_rate):
                break

            time_start = time.time()
            cum_grad = None
            record_selection_mse_loss = utils.AverageMeter()
            record_selection_softmax_loss = utils.AverageMeter()
            record_selection_loss = utils.AverageMeter()
            img_count = 0
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.cuda()
                labels = labels.cuda()
                net_origin_parallel(images)
                output = net_pruned_parallel(images)
                softmax_loss = criterion_softmax(aux_fc(output), labels)

                origin_feature = self._concat_gpu_data(self.feature_cache_origin)
                self.feature_cache_origin = {}
                pruned_feature = self._concat_gpu_data(self.feature_cache_pruned)
                self.feature_cache_pruned = {}
                mse_loss = criterion_mse(pruned_feature, origin_feature)

                loss = mse_loss * self.settings.mse_weight + softmax_loss * self.settings.softmax_weight
                loss.backward()
                record_selection_loss.update(loss.item(), images.size(0))
                record_selection_mse_loss.update(mse_loss.item(), images.size(0))
                record_selection_softmax_loss.update(softmax_loss.item(), images.size(0))

                if cum_grad is None:
                    cum_grad = layer.pruned_weight.grad.data.clone()
                else:
                    cum_grad.add_(layer.pruned_weight.grad.data)
                    layer.pruned_weight.grad = None

                img_count += images.size(0)
                if self.settings.max_samples != -1 and img_count >= self.settings.max_samples:
                    break

            self.v_logger.scalar("F-block-{}_{}_LossAll".format(block_count, layer_name),
                                                   logger_counter,
                                                   [record_selection_loss.avg])

            self.v_logger.scalar("F-block-{}_{}_MSELoss".format(block_count, layer_name),
                                                   logger_counter,
                                                   [record_selection_mse_loss.avg])

            self.v_logger.scalar("F-block-{}_{}_SoftmaxLoss".format(block_count, layer_name),
                                                   logger_counter,
                                                   [record_selection_softmax_loss.avg])

            cum_grad.abs_()
            # calculate gradient F norm
            grad_fnorm = cum_grad.mul(cum_grad).sum((2, 3)).sqrt().sum(0)

            # find grad_fnorm with maximum absolute gradient
            while True:
                _, max_index = torch.topk(grad_fnorm, 1)
                if layer.d[max_index[0]] == 0:
                    layer.d[max_index[0]] = 1
                    layer.pruned_weight.data[:, max_index[0], :, :] = layer.weight[:, max_index[0], :, :].data.clone()
                    break
                else:
                    grad_fnorm[max_index[0]] = -1

            # fine-tune average meter
            record_finetune_softmax_loss = utils.AverageMeter()
            record_finetune_mse_loss = utils.AverageMeter()
            record_finetune_loss = utils.AverageMeter()

            record_finetune_top1_error = utils.AverageMeter()
            record_finetune_top5_error = utils.AverageMeter()

            # define optimizer
            params_list = []
            params_list.append({"params": layer.pruned_weight, "lr": self.settings.layer_wise_lr})
            if layer.bias is not None:
                layer.bias.requires_grad = True
                params_list.append({"params": layer.bias, "lr": 0.001})
            optimizer = torch.optim.SGD(params=params_list,
                                        weight_decay=self.settings.weight_decay,
                                        momentum=self.settings.momentum,
                                        nesterov=True)
            img_count = 0
            for epoch in range(1):
                for i, (images, labels) in enumerate(self.train_loader):
                    images = images.cuda()
                    labels = labels.cuda()
                    features = net_pruned_parallel(images)
                    net_origin_parallel(images)
                    output = aux_fc(features)
                    softmax_loss = criterion_softmax(output, labels)

                    origin_feature = self._concat_gpu_data(self.feature_cache_origin)
                    self.feature_cache_origin = {}
                    pruned_feature = self._concat_gpu_data(self.feature_cache_pruned)
                    self.feature_cache_pruned = {}
                    mse_loss = criterion_mse(pruned_feature, origin_feature)

                    top1_error, _, top5_error = utils.compute_singlecrop(
                        outputs=output, labels=labels,
                        loss=softmax_loss, top5_flag=True, mean_flag=True)

                    # update parameters
                    optimizer.zero_grad()
                    loss = mse_loss * self.settings.mse_weight + softmax_loss * self.settings.softmax_weight
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=10.0)
                    layer.pruned_weight.grad.data.mul_(
                        layer.d.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(layer.pruned_weight))
                    optimizer.step()
                    # update record info
                    record_finetune_softmax_loss.update(softmax_loss.item(), images.size(0))
                    record_finetune_mse_loss.update(mse_loss.item(), images.size(0))
                    record_finetune_loss.update(loss.item(), images.size(0))
                    record_finetune_top1_error.update(top1_error, images.size(0))
                    record_finetune_top5_error.update(top5_error, images.size(0))

                    img_count += images.size(0)
                    if self.settings.max_samples != -1 and img_count >= self.settings.max_samples:
                        break

            layer.pruned_weight.grad = None
            if layer.bias is not None:
                layer.bias.requires_grad = False

            self.v_logger.scalar("F-block-{}_{}_SoftmaxLoss".format(block_count, layer_name),
                                                   logger_counter,
                                                   [record_finetune_softmax_loss.avg])

            self.v_logger.scalar("F-block-{}_{}_Loss".format(block_count, layer_name),
                                                   logger_counter,
                                                   [record_finetune_mse_loss.avg])

            self.v_logger.scalar("F-block-{}_{}_MSELoss".format(block_count, layer_name),
                                                   logger_counter,
                                                   [record_finetune_loss.avg])

            self.v_logger.scalar("F-block-{}_{}_Top1Error".format(block_count, layer_name),
                                                   logger_counter,
                                                   [record_finetune_top1_error.avg])

            self.v_logger.scalar("F-block-{}_{}_Top5Error".format(block_count, layer_name),
                                                   logger_counter,
                                                   [record_finetune_top5_error.avg])

            # write log information to file
            self._write_log(
                dir_name=os.path.join(self.settings.save_path, "log"),
                file_name="log_block-{:0>2d}_{}.txt".format(block_count, layer_name),
                log_str="{:d}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t\n".format(
                    int(layer.d.sum()),
                    record_selection_loss.avg,
                    record_selection_mse_loss.avg,
                    record_selection_softmax_loss.avg,
                    record_finetune_loss.avg,
                    record_finetune_mse_loss.avg,
                    record_finetune_softmax_loss.avg,
                    record_finetune_top1_error.avg,
                    record_finetune_top5_error.avg))
            log_str = "Block-{:0>2d}-{}\t#channels: [{:0>4d}|{:0>4d}]\t".format(
                block_count, layer_name,
                int(layer.d.sum()), layer.d.size(0))
            log_str += "[selection]loss: {:4f}\tmseloss: {:4f}\tsoftmaxloss: {:4f}\t".format(
                record_selection_loss.avg,
                record_selection_mse_loss.avg,
                record_selection_softmax_loss.avg)
            log_str += "[fine-tuning]loss: {:4f}\tmseloss: {:4f}\tsoftmaxloss: {:4f}\t".format(
                record_finetune_loss.avg,
                record_finetune_mse_loss.avg,
                record_finetune_softmax_loss.avg)
            log_str += "top1error: {:4f}\ttop5error: {:4f}".format(
                record_finetune_top1_error.avg,
                record_finetune_top5_error.avg)
            self.logger.info(log_str)

            logger_counter += 1
            time_interval = time.time() - time_start
            record_time.update(time_interval)

        for params in net_origin_parallel.parameters():
            params.requires_grad = True
        for params in net_pruned_parallel.parameters():
            params.requires_grad = True

        # remove hook
        hook_origin.remove()
        hook_pruned.remove()
        log_str = "|===>Select channel from block-{:d}_{}: time_total:{} time_avg: {}".format(
            block_count, layer_name,
            str(datetime.timedelta(seconds=record_time.sum)),
            str(datetime.timedelta(seconds=record_time.avg)))
        self.logger.info(log_str)
        log_str = "|===>fine-tuning result: loss: {:f}, mse_loss: {:f}, softmax_loss: {:f}, top1error: {:f} top5error: {:f}".format(
            record_finetune_loss.avg,
            record_finetune_mse_loss.avg,
            record_finetune_softmax_loss.avg,
            record_finetune_top1_error.avg,
            record_finetune_top5_error.avg)
        self.logger.info(log_str)

        self.logger.info("|===>remove hook")

    @staticmethod
    def _write_log(dir_name, file_name, log_str):
        """
        Write log to file
        :param dir_name:  the path of directory
        :param file_name: the name of the saved file
        :param log_str: the string that need to be saved
        """

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        with open(os.path.join(dir_name, file_name), "a+") as f:
            f.write(log_str)

    def _seg_channel_selection(self, net_origin, net_pruned, aux_fc, pivot_index, index):
        """
        conduct segment channel selection
        :param net_origin: original network segments
        :param net_pruned: pruned network segments
        :param aux_fc: auxiliary fully-connected layer
        :param pivot_index: the layer index of the additional loss
        :param index: the index of segment
        :return:
        """
        block_count = 0
        if self.settings.net_type in ["preresnet", "resnet"]:
            for module in net_pruned.modules():
                if isinstance(module, (PreBasicBlock, BasicBlock)):
                    block_count += 1
                    # We will not prune the pruned blocks again
                    if not isinstance(module.conv2, MaskConv2d):
                        self._layer_channel_selection(
                            net_origin=net_origin, net_pruned=net_pruned,
                            aux_fc=aux_fc, module=module, block_count=block_count,
                            layer_name="conv2")
                        self.logger.info("|===>checking layer type: {}".format(type(module.conv2)))

                        self.checkpoint.save_model(self.ori_model, self.pruned_model,
                                                   self.segment_wise_trainer.aux_fc,
                                                   pivot_index, channel_selection=True,
                                                   index=index, block_count=block_count)
                        self.checkpoint.save_checkpoint(self.ori_model, self.pruned_model,
                                                        self.segment_wise_trainer.aux_fc,
                                                        self.segment_wise_trainer.fc_optimizer,
                                                        self.segment_wise_trainer.seg_optimizer,
                                                        pivot_index,
                                                        channel_selection=True,
                                                        index=index, block_count=block_count)

                elif isinstance(module, Bottleneck):
                    block_count += 1
                    if not isinstance(module.conv2, MaskConv2d):
                        self._layer_channel_selection(
                            net_origin=net_origin, net_pruned=net_pruned,
                            aux_fc=aux_fc, module=module, block_count=block_count,
                            layer_name="conv2")

                    if not isinstance(module.conv3, MaskConv2d):
                        self._layer_channel_selection(
                            net_origin=net_origin, net_pruned=net_pruned,
                            aux_fc=aux_fc, module=module, block_count=block_count,
                            layer_name="conv3")

                        self.checkpoint.save_model(self.ori_model, self.pruned_model,
                                                   self.segment_wise_trainer.aux_fc,
                                                   pivot_index, channel_selection=True,
                                                   index=index, block_count=block_count)
                        self.checkpoint.save_checkpoint(self.ori_model, self.pruned_model,
                                                        self.segment_wise_trainer.aux_fc,
                                                        self.segment_wise_trainer.fc_optimizer,
                                                        self.segment_wise_trainer.seg_optimizer,
                                                        pivot_index,
                                                        channel_selection=True,
                                                        index=index, block_count=block_count)


def main():
    parser = argparse.ArgumentParser(description="Discrimination-aware channel pruning")
    parser.add_argument('conf_path', type=str, metavar='conf_path',
                        help='configuration path')
    args = parser.parse_args()

    option = Option(args.conf_path)

    experiment = Experiment(option)
    experiment.channel_selection()


if __name__ == '__main__':
    main()
