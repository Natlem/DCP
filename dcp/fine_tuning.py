import argparse
import logging
import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from checkpoint import CheckPoint
from option import Option
from pruning import ResModelPrune
from trainer import NetworkWiseTrainer

import dcp.models as md
from dcp.mask_conv import MaskConv2d
from dcp.models.preresnet import PreBasicBlock
from dcp.models.resnet import BasicBlock, Bottleneck
from dcp.utils.tensorboard_logger import TensorboardLogger


class Experiment(object):
    """
    run experiments with pre-defined pipeline
    """

    def __init__(self, options=None, conf_path=None):
        self.settings = options or Option(conf_path)
        self.checkpoint = None
        self.train_loader = None
        self.val_loader = None
        self.pruned_model = None
        self.network_wise_trainer = None
        self.optimizer_state = None

        os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.gpu

        self.settings.set_save_path()
        self.write_settings()
        self.logger = self.set_logger()
        self.tensorboard_logger = TensorboardLogger(self.settings.save_path)

        self.epoch = 0

        self.prepare()

    def write_settings(self):
        """
        save expriment settings to a file
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
        torch.cuda.set_device(0)
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

    def _set_model(self):
        """
        get model
        """

        if self.settings.dataset in ["cifar10", "cifar100"]:
            if self.settings.net_type == "preresnet":
                self.pruned_model = md.PreResNet(depth=self.settings.depth,
                                                 num_classes=self.settings.n_classes)
            else:
                assert False, "use {} data while network is {}".format(self.settings.dataset, self.settings.net_type)

        elif self.settings.dataset in ["imagenet", "imagenet_mio"]:
            if self.settings.net_type == "resnet":
                self.pruned_model = md.ResNet(
                    depth=self.settings.depth,
                    num_classes=self.settings.n_classes)
            else:
                assert False, "use {} data while network is {}".format(self.settings.dataset, self.settings.net_type)

        else:
            assert False, "unsupported data set: {}".format(self.settings.dataset)

        # replace the conv layer in resnet with mask_conv
        if self.settings.net_type in ["preresnet", "resnet"]:
            for module in self.pruned_model.modules():
                if isinstance(module, (PreBasicBlock, BasicBlock, Bottleneck)):
                    # replace conv2
                    temp_conv = MaskConv2d(
                        in_channels=module.conv2.in_channels,
                        out_channels=module.conv2.out_channels,
                        kernel_size=module.conv2.kernel_size,
                        stride=module.conv2.stride,
                        padding=module.conv2.padding,
                        bias=(module.conv2.bias is not None))

                    temp_conv.weight.data.copy_(module.conv2.weight.data)
                    if module.conv2.bias is not None:
                        temp_conv.bias.data.copy_(module.conv2.bias.data)
                    module.conv2 = temp_conv

                    if isinstance(module, (Bottleneck)):
                        # replace conv3
                        temp_conv = MaskConv2d(
                            in_channels=module.conv3.in_channels,
                            out_channels=module.conv3.out_channels,
                            kernel_size=module.conv3.kernel_size,
                            stride=module.conv3.stride,
                            padding=module.conv3.padding,
                            bias=(module.conv3.bias is not None))

                        temp_conv.weight.data.copy_(module.conv3.weight.data)
                        if module.conv3.bias is not None:
                            temp_conv.bias.data.copy_(module.conv3.bias.data)
                        module.conv3 = temp_conv

    def _set_checkpoint(self):
        """
        load pre-trained model or resume checkpoint
        """

        assert self.pruned_model is not None, "please create model first"

        self.checkpoint = CheckPoint(self.settings.save_path, self.logger)
        self._load_pretrained()
        self._load_resume()

    def _load_pretrained(self):
        """
        load pre-trained model
        """

        if self.settings.retrain is not None:
            check_point_params = torch.load(self.settings.retrain)
            model_state = check_point_params["pruned_model"]
            self.pruned_model = self.checkpoint.load_state(self.pruned_model, model_state)
            self.logger.info("|===>load restrain file: {}".format(self.settings.retrain))

    def _load_resume(self):
        """
        load resume checkpoint
        """

        if self.settings.resume is not None:
            check_point_params = torch.load(self.settings.resume)
            pruned_model_state = check_point_params["pruned_model"]
            self.optimizer_state = check_point_params["optimizer_state"]
            self.epoch = check_point_params["epoch"]
            self.pruned_model = self.checkpoint.load_state(self.pruned_model, pruned_model_state)
            self.logger.info("|===>load resume file: {}".format(self.settings.resume))

    def _set_trainier(self):
        """
        initialize network-wise trainer
        """

        self.network_wise_trainer = NetworkWiseTrainer(pruned_model=self.pruned_model,
                                                       train_loader=self.train_loader,
                                                       val_loader=self.val_loader,
                                                       settings=self.settings,
                                                       logger=self.logger,
                                                       tensorboard_logger=self.tensorboard_logger)


    def pruning(self):
        """
        prune channels
        """

        self.logger.info(self.pruned_model)
        self.network_wise_trainer.val(0)

        if self.settings.net_type in ["preresnet", "resnet"]:
            model_prune = ResModelPrune(model=self.pruned_model,
                                        net_type=self.settings.net_type,
                                        depth=self.settings.depth)
        else:
            assert False, "unsupport net_type: {}".format(self.settings.net_type)

        model_prune.run()
        self.network_wise_trainer.update_model(model_prune.model, self.optimizer_state)

        self.network_wise_trainer.val(0)
        self.logger.info(self.pruned_model)

    def fine_tuning(self):
        """
        conduct network-wise fine-tuning
        """

        best_top1 = 100
        best_top5 = 100

        start_epoch = 0
        if self.epoch != 0:
            start_epoch = self.epoch + 1
            self.epoch = 0

        self.network_wise_trainer.val(0)

        for epoch in range(start_epoch, self.settings.network_wise_n_epochs):
            train_error, train_loss, train5_error = self.network_wise_trainer.train(epoch)
            val_error, val_loss, val5_error = self.network_wise_trainer.val(epoch)

            for module in self.pruned_model.modules():
                if isinstance(module, MaskConv2d):
                    print(module.pruned_weight[0, :, 0, 0].eq(0).sum())

            # write and print result
            best_flag = False
            if best_top1 >= val_error:
                best_top1 = val_error
                best_top5 = val5_error
                best_flag = True

            if best_flag:
                self.checkpoint.save_network_wise_fine_tune_model(self.pruned_model, best_flag)

            self.logger.info("|===>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}\n".format(best_top1, best_top5))
            self.logger.info("|==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}\n".format(100 - best_top1,
                                                                                                     100 - best_top5))

            if self.settings.dataset in ["imagenet"]:
                self.checkpoint.save_network_wise_fine_tune_checkpoint(
                    self.pruned_model, self.network_wise_trainer.optimizer, epoch, epoch)
            else:
                self.checkpoint.save_network_wise_fine_tune_checkpoint(
                    self.pruned_model, self.network_wise_trainer.optimizer, epoch)


def main():
    """
    main func
    """
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('conf_path', type=str, metavar='conf_path',
                        help='input batch size for training (default: 64)')
    args = parser.parse_args()

    option = Option(args.conf_path)

    experiment = Experiment(option)
    experiment.pruning()
    experiment.fine_tuning()


if __name__ == '__main__':
    main()
