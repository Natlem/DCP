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
from trainer import NetworkWiseTrainer

from dcp.models.pruned_preresnet import PrunedPreResNet
from dcp.models.pruned_resnet import PrunedResNet


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
        self.logger = self.set_logger()
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
        console_formatter = logging.Formatter('%(message)s')

        # console log
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)

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
                self.pruned_model = PrunedPreResNet(depth=self.settings.depth,
                                                    pruning_rate=self.settings.pruning_rate,
                                                    num_classes=self.settings.n_classes)
            else:
                assert False, "use {} data while network is {}".format(self.settings.dataset, self.settings.net_type)

        elif self.settings.dataset in ["imagenet", "imagenet_mio"]:
            if self.settings.net_type == "resnet":
                self.pruned_model = PrunedResNet(
                    depth=self.settings.depth,
                    pruning_rate=self.settings.pruning_rate,
                    num_classes=self.settings.n_classes)
            else:
                assert False, "use {} data while network is {}".format(self.settings.dataset, self.settings.net_type)

        else:
            assert False, "unsupported data set: {}".format(self.settings.dataset)

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
            model_state = check_point_params
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
                                                       tensorboard_logger=None)

    def evaluation(self):
        """
        conduct network-wise fine-tuning
        """

        val_error, val_loss, val5_error = self.network_wise_trainer.val(0)


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
    experiment.evaluation()


if __name__ == '__main__':
    main()
