from sacred import Experiment
ex = Experiment()

from sacred.observers import MongoObserver
ex.observers.append(MongoObserver.create(url="10.180.177.56:27017"))
from option import Option

class LoggerForSacred():
    def __init__(self, visdom_logger, id, ex_logger=None):
        self.visdom_logger = visdom_logger
        self.ex_logger = ex_logger
        self.id = id


    def log_scalar(self, metrics_name, value, step):
        if self.visdom_logger is not None:
            self.visdom_logger.scalar(metrics_name + "_{}".format(self.id), step, [value])
        if self.ex_logger is not None:
            self.ex_logger.log_scalar(metrics_name + "_{}".format(self.id), value, step)

from dcp.channel_pruning import Experiment as exp_cp

@ex.config
def exp_config():

    # Hyper Parameters Config
    conf_path = "cifat_resnet_03.hocon"

@ex.capture()
def exp_basic_train(conf_path, id):

    logger = ex
    l1 = LoggerForSacred(None, id, logger)
    option = Option(conf_path)

    experiment = exp_cp(option, None, l1)
    experiment.channel_selection()


@ex.main
def run_exp():
    exp_basic_train()

if __name__ == "__main__":

    ex.run(config_updates={'conf_path': "dcp/cifar10_resnet_03.hocon" , "id": "0.3"},
           options={'--name': 'DCP_cifar10_resnet_56_cp_prune_03'})
    ex.run(config_updates={'conf_path': "dcp/cifar10_resnet_05.hocon" , "id": "0.5"},
           options={'--name': 'DCP_cifar10_resnet_56_cp_prune_05'})
    ex.run(config_updates={'conf_path': "dcp/cifar10_resnet_07.hocon" , "id": "0.7"},
           options={'--name': 'DCP_cifar10_resnet_56_cp_prune_07'})
    ex.run(config_updates={'conf_path': "dcp/cifar10_resnet_09.hocon" , "id": "0.9"},
           options={'--name': 'DCP_cifar10_resnet_56_cp_prune_09'})

