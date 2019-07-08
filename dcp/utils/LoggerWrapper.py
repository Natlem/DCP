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
