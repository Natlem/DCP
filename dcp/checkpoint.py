import os
import dcp.utils as utils

import torch
import torch.nn as nn

__all__ = ["CheckPoint"]


class CheckPoint(object):
    """
    save model state to file
    check_point_params: model, optimizer, epoch
    """

    def __init__(self, save_path, logger):

        self.save_path = os.path.join(save_path, "check_point")
        self.check_point_params = {'model': None,
                                   'optimizer': None,
                                   'epoch': None}
        self.logger = logger

        # make directory
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def load_state(self, model, state_dict):
        """
        load state_dict to model
        :params model:
        :params state_dict:
        :return: model
        """
        model.eval()
        model = utils.list2sequential(model)
        model_dict = model.state_dict()

        for key, value in list(state_dict.items()):
            if key in list(model_dict.keys()):
                model_dict[key] = value
            else:
                self.logger.error("key error: {} {}".format(key, value.size))
                # assert False
        model.load_state_dict(model_dict)
        return model

    def load_model(self, model_path):
        """
        load model
        :params model_path: path to the model
        :return: model_state_dict
        """
        if os.path.isfile(model_path):
            self.logger.info("|===>Load retrain model from: {}".format(model_path))
            model_state_dict = torch.load(model_path, map_location={'cuda:1': 'cuda:0'})
            return model_state_dict
        else:
            assert False, "file not exits, model path: " + model_path

    def load_checkpoint(self, checkpoint_path):
        """
        load checkpoint file
        :params checkpoint_path: path to the checkpoint file
        :return: model_state_dict, optimizer_state_dict, epoch
        """
        if os.path.isfile(checkpoint_path):
            self.logger.info("|===>Load resume check-point from: {}".format(checkpoint_path))
            self.check_point_params = torch.load(checkpoint_path)
            model_state_dict = self.check_point_params['model']
            optimizer_state_dict = self.check_point_params['optimizer']
            epoch = self.check_point_params['epoch']
            return model_state_dict, optimizer_state_dict, epoch
        else:
            assert False, "file not exits" + checkpoint_path

    def save_checkpoint(self, ori_model, pruned_model, aux_fc=None, aux_fc_opt=None, seg_opt=None,
                        current_pivot=None, segment_wise_finetune=False, channel_selection=False,
                        index=0, epoch=0, block_count=0):
        # save state of the network
        check_point_params = {}
        ori_model = utils.list2sequential(ori_model)
        if isinstance(ori_model, nn.DataParallel):
            check_point_params["ori_model"] = ori_model.module.state_dict()
        else:
            check_point_params["ori_model"] = ori_model.state_dict()
        pruned_model = utils.list2sequential(pruned_model)
        if isinstance(pruned_model, nn.DataParallel):
            check_point_params["pruned_model"] = pruned_model.module.state_dict()
        else:
            check_point_params["pruned_model"] = pruned_model.state_dict()
        aux_fc_state = []
        aux_fc_opt_state = []
        seg_opt_state = []
        if aux_fc:
            for i in range(len(aux_fc)):
                if isinstance(aux_fc[i], nn.DataParallel):
                    temp_state = aux_fc[i].module.state_dict()
                else:
                    temp_state = aux_fc[i].state_dict()
                aux_fc_state.append(temp_state)
                if aux_fc_opt:
                    aux_fc_opt_state.append(aux_fc_opt[i].state_dict())
                if seg_opt:
                    seg_opt_state.append(seg_opt[i].state_dict())
        check_point_params["aux_fc"] = aux_fc_state
        check_point_params["aux_fc_opt"] = aux_fc_opt_state
        check_point_params["seg_opt"] = seg_opt_state
        check_point_params["current_pivot"] = current_pivot
        check_point_params["segment_wise_finetune"] = segment_wise_finetune
        check_point_params["channel_selection"] = channel_selection
        check_point_params["epoch"] = epoch
        check_point_params["current_block_count"] = block_count
        checkpoint_save_name = "checkpoint_{:0>3d}.pth".format(index)
        if segment_wise_finetune:
            checkpoint_save_name = "checkpoint_{:0>3d}_swft_{}.pth".format(index, epoch)
        if channel_selection:
            checkpoint_save_name = "checkpoint_{:0>3d}_cs_{:0>3d}.pth".format(index, block_count)
        torch.save(check_point_params, os.path.join(self.save_path, checkpoint_save_name))

    def save_network_wise_fine_tune_checkpoint(self, pruned_model, optimizer=None, epoch=0, index=0):
        # save state of the network
        check_point_params = {}
        pruned_model = utils.list2sequential(pruned_model)
        if isinstance(pruned_model, nn.DataParallel):
            check_point_params["pruned_model"] = pruned_model.module.state_dict()
        else:
            check_point_params["pruned_model"] = pruned_model.state_dict()
        check_point_params["optimizer_state"] = optimizer.state_dict()
        check_point_params["epoch"] = epoch
        checkpoint_save_name = "checkpoint_ft_{:0>3d}.pth".format(index)
        torch.save(check_point_params, os.path.join(self.save_path, checkpoint_save_name))

    def save_model(self, ori_model, pruned_model, aux_fc=None, current_pivot=None, segment_wise_finetune=False,
                   channel_selection=False, index=0, block_count=0):
        # save final model
        check_point_params = {}
        ori_model = utils.list2sequential(ori_model)
        if isinstance(ori_model, nn.DataParallel):
            check_point_params["ori_model"] = ori_model.module.state_dict()
        else:
            check_point_params["ori_model"] = ori_model.state_dict()
        pruned_model = utils.list2sequential(pruned_model)
        if isinstance(pruned_model, nn.DataParallel):
            check_point_params["pruned_model"] = pruned_model.module.state_dict()
        else:
            check_point_params["pruned_model"] = pruned_model.state_dict()
        aux_fc_state = []
        if aux_fc:
            for i in range(len(aux_fc)):
                if isinstance(aux_fc[i], nn.DataParallel):
                    aux_fc_state.append(aux_fc[i].module.state_dict())
                else:
                    aux_fc_state.append(aux_fc[i].state_dict())
        check_point_params["aux_fc"] = aux_fc_state
        check_point_params["current_pivot"] = current_pivot
        check_point_params["segment_wise_finetune"] = segment_wise_finetune
        check_point_params["channel_selection"] = channel_selection
        model_save_name = "model_{:0>3d}.pth".format(index)
        if segment_wise_finetune:
            model_save_name = "model_{:0>3d}_swft.pth".format(index)
        if channel_selection:
            model_save_name = "model_{:0>3d}_cs_{:0>3d}.pth".format(index, block_count)
        torch.save(check_point_params, os.path.join(self.save_path, model_save_name))

    def save_network_wise_fine_tune_model(self, pruned_model, best_flag=False, index=0, tag=""):
        """
        :params model: model to save
        :params best_flag: if True, the saved model is the one that gets best performance
        """
        # get state dict
        pruned_model = utils.list2sequential(pruned_model)
        if isinstance(pruned_model, nn.DataParallel):
            pruned_model = pruned_model.module
        pruned_model = pruned_model.state_dict()
        if best_flag:
            if tag != "":
                torch.save(pruned_model, os.path.join(self.save_path, "{}_best_model.pth".format(tag)))
            else:
                torch.save(pruned_model, os.path.join(self.save_path, "best_model.pth"))
        else:
            if tag != "":
                torch.save(pruned_model, os.path.join(self.save_path, "{}_model_{:0>3d}.pth".format(tag, index)))
            else:
                torch.save(pruned_model, os.path.join(self.save_path, "model_{:0>3d}.pth".format(index)))
