
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.metrics import confusion_matrix
import pickle
import random
from copy import deepcopy
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_physicist.theory_learning.models import Loss_Fun_Cumu, get_Lagrangian_loss
from AI_physicist.theory_learning.util_theory import forward, logplus, Loss_Decay_Scheduler, count_metrics_pytorch, plot3D, plot_indi_domain, to_one_hot, load_info_dict, get_piecewise_dataset, get_group_norm
from AI_physicist.settings.filepath import theory_PATH
from AI_physicist.settings.global_param import COLOR_LIST, PrecisionFloorLoss
from AI_physicist.pytorch_net.util import Loss_Fun, make_dir, Early_Stopping, record_data, plot_matrices, get_args, base_repr, base_repr_2_int
from AI_physicist.pytorch_net.util import sort_two_lists, to_string, Loss_with_uncertainty, get_optimizer, Gradient_Noise_Scale_Gen, to_np_array, to_Variable, to_Boolean, get_criterion
from AI_physicist.pytorch_net.net import MLP, Net_Ensemble, load_model_dict, construct_net_ensemble_from_nets, train_simple


# In[ ]:


def get_pred_with_uncertainty(preds, uncertainty_nets, X):
    log_std = uncertainty_nets(X)
    info_list = torch.exp(-2 * (F.relu(log_std + 20) - 20)) + 1e-20
    pred_with_uncertainty = (preds * info_list).sum(1, keepdim = True) / info_list.sum(1, keepdim = True)
    return pred_with_uncertainty, info_list


def get_best_model_idx(net_dict, X, y, loss_fun_cumu, forward_steps = 1, mode = "first_step", is_Lagrangian = False):
    preds, _ = get_preds_valid(net_dict, X, forward_steps = forward_steps, is_Lagrangian = is_Lagrangian)
    loss_indi = loss_fun_cumu(preds, y, cumu_mode = "original", neglect_threshold_on = False, is_mean = False)
    num_theories = net_dict["pred_nets"].num_models
    if mode == "first_step":
        return loss_indi.min(1)[1] / num_theories ** (forward_steps - 1)
    elif mode == "expanded":
        return loss_indi.min(1)[1]
    else:
        raise Exception("mode {0} not recognized!".format(mode))


def get_valid_prob(X, domain_net, num_theories, domain_pred_mode = "onehot"):
    if domain_net is not None:
        if domain_pred_mode == "prob":
            valid_prob = nn.Softmax(dim = 1)(domain_net(X))
        elif domain_pred_mode == "onehot":
            valid_prob = nn.Softmax(dim = 1)(domain_net(X)).max(1)[1]
            valid_prob = to_one_hot(valid_prob, num = num_theories).float()
        else:
            raise Exception("domain_pred_mode {0} not recognized!".format(domain_pred_mode))
    else:
        valid_prob = None
    return valid_prob


def get_preds_valid(net_dict, X, forward_steps = 1, domain_net = None, domain_pred_mode = "onehot", is_Lagrangian = False, disable_autoencoder = False):
    if "autoencoder" in net_dict and len(X.shape) == 4 and not disable_autoencoder:
        autoencoder = net_dict["autoencoder"]
        X = autoencoder.encode(X)
    pred_nets = net_dict["pred_nets"]
    num_theories = pred_nets.num_models
    if is_Lagrangian:
        assert forward_steps == 1
        preds = get_Lagrangian_loss(pred_nets, X)
    else:
        preds = pred_nets(X)
        
    valid_prob = get_valid_prob(X, domain_net, num_theories = num_theories, domain_pred_mode = domain_pred_mode)
    if forward_steps > 1:
        num_output_dims = preds.size(-1)
        input_dim = int(X.size(1) / num_output_dims)
        pred_list_all = [preds]
        for i in range(forward_steps - 1):
            new_pred_list = []
            valid_prob_list = []
            for k in range(pred_list_all[-1].size(1)):
                prev_idx = base_repr(k, num_theories, len(pred_list_all))
                prev_len = len(prev_idx)
                current_input = []
                source_dim = input_dim - prev_len
                if source_dim > 0:
                    current_input.append(X[:, -source_dim * num_output_dims:].contiguous().view(-1, source_dim, num_output_dims))
                len_backward = min(input_dim, prev_len)
                for j in range(len_backward):
                    idx_j = base_repr_2_int(prev_idx[:len(prev_idx)-len_backward+j+1], base = num_theories)
                    current_input.append(pred_list_all[-len_backward + j][:, idx_j:idx_j+1])
                current_input = torch.cat(current_input, 1)
                current_input_flattened = current_input.view(-1, input_dim * num_output_dims)
                if domain_net is not None:
                    new_valid_prob_ele = get_valid_prob(current_input_flattened, domain_net, num_theories = num_theories, domain_pred_mode = domain_pred_mode)
                    valid_prob_ele = valid_prob[:, k:k+1] * new_valid_prob_ele
                    valid_prob_list.append(valid_prob_ele)
                new_pred = net_dict["pred_nets"](current_input_flattened)
                new_pred_list.append(new_pred)
            new_pred_list = torch.cat(new_pred_list, 1)
            if domain_net is not None:
                valid_prob = torch.cat(valid_prob_list, 1)
            pred_list_all.append(new_pred_list)
        preds = pred_list_all[-1]
    if "autoencoder" in net_dict and not disable_autoencoder:
        preds = net_dict["autoencoder"].decode(preds)
    return preds, valid_prob


def get_loss(
    net_dict,
    X,
    y,
    loss_types, 
    forward_steps = 1, 
    domain_net = None, 
    domain_pred_mode = "onehot", 
    loss_fun_dict = {},
    replaced_loss_order = None, 
    is_Lagrangian = False,
    is_mean = True,
    ):
    """Evaluates the various loss metrics."""
    preds, valid_prob = get_preds_valid(net_dict,
                                        X,
                                        forward_steps = forward_steps,
                                        domain_net = domain_net, 
                                        domain_pred_mode = domain_pred_mode,
                                        is_Lagrangian = is_Lagrangian,
                                       )
    loss_dict = {}
    for loss_mode, loss_setting in loss_types.items():
        if loss_mode[:10] == "pred-based":
            loss_fun_cumu = loss_fun_dict["loss_fun_cumu"]
            loss_mode_split = loss_mode.split("_")
            try:
                cumu_mode = (loss_mode_split[1], float(loss_mode_split[2]))
            except:
                cumu_mode = loss_mode_split[1]
            if replaced_loss_order is not None and cumu_mode[0] == "generalized-mean" and ("decay_on" in loss_setting and loss_setting["decay_on"] is True):
                cumu_mode = (cumu_mode[0], replaced_loss_order[loss_mode])
            loss_dict[loss_mode] = loss_fun_cumu(preds, y, model_weights = valid_prob, cumu_mode = cumu_mode, is_mean = is_mean) * loss_setting["amp"]
        elif loss_mode == "uncertainty-based":
            assert forward_steps == 1
            loss_with_uncertainty = loss_fun_dict["loss_with_uncertainty"]
            loss_with_uncertainty.is_mean = is_mean
            pred_with_uncertainty, info_list = get_pred_with_uncertainty(preds, net_dict["uncertainty_nets"], X)
            std = info_list.sum(1) ** (-0.5)
            loss_dict[loss_mode] = loss_with_uncertainty(pred_with_uncertainty, y, std = std) * loss_setting["amp"]
        else:
            raise Exception("loss_mode {0} not recognized!".format(loss_mode))
    loss = 0
    for loss_mode, loss_ele in loss_dict.items():
        loss = loss + loss_ele
    return loss, loss_dict


def get_reg(net_dict, reg_dict, mode = "L2", is_cuda = False):   
    reg_value_dict = {}
    for net_target, reg_setting in reg_dict.items():
        if net_target == "pred_nets":
            for source_target, reg_amp in reg_setting.items():
                reg_value_dict["pred_nets"] = net_dict["pred_nets"].get_regularization(source = [source_target], mode = mode) * reg_amp
        elif net_target == "domain_net":
            for source_target, reg_amp in reg_setting.items():
                reg_value_dict["domain_net"] = net_dict["domain_net"].get_regularization(source = [source_target], mode = mode) * reg_amp
        elif net_target == "uncertainty_nets" and "uncertainty_nets" in net_dict:                
            for source_target, reg_amp in reg_setting.items():
                reg_value_dict["uncertainty_nets"] = net_dict["uncertainty_nets"].get_regularization(source = [source_target], mode = mode) * reg_amp
        else:
            raise
    reg = Variable(torch.FloatTensor([0]), requires_grad = False)
    if is_cuda:
        reg = reg.cuda()
    for net_target, reg_value in reg_value_dict.items():
        reg = reg + reg_value
    return reg, reg_value_dict


def combine_losses(loss_with_domain, loss_without_domain, loss_distribution_mode, isTorch = True, **kwargs):
    if isinstance(loss_distribution_mode, np.ndarray):
        k = kwargs["k"]
        loss = loss_with_domain * loss_distribution_mode[k] + loss_without_domain * (1 - loss_distribution_mode[k])
    elif isinstance(loss_distribution_mode, tuple):
        if loss_distribution_mode[0] == "generalized-mean":
            order = float(loss_distribution_mode[1])
            loss = ((loss_with_domain ** order + loss_without_domain ** order) / 2) ** (1 / order)
        else:
            raise
    elif loss_distribution_mode == "min":
        if isTorch:
            loss = torch.min(loss_with_domain, loss_without_domain)
        else:
            loss = np.minimum(loss_with_domain, loss_without_domain)
    elif loss_distribution_mode == "harmonic":
        loss = 2 / (1 / loss_with_domain + 1 / loss_without_domain)
    elif loss_distribution_mode == "mean":
        loss = (loss_with_domain + loss_without_domain) / 2
    else:
        raise Exception("loss_distribution_mode {0} not recognized!".format(loss_distribution_mode))
    return loss


def load_model_dict(model_dict):
    if model_dict["type"] == "Theory_Training":
        model = Theory_Training(num_theories = model_dict["num_theories"],
                                proposed_theory_models = None,
                                input_size = model_dict["input_size"],
                                struct_param_pred = model_dict["struct_param_pred"],
                                struct_param_domain = model_dict["struct_param_domain"],
                                struct_param_uncertainty = model_dict["struct_param_uncertainty"],
                                settings_pred = model_dict["settings_pred"],
                                settings_domain = model_dict["settings_domain"],
                                settings_uncertainty = model_dict["settings_uncertainty"],
                                autoencoder = load_model_dict(model_dict["autoencoder"]),
                                loss_types = model_dict["loss_types"],
                                loss_core = model_dict["loss_core"],
                                loss_order = model_dict["loss_order"] if "loss_order" in model_dict else -1,
                                is_Lagrangian = model_dict["is_Lagrangian"] if "is_Lagrangian" in model_dict else False,
                                neglect_threshold = model_dict["neglect_threshold"] if "neglect_threshold" in model_dict else None,
                                reg_multiplier = model_dict["reg_multiplier"] if "reg_multiplier" in model_dict else None,
                               )
        model.pred_nets.load_model_dict(model_dict["pred_nets"])
        model.domain_net.load_model_dict(model_dict["domain_net"])
        model.domain_net_on = model_dict["domain_net_on"]
        if model_dict["struct_param_uncertainty"] is not None:
            model.uncertainty_nets.load_model_dict(model_dict["uncertainty_nets"])
    else:
        raise Exception("type {0} not recognized!".format(model_dict["type"]))
    return model


class Theory_Training(nn.Module):
    def __init__(
        self,
        num_theories,
        proposed_theory_models,
        input_size,
        struct_param_pred,
        struct_param_domain,
        struct_param_uncertainty = None,
        settings_pred = {},
        settings_domain = {},
        settings_uncertainty = {},
        autoencoder = None,
        loss_types = {},
        loss_core = "mse",
        loss_order = -1,
        loss_balance_model_influence = False,
        loss_precision_floor = PrecisionFloorLoss,
        is_Lagrangian = False,
        neglect_threshold = None,
        reg_multiplier = None,
        is_cuda = False,
        ):
        super(Theory_Training, self).__init__()
        self.num_theories = num_theories
        self.input_size = input_size
        self.loss_types = loss_types
        self.loss_core = loss_core
        self.loss_order = loss_order
        self.loss_balance_model_influence = loss_balance_model_influence
        self.loss_precision_floor = loss_precision_floor
        self.is_Lagrangian = is_Lagrangian
        self.neglect_threshold = neglect_threshold
        self.reg_multiplier = deepcopy(reg_multiplier)
        if self.reg_multiplier is not None:
            self.reg_model_idx = -1
            self.reg_domain_idx = -1
        self.is_cuda = is_cuda
        
        if proposed_theory_models is not None and len(proposed_theory_models) > 0:
            # If proposed_theory_models is not None, use the proposed models to construct pred_nets:
            fraction_best_list = []
            proposed_model_list = []
            for name, theory_info in proposed_theory_models.items():
                fraction_best_list.append(theory_info["fraction_best"])
                proposed_model_list.append(theory_info["theory_model"])
            fraction_best_list, proposed_model_list = sort_two_lists(fraction_best_list, proposed_model_list, reverse = True)
            proposed_model_list = proposed_model_list[:self.num_theories]
            for i in range(self.num_theories - len(proposed_model_list)):
                net = MLP(input_size = self.input_size, struct_param = struct_param_pred, settings = settings_pred, is_cuda = self.is_cuda)
                proposed_model_list.append(net)
            self.pred_nets = construct_net_ensemble_from_nets(proposed_model_list)
        else:
            self.pred_nets = Net_Ensemble(num_models = self.num_theories, input_size = self.input_size if not self.is_Lagrangian else int(self.input_size / 2), 
                                          struct_param = struct_param_pred, settings = settings_pred, is_cuda = self.is_cuda)
        
        self.domain_net = MLP(input_size = self.input_size, struct_param = struct_param_domain, settings = settings_domain, is_cuda = self.is_cuda)
        self.domain_net_on = False
        self.loss_fun_cumu = Loss_Fun_Cumu(core = loss_core, cumu_mode = ("generalized-mean", loss_order), neglect_threshold = neglect_threshold, 
                                           balance_model_influence = loss_balance_model_influence, loss_precision_floor = self.loss_precision_floor,
                                          )
        self.net_dict = {}
        self.net_dict["pred_nets"] = self.pred_nets
        self.net_dict["domain_net"] = self.domain_net
        if autoencoder is not None:
            self.autoencoder = autoencoder
            self.net_dict["autoencoder"] = self.autoencoder
        self.loss_fun_dict = {}
        self.loss_fun_dict["loss_fun_cumu"] = self.loss_fun_cumu
        if self.struct_param_uncertainty is not None:
            self.uncertainty_nets = Net_Ensemble(num_models = self.num_theories, input_size = self.input_size, 
                                                 struct_param = struct_param_uncertainty, settings = settings_uncertainty, is_cuda = self.is_cuda)
            self.loss_with_uncertainty = Loss_with_uncertainty(core = self.loss_core)
            self.net_dict["uncertainty_nets"] = self.uncertainty_nets
            self.loss_fun_dict["loss_with_uncertainty"] = self.loss_with_uncertainty
    
    @property
    def struct_param_pred(self):
        return self.pred_nets.struct_param
    
    @property
    def struct_param_domain(self):
        return self.domain_net.struct_param
    
    @property
    def struct_param_uncertainty(self):
        if hasattr(self, "uncertainty_nets"):
            return self.uncertainty_nets.struct_param
        else:
            return None
    
    @property
    def settings_pred(self):
        return self.pred_nets.settings
    
    @property
    def settings_domain(self):
        return self.domain_net.settings
    
    @property
    def settings_uncertainty(self):
        if hasattr(self, "uncertainty_nets"):
            return self.uncertainty_nets.settings
        else:
            return None

    @property
    def model_dict(self):
        model_dict = {"type": "Theory_Training"}
        model_dict["num_theories"] = self.num_theories
        model_dict["input_size"] = self.input_size
        model_dict["struct_param_pred"] = self.struct_param_pred
        model_dict["struct_param_domain"] = self.struct_param_domain
        model_dict["struct_param_uncertainty"] = self.struct_param_uncertainty
        model_dict["settings_pred"] = self.settings_pred
        model_dict["settings_domain"] = self.settings_domain
        model_dict["settings_uncertainty"] = self.settings_uncertainty
        model_dict["loss_types"] = self.loss_types
        model_dict["loss_core"] = self.loss_core
        model_dict["loss_order"] = self.loss_order
        model_dict["is_Lagrangian"] = self.is_Lagrangian
        model_dict["neglect_threshold"] = self.neglect_threshold
        model_dict["reg_multiplier"] = self.reg_multiplier
        model_dict["pred_nets"] = self.pred_nets.model_dict
        model_dict["domain_net"] = self.domain_net.model_dict
        model_dict["domain_net_on"] = self.domain_net_on
        if hasattr(self, "autoencoder"):
            model_dict["autoencoder"] = self.autoencoder.model_dict
        if self.struct_param_uncertainty is not None:
            model_dict["uncertainty_nets"] = self.uncertainty_nets.model_dict
        return model_dict

    def load_model_dict(self, model_dict):
        new_model = load_model_dict(model_dict)
        self.__dict__.update(new_model.__dict__)

    @property
    def DL(self):
        return self.pred_nets.DL + self.domain_net.DL


    def get_fraction_list(self, X, y = None, mode = "best"):
        if mode == "best":
            best_theory_idx = get_best_model_idx(self.net_dict, X, y, loss_fun_cumu = self.loss_fun_cumu, is_Lagrangian = self.is_Lagrangian)
            fraction_list = to_one_hot(best_theory_idx, self.num_theories).sum(0).float() / len(best_theory_idx)
            fraction_list = to_np_array(fraction_list.view(-1), full_reduce = False)
        elif mode == "domain":
            if hasattr(self, "autoencoder"):
                X = self.autoencoder.encode(X)
            valid_idx = self.domain_net(X).max(1)[1]
            idx = to_one_hot(valid_idx, self.num_theories)
            fraction_list = to_np_array(idx.sum(0), full_reduce = False) / float(len(X))
        else:
            raise
        return fraction_list
    
    
    def get_data_based_on_model(self, model_id, X, y, mode = "best"):
        if mode == "best":
            domain_pred_idx = get_best_model_idx(self.net_dict, X, y, loss_fun_cumu = self.loss_fun_cumu, is_Lagrangian = self.is_Lagrangian)
        elif mode == "domain":
            domain_pred_idx = self.domain_net(X).max(1)[1]
        else:
            raise
        idx = (domain_pred_idx == model_id).unsqueeze(1)
        X_chosen = torch.masked_select(X, idx).view(-1, *X.size()[1:])
        y_chosen = torch.masked_select(y, idx).view(-1, *y.size()[1:])
        return X_chosen, y_chosen


    def get_loss_core(self):
        return deepcopy(self.loss_fun_cumu.loss_fun.core)


    def set_loss_core(self, loss_core, loss_precision_floor = None):
        self.loss_core = loss_core
        self.loss_fun_cumu.loss_fun.core = loss_core
        if loss_precision_floor is not None:
            self.loss_precision_floor = loss_precision_floor
            self.loss_fun_cumu.loss_precision_floor = loss_precision_floor
            self.loss_fun_cumu.loss_fun.loss_precision_floor = loss_precision_floor


    def remove_theories_based_on_data(self, X, y, threshold, criteria = ["best", "domain"]):
        to_prune_onehot = np.zeros(self.num_theories).astype(bool)
        fraction_list_whole = []
        for criteria_ele in criteria:
            fraction_list = self.get_fraction_list(X, y, criteria_ele)
            fraction_list_whole.append(fraction_list)
            to_prune_onehot = to_prune_onehot | (fraction_list < threshold)
        to_prune = to_prune_onehot.nonzero()[0].tolist()
        print("fraction_best: {0}".format(self.get_fraction_list(X, y, "best")))
        print("fraction_domain: {0}".format(self.get_fraction_list(X, y, "domain")))
        if len(to_prune) > 0:
            if len(to_prune) == self.num_theories:
                print("Cannot remove all theories!")
                fraction_list_whole = np.array(fraction_list_whole)
                to_prune = list(range(self.num_theories))
                to_prune.remove(fraction_list_whole.mean(0).argmax())
            self.remove_theories(to_prune)
            print("theories {0} removed!".format(to_prune))
        return to_prune


    def remove_theories(self, theory_ids):
        self.pred_nets.remove_models(theory_ids)
        self.domain_net.prune_neurons(layer_id = -1, neuron_ids = theory_ids)
        assert self.pred_nets.num_models == self.domain_net.struct_param[-1][0],             "pred_nets has {0} models, while domain_net has {1} output neurons!".format(self.pred_nets.num_models, self.domain_net.struct_param[-1][0])
        self.num_theories = self.pred_nets.num_models
    
    
    def add_theories(
        self,
        X,
        y,
        validation_data = None,
        criteria = ("loss_with_domain", 0),
        loss_threshold = 1e-5,
        fraction_threshold = 0.05,
        isplot = True,
        **kwargs
        ):
        if validation_data is None:
            validation_data = (X, y)
        if hasattr(self, "autoencoder"):
            X_lat = self.autoencoder.encode(X).detach()
            autoencoder = self.autoencoder
        else:
            X_lat = X
            autoencoder = None
        X_test, y_test = validation_data
        criterion = nn.MSELoss(reduce = False)
        fraction_list = self.get_fraction_list(X_test, y_test)
        valid_idx = self.domain_net(X_lat).max(1)[1]
        idx = to_Boolean(to_one_hot(valid_idx, self.num_theories))
        if len(X.shape) == 4:
            idx = idx.unsqueeze(-1).unsqueeze(-1)
        is_add = False
        info = {}

        for i in range(self.num_theories):
            if fraction_list[i] > 0.3:
                X_chosen = torch.masked_select(X, idx[:, i:i+1]).view(-1, *X.size()[1:])
                X_chosen_lat = torch.masked_select(X_lat, idx[:, i:i+1].view(-1, 1)).view(-1, *X_lat.size()[1:])
                y_chosen = torch.masked_select(y, idx[:, i:i+1]).view(-1, *y.size()[1:])
                if len(X_chosen) == 0:
                    continue
                model = self.pred_nets.fetch_model(i)
                pred = forward(model, X_chosen_lat, autoencoder = autoencoder, is_Lagrangian = self.is_Lagrangian)
                loss = criterion(pred, y_chosen)
                if len(loss.shape) == 4:
                    loss = loss.mean(-1, keepdim = True).mean(-2, keepdim = True)
                loss = loss.sum(1, keepdim = True)
                idx_large = (loss > loss_threshold).detach()
                large_fraction = to_np_array(idx_large.long().sum().float() / float(X.size(0)))
                if large_fraction > fraction_threshold:
                    info[i] = {}
                    print("%" * 40 + "\nThe large loss points for theory_{0} constitute a fraction of {1:.4f} of total points. Perform tentative splitting of theory_{2}.\n".format(i, large_fraction, self.num_theories + 1) + "%" * 40 + "\n")

                    all_losses_dict = self.get_losses(X_test, y_test)
                    # Perform tentative adding and adaptation of new theory:
                    new_model = deepcopy(model)
                    X_large = torch.masked_select(X_chosen, idx_large).view(-1, *X_chosen.size()[1:])
                    X_large_lat = torch.masked_select(X_chosen_lat, idx_large.view(-1, 1)).view(-1, *X_chosen_lat.size()[1:])
                    y_large = torch.masked_select(y_chosen, idx_large).view(-1, *y_chosen.size()[1:])
                    train_simple(new_model, X_large_lat, y_large, loss_type = self.get_loss_core(), loss_precision_floor = self.loss_precision_floor, autoencoder = autoencoder, is_Lagrangian = self.is_Lagrangian)

                    U = deepcopy(self)
                    U.pred_nets.add_models(new_model)
                    U.domain_net.add_neurons(-1, 1, (("copy", i), None))
                    U.num_theories += 1
                    U.re_init_optimizers()
                    if U.domain_net_on and "DL" not in U.get_loss_core():
                        if isplot:
                            print("\nPerform joint training of all models:\n")
                        data_record = U.fit_model_schedule(X, y, 
                                                  validation_data = validation_data,
                                                  reg_dict = U.reg_dict,
                                                  reg_mode = U.reg_mode,
                                                  forward_steps = U.forward_steps,
                                                  domain_pred_mode = U.domain_pred_mode,
                                                  epochs = 10000, 
                                                  patience = 200,
                                                  isplot = isplot, 
                                                  prefix = "Tentative splitting of theory_{0}, train_model:".format(i), 
                                                  num_phases = 2,
                                                  add_theory_quota = 0,
                                                  **kwargs
                                                 )
                        if isplot:
                            print("%" * 40 + "\nRefit domain to best model:\n")
                        data_record_domain = U.fit_domain(X, y, 
                                                          validation_data = validation_data,
                                                          reg_dict = U.reg_dict,
                                                          reg_mode = U.reg_mode,
                                                          forward_steps = U.forward_steps,
                                                          domain_pred_mode = U.domain_pred_mode,
                                                          epochs = 10000, 
                                                          patience = 200, 
                                                          isplot = isplot,
                                                          prefix = "Tentative splitting of theory_{0}, train_domain:".format(i), 
                                                          **kwargs
                                                         )
                    else:
                        if isplot:
                            print("\nPerform joint training of all models and domains:\n")
                        data_record = U.fit_model_schedule(X, y, 
                                                  validation_data = validation_data, 
                                                  reg_dict = U.reg_dict,
                                                  reg_mode = U.reg_mode,
                                                  forward_steps = U.forward_steps,
                                                  domain_pred_mode = U.domain_pred_mode,
                                                  epochs = 10000, 
                                                  patience = 200, 
                                                  domain_fit_setting = U.domain_fit_setting,
                                                  isplot = isplot, 
                                                  prefix = "Tentative splitting of theory_{0}, train_model and domain:".format(i),
                                                  num_phases = 2,
                                                  add_theory_quota = 0,
                                                  **kwargs
                                                 )
                    print("%" * 40)
                    U.set_loss_core(self.loss_core, self.loss_precision_floor)
                    all_losses_dict_new = U.get_losses(X_test, y_test)

                    # Passes:
                    if all_losses_dict_new[criteria[0]] <= all_losses_dict[criteria[0]] + criteria[1]:
                        print("The new {0} of {1} is smaller than the previous {2} + {3}, accept.".format(criteria[0], all_losses_dict_new[criteria[0]],
                                                                                                         all_losses_dict[criteria[0]], criteria[1]))
                        info[i]["fraction_best"] = fraction_list[i]
                        info[i]["large_fraction"] = large_fraction
                        is_add = True
                        self.set_net("pred_nets", U.pred_nets)
                        self.set_net("domain_net", U.domain_net)
                        info[i]["data_record"] = data_record
                        if "data_record_domain" in locals():
                            info[i]["data_record_domain"] = data_record_domain
                        fraction_list = self.get_fraction_list(X_test, y_test)

                        # Reinitialize the optimizers if already have one:
                        self.re_init_optimizers()
                    else:
                        print("The new {0} of {1} is larger than the previous {2} + {3}, revert.".format(criteria[0], all_losses_dict_new[criteria[0]],
                                                                                                         all_losses_dict[criteria[0]], criteria[1]))

        if not is_add:
            print("Do not add new theories.")
        return is_add, info
    
    
    def re_init_optimizers(self):
        if hasattr(self, "optimizer"):
            lr = self.optimizer.param_groups[0]["lr"]
            trainable_parameters = [parameter for parameter in self.pred_nets.parameters() if parameter.requires_grad]
            self.optimizer = get_optimizer(optim_type = self.optim_type[0], lr = lr, parameters = trainable_parameters)
            if hasattr(self, "scheduler"):
                if self.scheduler_settings[0] == "ReduceLROnPlateau":
                    scheduler_patience = self.scheduler_settings[1]
                    scheduler_factor = self.scheduler_settings[2]
                    self.scheduler = ReduceLROnPlateau(self.optimizer, factor = scheduler_factor, patience = scheduler_patience, verbose = True)
                else:
                    raise
        if hasattr(self, "optimizer_domain"):
            lr_domain = self.optimizer_domain.param_groups[0]["lr"]
            optim_domain_type = self.optim_domain_type[0] if hasattr(self, "optim_domain_type") else self.optim_type[0]
            self.optimizer_domain = get_optimizer(optim_type = optim_domain_type, lr = lr_domain, parameters = self.domain_net.parameters())
            if hasattr(self, "scheduler_domain"):
                if self.scheduler_settings[0] == "ReduceLROnPlateau":
                    scheduler_patience = self.scheduler_settings[1]
                    scheduler_factor = self.scheduler_settings[2]
                    self.scheduler_domain = ReduceLROnPlateau(self.optimizer_domain, factor = scheduler_factor, patience = scheduler_patience, verbose = True)
                else:
                    raise
    
    
    def pred_nets_forward(self, input):
        output, _ = get_preds_valid(self.net_dict, input, forward_steps = 1, is_Lagrangian = self.is_Lagrangian)
        return output
    
    
    def domain_net_forward(self, input):
        if hasattr(self, "autoencoder"):
            input = self.autoencoder.encode(input)
        return self.domain_net(input)


    def forward_one_step(self, input, domain_pred_mode = "onehot"):
        preds, _ = get_preds_valid(self.net_dict, input, forward_steps = 1, is_Lagrangian = self.is_Lagrangian, disable_autoencoder = True)
        if domain_pred_mode == "onehot":
            valid = self.domain_net(input).max(1)[1]
            valid_onehot = to_one_hot(valid, self.num_theories)
            output = torch.masked_select(preds, to_Boolean(valid_onehot).unsqueeze(-1)).view(-1, preds.size(-1))
        elif domain_pred_mode == "prob":
            valid_prob = nn.Softmax(dim = 1)(self.domain_net(input))
            output = (preds * valid_prob.unsqueeze(-1)).sum(1)
        else:
            raise Exception("domain_pred_mode {0} not recognized!".format(domain_pred_mode))
        return output


    def forward(self, input, forward_steps = 1, output_format = "last", domain_pred_mode = "onehot"):
        # Encode:
        if hasattr(self, "autoencoder"):
            output_lat = self.autoencoder.encode(input)
        else:
            output_lat = input
        
        # Make prediction:
        if forward_steps == 1:
            output_pred = self.forward_one_step(output_lat, domain_pred_mode = domain_pred_mode)
            if hasattr(self, "autoencoder"):
                output_pred = self.autoencoder.decode(output_pred)
            return output_pred
        dim = self.pred_nets.struct_param[0][-1][0]
        pred_list = []
        for i in range(forward_steps):
            new_pred = self.forward_one_step(output_lat, domain_pred_mode = domain_pred_mode)
            if output_format == "all":
                pred_list.append(new_pred)
            if i != forward_steps - 1:
                output_lat = torch.cat([output_lat[:, dim:], new_pred], 1)
        if output_format == "last":
            output_lat = new_pred
        elif output_format == "all":
            output_lat = torch.cat(pred_list, 1)
        else:
            raise Exception("output_format {0} not recognized!".format(output_format))

        # Decode:
        if hasattr(self, "autoencoder"):
            return self.autoencoder.decode(output_lat)
        else:
            return output_lat


    def get_losses(self, X, y, mode = "all", forward_steps = 1, domain_pred_mode = "onehot", output_format = "value", is_mean = True, **kwargs):
        if mode == "all":
            mode = ["loss", "loss_with_domain", "loss_without_domain", "mse_with_domain", "mse_without_domain", 
                    "loss_best", "loss_indi_theory", "loss_recons", "reg", "reg_smooth", "loss_domain", "reg_domain", "fraction_list_best", "fraction_list_domain",
                    "DL_pred_nets", "DL_domain_net", "DL_data", "DL_data_absolute", "loss_precision_floor", "metrics_big_domain"]
        if not isinstance(mode, list):
            mode = [mode]
        all_losses_dict = {}
        if domain_pred_mode is None:
            domain_pred_mode_used = self.domain_pred_mode
        else:
            domain_pred_mode_used = domain_pred_mode
        if hasattr(self, "autoencoder"):
            X_lat = self.autoencoder.encode(X)
        else:
            X_lat = X
        
        for mode_ele in mode:
            if mode_ele == "loss":
                loss, loss_dict = get_loss(net_dict = self.net_dict, X = X, y = y, loss_types = self.loss_types, 
                                           forward_steps = forward_steps,
                                           domain_net = self.domain_net if self.domain_net_on else None,
                                           domain_pred_mode = domain_pred_mode_used,
                                           loss_fun_dict = self.loss_fun_dict,
                                           is_Lagrangian = self.is_Lagrangian,
                                           is_mean = is_mean,
                                          )
                if output_format == "value":
                    loss = to_np_array(loss)
                    loss_dict = {key: to_np_array(value) for key, value in loss_dict.items()}
                all_losses_dict[mode_ele] = loss
                all_losses_dict["loss_dict"] = loss_dict
            elif mode_ele == "loss_with_domain":
                loss, loss_dict = get_loss(net_dict = self.net_dict, X = X, y = y, loss_types = self.loss_types, 
                                           forward_steps = forward_steps,
                                           domain_net = self.domain_net,
                                           domain_pred_mode = domain_pred_mode_used,
                                           loss_fun_dict = self.loss_fun_dict,
                                           is_Lagrangian = self.is_Lagrangian,
                                           is_mean = is_mean,
                                          )
                if output_format == "value":
                    loss = to_np_array(loss)
                all_losses_dict[mode_ele] = loss
            elif mode_ele == "loss_without_domain":
                loss, loss_dict = get_loss(net_dict = self.net_dict, X = X, y = y, loss_types = self.loss_types,
                                           forward_steps = forward_steps,
                                           domain_net = None,
                                           domain_pred_mode = domain_pred_mode_used,
                                           loss_fun_dict = self.loss_fun_dict,
                                           is_Lagrangian = self.is_Lagrangian,
                                           is_mean = is_mean,
                                          )
                if output_format == "value":
                    loss = to_np_array(loss)
                all_losses_dict[mode_ele] = loss
            elif mode_ele == "mse_with_domain":
                loss, loss_dict = get_loss(net_dict = self.net_dict, X = X, y = y, loss_types = {"pred-based_mean": {"amp": 1.}},
                                           forward_steps = forward_steps,
                                           domain_net = self.domain_net,
                                           domain_pred_mode = domain_pred_mode_used,
                                           loss_fun_dict = {"loss_fun_cumu": Loss_Fun_Cumu(core = "mse", cumu_mode = "mean", balance_model_influence = False, epsilon = 0)},
                                           is_Lagrangian = self.is_Lagrangian,
                                           is_mean = is_mean,
                                          )
                if output_format == "value":
                    loss = to_np_array(loss)
                all_losses_dict[mode_ele] = loss
            elif mode_ele == "mse_without_domain":
                loss, loss_dict = get_loss(net_dict = self.net_dict, X = X, y = y, loss_types = {"pred-based_mean": {"amp": 1.}},
                                           forward_steps = forward_steps,
                                           domain_net = None,
                                           domain_pred_mode = domain_pred_mode_used,
                                           loss_fun_dict = {"loss_fun_cumu": Loss_Fun_Cumu(core = "mse", cumu_mode = "mean")},
                                           is_Lagrangian = self.is_Lagrangian,
                                           is_mean = is_mean,
                                          )
                if output_format == "value":
                    loss = to_np_array(loss)
                all_losses_dict[mode_ele] = loss
            elif mode_ele == "loss_best":
                loss_best, _ = get_loss(net_dict = self.net_dict, X = X, y = y, loss_types = {"pred-based_min": {"amp": 1.}},
                                        forward_steps = forward_steps, 
                                        loss_fun_dict = self.loss_fun_dict,
                                        is_Lagrangian = self.is_Lagrangian,
                                        is_mean = is_mean,
                                       )
                if output_format == "value":
                    loss_best = to_np_array(loss_best)
                all_losses_dict["loss_best"] = loss_best
            elif mode_ele == "loss_indi_theory":
                # Get individual theory loss:
                loss_indi_theory = {}
                preds, _ = get_preds_valid(self.net_dict, X_lat, forward_steps = forward_steps, is_Lagrangian = self.is_Lagrangian)
                for i in range(self.num_theories):
                    valid_onehot = to_one_hot(nn.Softmax(dim = 1)(self.domain_net(X_lat)).max(1)[1], self.num_theories).float()
                    if self.is_cuda:
                        valid_onehot = valid_onehot.cuda()
                    loss_indi_theory_i = self.loss_fun_cumu(preds[:, i:i+1], y, sample_weights = valid_onehot[:,i:i+1], is_mean = is_mean)
                    if output_format == "value":
                        loss_indi_theory_i = to_np_array(loss_indi_theory_i)
                    loss_indi_theory[i] = loss_indi_theory_i
                all_losses_dict["loss_indi_theory"] = loss_indi_theory
            elif mode_ele == "loss_recons":
                if hasattr(self, "autoencoder"):
                    loss_recons = self.autoencoder.get_loss(X, X, nn.MSELoss()) * self.optim_autoencoder_type[2]
                    if output_format == "value":
                        loss_recons = to_np_array(loss_recons)
                    all_losses_dict["loss_recons"] = loss_recons
            elif mode_ele == "reg":
                reg, reg_dict = get_reg(net_dict = self.net_dict, reg_dict = self.reg_dict if hasattr(self, "reg_dict") else {}, 
                                        mode = self.reg_mode if hasattr(self, "reg_dict") else None, is_cuda = self.is_cuda)
                if output_format == "value":
                    reg = to_np_array(reg)
                    reg_dict = {key: to_np_array(value) for key, value in reg_dict.items()}
                if hasattr(self, "reg_multiplier_model"):
                    reg = reg * self.reg_multiplier_model
                all_losses_dict["reg"] = reg
                all_losses_dict["reg_dict"] = reg_dict
            elif mode_ele == "reg_smooth":
                reg_smooth_in = kwargs["reg_smooth"] if "reg_smooth" in kwargs else None
                if reg_smooth_in is not None:
                    reg_smooth = reg_smooth_in
                else:
                    reg_smooth = (0.05, 2, 10, 1e-6, 1)
                num_samples = reg_smooth[4]
                input_noise_scale = reg_smooth[0]
                diff_list = []
                for _ in range(num_samples):
                    input_perturb = Variable(torch.randn(*X.size()) * input_noise_scale)
                    if self.is_cuda:
                        input_perturb = input_perturb.cuda()
                    diff = self.pred_nets_forward(X + input_perturb) - self.pred_nets_forward(X)
                    diff_list.append(diff)
                diff_list = torch.stack(diff_list, 2)
                smooth_norms = get_group_norm(diff_list, reg_smooth[1], reg_smooth[2])
                reg_smooth_amp = reg_smooth[3] if reg_smooth_in is not None else 0
                reg_smooth_value = smooth_norms.mean() * reg_smooth[3]
                if hasattr(self, "reg_multiplier_model"):
                    reg_smooth_value = reg_smooth_value * self.reg_multiplier_model
                if output_format == "value":
                    smooth_norms = to_np_array(smooth_norms)
                    reg_smooth_value = to_np_array(reg_smooth_value)
                all_losses_dict["smooth_norms"] = smooth_norms
                all_losses_dict["reg_smooth_value"] = reg_smooth_value        
#             elif mode_ele == "reg_grad":
#                 reg_grad = kwargs["reg_grad"] if "reg_grad" in kwargs else None
#                 if reg_grad is not None:
#                     X.requires_grad = True
#                     loss_indi = self.loss_fun_cumu(self.pred_nets(X), y, cumu_mode = "original", is_mean = False).mean(0)
#                     grad_norms = torch.cat([get_group_norm(grad(loss_indi[i], X, create_graph = True)[0], reg_grad[0], reg_grad[1]) for i in range(self.num_theories)])
#                     X.requires_grad = False
#                     reg_grad_value = grad_norms.mean() * reg_grad[2]
#                     if hasattr(self, "reg_multiplier_model"):
#                         reg_grad_value = reg_grad_value * self.reg_multiplier_model
#                     if output_format == "value":
#                         grad_norms = to_np_array(grad_norms)
#                         reg_grad_value = to_np_array(reg_grad_value)
#                     all_losses_dict["grad_norms"] = grad_norms
#                     all_losses_dict["reg_grad_value"] = reg_grad_value
            elif mode_ele == "loss_domain":
                best_theory_idx_test = get_best_model_idx(self.net_dict, X, y, loss_fun_cumu = self.loss_fun_cumu, forward_steps = forward_steps, is_Lagrangian = self.is_Lagrangian)
                loss_domain = nn.CrossEntropyLoss(size_average = is_mean)(self.domain_net(X_lat), best_theory_idx_test)
                if output_format == "value":
                    loss_domain = to_np_array(loss_domain)
                all_losses_dict["loss_domain"] = loss_domain
            elif mode_ele == "reg_domain":
                reg_domain, reg_domain_dict = get_reg(net_dict = self.net_dict, reg_dict = self.reg_domain_dict if hasattr(self, "reg_domain_dict") else {}, 
                                                      mode = self.reg_domain_mode if hasattr(self, "reg_domain_dict") else None, is_cuda = self.is_cuda)
                if output_format == "value":
                    reg_domain = to_np_array(reg_domain)
                    reg_domain_dict = {key: to_np_array(value) for key, value in reg_domain_dict.items()}
                if hasattr(self, "reg_multiplier_domain"):
                    reg_domain = reg_domain * self.reg_multiplier_domain
                all_losses_dict["reg_domain"] = reg_domain
                all_losses_dict["reg_domain_dict"] = reg_domain_dict
            elif mode_ele == "fraction_list_best":
                all_losses_dict["fraction_list_best"] = self.get_fraction_list(X, y, mode = "best")
            elif mode_ele == "fraction_list_domain":
                all_losses_dict["fraction_list_domain"] = self.get_fraction_list(X, mode = "domain")
            elif mode_ele == "DL_pred_nets":
                all_losses_dict["DL_pred_nets"] = self.pred_nets.DL
            elif mode_ele == "DL_domain_net":
                all_losses_dict["DL_domain_net"] = self.domain_net.DL
            elif mode_ele == "DL_data":
                pred = self(X)
                DL_mode = kwargs["DL_mode"] if "DL_mode" in kwargs else "DLs"
                DL_criterion = Loss_Fun(core = DL_mode, loss_precision_floor = self.loss_precision_floor, DL_sum = True)
                DL_data = DL_criterion(pred, y)
                all_losses_dict["DL_data"] = to_np_array(DL_data)
            elif mode_ele == "DL_data_absolute":
                pred = self(X)
                DL_mode = kwargs["DL_mode"] if "DL_mode" in kwargs else "DLs"
                DL_criterion = Loss_Fun(core = DL_mode, loss_precision_floor = PrecisionFloorLoss, DL_sum = True)
                DL_data = DL_criterion(pred, y)
                all_losses_dict["DL_data_absolute"] = to_np_array(DL_data)
            elif mode_ele == "loss_precision_floor":
                all_losses_dict["loss_precision_floor"] = deepcopy(self.loss_precision_floor)
            elif mode_ele == "metrics_big_domain":
                if "big_domain_ids" in kwargs and "true_domain_test" in kwargs and kwargs["big_domain_ids"] is not None and kwargs["true_domain_test"] is not None:
                    predicted_domain = self.domain_net(X_lat).max(1)[1]
                    (union, predicted_big_domains, true_big_domains, intersection), _ = count_metrics_pytorch(predicted_domain, true_domain = kwargs["true_domain_test"], big_domain_ids = kwargs["big_domain_ids"], verbose = False)

                    true_domain_np = to_np_array(kwargs["true_domain_test"]).flatten()
                    idx_in_big = torch.LongTensor(np.array([i for i in range(len(true_domain_np)) if int(true_domain_np[i]) in kwargs["big_domain_ids"]]))
                    predicted_domain_in_big = predicted_domain[idx_in_big]
                    true_domain_in_big = kwargs["true_domain_test"][idx_in_big]
                    (union_in_big, predicted_big_domains_in_big, true_big_domains_in_big, intersection_in_big), _ = count_metrics_pytorch(predicted_domain_in_big, true_domain = true_domain_in_big, big_domain_ids = kwargs["big_domain_ids"], verbose = True if "verbose" in kwargs and kwargs["verbose"] is True else False)
                    
                    assert union_in_big == true_big_domains_in_big, "For in_big, the three quantities must be equal!"
                    if true_big_domains is not None:
                        assert union_in_big == true_big_domains, "For in_big, the three quantities must be equal!"

                    metrics = {"union": union,
                               "predicted_big_domains": predicted_big_domains,
                               "true_big_domains": union_in_big,
                               "intersection": intersection,
                               "intersection_in_big": intersection_in_big,
                              }
                    is_big_domain = to_Boolean(torch.zeros(X.size(0), 1))
                    if X.is_cuda:
                        is_big_domain = is_big_domain.cuda()
                    true_domain_test = kwargs["true_domain_test"]
                    for big_domain_id in kwargs["big_domain_ids"]:
                        is_big_domain = is_big_domain | to_Boolean(true_domain_test == big_domain_id)
                    if len(X.shape) == 4:
                        is_big_domain = is_big_domain.unsqueeze(-1).unsqueeze(-1)
                    is_big_domain = to_Variable(is_big_domain, is_cuda = self.is_cuda)
                    X_big_domain = torch.masked_select(X, is_big_domain).view(-1, *X.size()[1:])
                    y_big_domain = torch.masked_select(y, is_big_domain).view(-1, *y.size()[1:])
                    X_lat_big_domain = torch.masked_select(X_lat, is_big_domain.view(-1, 1)).view(-1, *X_lat.size()[1:])
                    
                    # Loss:
                    loss_big_domain, loss_dict_big_domain = get_loss(net_dict = self.net_dict, X = X_big_domain, y = y_big_domain, loss_types = self.loss_types, 
                                                                       forward_steps = forward_steps,
                                                                       domain_net = self.domain_net if self.domain_net_on else None,
                                                                       domain_pred_mode = domain_pred_mode_used,
                                                                       loss_fun_dict = self.loss_fun_dict,
                                                                       is_Lagrangian = self.is_Lagrangian,
                                                                       is_mean = is_mean,
                                                                      )
                    if output_format == "value":
                        loss_big_domain = to_np_array(loss_big_domain)
                        loss_dict_big_domain = {key: to_np_array(value) for key, value in loss_dict_big_domain.items()}
                    metrics["loss_big_domain"] = loss_big_domain
                    metrics["loss_dict_big_domain"] = loss_dict_big_domain
                    
                    # loss_with_domain:
                    loss_big_domain, _ = get_loss(net_dict = self.net_dict, X = X_big_domain, y = y_big_domain, loss_types = {"pred-based_mean": {"amp": 1.}},
                                                   forward_steps = forward_steps,
                                                   domain_net = self.domain_net,
                                                   domain_pred_mode = domain_pred_mode_used,
                                                   loss_fun_dict = {"loss_fun_cumu": Loss_Fun_Cumu(core = "mse", cumu_mode = "mean", balance_model_influence = False, epsilon = 0)},
                                                   is_Lagrangian = self.is_Lagrangian,
                                                   is_mean = is_mean,
                                                  )
                    if output_format == "value":
                        loss_big_domain = to_np_array(loss_big_domain)
                    metrics["loss_with_domain_big_domain"] = loss_big_domain
                    
                    # mse_with_domain:
                    loss_big_domain, _ = get_loss(net_dict = self.net_dict, X = X_big_domain, y = y_big_domain, loss_types = {"pred-based_mean": {"amp": 1.}},
                                                   forward_steps = forward_steps,
                                                   domain_net = self.domain_net,
                                                   domain_pred_mode = domain_pred_mode_used,
                                                   loss_fun_dict = {"loss_fun_cumu": Loss_Fun_Cumu(core = "mse", cumu_mode = "mean", balance_model_influence = False, epsilon = 0)},
                                                   is_Lagrangian = self.is_Lagrangian,
                                                   is_mean = is_mean,
                                                  )
                    if output_format == "value":
                        loss_big_domain = to_np_array(loss_big_domain)
                    metrics["mse_with_domain_big_domain"] = loss_big_domain
                    
                    # mse_indi_theory_big_domain:
                    mse_indi_theory_big_domain = {}
                    preds_big_domain, _ = get_preds_valid(self.net_dict, X_lat_big_domain, forward_steps = forward_steps, is_Lagrangian = self.is_Lagrangian)
                    for i in range(self.num_theories):
                        valid_onehot_big_domain = to_one_hot(nn.Softmax(dim = 1)(self.domain_net(X_lat_big_domain)).max(1)[1], self.num_theories).float()
                        if self.is_cuda:
                            valid_onehot_big_domain = valid_onehot_big_domain.cuda()
                        loss_fun_cumu = Loss_Fun_Cumu(core = "mse", cumu_mode = "mean", balance_model_influence = False, epsilon = 0)
                        mse_indi_theory_i_big_domain = loss_fun_cumu(preds_big_domain[:, i:i+1], y_big_domain, sample_weights = valid_onehot_big_domain[:,i:i+1], is_mean = is_mean)
                        if output_format == "value":
                            mse_indi_theory_i_big_domain = to_np_array(mse_indi_theory_i_big_domain)
                        mse_indi_theory_big_domain[i] = mse_indi_theory_i_big_domain
                    all_losses_dict["mse_indi_theory_big_domain"] = mse_indi_theory_big_domain
                    all_losses_dict["metrics_big_domain"] = metrics
            else:
                raise Exception("mode {0} not recognized!".format(mode_ele))
        return all_losses_dict


    def get_adaptive_precision_floor(self, X, y, range = (1e-6, 1e-1), nonzero_ratio = 0.5):
        num_zero_loss_list = []
        U = deepcopy(self)
        for dl in np.logspace(np.log10(range[1]), np.log10(range[0]), 200):
            U.set_loss_core("DL", dl)
            num_zero_loss = (U.get_losses(X, y, mode = "loss_with_domain", is_mean = False)["loss_with_domain"] < 2 ** (-32)).sum()
            if num_zero_loss < len(X) * nonzero_ratio:
                break
        return dl


    def fit_model_schedule(
        self,
        X_train,
        y_train,
        validation_data = None,
        optim_type = ("adam", 5e-3),
        reg_dict = {},
        reg_mode = "L1",
        domain_fit_setting = None,
        forward_steps = 1,
        domain_pred_mode = "onehot",
        grad_clipping = None,
        scheduler_settings = ("ReduceLROnPlateau", 30, 0.1),
        loss_order_decay = None,
        gradient_noise = None,
        epochs = None,
        batch_size = None,
        inspect_interval = 1000,
        patience = None,
        change_interval = 1,
        record_interval = None,
        isplot = True,
        filename = None,
        view_init = (10, 190),
        raise_nan = True,
        add_theory_quota = 1,
        add_theory_criteria = ("mse_with_domain", 0),
        add_theory_loss_threshold = None,
        theory_remove_fraction_threshold = None,
        loss_floor = 1e-12,
        prefix = None,
        num_phases = 3,
        **kwargs
        ):
        """Implements steps 2 to 6 in Alg. 2 in Wu and Tegmark (2019)"""
        if self.get_loss_core() == "mse":
            return self.fit_model(
                                X_train = X_train,
                                y_train = y_train,
                                validation_data = validation_data,
                                optim_type = optim_type,
                                reg_dict = reg_dict,
                                reg_mode = reg_mode,
                                domain_fit_setting = domain_fit_setting,
                                forward_steps = forward_steps,
                                domain_pred_mode = domain_pred_mode,
                                grad_clipping = grad_clipping,
                                scheduler_settings = scheduler_settings,
                                loss_order_decay = loss_order_decay,
                                gradient_noise = gradient_noise,
                                epochs = epochs,
                                batch_size = batch_size,
                                inspect_interval = inspect_interval,
                                patience = patience,
                                change_interval = change_interval,
                                record_interval = record_interval,
                                isplot = isplot,
                                filename = filename,
                                view_init = view_init,
                                raise_nan = raise_nan,
                                add_theory_quota = add_theory_quota,
                                add_theory_criteria = add_theory_criteria,
                                add_theory_loss_threshold = add_theory_loss_threshold,
                                loss_floor = loss_floor,
                                prefix = prefix,
                                **kwargs
                                )
        elif "DL" in self.get_loss_core():
            if validation_data is None:
                validation_data = (X_train, y_train)
            X_test, y_test = validation_data
            data_record_whole = []
            for ii in range(num_phases):
                U = deepcopy(self)
                if "fix_adaptive_precision_floor" in kwargs and kwargs["fix_adaptive_precision_floor"] is True:
                    dl = U.loss_fun_cumu.loss_fun.loss_precision_floor
                else:
                    dl = U.get_adaptive_precision_floor(X_test, y_test, range = (1e-6, 1e-1), nonzero_ratio = 0.5)
                    U.set_loss_core(U.loss_core, dl)
                print("## Phase {0}:\tcurrent DL precision_floor: {1:.9f}".format(ii, dl))
                # Tentative fitting:
                loss_dict = U.get_losses(X_test, y_test, forward_steps = forward_steps, **kwargs)
                data_record = U.fit_model(
                                X_train = X_train,
                                y_train = y_train,
                                validation_data = validation_data,
                                optim_type = optim_type,
                                reg_dict = reg_dict,
                                reg_mode = reg_mode,
                                domain_fit_setting = domain_fit_setting,
                                forward_steps = forward_steps,
                                domain_pred_mode = domain_pred_mode,
                                grad_clipping = grad_clipping,
                                scheduler_settings = scheduler_settings,
                                loss_order_decay = loss_order_decay,
                                gradient_noise = gradient_noise,
                                epochs = epochs,
                                batch_size = batch_size,
                                inspect_interval = inspect_interval,
                                patience = patience,
                                change_interval = change_interval,
                                record_interval = record_interval,
                                isplot = isplot,
                                filename = filename,
                                view_init = view_init,
                                raise_nan = raise_nan,
                                add_theory_quota = add_theory_quota,
                                add_theory_criteria = add_theory_criteria,
                                add_theory_loss_threshold = add_theory_loss_threshold,
                                loss_floor = loss_floor,
                                prefix = prefix,
                                **kwargs
                                )
                data_record["loss_precision_floor"] = dl
                loss_dict_new = U.get_losses(X_test, y_test, forward_steps = forward_steps, **kwargs)
                if theory_remove_fraction_threshold is not None:
                    data_record["removed_theories"] = U.remove_theories_based_on_data(X_test, y_test, threshold = theory_remove_fraction_threshold)
                if loss_dict_new["loss_with_domain"] >= loss_dict["loss_with_domain"]:
                    print("The loss_with_domain {0} is larger than previous {1}. Revert and abort fitting.".format(loss_dict_new["loss_with_domain"], loss_dict["loss_with_domain"]))
                    break
                else:
                    print("The loss_with_domain {0} decrease from previous {1}. Accept and continue training.".format(loss_dict_new["loss_with_domain"], loss_dict["loss_with_domain"]))
                    data_record_whole.append(data_record)
                    self.__dict__.update(U.__dict__)
                if loss_dict_new["mse_with_domain"] < loss_floor:
                    print("mse_with_domain = {0} is below the floor level {1}, stop.".format(loss_dict_new["mse_with_domain"], loss_floor))
                    break
            return data_record_whole
        else:
            raise
    

    def fit_model(
        self,
        X_train,
        y_train,
        validation_data = None,
        optim_type = ("adam", 5e-3),
        reg_dict = {},
        reg_mode = "L1",
        domain_fit_setting = None,
        forward_steps = 1,
        domain_pred_mode = "onehot",
        grad_clipping = None,
        scheduler_settings = ("ReduceLROnPlateau", 30, 0.1),
        loss_order_decay = None,
        gradient_noise = None,
        epochs = None,
        inspect_interval = 1000,
        patience = None,
        change_interval = 1,
        record_interval = None,
        isplot = True,
        filename = None,
        view_init = (10, 190),
        raise_nan = True,
        add_theory_quota = 1,
        add_theory_criteria = ("mse_with_domain", 0),
        add_theory_loss_threshold = None,
        loss_floor = 1e-12,
        prefix = None,
        **kwargs
        ):
        """Implements the IterativeTrain algorithm in Alg. 2 in Wu and Tegmark (2019)"""
        X_test, y_test = validation_data
        self.optim_type = optim_type
        self.reg_dict = reg_dict
        self.reg_mode = reg_mode
        self.forward_steps = forward_steps
        self.domain_pred_mode = domain_pred_mode
        self.grad_clipping = grad_clipping
        self.scheduler_settings = scheduler_settings
        self.loss_order_decay = loss_order_decay
        self.gradient_noise = gradient_noise
        self.domain_fit_setting = domain_fit_setting
        print("scheduler_settings:", self.scheduler_settings)
        print("grad_clipping:", self.grad_clipping)
        print("forward_steps:", forward_steps)
        print("loss_order_decay:", loss_order_decay)
        print("gradient_noise:", gradient_noise)
        print("domain_fit_setting:", domain_fit_setting)
        print("loss_types: ", self.loss_types)
        self.optim_autoencoder_type = kwargs["optim_autoencoder_type"] if "optim_autoencoder_type" in kwargs else None
        print("optim_autoencoder_type: ", self.optim_autoencoder_type)
        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else None
        print("batch_size:", batch_size)
        record_mode = kwargs["record_mode"] if "record_mode" in kwargs else 1
        print("record_mode: {0}".format(record_mode))
        reg_smooth = kwargs["reg_smooth"] if "reg_smooth" in kwargs else None
        print("reg_smooth: {0}".format(reg_smooth))
#         reg_grad = kwargs["reg_grad"] if "reg_grad" in kwargs else None
#         print("reg_grad: {0}".format(reg_grad))
        if "big_domain_ids" in kwargs:
            print("big_domain_ids: ", kwargs["big_domain_ids"])
        add_theory_limit = kwargs["add_theory_limit"] if "add_theory_limit" in kwargs else None
        print("add_theory_limit: {0}".format(add_theory_limit))
        print()


        if validation_data is None:
            validation_data = (X_train, y_train)

        # Setting up optimizer:
        if not hasattr(self, "optimizer"):
            trainable_parameters = [parameter for parameter in self.pred_nets.parameters() if parameter.requires_grad]
            self.optimizer = get_optimizer(optim_type = self.optim_type[0], lr = self.optim_type[1], parameters = trainable_parameters)
        else:
            new_lr = np.sqrt(self.optimizer.param_groups[0]["lr"] * self.optim_type[1])
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
        if domain_fit_setting is not None:
            if not hasattr(self, "optimizer_domain"):
                self.optimizer_domain = get_optimizer(optim_type = domain_fit_setting["optim_domain_type"][0], lr = domain_fit_setting["optim_domain_type"][1], parameters = self.domain_net.parameters())
        if hasattr(self, "autoencoder"):
            self.optimizer_autoencoder = get_optimizer(optim_type = self.optim_autoencoder_type[0], lr = self.optim_autoencoder_type[1], parameters = self.autoencoder.parameters())
                
        if batch_size is None:
            if record_interval is None:
                record_interval = 10
            if self.optim_type[0] == "LBFGS":
                num_iter = 5000
            else:
                num_iter = 15000
        else:
            if record_interval is None:
                record_interval = 1
            if self.optim_type[0] == "LBFGS":
                num_iter = 250
            else:
                num_iter = 1000
        assert inspect_interval % record_interval == 0
        if epochs is not None:
            num_iter = epochs
        if batch_size is not None:
            dataset_train = data_utils.TensorDataset(X_train.data, y_train.data)
            self.train_loader_model = data_utils.DataLoader(dataset_train, batch_size = batch_size, shuffle = True)

        # Setting up lr scheduler:
        if self.scheduler_settings is not None:
            if self.scheduler_settings[0] == "LambdaLR":
                function_type = scheduler_settings[1]
                decay_scale = scheduler_settings[2]
                scheduler_continue_decay = scheduler_settings[3]
                if function_type == "exp":
                    lambda_pred = lambda epoch: (1 - 1 / float(num_iter / change_interval / decay_scale)) ** epoch
                elif function_type == "poly":
                    lambda_pred = lambda epoch: 1 / (1 + 0.01 * epoch * change_interval * decay_scale)
                else:
                    raise
                if scheduler_continue_decay:
                    if not hasattr(self, "scheduler"):
                        self.scheduler = LambdaLR(self.optimizer, lr_lambda = lambda_pred)
                else:
                    self.scheduler = LambdaLR(self.optimizer, lr_lambda = lambda_pred)

                if domain_fit_setting is not None:
                    if scheduler_continue_decay:
                        if not hasattr(self, "scheduler_domain"):
                            self.scheduler_domain = LambdaLR(self.optimizer_domain, lr_lambda = lambda_pred) 
                    else:
                        self.scheduler_domain = LambdaLR(self.optimizer_domain, lr_lambda = lambda_pred)
            elif self.scheduler_settings[0] == "ReduceLROnPlateau":
                scheduler_patience = self.scheduler_settings[1]
                scheduler_factor = self.scheduler_settings[2]
                self.scheduler = ReduceLROnPlateau(self.optimizer, factor = scheduler_factor, patience = scheduler_patience, verbose = True)
                if domain_fit_setting is not None:
                    self.scheduler_domain = ReduceLROnPlateau(self.optimizer_domain, factor = scheduler_factor, patience = scheduler_patience, verbose = True)
            else:
                raise

        # Setting up gradient noise:
        if self.gradient_noise is not None:
            self.scale_gen = Gradient_Noise_Scale_Gen(gamma = self.gradient_noise["gamma"],
                                                      eta = self.gradient_noise["eta"],
                                                      gradient_noise_interval_batch = self.gradient_noise["gradient_noise_interval_batch"],
                                                      batch_size = len(y_train),
                                                     )
            gradient_noise_scale = self.scale_gen.generate_scale(epochs = num_iter, num_examples = len(y_train), verbose = True)
        else:
            self.scale_gen = None

        # Setting up loss_order_decay:
        if loss_order_decay is not None:
            self.loss_decay_scheduler = Loss_Decay_Scheduler(self.loss_types, lambda_loss_decay = loss_order_decay)
        else:
            self.loss_decay_scheduler = None

        figsize = (10, 8)
        self.model_dict_last = {}
        self.model_dict_second_last = {}
        self.data_record = {}
        if patience is not None:
            self.early_stopping = Early_Stopping(patience = patience, epsilon = 1e-10)
        if domain_fit_setting is not None:
            if patience is not None:
                self.early_stopping_domain = Early_Stopping(patience = patience, epsilon = 1e-10)
            to_stop_domain = False
        to_stop = False

        def show(k):
            all_losses_dict = self.get_losses(X_test, y_test, forward_steps = forward_steps, **kwargs)
            if prefix is not None:
                print(prefix)
            print("iter {0}  lr = {1:.9f}".format(k, self.optimizer.param_groups[0]["lr"]))
            if domain_fit_setting is not None:
                print("lr_domain:\t{0:.9f}\nloss_domain:\t{1:.9f}\nreg_domain:\t{2:.9f}".format(self.optimizer_domain.param_groups[0]["lr"], all_losses_dict["loss_domain"], all_losses_dict["reg_domain"]))
            if hasattr(self, "autoencoder"):
                print("loss_recons:\t{0:.9f}".format(all_losses_dict["loss_recons"]))
            print("loss_best:\t{0:.9f}\nloss_model:\t{1:.9f}\nloss_with_domain:\t{2:.9f}".format(all_losses_dict["loss_best"], all_losses_dict["loss"], all_losses_dict["loss_with_domain"]))
            for loss_mode in all_losses_dict["loss_dict"]:
                print("loss_{0}:\t{1:.9f}".format(loss_mode, all_losses_dict["loss_dict"][loss_mode]))
            print("mse_with_domain:\t{0:.9f}\nmse_without_domain:\t{1:.9f}".format(all_losses_dict["mse_with_domain"], all_losses_dict["mse_without_domain"]))
            for i in range(self.num_theories):
                print("{0}_theory_{1}:\t{2:.9f}\tfraction best: {3:.5f}\t domain: {4:.5f}".format(self.loss_fun_cumu.loss_fun.core, i, all_losses_dict["loss_indi_theory"][i], all_losses_dict["fraction_list_best"][i], all_losses_dict["fraction_list_domain"][i]))
            print("reg:\t{0:.9f}".format(all_losses_dict["reg"]))
            print("reg_smooth_value: {0:.9f}\tsmooth_norms: {1}".format(all_losses_dict["reg_smooth_value"], all_losses_dict["smooth_norms"]))
#             if reg_grad is not None:
#                 print("reg_grad_value: {0:.9f}\tgrad_norms: {1}".format(all_losses_dict["reg_grad_value"], all_losses_dict["grad_norms"]))
            if self.reg_multiplier is not None:
                print("reg_multiplier_model iter {0}:\t{1:.9f}\nreg_multiplier_domain iter {2}\t{3:.9f}".format(self.reg_model_idx, self.reg_multiplier_model, self.reg_domain_idx, self.reg_multiplier_domain))
            if loss_order_decay is not None:
                print("loss_order_current:\t{0}".format(replaced_loss_order))
            if self.scale_gen is not None:
                print("current gradient noise scale: {0:.9f}".format(current_gradient_noise_scale))
            if "big_domain_ids" in kwargs and kwargs["big_domain_ids"] is not None:
                if "metrics_big_domain" in all_losses_dict:
                    union = all_losses_dict["metrics_big_domain"]["union"]
                    predicted_big_domains = all_losses_dict["metrics_big_domain"]["predicted_big_domains"]
                    true_big_domains = all_losses_dict["metrics_big_domain"]["true_big_domains"]
                    intersection = all_losses_dict["metrics_big_domain"]["intersection"]
                    intersection_in_big = all_losses_dict["metrics_big_domain"]["intersection_in_big"]
                    if union is not None:
                        precision = intersection / float(predicted_big_domains)
                        recall = intersection / float(true_big_domains)
                        F1 = 2 / (1 / precision  + 1 / recall)
                        IoU = intersection / float(union)
                        print("union: {0}\tpredicted_big_domains: {1}\ttrue_big_domains: {2}\tintersection_in_big: {3}\tintersection: {4}".format(union, predicted_big_domains, true_big_domains, intersection_in_big, intersection))
                        print("Precision: {0:.4f}\tRecall: {1:.4f}\tF1: {2:.4f}\tIoU: {3:.4f}".format(precision, recall, F1, IoU))
                    print("loss_big_domain: {0:.9f}".format(all_losses_dict["metrics_big_domain"]["loss_big_domain"]))
                    print("loss_with_domain_big_domain: {0:.9f}".format(all_losses_dict["metrics_big_domain"]["loss_with_domain_big_domain"]))
                    print("mse_with_domain_big_domain: {0:.9f}".format(all_losses_dict["metrics_big_domain"]["mse_with_domain_big_domain"]))
            print()
            try:
                sys.stdout.flush()
            except:
                pass
            # Plotting:
            if isplot or filename is not None:
                self.plot(X_test, y_test, forward_steps = forward_steps, view_init = view_init, figsize = figsize, is_show = isplot, filename = filename + "_{0}".format(k) if filename is not None else None,
                          true_domain = kwargs["true_domain_test"] if "true_domain_test" in kwargs else None, num_output_dims = kwargs["num_output_dims"],
                          show_3D_plot = kwargs["show_3D_plot"] if "show_3D_plot" in kwargs else False, 
                          show_vs = kwargs["show_vs"] if "show_vs" in kwargs else False,
                         )
            print("=" * 100 + "\n\n")

        add_theory_count = 0
        for k in range(num_iter):
            # Configure reg_multiplier:
            if self.reg_multiplier is not None:
                if not to_stop:
                    self.reg_model_idx += 1
                    self.reg_multiplier_model = self.reg_multiplier[self.reg_model_idx] if self.reg_model_idx < len(self.reg_multiplier) else self.reg_multiplier[-1]
                if domain_fit_setting is not None:
                    self.reg_domain_idx += 1
                    self.reg_multiplier_domain = self.reg_multiplier[self.reg_domain_idx] if self.reg_domain_idx < len(self.reg_multiplier) else self.reg_multiplier[-1]
            else:
                self.reg_multiplier_model = 1
                if domain_fit_setting is not None:
                    self.reg_multiplier_domain = 1

            # Record and visualization:
            if k % record_interval == 0:
                all_losses_dict = self.get_losses(X_test, y_test, forward_steps = forward_steps, **kwargs)
                record_data(self.data_record, [k, self.optimizer.param_groups[0]["lr"], all_losses_dict, None], ["iter", "lr", "all_losses_dict", "event"])
                if record_mode >= 2:
                    record_data(self.data_record, [self.pred_nets.model_dict, self.domain_net.model_dict], ["pred_nets_model_dict", "domain_net_model_dict"])
            if k % inspect_interval == 0:
                show(k)

            # Pre-optimization setting:
            self.model_dict_second_last = deepcopy(self.model_dict_last)
            self.model_dict_last["pred_nets"] = self.net_dict["pred_nets"].model_dict
            if "uncertainty_nets" in self.net_dict:
                self.model_dict_last["uncertainty_nets"] = self.net_dict["uncertainty_nets"].model_dict
            
            # Update domain target and lr every change_interval:
            if k % change_interval == 0:
                if scheduler_settings is not None:
                    if scheduler_settings[0] == "ReduceLROnPlateau":
                        loss_test = self.get_losses(X_test, y_test, mode = ["loss"], forward_steps = forward_steps)["loss"]
                        self.scheduler.step(loss_test)
                    else:
                        self.scheduler.step()
                if domain_fit_setting is not None:
                    if scheduler_settings is not None:
                        if scheduler_settings[0] == "ReduceLROnPlateau":
                            if self.domain_net_on:
                                loss_domain_test = self.get_losses(X_test, y_test, mode = ["loss_domain"], forward_steps = forward_steps)["loss_domain"]
                                self.scheduler_domain.step(loss_domain_test)
                        else:
                            self.scheduler_domain.step()
                    self.best_theory_idx = get_best_model_idx(self.net_dict, X_train, y_train, loss_fun_cumu = self.loss_fun_cumu, forward_steps = forward_steps, is_Lagrangian = self.is_Lagrangian)
                    self.best_theory_idx_test = get_best_model_idx(self.net_dict, X_test, y_test, loss_fun_cumu = self.loss_fun_cumu, forward_steps = forward_steps, is_Lagrangian = self.is_Lagrangian)
                    dataset_domain_train = data_utils.TensorDataset(X_train.data, self.best_theory_idx.data)
                    self.train_loader_domain = data_utils.DataLoader(dataset_domain_train, batch_size = batch_size, shuffle = True)

            # Loss-order decay:
            if loss_order_decay is not None:
                replaced_loss_order = self.loss_decay_scheduler.step()
            else:
                replaced_loss_order = None

            # Gradient noise:
            if self.scale_gen is not None:
                hook_handle_list = []
                if k % self.scale_gen.gradient_noise_interval_batch == 0:
                    for h in hook_handle_list:
                        h.remove()
                    hook_handle_list = []
                    scale_idx = int(k / self.scale_gen.gradient_noise_interval_batch)
                    if scale_idx >= len(gradient_noise_scale):
                        current_gradient_noise_scale = gradient_noise_scale[-1]
                    else:
                        current_gradient_noise_scale = gradient_noise_scale[scale_idx]
                    for parameter in self.pred_nets.parameters():
                        if parameter.requires_grad:
                            h = parameter.register_hook(lambda grad: grad + Variable(torch.normal(means = torch.zeros(grad.size()),
                                                                                     std = current_gradient_noise_scale * torch.ones(grad.size()))))
                            hook_handle_list.append(h)

            # Calculate loss and gradient:
            if batch_size is None:
                train_loader_model = [[X_train.data, y_train.data]]
                if domain_fit_setting is not None:
                    train_loader_domain = [[X_train.data, self.best_theory_idx.data]]
            else:
                train_loader_model = self.train_loader_model
                if domain_fit_setting is not None:
                    train_loader_domain = self.train_loader_domain

            # Training:
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader_model):
                # Trainging model:
                X_batch = Variable(X_batch, requires_grad = False)
                y_batch = Variable(y_batch, requires_grad = False)
                if not to_stop:
                    if self.optim_type[0] == "LBFGS":
                        def closure():
                            self.optimizer.zero_grad()
                            loss, _ = get_loss(net_dict = self.net_dict, X = X_batch, y = y_batch, loss_types = self.loss_types,
                                               forward_steps = forward_steps,
                                               domain_net = self.domain_net if self.domain_net_on else None, 
                                               domain_pred_mode = domain_pred_mode,
                                               loss_fun_dict = self.loss_fun_dict,
                                               replaced_loss_order = replaced_loss_order,
                                               is_Lagrangian = self.is_Lagrangian,
                                              )
                            if hasattr(self, "autoencoder"):
                                scale_autoencoder = self.optim_autoencoder_type[2]
                                loss_recons = self.autoencoder.get_loss(X_train, X_train, nn.MSELoss()) * scale_autoencoder
                                loss = loss + loss_recons
                            reg, _ = get_reg(net_dict = self.net_dict, reg_dict = self.reg_dict, mode = self.reg_mode, is_cuda = self.is_cuda)
                            if reg_smooth is not None:
                                input_noise_scale = reg_smooth[0]
                                num_samples = reg_smooth[4]
                                diff_list = []
                                for _ in range(num_samples):
                                    input_perturb = Variable(torch.randn(*X_batch.size()) * input_noise_scale)
                                    if self.is_cuda:
                                        input_perturb = input_perturb.cuda()
                                    diff = get_preds_valid(self.net_dict, X_batch + input_perturb, forward_steps = 1, is_Lagrangian = self.is_Lagrangian)[0] -                                            get_preds_valid(self.net_dict, X_batch, forward_steps = 1, is_Lagrangian = self.is_Lagrangian)[0]
                                    diff_list.append(diff)
                                diff_list = torch.stack(diff_list, 2)
                                smooth_norms = get_group_norm(diff, reg_smooth[1], reg_smooth[2])
                                reg_smooth_value = smooth_norms.mean() * reg_smooth[3]
                                reg = reg + reg_smooth_value
#                             if reg_grad is not None:
#                                 X_batch.requires_grad = True
#                                 loss_indi = self.loss_fun_cumu(self.pred_nets(X_batch), y_batch, cumu_mode = "original", is_mean = False).mean(0)
#                                 grad_norms = torch.cat([get_group_norm(grad(loss_indi[i], X_batch, create_graph = True)[0], reg_grad[0], reg_grad[1]) for i in range(self.num_theories)])
#                                 X_batch.requires_grad = False
#                                 reg_grad_value = grad_norms.mean() * reg_grad[2]
#                                 reg = reg + reg_grad_value
                            loss = loss + reg * self.reg_multiplier_model
                            loss.backward()
                            if self.grad_clipping is not None:
                                total_norm = torch.nn.utils.clip_grad_norm(self.pred_nets.parameters(), self.grad_clipping)
                            if np.isnan(to_np_array(loss)):
                                self.data_record["is_nan"] = True
                                raise Exception("NaN encountered!")
                            return loss
                        self.optimizer.step(closure)
                        if hasattr(self, "autoencoder"):
                            self.optimizer_autoencoder.step(closure)
                    else:
                        self.optimizer.zero_grad()
                        loss, _ = get_loss(net_dict = self.net_dict, X = X_batch, y = y_batch, loss_types = self.loss_types,
                                           forward_steps = forward_steps,
                                           domain_net = self.domain_net if self.domain_net_on else None,
                                           domain_pred_mode = domain_pred_mode,
                                           loss_fun_dict = self.loss_fun_dict,
                                           replaced_loss_order = replaced_loss_order,
                                           is_Lagrangian = self.is_Lagrangian,
                                          )
                        if hasattr(self, "autoencoder"):
                            scale_autoencoder = self.optim_autoencoder_type[2]
                            loss_recons = self.autoencoder.get_loss(X_train, X_train, nn.MSELoss()) * scale_autoencoder
                            loss = loss + loss_recons
                        reg, _ = get_reg(net_dict = self.net_dict, reg_dict = self.reg_dict, mode = self.reg_mode, is_cuda = self.is_cuda)
                        if reg_smooth is not None:
                            input_noise_scale = reg_smooth[0]
                            num_samples = reg_smooth[4]
                            diff_list = []
                            for _ in range(num_samples):
                                input_perturb = Variable(torch.randn(*X_batch.size()) * input_noise_scale)
                                if self.is_cuda:
                                    input_perturb = input_perturb.cuda()
                                diff = get_preds_valid(self.net_dict, X_batch + input_perturb, forward_steps = 1, is_Lagrangian = self.is_Lagrangian)[0] -                                        get_preds_valid(self.net_dict, X_batch, forward_steps = 1, is_Lagrangian = self.is_Lagrangian)[0]
                                diff_list.append(diff)
                            diff_list = torch.stack(diff_list, 2)
                            smooth_norms = get_group_norm(diff, reg_smooth[1], reg_smooth[2])
                            reg_smooth_value = smooth_norms.mean() * reg_smooth[3]
                            reg = reg + reg_smooth_value
#                         if reg_grad is not None:
#                             X_batch.requires_grad = True
#                             loss_indi = self.loss_fun_cumu(self.pred_nets(X_batch), y_batch, cumu_mode = "original", is_mean = False).mean(0)
#                             grad_norms = torch.cat([get_group_norm(grad(loss_indi[i], X_batch, create_graph = True)[0], reg_grad[0], reg_grad[1]) for i in range(self.num_theories)])
#                             X_batch.requires_grad = False
#                             reg_grad_value = grad_norms.mean() * reg_grad[2]
#                             reg = reg + reg_grad_value
                        loss = loss + reg * self.reg_multiplier_model
                        loss.backward()
                        if self.grad_clipping is not None:
                            total_norm = torch.nn.utils.clip_grad_norm(self.pred_nets.parameters(), self.grad_clipping)
                        if np.isnan(to_np_array(loss)):
                            if raise_nan:
                                self.data_record["is_nan"] = True
                                raise Exception("NaN encountered!")
                            else:
                                self.data_record["is_nan"] = True
                                print("NaN encountered!")
                                self.pred_nets.load_model_dict(self.model_dict_second_last["pred_nets"])
                                self.net_dict["pred_nets"] = self.pred_nets
                                return deepcopy(self.data_record)
                        self.optimizer.step()
                        if hasattr(self, "autoencoder"):
                            self.optimizer_autoencoder.step()

                # Training domain:
                if domain_fit_setting is not None:
                    X_batch_domain, best_idx_domain = list(self.train_loader_domain)[batch_idx]
                    X_batch_domain = Variable(X_batch_domain, requires_grad = False)
                    best_idx_domain = Variable(best_idx_domain, requires_grad = False)
                    if self.is_cuda:
                        X_batch_domain = X_batch_domain.cuda()
                        best_idx_domain = best_idx_domain.cuda()
                    if hasattr(self, "autoencoder"):
                         X_batch_domain = self.autoencoder.encode(X_batch_domain)
                    if domain_fit_setting["optim_domain_type"][0] == "LBFGS":
                        def closure_domain():
                            self.optimizer_domain.zero_grad()
                            loss_domain = nn.CrossEntropyLoss()(self.domain_net(X_batch_domain), best_idx_domain)
                            reg_domain, _ = get_reg(net_dict = self.net_dict, reg_dict = domain_fit_setting["reg_domain_dict"], mode = domain_fit_setting["reg_domain_mode"], is_cuda = self.is_cuda)
                            loss_domain = loss_domain + reg_domain * self.reg_multiplier_domain
                            loss_domain.backward()
                            return loss_domain
                        self.optimizer_domain.step(closure_domain)
                    else:
                        self.optimizer_domain.zero_grad()
                        loss_domain = nn.CrossEntropyLoss()(self.domain_net(X_batch_domain), best_idx_domain)
                        reg_domain, _ = get_reg(net_dict = self.net_dict, reg_dict = domain_fit_setting["reg_domain_dict"], mode = domain_fit_setting["reg_domain_mode"], is_cuda = self.is_cuda)
                        loss_domain = loss_domain + reg_domain * self.reg_multiplier_domain
                        loss_domain.backward()
                        self.optimizer_domain.step()

            # Early stopping:
            loss_dict = self.get_losses(X_test, y_test, mode = ["loss", "mse_with_domain", "loss_domain"], forward_steps = forward_steps)
            if loss_dict["mse_with_domain"] < loss_floor:
                print("loss = {0} is below the floor level {1}, stop.".format(loss_dict["mse_with_domain"], loss_floor))
                break
            if not to_stop:
                if patience is not None:
                    to_stop = self.early_stopping.monitor(loss_dict["loss"])
                if to_stop:
                    if add_theory_loss_threshold is not None and (add_theory_quota is None or (add_theory_quota is not None and add_theory_count < add_theory_quota))                                                              and (add_theory_limit is None or (add_theory_limit is not None and self.num_theories < add_theory_limit)):
                        is_add, add_theories_info = self.add_theories(X_train, y_train,
                                                                      validation_data = (X_test, y_test),
                                                                      criteria = add_theory_criteria,
                                                                      loss_threshold = add_theory_loss_threshold,
                                                                      **kwargs
                                                                     )
                        if is_add:
                            print("At iteration {0}".format(k))
                            to_stop = False
                            add_theory_count += 1
                            self.early_stopping.reset()
                            all_losses_dict = self.get_losses(X_test, y_test, forward_steps = forward_steps, **kwargs)
                            record_data(self.data_record, [k, self.optimizer.param_groups[0]["lr"], all_losses_dict, ("add_theories", add_theories_info)], ["iter", "lr", "all_losses_dict", "event"])
                            if record_mode >= 2:
                                record_data(self.data_record, [self.pred_nets.model_dict, self.domain_net.model_dict], ["pred_nets_model_dict", "domain_net_model_dict"])
                            show(k)
                    else:
                        is_add = False
                    if not is_add:
                        all_losses_dict = self.get_losses(X_test, y_test, forward_steps = forward_steps, **kwargs)
                        record_data(self.data_record, [k, self.optimizer.param_groups[0]["lr"], all_losses_dict, "Model training complete."], ["iter", "lr", "all_losses_dict", "event"])
                        if record_mode >= 2:
                            record_data(self.data_record, [self.pred_nets.model_dict, self.domain_net.model_dict], ["pred_nets_model_dict", "domain_net_model_dict"])
                        print("model training complete with early stopping at iteration {0}, with loss = {1:.9f}. Continue domain training.".format(k, loss_dict["loss"]))
            if domain_fit_setting is not None:
                if patience is not None:
                    to_stop_domain = self.early_stopping_domain.monitor(loss_dict["loss_domain"])
            if to_stop and (domain_fit_setting is None or (domain_fit_setting is not None and to_stop_domain)):
                print("the loss does not decrease for {0} consecutive iterations, stop at iteration {1}. Latest loss: {2}".format(patience, k, loss_dict["loss"]))
                break

        all_losses_dict = self.get_losses(X_test, y_test, forward_steps = forward_steps, **kwargs)
        record_data(self.data_record, [k + 1, self.optimizer.param_groups[0]["lr"], all_losses_dict, "end"], ["iter", "lr", "all_losses_dict", "event"])
        if record_mode >= 2:
            record_data(self.data_record, [self.pred_nets.model_dict, self.domain_net.model_dict], ["pred_nets_model_dict", "domain_net_model_dict"])
        show(k + 1)
        self.data_record["is_nan"] = False
        print("completed!")
        return deepcopy(self.data_record)
    
    
    def fit_domain(
        self,
        X_train,
        y_train,
        validation_data = None,
        optim_domain_type = ("adam", 1e-3),
        reg_domain_dict = {},
        reg_domain_mode = "L1",
        forward_steps = 1,
        domain_pred_mode = "onehot",
        scheduler_settings = ("ReduceLROnPlateau", 30, 0.1),
        epochs = None,
        patience = None,
        inspect_interval = None,
        change_interval = 1,
        record_interval = None,
        isplot = True,
        filename = None,
        view_init = (10, 190),
        loss_floor = 1e-12,
        prefix = None,
        **kwargs
        ):
        self.domain_net_on = True
        X_test, y_test = validation_data
        self.optim_domain_type = optim_domain_type
        self.reg_domain_dict = reg_domain_dict
        self.reg_domain_mode = reg_domain_mode
        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else None
        record_mode = kwargs["record_mode"] if "record_mode" in kwargs else 1
        
        if validation_data is None:
            validation_data = (X_train, y_train)

        if not hasattr(self, "optimizer_domain"):
            self.optimizer_domain = get_optimizer(optim_type = self.optim_domain_type[0], lr = self.optim_domain_type[1], parameters = self.domain_net.parameters())
        else:
            new_lr = np.sqrt(self.optimizer_domain.param_groups[0]["lr"] * self.optim_domain_type[1])
            for param_group in self.optimizer_domain.param_groups:
                param_group["lr"] = new_lr
    
        if batch_size is None:
            if record_interval is None:
                record_interval = 50
            if self.optim_domain_type[0] == "LBFGS":
                num_iter_domain = 5000
                inspect_interval_domain = 100
            else:
                num_iter_domain = 30000
                inspect_interval_domain = 2000
        else:
            if record_interval is None:
                record_interval = 1
            if self.optim_domain_type[0] == "LBFGS":
                num_iter_domain = 250
                inspect_interval_domain = 10
            else:
                num_iter_domain = 1000
                inspect_interval_domain = 20              
        if epochs is not None:
            num_iter_domain = epochs
        if inspect_interval is None:
            inspect_interval = inspect_interval_domain
        assert inspect_interval % record_interval == 0
        
        figsize = (10, 8)

        self.data_record_domain = {}
        self.best_theory_idx = get_best_model_idx(self.net_dict, X_train, y_train, loss_fun_cumu = self.loss_fun_cumu, forward_steps = forward_steps, is_Lagrangian = self.is_Lagrangian)
        self.best_theory_idx_test = get_best_model_idx(self.net_dict, X_test, y_test, loss_fun_cumu = self.loss_fun_cumu, forward_steps = forward_steps, is_Lagrangian = self.is_Lagrangian)
        if hasattr(self, "autoencoder"):
            X_train_lat = Variable(self.autoencoder.encode(X_train).data, requires_grad = False)
            X_test_lat = Variable(self.autoencoder.encode(X_test).data, requires_grad = False)
        else:
            X_train_lat, X_test_lat = X_train, X_test
        if batch_size is not None:
            dataset_domain_train = data_utils.TensorDataset(X_train_lat.data, self.best_theory_idx.data)
            self.train_loader_domain = data_utils.DataLoader(dataset_domain_train, batch_size = batch_size, shuffle = True)
            
        # Setting up lr_scheduler:
        if scheduler_settings is not None:
            if scheduler_settings[0] == "LambdaLR":
                function_type = scheduler_settings[1]
                decay_scale = scheduler_settings[2]
                scheduler_continue_decay = scheduler_settings[3]
                if function_type == "exp":
                    lambda_domain = lambda epoch: (1 - 1 / float(num_iter_domain / change_interval / decay_scale)) ** epoch
                elif function_type == "poly":
                    lambda_domain = lambda epoch: 1 / (1 + 0.01 * epoch * change_interval * decay_scale)
                else:
                    raise
                if scheduler_continue_decay:
                    if not hasattr(self, "scheduler_domain"):
                        self.scheduler_domain = LambdaLR(self.optimizer_domain, lr_lambda = lambda_domain) 
                else:
                    self.scheduler_domain = LambdaLR(self.optimizer_domain, lr_lambda = lambda_domain)
            elif scheduler_settings[0] == "ReduceLROnPlateau":
                scheduler_patience = scheduler_settings[1]
                scheduler_factor = scheduler_settings[2]
                self.scheduler_domain = ReduceLROnPlateau(self.optimizer_domain, factor = scheduler_factor, patience = scheduler_patience, verbose = True)
            else:
                raise

        self.early_stopping_domain = Early_Stopping(patience = patience, epsilon = 1e-10)
        to_stop_domain = False
        
        def show(k):
            all_losses_dict = self.get_losses(X_test, y_test, forward_steps = forward_steps, **kwargs)
            if prefix is not None:
                print(prefix)
            print("domain_iter {0}\tlr = {1:.9f}\nloss_domain:\t{2:.9f}\nreg_domain:\t{3:.9f}".format(k, self.optimizer_domain.param_groups[0]["lr"], 
                                                                                                  all_losses_dict["loss_domain"], all_losses_dict["reg_domain"]))
            print("loss_best:\t{0:.9f}\nloss_model:\t{1:.9f}\nloss_with_domain:\t{2:.9f}".format(all_losses_dict["loss_best"], all_losses_dict["loss"], all_losses_dict["loss_with_domain"]))
            for loss_mode in all_losses_dict["loss_dict"]:
                print("loss_{0}:\t{1:.9f}".format(loss_mode, all_losses_dict["loss_dict"][loss_mode]))
            for i in range(self.num_theories):
                print("{0}_theory_{1}:\t{2:.9f}\tfraction best: {3:.5f} \tfraction domain: {4:.5f}".format(self.loss_fun_cumu.loss_fun.core, i, all_losses_dict["loss_indi_theory"][i], all_losses_dict["fraction_list_best"][i], all_losses_dict["fraction_list_domain"][i]))
            print("mse_with_domain:\t{0:.9f}\nmse_without_domain:\t{1:.9f}".format(all_losses_dict["mse_with_domain"], all_losses_dict["mse_without_domain"]))
            if self.reg_multiplier is not None:
                print("reg_multiplier_domain iter {0}\t{1:.9f}".format(self.reg_domain_idx, self.reg_multiplier_domain))
            if "big_domain_ids" in kwargs and kwargs["big_domain_ids"] is not None:
                if "metrics_big_domain" in all_losses_dict and all_losses_dict["metrics_big_domain"] is not None:
                    union = all_losses_dict["metrics_big_domain"]["union"]
                    predicted_big_domains = all_losses_dict["metrics_big_domain"]["predicted_big_domains"]
                    true_big_domains = all_losses_dict["metrics_big_domain"]["true_big_domains"]
                    intersection = all_losses_dict["metrics_big_domain"]["intersection"]
                    intersection_in_big = all_losses_dict["metrics_big_domain"]["intersection_in_big"]
                    if union is not None:
                        precision = intersection / float(predicted_big_domains)
                        recall = intersection / float(true_big_domains)
                        F1 = 2 / (1 / precision  + 1 / recall)
                        IoU = intersection / float(union)
                        print("union: {0}\tpredicted_big_domains: {1}\ttrue_big_domains: {2}\tintersection_in_big: {3}\tintersection: {4s}".format(union, predicted_big_domains, true_big_domains, intersection_in_big, intersection))
                        print("Precision: {0:.4f}\tRecall: {1:.4f}\tF1: {2:.4f}\tIoU: {3:.4f}".format(precision, recall, F1, IoU))
                    print("loss_big_domain: {0:.9f}".format(all_losses_dict["metrics_big_domain"]["loss_big_domain"]))
                    print("loss_with_domain_big_domain: {0:.9f}".format(all_losses_dict["metrics_big_domain"]["loss_with_domain_big_domain"]))
                    print("mse_with_domain_big_domain: {0:.9f}".format(all_losses_dict["metrics_big_domain"]["mse_with_domain_big_domain"]))
            print()
            try:
                sys.stdout.flush()
            except:
                pass
            if isplot or filename is not None:
                self.plot(X_test, y_test, forward_steps = forward_steps, view_init = view_init, figsize = figsize, is_show = isplot, filename = filename + "_{0}".format(k) if filename is not None else None,
                          true_domain = kwargs["true_domain_test"] if "true_domain_test" in kwargs else None,
                          show_3D_plot = kwargs["show_3D_plot"] if "show_3D_plot" in kwargs else False, 
                          show_vs = kwargs["show_vs"] if "show_vs" in kwargs else False,
                         )
            print("=" * 100 + "\n\n")
        
        for k in range(num_iter_domain):
            if self.reg_multiplier is not None:
                self.reg_domain_idx += 1
                self.reg_multiplier_domain = self.reg_multiplier[self.reg_domain_idx] if self.reg_domain_idx < len(self.reg_multiplier) else self.reg_multiplier[-1]
            else:
                self.reg_multiplier_domain = 1

            # Record:
            if k % record_interval == 0:  
                all_losses_dict = self.get_losses(X_test, y_test, forward_steps = forward_steps, **kwargs)
                record_data(self.data_record_domain, [k, self.optimizer_domain.param_groups[0]["lr"], all_losses_dict, None], ["iter", "lr", "all_losses_dict", "event"])
                if record_mode >= 2:
                    record_data(self.data_record_domain, [self.pred_nets.model_dict, self.domain_net.model_dict], ["pred_nets_model_dict", "domain_net_model_dict"])
            if k % inspect_interval == 0:
                show(k)
            if k % change_interval == 0:
                if scheduler_settings is not None:
                    if scheduler_settings[0] == "ReduceLROnPlateau":
                        loss_domain_test = self.get_losses(X_test, y_test, mode = ["loss_domain"], forward_steps = forward_steps)["loss_domain"]
                        self.scheduler_domain.step(loss_domain_test)
                    else:
                        self.scheduler_domain.step()

            if batch_size is None:
                train_loader_domain = [[X_train_lat.data, self.best_theory_idx.data]]
            else:
                train_loader_domain = self.train_loader_domain
            for batch_idx, (X_domain_batch, best_idx_domain) in enumerate(train_loader_domain):
                X_domain_batch = Variable(X_domain_batch, requires_grad = False)
                best_idx_domain = Variable(best_idx_domain, requires_grad = False)
                if self.is_cuda:
                    X_domain_batch = X_domain_batch.cuda()
                    best_idx_domain = best_idx_domain.cuda()

                if self.optim_domain_type[0] == "LBFGS":
                    def closure_domain():
                        self.optimizer_domain.zero_grad()
                        loss_domain = nn.CrossEntropyLoss()(self.domain_net(X_domain_batch), best_idx_domain)
                        reg_domain, _ = get_reg(net_dict = self.net_dict, reg_dict = self.reg_domain_dict, mode = self.reg_domain_mode, is_cuda = self.is_cuda)
                        loss_domain = loss_domain + reg_domain * self.reg_multiplier_domain
                        loss_domain.backward()
                        return loss_domain
                    self.optimizer_domain.step(closure_domain)
                else:
                    self.optimizer_domain.zero_grad()
                    loss_domain = nn.CrossEntropyLoss()(self.domain_net(X_domain_batch), best_idx_domain)
                    reg_domain, _ = get_reg(net_dict = self.net_dict, reg_dict = self.reg_domain_dict, mode = self.reg_domain_mode, is_cuda = self.is_cuda)
                    loss_domain = loss_domain + reg_domain * self.reg_multiplier_domain
                    loss_domain.backward()
                    self.optimizer_domain.step()

            loss_dict = self.get_losses(X_test, y_test, mode = ["mse_with_domain", "loss_domain"], forward_steps = forward_steps)
            if loss_dict["mse_with_domain"] < loss_floor:
                print("loss = {0} is below the loss floor level {1}, stop.".format(loss_dict["mse_with_domain"], loss_floor))
                break

            loss_domain_test = loss_dict["loss_domain"]
            to_stop_domain = self.early_stopping_domain.monitor(loss_domain_test)
            if to_stop_domain:
                print("the loss_domain does not decrease for {0} consecutive iterations, stop at iteration {1}. loss_domain: {2}".format(patience, k, loss_domain_test))
                break
        print("completed")
        # Record:
        all_losses_dict = self.get_losses(X_test, y_test, forward_steps = forward_steps, **kwargs)
        record_data(self.data_record_domain, [k + 1, self.optimizer_domain.param_groups[0]["lr"], all_losses_dict, "end"], ["iter", "lr", "all_losses_dict", "event"])
        if record_mode >= 2:
            record_data(self.data_record_domain, [self.pred_nets.model_dict, self.domain_net.model_dict], ["pred_nets_model_dict", "domain_net_model_dict"])
        show(k + 1)
        self.data_record_domain["is_nan"] = False
        return deepcopy(self.data_record_domain)


    def plot(
        self,
        X,
        y, 
        forward_steps = 1,
        view_init = (10, 190),
        figsize = (10, 8), 
        show_3D_plot = False,
        show_vs = False,
        show_loss_histogram = True,
        is_show = True,
        filename = None,
        **kwargs
        ):
        if not is_show:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if hasattr(self, "autoencoder"):
            X_lat = self.autoencoder.encode(X)
        else:
            X_lat = X
        preds, valid_onehot = get_preds_valid(self.net_dict, X, forward_steps = forward_steps, domain_net = self.domain_net, domain_pred_mode = "onehot", is_Lagrangian = self.is_Lagrangian)
        best_theory_idx = get_best_model_idx(self.net_dict, X, y, loss_fun_cumu = self.loss_fun_cumu, forward_steps = forward_steps, mode = "expanded", is_Lagrangian = self.is_Lagrangian)
        best_theory_onehot = to_one_hot(best_theory_idx, valid_onehot.size(1))
        true_domain = kwargs["true_domain"] if "true_domain" in kwargs else None
        if true_domain is not None:
            uniques = np.unique(to_np_array(true_domain))
            num_uniques = int(max(np.max(uniques) + 1, len(uniques)))
            true_domain_onehot = to_one_hot(true_domain, num_uniques)
        else:
            true_domain_onehot = None
        if show_3D_plot:
            from mpl_toolkits.mplot3d import Axes3D
            loss_dict = self.get_losses(X, y, mode = "all")
            if self.input_size > 1:
                for i in range(y.size(1)):
                    if is_show:
                        print("target with axis {0}:".format(i))
                    if true_domain is not None:
                        axis_lim = plot3D(X, y[:,i:i+1].repeat(1, num_uniques), true_domain_onehot, view_init = view_init, figsize = figsize, is_show = is_show, filename = filename + "_target_ax_{0}.png".format(i) if filename is not None else None) 
                    else:
                        axis_lim = plot3D(X, y[:,i:i+1], view_init = view_init, figsize = figsize, is_show = is_show, filename = filename + "_target_ax_{0}.png".format(i) if filename is not None else None) 
                for i in range(preds.size(2)):
                    if is_show:
                        print("best_prediction with axis {0}:".format(i))
                    plot3D(X, preds[:,:,i], best_theory_onehot, view_init = view_init, axis_lim = axis_lim, 
                           axis_title = ["loss_best = {0:.9f}".format(loss_dict["loss_best"]),
                                          "\n".join(["{2}_theory_{0}: {1:.9f}".format(j, loss_dict['loss_indi_theory'][j], self.loss_fun_cumu.loss_fun.core) for j in range(len(loss_dict['loss_indi_theory']))]),
                                        ], 
                           figsize = figsize, is_show = is_show, filename = filename + "_best-prediction_ax_{0}.png".format(i) if filename is not None else None)
                for i in range(preds.size(2)):
                    if is_show:
                        print("all theory prediction with axis {0}:".format(i))
                    plot3D(X, preds[:,:,i], valid_onehot, view_init = view_init, axis_lim = axis_lim, 
                           axis_title = ["mse_with_domain = {0:.9f}".format(loss_dict['mse_with_domain']), "loss_total = {0:.9f}".format(loss_dict["loss"])], 
                           figsize = figsize, is_show = is_show, filename = filename + "_all-prediction_ax_{0}.png".format(i) if filename is not None else None)
            else:
                ylim = (np.floor(preds.data.min()) - 3, np.ceil(preds.data.max()) + 3)        
                if "uncertainty_nets" in self.net_dict:
                    pred_with_uncertainty, info_list = get_pred_with_uncertainty(preds, self.net_dict["uncertainty_nets"], X)
                    fig = plt.figure(figsize = (6, 5))
                    sigma = info_list.sum(1) ** (-0.5)
                    plt.errorbar(to_np_array(X), to_np_array(pred_with_uncertainty), yerr = to_np_array(sigma), fmt='ob', markersize= 1, alpha = 0.4, label = "theory_whole")
                    plt.ylim(ylim)
                    plt.plot(to_np_array(X), to_np_array(y), ".k", markersize = 2, alpha = 0.9)
                else:
                    for j in range(self.num_theories):
                        plt.plot(to_np_array(X), to_np_array(preds[:,j]), color = COLOR_LIST[j % len(COLOR_LIST)], marker = ".", markersize = 1, alpha = 0.6, label = "theory_{0}".format(j))
                    plt.plot(to_np_array(X), to_np_array(y), ".k", markersize = 2, alpha = 0.9)
                    plt.legend()
                    plt.ylim(ylim)
                plt.legend()
                plt.show()

                fig = plt.figure(figsize = (self.num_theories * 6, 5))
                for j in range(self.num_theories):
                    plt.subplot(1, self.num_theories, j + 1)
                    if "uncertainty-based" in self.loss_types:
                        plt.errorbar(to_np_array(X), to_np_array(preds[:, j]), yerr = to_np_array((info_list ** (-0.5))[:, j]), fmt='o{0}'.format(
                                     COLOR_LIST[j % len(COLOR_LIST)]), markersize= 1, alpha = 0.2, label = "theory_{0}".format(j))
                    else:
                        plt.plot(to_np_array(X), to_np_array(preds[:,j]), color = COLOR_LIST[j % len(COLOR_LIST)], marker = ".", markersize = 1, alpha = 0.6, label = "theory_{0}".format(j))
                    plt.ylim(ylim)
                    plt.plot(to_np_array(X), to_np_array(y), ".k", markersize = 2, alpha = 0.9)
                    plt.legend()
                plt.show()

        # Plotting the domain on a 2D plane:
        if "true_domain" in kwargs and kwargs["true_domain"] is not None and self.input_size % 2 == 0 and             ("num_output_dims" not in kwargs or ("num_output_dims" in kwargs and kwargs["num_output_dims"] in [1, 2, 4])) and             self.is_Lagrangian is not True:
            if "num_output_dims" in kwargs and kwargs["num_output_dims"] == 4:
                idx = Variable(torch.LongTensor(np.array([0,2])))
            else:
                idx = None
            self.get_domain_plot(X, y, X_lat = X_lat, true_domain = kwargs["true_domain"], 
                                 X_idx = idx, y_idx = idx, 
                                 is_plot_loss = False if len(y.shape) == 4 else True, 
                                 is_plot_indi_domain = False, 
                                 is_show = is_show, 
                                 filename = filename + "_domain-plot.png" if filename is not None else None,
                                 is_Lagrangian = self.is_Lagrangian,
                                )

        # Plotting pred vs. target:
        if show_vs:
            for i in range(preds.size(2)):
                _, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize=(14,6))       
                self.plot_pred_vs_y(preds[:,:,i], y[:,i:i+1], best_theory_onehot, title = "best_prediction_ax_{0}".format(i), ax = ax1, is_color = True, is_show = False, filename = filename + "_pred-vs-target_ax_{0}.png".format(i) if filename is not None else None, is_close = False)
                self.plot_pred_vs_y(preds[:,:,i], y[:,i:i+1], valid_onehot, title = "domain_prediction_ax_{0}".format(i), ax = ax2, is_color = True, is_show = is_show, filename = filename + "_pred-vs-target_ax_{0}.png".format(i) if filename is not None else None)

        if is_show and hasattr(self, "autoencoder"):
            loss_indi_theory = []
            for k in range(self.num_theories):
                loss_indi_theory.append(to_np_array(self.loss_fun_cumu(preds[:, k:k+1], y, is_mean = False)))
            loss_indi_theory = np.concatenate(loss_indi_theory, 1)
            domains = to_np_array(self.domain_net(X_lat).max(1)[1])
            X_recons = self.autoencoder(X)
            print("reconstruct:")
            for i in np.random.randint(len(X), size = 2):
                plot_matrices(torch.cat([X[i], X_recons[i]], 0), images_per_row = 5)
            print("prediction:")
            for i in np.random.randint(len(X), size = 10):
                print("losses: {0}".format(to_string(loss_indi_theory[i], connect = "\t", num_digits = 6)))
                print("best_idx: {0}\tdomain_idx: {1}".format(to_np_array(best_theory_idx[i]), domains[i]))
                plot_matrices(torch.cat([y[i], preds[i]], 0), images_per_row = 5)

        if show_loss_histogram:
            self.plot_loss_histogram(X, y, X_lat = X_lat, mode = "log-mse", forward_steps = forward_steps, is_show = is_show, filename = filename)
            self.plot_loss_histogram(X, y, X_lat = X_lat, mode = "DL" if not ("DL" in self.loss_core and self.loss_core != "DL") else self.loss_core, forward_steps = forward_steps, is_show = is_show, filename = filename)

        plt.clf()
        plt.close()


    def plot_pred_vs_y(self, preds, y, valid_onehot, title = None, ax = None, is_color = False, is_show = True, filename = None, is_close = True):
        if ax is not None and not is_show:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        valid_onehot = to_Boolean(valid_onehot)
        pred_chosen = torch.masked_select(preds, valid_onehot)
        if ax is None:
            if not is_color:
                plt.plot(to_np_array(y), to_np_array(pred_chosen), ".", markersize = 1)
            else:
                for i in range(valid_onehot.size(1)):
                    plt.plot(to_np_array(y[:, 0][valid_onehot[:, i]]), to_np_array(preds[:, i][valid_onehot[:, i]]), ".", color = COLOR_LIST[i % len(COLOR_LIST)], markersize = 1.5, alpha = 0.6)
            plt.xlabel("y")
            plt.ylabel("pred")
            if title is not None:
                plt.title(title)
        else:
            if not is_color:
                ax.plot(to_np_array(y), to_np_array(pred_chosen), ".", markersize = 1)
            else:
                for i in range(valid_onehot.size(1)):
                    ax.plot(to_np_array(y[:, 0][valid_onehot[:, i]]), to_np_array(preds[:, i][valid_onehot[:, i]]), ".", color = COLOR_LIST[i % len(COLOR_LIST)], markersize = 1.5, alpha = 0.6)
            ax.set_xlabel("y")
            ax.set_ylabel("pred")
            if title is not None:
                ax.set_title(title)
        if filename is not None:
            plt.savefig(filename)
        if is_show:
            plt.show()
        if is_close:
            plt.clf()
            plt.close()


    def plot_loss_histogram(self, X, y, X_lat = None, mode = "log-mse", forward_steps = 1, is_show = True, filename = None, **kwargs):
        if not is_show:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if mode == "log-mse":
            loss_list = torch.log(get_loss(self.net_dict, X, y, loss_types = {"pred-based_mean": {"amp": 1.}}, forward_steps = forward_steps, domain_net = self.domain_net, 
                                           loss_fun_dict = {"loss_fun_cumu": Loss_Fun_Cumu(core = "mse", cumu_mode = "mean", balance_model_influence = False, loss_precision_floor = self.loss_precision_floor)},
                                           is_Lagrangian = self.is_Lagrangian, is_mean = False)[0]) / np.log(10)
            loss_list = to_np_array(loss_list)
            range = kwargs["range"] if "range" in kwargs else (-10, 1)
        elif "DL" in mode:
            loss_list = get_loss(self.net_dict, X, y, loss_types = {"pred-based_mean": {"amp": 1.}}, forward_steps = forward_steps, domain_net = self.domain_net, 
                                 loss_fun_dict = {"loss_fun_cumu": Loss_Fun_Cumu(core = mode, cumu_mode = "mean", balance_model_influence = False, loss_precision_floor = self.loss_precision_floor)},
                                 is_Lagrangian = self.is_Lagrangian, is_mean = False)[0]
            loss_list = to_np_array(loss_list)
            if "range" in kwargs:
                range = kwargs["range"]
            else:
                range_max = int(np.ceil(loss_list.max() / 2) * 2)
                range = (0, range_max)
        else:
            raise Exception("loss mode {0} not recognized!".format(mode))
        plt.figure(figsize = (16, 6))
        plt.subplot(1, 2, 1)
        plt.hist(loss_list, range = range, bins = 40)
        if "DL" in mode:
            plt.title("{0} histogram, loss_precision_floor = {1}".format(mode, self.loss_precision_floor))
        else:
            plt.title("{} histogram".format(mode))

        plt.subplot(1, 2, 2)
        domain_pred = self.domain_net(X_lat).max(1)[1]
        for idx in np.unique(to_np_array(domain_pred)):
            is_in = to_np_array(domain_pred == int(idx)).astype(bool)
            plt.hist(loss_list[is_in], range = range, bins = 40, alpha = kwargs["alpha"] if "alpha" in kwargs else 0.5, color = COLOR_LIST[idx % len(COLOR_LIST)], label = "{0}".format(idx))
        plt.legend()
        plt.title("{0} per theory".format(mode))
        if filename is not None:
            plt.savefig(filename + "_{0}_hist.png".format(mode))
        if is_show:
            plt.show()
        plt.clf()
        plt.close()

    
    def get_domain_plot(self, X, y, X_lat, true_domain, X_idx = None, y_idx = None, is_plot_loss = True, forward_steps = 1, is_plot_indi_domain = False, is_show = True, filename = None, is_Lagrangian = False):
        if not is_show:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pylab as plt
        if self.domain_net.is_cuda:
            X = X.cuda()
            y = y.cuda()
        domain_net_pred = to_np_array(self.domain_net(X_lat).max(1)[1])
        best_model_pred = to_np_array(get_best_model_idx(self.net_dict, X, y, loss_fun_cumu = self.loss_fun_cumu, is_Lagrangian = self.is_Lagrangian))
        true_domain = to_np_array(true_domain.squeeze())
        if is_plot_loss:
            preds, _ = get_preds_valid(self.net_dict, X, forward_steps = forward_steps, domain_net = None, is_Lagrangian = self.is_Lagrangian)
            
#             loss_fun_cumu = Loss_Fun_Cumu(core = "mse", cumu_mode = "mean", balance_model_influence = False, loss_precision_floor = self.loss_precision_floor)
#             log_loss_best = to_np_array(torch.log(loss_fun_cumu(preds, y, model_weights = None, cumu_mode = "min", is_mean = False)) / np.log(10))
#             log_loss_domain = to_np_array(torch.log(get_loss(self.net_dict, X, y, loss_types = {"pred-based_mean": {"amp": 1.}}, forward_steps = forward_steps, domain_net = self.domain_net, loss_fun_dict = {"loss_fun_cumu": loss_fun_cumu}, is_mean = False)[0]) / np.log(10))
            
            loss_fun_cumu = Loss_Fun_Cumu(core = "DLs", cumu_mode = "mean", balance_model_influence = False, loss_precision_floor = self.loss_precision_floor)
            log_loss_best = to_np_array(loss_fun_cumu(preds, y, model_weights = None, cumu_mode = "min", is_mean = False))
            log_loss_domain = get_loss(self.net_dict, X, y, loss_types = {"pred-based_mean": {"amp": 1.}}, forward_steps = forward_steps, domain_net = self.domain_net, loss_fun_dict = {"loss_fun_cumu": loss_fun_cumu}, is_Lagrangian = self.is_Lagrangian, is_mean = False)[0]
        else:
            log_loss_best = None
            log_loss_domain = None
        
        if X_idx is not None:
            X_idx = to_Variable(X_idx).long()
            X = torch.index_select(X, -1, X_idx)
        if y_idx is not None:
            y_idx = to_Variable(y_idx).long()
            y = torch.index_select(y, -1, y_idx)

        if is_plot_indi_domain:
            plot_indi_domain(X_lat, domain = true_domain, is_show = is_show, filename = filename)

        def plot_domains(
            X,
            domain,
            y = y if is_plot_loss else None,
            log_loss = None,
            title = None,
            is_legend = True,
            ):
            X = to_np_array(X).reshape(X.shape[0], -1, 2)
            if y is not None:
                y = to_np_array(y).reshape(y.shape[0], 1, 2)
                Xy = np.concatenate([X, y], 1)

            for idx in np.unique(domain):
                domain_idx = (domain == int(idx)).astype(bool)
                if y is not None:
                    X_domain = Xy[domain_idx]
                    y_domain = y[domain_idx]
                    if log_loss is not None:
                        log_loss_domain = to_np_array(log_loss)[domain_idx]
                else:
                    X_domain = X[domain_idx]

                for i in range(len(X_domain)):
                    if i == 0:
                        plt.plot(X_domain[i, :, 0], X_domain[i, :, 1], ".-", color = COLOR_LIST[idx % len(COLOR_LIST)], alpha = 0.5, markersize = 1, linewidth = 1, label = str(idx))
                    else:
                        plt.plot(X_domain[i, :, 0], X_domain[i, :, 1], ".-", color = COLOR_LIST[idx % len(COLOR_LIST)], alpha = 0.5, markersize = 1, linewidth = 1)
                if y is not None and log_loss is not None:
                    plt.scatter(y_domain[:, 0, 0], y_domain[:, 0, 1], s = 4 * np.sqrt(log_loss_domain), color = COLOR_LIST[idx % len(COLOR_LIST)])
            if is_legend:
                plt.legend(bbox_to_anchor = (1, 0.9, 0.15 ,0.1))
            if title is not None:
                plt.title(title)

        plt.figure(figsize = (19,16))
        plt.subplot(2, 2, 1)
        plot_domains(X_lat, true_domain, title = "True", is_legend = False)

        plt.subplot(2, 2, 2)
        plot_domains(X_lat, best_model_pred, log_loss = log_loss_best, title = "Best")

        plt.subplot(2, 2, 3)
        plot_domains(X_lat, domain_net_pred, log_loss = log_loss_domain, title = "Domain: precision-floor: {0}".format(self.loss_precision_floor))

        if filename is not None:
            plt.savefig(filename)
        if is_show:
            plt.show()
        plt.clf()
        plt.close()


    def set_net(self, net_name, net):
        setattr(self, net_name, net)
        self.net_dict[net_name] = net
        self.num_theories = self.pred_nets.num_models

