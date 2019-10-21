
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_physicist.pytorch_net.net import MLP
from AI_physicist.pytorch_net.util import get_criterion, MAELoss, to_np_array, to_Variable
from AI_physicist.settings.global_param import PrecisionFloorLoss, Dt
from AI_physicist.theory_learning.util_theory import logplus


# ## Important loss functions:

# In[ ]:


class Loss_Fun_Cumu(nn.Module):
    def __init__(
        self,
        core,
        cumu_mode,
        neglect_threshold = None,
        epsilon = 1e-10,
        balance_model_influence = False,
        balance_model_influence_epsilon = 0.03, 
        loss_precision_floor = None,
        ):
        super(Loss_Fun_Cumu, self).__init__()
        self.name = "Loss_Fun_Cumu"
        self.loss_fun = Loss_Fun(core = core, epsilon = epsilon, loss_precision_floor = loss_precision_floor)
        self.cumu_mode = cumu_mode
        self.neglect_threshold = neglect_threshold
        self.epsilon = epsilon
        self.balance_model_influence = balance_model_influence
        self.balance_model_influence_epsilon = balance_model_influence_epsilon
        self.loss_precision_floor = loss_precision_floor

    def forward(
        self,
        pred,
        target,
        model_weights = None,
        sample_weights = None,
        neglect_threshold_on = True,
        is_mean = True,
        cumu_mode = None,
        balance_model_influence = None,
        ):
        num = pred.size(1)
        if num == 1:
            return self.loss_fun(pred, target, sample_weights = sample_weights, is_mean = is_mean)

        if model_weights is not None:
            model_weights = model_weights.float() / model_weights.float().sum(1, keepdim = True)
        loss_list = torch.cat([self.loss_fun(pred[:, i:i+1], target, is_mean = False) for i in range(num)], 1)

        # Modify model_weights according to neglect_threshold if stipulated:
        if neglect_threshold_on:
            neglect_threshold = self.neglect_threshold
        else:
            neglect_threshold = None
        if neglect_threshold is not None:
            valid_candidate = (loss_list <= neglect_threshold).long()
            renew_id = valid_candidate.sum(1) < 1
            valid_weights = valid_candidate.clone().masked_fill_(renew_id.unsqueeze(1), 1).float()
            if model_weights is None:
                model_weights = valid_weights
            else:
                model_weights = model_weights * valid_weights + self.epsilon

        # Setting cumu_mode:
        if cumu_mode is None:
            cumu_mode = self.cumu_mode
        if cumu_mode[0] == "generalized-mean" and cumu_mode[1] == 1:
            cumu_mode = "mean"
        elif cumu_mode[0] == "generalized-mean" and cumu_mode[1] == 0:
            cumu_mode = "geometric"
        elif cumu_mode[0] == "generalized-mean" and cumu_mode[1] == -1:
            cumu_mode = "harmonic"

        # Obtain loss:
        if cumu_mode == "original":
            loss = loss_list
            if model_weights is not None:
                loss = loss * model_weights
        elif cumu_mode == "mean":
            if model_weights is None:
                loss = loss_list.mean(1)
            else:
                loss = (loss_list * model_weights).sum(1)
        elif cumu_mode == "min":
            loss = loss_list.min(1)[0]
        elif cumu_mode == "max":
            loss = loss_list.max(1)[0]
        elif cumu_mode == "harmonic":
            if model_weights is None:
                loss = num / (1 / (loss_list + self.epsilon)).sum(1)
            else:
                loss = 1 / (model_weights / (loss_list + self.epsilon)).sum(1)
        elif cumu_mode == "geometric":
            if model_weights is None:
                loss = (loss_list + self.epsilon).prod(1) ** (1 / float(num))
            else:
                loss = torch.exp((model_weights * torch.log(loss_list + self.epsilon)).sum(1))
        elif cumu_mode[0] == "generalized-mean":
            order = cumu_mode[1]
            if model_weights is None:
                loss = (((loss_list + self.epsilon) ** order).clamp(max = 1e30).mean(1)) ** (1 / float(order))
            else:
                loss = ((model_weights * (loss_list + self.epsilon) ** order).clamp(max = 1e30).sum(1)) ** (1 / float(order))
        elif cumu_mode[0] == "DL-generalized-mean":
            if self.loss_precision_floor is None:
                loss_precision_floor = PrecisionFloorLoss
            else:
                loss_precision_floor = self.loss_precision_floor
            order = cumu_mode[1]
            if model_weights is None:
                loss = logplus((((loss_list + self.epsilon) ** order).clamp(max = 1e30).mean(1)) ** (1 / float(order)) / loss_precision_floor)
            else:
                loss = logplus(((model_weights * (loss_list + self.epsilon) ** order).clamp(max = 1e30).sum(1)) ** (1 / float(order)) / loss_precision_floor)
        else:
            raise Exception("cumu_mode {0} not recognized!".format(cumu_mode))

        # Balance model influence so that each model has the same total weights on the samples:
        if balance_model_influence is None:
            balance_model_influence = self.balance_model_influence
        if model_weights is not None and balance_model_influence is True: 
            num_examples = len(model_weights)
            if sample_weights is None:
                sample_weights = (model_weights.float() / (self.balance_model_influence_epsilon * num_examples + model_weights.float().sum(0, keepdim = True))).sum(1)
            else:
                sample_weights = (sample_weights * model_weights.float() / (self.balance_model_influence_epsilon * num_examples + product.sum(0, keepdim = True))).sum(1)
            sample_weights = sample_weights / sample_weights.sum() * num_examples

        # Multiply by sample weights:
        if sample_weights is not None:
            assert tuple(sample_weights.size()) == tuple(loss.size()), "sample_weights must have the same size as the accumulated loss!"
            loss = loss * sample_weights

        # Calculate the mean:
        if is_mean:
            loss = loss.mean()
        return loss
    
    
def get_Lagrangian_loss_indi(model, X, dt = Dt, force_kinetic_term = False, normalize = True):
    """Obtain individual loss for Euler-Lagrangian Equation
    qx1, qdotx1, qy1, qdoty1, qx2, qdotx2, qy2, qdoty2 are the 8 dimensions of the X.
    """
    assert X.shape[-1] == 8, "The input dimension for the Lagragian must be 8!"
    X = X.detach()
    X.requires_grad = True
    if force_kinetic_term:
        pred1 = model(X[:,:4]) + (X[:,1:2] ** 2 + X[:,3:4] ** 2) / 2
        pred2 = model(X[:,4:]) + (X[:,5:6] ** 2 + X[:,7:8] ** 2) / 2
    else:
        pred1 = model(X[:,:4])
        pred2 = model(X[:,4:])
    pred_X = grad((pred1 + pred2).sum(), X, create_graph = True)[0]
    eq_x = pred_X[:, 5:6] - pred_X[:, 1:2] - dt * (pred_X[:, 4:5] + pred_X[:, 0:1]) / 2
    eq_y = pred_X[:, 7:8] - pred_X[:, 3:4] - dt * (pred_X[:, 6:7] + pred_X[:, 2:3]) / 2
    eq = torch.cat([eq_x, eq_y], 1)
    
    if normalize:
        eq = eq / model(X[:,4:])
    return eq



def get_Lagrangian_loss(model, X, dt = Dt, force_kinetic_term = True):
    """Obtain individual loss for Euler-Lagrangian Equation for either individual model or a model ensemble"""
    if model.__class__.__name__ != "Net_Ensemble":
        return get_Lagrangian_loss_indi(model, X, dt = dt, force_kinetic_term = force_kinetic_term)
    eq_list = []
    for i in range(model.num_models):
        eq = get_Lagrangian_loss(getattr(model, "model_{0}".format(i)), X, dt = dt, force_kinetic_term = force_kinetic_term)
        eq_list.append(eq)
    preds = torch.stack(eq_list, 1)
    return preds


# ## Models:

# In[ ]:


## The Statistics_Net and Generative_Net is a variant of the architecture in 
## Wu, Tailin, et al. "Meta-learning autoencoders for few-shot prediction." arXiv preprint arXiv:1807.09912 (2018).
class Statistics_Net(nn.Module):
    def __init__(self, input_size, pre_pooling_neurons, struct_param_pre, struct_param_post, struct_param_post_logvar = None, pooling = "max", settings = {"activation": "leakyRelu"}, layer_type = "Simple_layer", is_cuda = False):
        super(Statistics_Net, self).__init__()
        self.input_size = input_size
        self.pre_pooling_neurons = pre_pooling_neurons
        self.struct_param_pre = struct_param_pre
        self.struct_param_post = struct_param_post
        self.struct_param_post_logvar = struct_param_post_logvar
        self.pooling = pooling
        self.settings = settings
        self.layer_type = layer_type
        self.is_cuda = is_cuda

        self.encoding_statistics_Net = MLP(input_size = self.input_size, struct_param = self.struct_param_pre, settings = self.settings, is_cuda = is_cuda)
        self.post_pooling_Net = MLP(input_size = self.pre_pooling_neurons, struct_param = self.struct_param_post, settings = self.settings, is_cuda = is_cuda)
        if self.struct_param_post_logvar is not None:
            self.post_pooling_logvar_Net = MLP(input_size = self.pre_pooling_neurons, struct_param = self.struct_param_post_logvar, settings = self.settings, is_cuda = is_cuda)
        if self.is_cuda:
            self.cuda()

    @property
    def model_dict(self):
        model_dict = {"type": "Statistics_Net"}
        model_dict["input_size"] = self.input_size
        model_dict["pre_pooling_neurons"] = self.pre_pooling_neurons
        model_dict["struct_param_pre"] = self.struct_param_pre
        model_dict["struct_param_post"] = self.struct_param_post
        model_dict["struct_param_post_logvar"] = self.struct_param_post_logvar
        model_dict["pooling"] = self.pooling
        model_dict["settings"] = self.settings
        model_dict["layer_type"] = self.layer_type
        model_dict["encoding_statistics_Net"] = self.encoding_statistics_Net.model_dict
        model_dict["post_pooling_Net"] = self.post_pooling_Net.model_dict
        if self.struct_param_post_logvar is not None:
            model_dict["post_pooling_logvar_Net"] = self.post_pooling_logvar_Net.model_dict
        return model_dict
    
    def load_model_dict(self, model_dict):
        new_net = load_model_dict(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_net.__dict__)

    def forward(self, input):
        encoding = self.encoding_statistics_Net(input)
        if self.pooling == "mean":
            pooled = encoding.mean(0)
        elif self.pooling == "max":
            pooled = encoding.max(0)[0]
        else:
            raise Exception("pooling {0} not recognized!".format(self.pooling))
        output = self.post_pooling_Net(pooled.unsqueeze(0))
        if self.struct_param_post_logvar is None:
            return output
        else:
            logvar = self.post_pooling_logvar_Net(pooled.unsqueeze(0))
            return output, logvar
    
    def forward_inputs(self, X, y):
        return self(torch.cat([X, y], 1))
    

    def get_regularization(self, source = ["weight", "bias"], mode = "L1"):
        reg = self.encoding_statistics_Net.get_regularization(source = source, mode = mode) +               self.post_pooling_Net.get_regularization(source = source, mode = mode)
        if self.struct_param_post_logvar is not None:
            reg = reg + self.post_pooling_logvar_Net.get_regularization(source = source, mode = mode)
        return reg


class Generative_Net(nn.Module):
    def __init__(
        self, 
        input_size,
        W_struct_param_list,
        b_struct_param_list, 
        num_context_neurons = 0, 
        settings_generative = {"activation": "leakyRelu"}, 
        settings_model = {"activation": "leakyRelu"}, 
        learnable_latent_param = False,
        last_layer_linear = True,
        is_cuda = False,
        ):
        super(Generative_Net, self).__init__()
        assert(len(W_struct_param_list) == len(b_struct_param_list))
        self.input_size = input_size
        self.W_struct_param_list = W_struct_param_list
        self.b_struct_param_list = b_struct_param_list
        self.num_context_neurons = num_context_neurons
        self.settings_generative = settings_generative
        self.settings_model = settings_model
        self.learnable_latent_param = learnable_latent_param
        self.last_layer_linear = last_layer_linear
        self.is_cuda = is_cuda

        for i, W_struct_param in enumerate(self.W_struct_param_list):
            setattr(self, "W_gen_{0}".format(i), MLP(input_size = self.input_size + num_context_neurons, struct_param = W_struct_param, settings = self.settings_generative, is_cuda = is_cuda))
            setattr(self, "b_gen_{0}".format(i), MLP(input_size = self.input_size + num_context_neurons, struct_param = self.b_struct_param_list[i], settings = self.settings_generative, is_cuda = is_cuda))
        # Setting up latent param and context param:
        self.latent_param = nn.Parameter(torch.randn(1, self.input_size)) if learnable_latent_param else None
        if self.num_context_neurons > 0:
            self.context = nn.Parameter(torch.randn(1, self.num_context_neurons))
        if self.is_cuda:
            self.cuda()

    @property
    def model_dict(self):
        model_dict = {"type": "Generative_Net"}
        model_dict["input_size"] = self.input_size
        model_dict["W_struct_param_list"] = self.W_struct_param_list
        model_dict["b_struct_param_list"] = self.b_struct_param_list
        model_dict["num_context_neurons"] = self.num_context_neurons
        model_dict["settings_generative"] = self.settings_generative
        model_dict["settings_model"] = self.settings_model
        model_dict["learnable_latent_param"] = self.learnable_latent_param
        model_dict["last_layer_linear"] = self.last_layer_linear
        for i, W_struct_param in enumerate(self.W_struct_param_list):
            model_dict["W_gen_{0}".format(i)] = getattr(self, "W_gen_{0}".format(i)).model_dict
            model_dict["b_gen_{0}".format(i)] = getattr(self, "b_gen_{0}".format(i)).model_dict
        if self.latent_param is None:
            model_dict["latent_param"] = None
        else:
            model_dict["latent_param"] = self.latent_param.cpu().data.numpy() if self.is_cuda else self.latent_param.data.numpy()
        if hasattr(self, "context"):
            model_dict["context"] = self.context.data.numpy() if not self.is_cuda else self.context.cpu().data.numpy()
        return model_dict
    
    def set_latent_param_learnable(self, mode):
        if mode == "on":
            if not self.learnable_latent_param:
                self.learnable_latent_param = True
                if self.latent_param is None:
                    self.latent_param = nn.Parameter(torch.randn(1, self.input_size))
                else:
                    self.latent_param = nn.Parameter(self.latent_param.data)
            else:
                assert isinstance(self.latent_param, nn.Parameter)
        elif mode == "off":
            if self.learnable_latent_param:
                assert isinstance(self.latent_param, nn.Parameter)
                self.learnable_latent_param = False
                self.latent_param = Variable(self.latent_param.data, requires_grad = False)
            else:
                assert isinstance(self.latent_param, Variable) or self.latent_param is None
        else:
            raise

    def load_model_dict(self, model_dict):
        new_net = load_model_dict(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_net.__dict__)

    def init_weights_bias(self, latent_param):
        if self.num_context_neurons > 0:
            latent_param = torch.cat([latent_param, self.context], 1)
        for i in range(len(self.W_struct_param_list)):
            setattr(self, "W_{0}".format(i), (getattr(self, "W_gen_{0}".format(i))(latent_param)).squeeze(0))
            setattr(self, "b_{0}".format(i), getattr(self, "b_gen_{0}".format(i))(latent_param))       

    def get_weights_bias(self, W_source = None, b_source = None, isplot = False, latent_param = None):
        if latent_param is not None:
            self.init_weights_bias(latent_param)
        W_list = []
        b_list = []
        if W_source is not None:
            for k in range(len(self.W_struct_param_list)):
                if W_source == "core":
                    W = getattr(self, "W_{0}".format(k)).data.numpy()
                else:
                    raise Exception("W_source '{0}' not recognized!".format(W_source))
                W_list.append(W)
        if b_source is not None:
            for k in range(len(self.b_struct_param_list)):
                if b_source == "core":
                    b = getattr(self, "b_{0}".format(k)).data.numpy()
                else:
                    raise Exception("b_source '{0}' not recognized!".format(b_source))
                b_list.append(b)
        if isplot:
            if W_source is not None:
                print("weight {0}:".format(W_source))
                plot_matrices(W_list)
            if b_source is not None:
                print("bias {0}:".format(b_source))
                plot_matrices(b_list)
        return W_list, b_list

    
    def set_latent_param(self, latent_param):
        assert isinstance(latent_param, Variable), "The latent_param must be a Variable!"
        if self.learnable_latent_param:
            self.latent_param.data.copy_(latent_param.data)
        else:
            self.latent_param = latent_param
    
    
    def latent_param_quick_learn(self, X, y, validation_data, loss_core = "huber", epochs = 10, batch_size = 128, lr = 1e-2, optim_type = "LBFGS"):
        assert self.learnable_latent_param is True, "To quick-learn latent_param, you must set learnable_latent_param as True!"
        self.latent_param_optimizer = get_optimizer(optim_type = optim_type, lr = lr, parameters = [self.latent_param])
        self.criterion = get_criterion(loss_core)
        loss_list = []
        X_test, y_test = validation_data
        batch_size = min(batch_size, len(X))
        if isinstance(X, Variable):
            X = X.data
        if isinstance(y, Variable):
            y = y.data

        dataset_train = data_utils.TensorDataset(X, y)
        train_loader = data_utils.DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
        
        y_pred_test = self(X_test)
        loss = get_criterion("mse")(y_pred_test, y_test)
        loss_list.append(loss.data[0])
        for i in range(epochs):
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch = Variable(X_batch)
                y_batch = Variable(y_batch)
                if optim_type == "LBFGS":
                    def closure():
                        self.latent_param_optimizer.zero_grad()
                        y_pred = self(X_batch)
                        loss = self.criterion(y_pred, y_batch)
                        loss.backward()
                        return loss
                    self.latent_param_optimizer.step(closure)
                else:
                    self.latent_param_optimizer.zero_grad()
                    y_pred = self(X_batch)
                    loss = self.criterion(y_pred, y_batch)
                    loss.backward()
                    self.latent_param_optimizer.step()
            y_pred_test = self(X_test)
            loss = get_criterion("mse")(y_pred_test, y_test)
            loss_list.append(loss.data[0])
        loss_list = np.array(loss_list)
        return loss_list


    def forward(self, input, latent_param = None):
        if latent_param is None:
            latent_param = self.latent_param
        self.init_weights_bias(latent_param)
        output = input
        for i in range(len(self.W_struct_param_list)):
            output = torch.matmul(output, getattr(self, "W_{0}".format(i))) + getattr(self, "b_{0}".format(i))
            if i == len(self.W_struct_param_list) - 1 and hasattr(self, "last_layer_linear") and self.last_layer_linear:
                activation = "linear"
            else:
                activation = self.settings_model["activation"] if "activation" in self.settings_model else "leakyRelu"
            output = get_activation(activation)(output)
        return output


    def get_regularization(self, source = ["weight", "bias"], mode = "L1"):
        reg = Variable(torch.FloatTensor(np.array([0])), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        for reg_type in source:
            if reg_type == "weight":
                for i in range(len(self.W_struct_param_list)):
                    if mode == "L1":
                        reg = reg + getattr(self, "W_{0}".format(i)).abs().sum()
                    else:
                        raise
            elif reg_type == "bias":
                for i in range(len(self.W_struct_param_list)):
                    if mode == "L1":
                        reg = reg + getattr(self, "b_{0}".format(i)).abs().sum()
                    else:
                        raise
            elif reg_type == "W_gen":
                for i in range(len(self.W_struct_param_list)):
                    reg = reg + getattr(self, "W_gen_{0}".format(i)).get_regularization(source = source, mode = mode)
            elif reg_type == "b_gen":
                for i in range(len(self.W_struct_param_list)):
                    reg = reg + getattr(self, "b_gen_{0}".format(i)).get_regularization(source = source, mode = mode)
            else:
                raise Exception("source {0} not recognized!".format(reg_type))
        return reg

