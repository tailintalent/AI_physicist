
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from copy import deepcopy
import itertools
from collections import OrderedDict
import datetime
import pandas as pd
import pprint as pp
import scipy
from sklearn.cluster import KMeans
import sympy
from sympy import Symbol

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_physicist.theory_learning.theory_model import Theory_Training, get_loss, load_info_dict, get_best_model_idx
from AI_physicist.theory_learning.models import Loss_Fun_Cumu, get_Lagrangian_loss
from AI_physicist.theory_learning.util_theory import plot_theories, plot3D, plot_indi_domain, to_one_hot, get_piecewise_dataset
from AI_physicist.theory_learning.models import Statistics_Net, Generative_Net
from AI_physicist.settings.filepath import theory_PATH
from AI_physicist.pytorch_net.util import Loss_Fun, Loss_with_uncertainty, Batch_Generator, get_criterion, to_np_array, make_dir, Early_Stopping
from AI_physicist.pytorch_net.util import record_data, plot_matrices, get_args, sort_two_lists, get_param_name_list
from AI_physicist.pytorch_net.net import MLP, Model_Ensemble, combine_model_ensembles, load_model_dict, load_model_dict_net, load_model_dict_model_ensemble, construct_model_ensemble_from_nets


# ## Helper functions for symbolic unification:

# In[1]:


def unification_symbolic(
    theory_collection,
    num_clusters,
    fraction_threshold=1,
    relative_diff_threshold=0,
    verbose=True,
    ):
    """Unification of symbolic theories, implementing the Alg. 4 in (Wu and Tegmark, 2019)."""
    df_dict_list = []
    skeleton_dict = {}
    for key, theory in theory_collection.items():
        # Record all different kinds of skeletons
        if theory.pred_net.layer_0.__class__.__name__ == "Symbolic_Layer": # Only unify models with Symbolic_Layer
            df_dict = theory.pred_net.get_sympy_expression(verbose = False)[0]
            if df_dict is None:
                print("{0} is not a symbolic net!".format(key))
                continue
            # Canonicalize (each expression is already in tree-form in sympy), so only have to generate the skeleton:
            skeleton_list = []
            param_tree_list = []
            for expression in df_dict["numerical_expression"]:
                skeleton, param_tree = get_skeleton(expression)
                skeleton_list.append(skeleton)
                param_tree_list.append(param_tree)
            df_dict["skeleton"] = skeleton_list
            df_dict["param_tree"] = param_tree_list

            # Assigning each structure of skeleton a different ID, for future calculation of the mode skeleton in a subset:
            is_in = False
            for skeleton_key, item in skeleton_dict.items():
                if skeleton_list == item:
                    df_dict["skeleton_id"] = skeleton_key
                    is_in = True
                    break
            if not is_in:
                df_dict["skeleton_id"] = len(skeleton_dict)
                skeleton_dict[len(skeleton_dict)] = skeleton_list

            # Other information:
            df_dict["theory_name"] = key
            df_dict["mse_train"], df_dict["mse_test"] = theory.get_loss()
            df_dict_list.append(df_dict)
    df = pd.DataFrame(df_dict_list).sort_values(by = "DL", ascending = True)
    df = df.rename(columns = {"DL": "pred_net_DL"})

    # Cluster the expressions according to DL:
    DLs = np.expand_dims(df["pred_net_DL"].values, 1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(DLs)
    df["cluster_id"] = kmeans.labels_
    df_sympy = df.reset_index()[["theory_name", "pred_net_DL", "cluster_id", "numerical_expression", "symbolic_expression", "skeleton", "param_tree", "skeleton_id", "mse_train", "mse_test", "param_dict"]]

    exprs_unified_list = []
    for cluster_id in range(num_clusters):
        # Find the mode of skeleton for each cluster:
        df_cluster = df[df["cluster_id"] == cluster_id]
        skeleton_mode = df_cluster["skeleton_id"].mode()
        if len(skeleton_mode) > 1:
            print("there are more than one modes in cluster {}".format(cluster_id))
        df_cluster_mode = df_cluster[df_cluster["skeleton_id"] == skeleton_mode[0]]
        param_tree_list = list(df_cluster_mode["param_tree"].values)
        # Jointly traverse all the numerical expressions with the same skeleton, replacing different
        # numerical values with a new parameter:
        param_unification, param_list = joint_traverse(param_tree_list,
                                                       param_list=[],
                                                       fraction_threshold=fraction_threshold,
                                                       relative_diff_threshold=relative_diff_threshold,
                                                      )
        exprs_unified = []
        for k in range(len(df_cluster_mode.iloc[0]["skeleton"])):
            is_valid, expr = assign_param_to_skeleton(df_cluster_mode.iloc[0]["skeleton"][k], param_unification[k])
            assert is_valid, "The unification is not valid. Check code!"
            exprs_unified.append(expr)
        exprs_unified_list.append(exprs_unified)

    if verbose:
        # Printing:
        print("Unified prediction functions:")
        pp.pprint(exprs_unified_list)
        print("\nTable of symbolic prediction functions:")
        display(df)
    return df_sympy, exprs_unified_list



def get_skeleton(expr):
    """Recursively obtain the skeleton and corresponding parameter-tree (in terms of lists of lists) of a symbolic expression, 
       where the skeleton replaces all numerical values by a symbol 'p', and the param_tree records the corresponding numerical parameter.
    """
    args = []
    param_tree = []
    for arg in expr.args:
        if isinstance(arg, sympy.numbers.Float) or isinstance(arg, sympy.numbers.Integer):
            arg_store = Symbol('p')
            param_value = arg
        elif isinstance(arg, sympy.symbol.Symbol):
            arg_store = arg
            param_value = None
        else:
            # Recursion:
            arg_store, param_value = get_skeleton(arg)
        args.append(arg_store)
        param_tree.append(param_value)
    skeleton = expr.func(*args)
    
    # From all the permutations on the branch of param_tree, find the one that has the same order as the skeleton:
    is_match = False
    for param_tree_permute in itertools.permutations(deepcopy(param_tree)):
        is_valid, assigned_expr = assign_param_to_skeleton(skeleton, param_tree_permute)
        if is_valid and assigned_expr == expr:
            is_match = True
            break
    if not is_match:
        raise Exception("Matching param_tree is not found!")
    param_tree = list(param_tree_permute)
    return skeleton, param_tree


def joint_traverse(exprs, param_list, fraction_threshold=1, relative_diff_threshold=0, pivot_type="mode"):
    """Recursively jointly traverse all the exprs with the same skeleton, replacing different numerical values with
       a unification parameter "p{}".
    """
    # Base case:
    if exprs[0] is None:
        return exprs[0], param_list
    
    elif isinstance(exprs[0], sympy.numbers.Float) or isinstance(exprs[0], sympy.numbers.Integer):
        is_same = check_same_number(exprs, 
                                    fraction_threshold=fraction_threshold,
                                    relative_diff_threshold=relative_diff_threshold,
                                    pivot_type=pivot_type,
                                   )
        # If not the same number, replace by a parameter:
        if not is_same:
            new_param = Symbol('p{}'.format(len(param_list)))
            param_list.append(new_param)
            return new_param, param_list
        # Otherwise keep this same number:
        else:
            return exprs[0], param_list

    # If it is a single symbol, keep this symbol:
    elif isinstance(exprs[0], sympy.symbol.Symbol):
        is_same = check_same_exact(exprs)
        if not is_same:
            print("The symbols at the same position are not the same!")
        return exprs[0], param_list
    
    # Obtain the args_same_pos_list, i.e. the list of args_same_pos which are at the same position for all exprs:
    num_args = len(exprs[0])
    args_same_pos_list = [[] for _ in range(num_args)]
    for args in exprs:
        for i in range(num_args):
            args_same_pos_list[i].append(args[i])
    
    # jointly traverse each individual arg_same_pos:
    param_uni_list = []
    for args_same_pos in args_same_pos_list:
        param_uni, param_list = joint_traverse(args_same_pos,
                                               param_list,
                                               fraction_threshold=fraction_threshold,
                                               relative_diff_threshold=relative_diff_threshold,
                                               pivot_type=pivot_type,
                                              )
        param_uni_list.append(param_uni)
    return param_uni_list, param_list


def assign_param_to_skeleton(skeleton, param_tree):
    """Recursively assign param_tree to skeleton"""
    from numbers import Number
    if isinstance(skeleton, sympy.symbol.Symbol):
        if skeleton == Symbol("p"):
            if isinstance(param_tree, sympy.numbers.Float) or isinstance(param_tree, sympy.numbers.Integer) or isinstance(param_tree, Number):
                # Assigning a numerical value to "p":
                is_valid = True
                return is_valid, param_tree
            elif isinstance(param_tree, sympy.symbol.Symbol) and param_tree.name.startswith("p"):
                # Assigning a symbol starting with "p" (e.g. "p1", "p2") to "p":
                is_valid = True
                return is_valid, param_tree
            else:
                # Otherwise the assignment is not valid:
                is_valid = False
                return is_valid, skeleton
        else:
            if param_tree is None:
                is_valid = True
                return is_valid, skeleton
            else:
                is_valid = False
                return is_valid, skeleton
    elif isinstance(skeleton, sympy.numbers.Float) or isinstance(skeleton, sympy.numbers.Integer):
        raise Exception("Skeleton cannot have numerical parameters!")
    else:
        if not isinstance(param_tree, list) and not isinstance(param_tree, tuple):
            return False, skeleton
        if len(skeleton.args) != len(param_tree):
            return False, skeleton
        is_valid = True
        sub_skeleton_list = []
        for sub_skeleton, param in zip(skeleton.args, param_tree):
            is_valid_sub, sub_skeleton_assigned = assign_param_to_skeleton(sub_skeleton, param)
            is_valid = is_valid and is_valid_sub
            sub_skeleton_list.append(sub_skeleton_assigned)
        return is_valid, skeleton.func(*sub_skeleton_list)


def check_same_exact(args):
    """Check if all the expressions in args has the same numerical value or symbol"""
    number = args[0]
    is_same = True
    for arg in args:
        if arg != number:
            is_same = False
            break
    return is_same


def check_same_number(args, fraction_threshold=1, relative_diff_threshold=0, pivot_type="mode"):
    """Check if all the expressions in args_same_pos has the same numerical value, where the fraction of numbers
    that is within relative_diff_threshold with the pivot_type (choose from 'mode' or 'mean') is above the fraction_threshold.
    The function with default parameters has the same behavior as check_same_exact().
    """
    args_np = []
    for arg in args:
        if isinstance(arg, sympy.numbers.Float):
            args_np.append(float(arg))
        elif isinstance(arg, sympy.numbers.Integer):
            args_np.append(int(arg))
        else:
            args_np.append(arg)
    args_np = np.array(args_np)
    if pivot_type == "mean":
        pivot = np.mean(args)
    elif pivot_type == "mode":
        pivot = scipy.stats.mode(args)[0][0]
    else:
        raise
    
    count = (np.abs((args_np - pivot) / pivot) <= relative_diff_threshold).sum()
    if count / len(args) >= fraction_threshold:
        is_same = True
    else:
        is_same = False
    return is_same


# ## Classes for theory, master_theory and theory_hub:

# In[2]:


class Theory_Tuple(object):
    """A theory_tuple contains the individual prediction function (pred_net, either in neural network or symbolic format), 
       its corresponding domain_net (a subclassifier that only takes one of the logit of the full domain classifier),
       and the dataset it is based on.
    """
    def __init__(self, pred_net, domain_net, dataset, is_Lagrangian = False, is_cuda = False):
        if isinstance(pred_net, dict):
            pred_net = load_model_dict_net(pred_net, is_cuda = is_cuda)
        if isinstance(domain_net, dict):
            domain_net = load_model_dict_net(domain_net, is_cuda = is_cuda)
        assert pred_net.__class__.__name__ == "MLP"
        assert domain_net.__class__.__name__ == "MLP"
        self.pred_net = pred_net
        self.domain_net = domain_net
        self.dataset = dataset
        self.is_Lagrangian = is_Lagrangian
        self.is_cuda = is_cuda

    @property
    def model_dict(self):
        model_dict = {"type": "Theory_Tuple"}
        model_dict["pred_net"] = deepcopy(self.pred_net.model_dict)
        model_dict["domain_net"] = deepcopy(self.domain_net.model_dict)
        model_dict["dataset"] = self.dataset
        model_dict["is_Lagrangian"] = self.is_Lagrangian
        return model_dict
    
    def load_model_dict(self, model_dict):
        new_theory_tuple = load_model_dict_theory_tuple(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_theory_tuple.__dict__)
    
    def get_loss(self):
        ((X_train, y_train), (X_test, y_test), _), _ = self.dataset
        if len(X_train.size()) == 0:
            mse_train = None
        else:
            if self.is_Lagrangian:
                mse_train = to_np_array(get_criterion("mse")(get_Lagrangian_loss(self.pred_net, X_train), y_train))
            else:
                mse_train = to_np_array(get_criterion("mse")(self.pred_net(X_train), y_train))
        if len(X_test.size()) == 0:
            mse_test = None
        else:
            if self.is_Lagrangian:
                mse_test = to_np_array(get_criterion("mse")(get_Lagrangian_loss(self.pred_net, X_test), y_test))
            else:
                mse_test = to_np_array(get_criterion("mse")(self.pred_net(X_test), y_test))
        return mse_train, mse_test
    
    def plot(self, is_train = True, **kwargs):
        ((X_train, y_train), (X_test, y_test), _), _ = self.dataset
        X = X_train if is_train else X_test
        pred = self.pred_net(X)
        plot3D(X, pred, figsize = (6,6))
        plot_indi_domain(X, torch.zeros(len(X)).long(), images_per_row = 2, row_width = 12, row_height = 5, **kwargs)

    def simplify(self, mode, **kwargs):
        ((X_train, y_train), _, _), _ = self.dataset
        self.pred_net.simplify(X_train, y_train, mode = mode, **kwargs)


class Master_Theory_Tuple(object):
    """A mster_theory_tuple contains the master theory and the theory_tuples it is based on."""
    def __init__(self, master_theory, theory_tuples, is_cuda = False):
        assert isinstance(theory_tuples, dict)
        self.master_theory = master_theory
        self.theory_tuples = theory_tuples
        self.is_cuda = is_cuda

    @property
    def model_dict(self):
        model_dict = {"type": "Master_Theory_Tuple"}
        model_dict["master_theory"] = deepcopy(self.master_theory.model_dict)
        model_dict["theory_tuples"] = deepcopy({name: theory_tuple.model_dict for name, theory_tuple in self.theory_tuples.items()})
        return model_dict
    
    def load_model_dict(self, model_dict):
        new_master_theory_tuple = load_model_dict_master_theory_tuple(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_master_theory_tuple.__dict__)



class Master_Theory(nn.Module):
    """Master_theory can generate a continuum of theories."""
    def __init__(
        self,
        input_size,
        output_size,
        pre_pooling_neurons,
        struct_param_statistics_Net,
        struct_param_classifier,
        pooling = "max",
        settings_statistics_Net = {"activation": "leakyRelu"},
        settings_classifier = {"activation": "leakyRelu"},
        is_cuda = False,
        ):
        super(Master_Theory, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.statistics_Net = Statistics_Net(input_size = input_size + output_size,
                                             pre_pooling_neurons = pre_pooling_neurons,
                                             struct_param_pre = struct_param_statistics_Net[0],
                                             struct_param_post = struct_param_statistics_Net[1],
                                             pooling = pooling,
                                             settings = settings_statistics_Net,
                                             is_cuda = is_cuda,
                                            )
        self.classifier = MLP(input_size = input_size,
                              struct_param = struct_param_classifier,
                              settings = settings_classifier,
                              is_cuda = is_cuda,
                             )
        self.latent_param = None
        self.is_cuda = is_cuda
        if self.is_cuda:
            self.cuda()


    @property
    def model_dict(self):
        model_dict = {"type": "Master_Theory"}
        model_dict["input_size"] = self.input_size
        model_dict["master_model"] = self.master_model.model_dict if hasattr(self, "master_model") else None
        model_dict["master_model_type"] = self.master_model_type if hasattr(self, "master_model_type") else None
        model_dict["statistics_Net"] = self.statistics_Net.model_dict
        model_dict["classifier"] = self.classifier.model_dict
        model_dict["latent_param"] = self.latent_param
        return model_dict


    def load_model_dict(self, model_dict):
        new_master_theory = load_model_dict_master_theory(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_master_theory.__dict__)


    def forward(self, input, latent_param = None, X = None, y = None):
        if X is not None or y is not None:
            assert latent_param is None
            latent_param = self.calculate_latent_param(X, y)
        elif latent_param is None:
            latent_param = self.latent_param
        if self.master_model_type == "symbolic":
            latent_param = {0: {param_name: latent_param[0, k] for k, param_name in enumerate(self.param_name_list)}}
        elif self.master_model_type == "regulated-Net":
            latent_param = get_regulated_latent_param(self.master_model, latent_param)
        return self.master_model(input, latent_param)


    def get_regularization(self, targets, source = ["weight", "bias"], mode = "L1"):
        reg = Variable(torch.FloatTensor(np.array([0])))
        if self.is_cuda:
            reg = reg.cuda()
        for target in targets:
            if target == "master_model":
                reg = reg + self.master_model.get_regularization(source = source, mode = mode)
            elif target == "statistics_Net":
                reg = reg + self.statistics_Net.get_regularization(source = source, mode = mode)
            elif target == "classifier":
                reg = reg + self.classifier.get_regularization(source = source, mode = mode)
            else:
                raise Exception("target {0} not recognized!".format(target))
        return reg


    def propose_master_model(
        self,
        theory_collection,
        input_size,
        statistics_output_neurons,
        master_model_type="regulated-Net",
        symbolic_expression=None,
        **kwargs
        ):
        """Propose the master model (the master prediction function that unifies multiple prediction functions in theories)
        based on a collection of theories.
        """
        self.master_model_type = master_model_type
        
        if master_model_type == "regulated-Net":
            struct_param_regulated_Net = kwargs["struct_param_regulated_Net"] if "struct_param_regulated_Net" in kwargs else [
                    [8, "Simple_Layer", {}],
                    [8, "Simple_Layer", {}],
                    [self.output_size, "Simple_Layer", {"activation": "linear"}],
            ]
            assert len(struct_param_regulated_Net) * 2 == statistics_output_neurons or len(struct_param_regulated_Net) == statistics_output_neurons
            activation_regulated_Net = kwargs["activation_regulated_Net"] if "activation_regulated_Net" in kwargs else "leakyRelu"
            self.master_model = MLP(input_size = input_size,
                                    struct_param = struct_param_regulated_Net,
                                    settings = {"activation": activation_regulated_Net},
                                    is_cuda = self.is_cuda,
                                   )            
        elif master_model_type == "symbolic":
            model_dict = {
                "type": "MLP",
                "input_size": input_size,
                "struct_param": [[len(symbolic_expression), "Symbolic_Layer", {"symbolic_expression": str(symbolic_expression)}]]
            }
            self.param_name_list = get_param_name_list(symbolic_expression)
            self.master_model = load_model_dict(model_dict, is_cuda=self.is_cuda)
        elif master_model_type == "generative_Net":
            struct_param_gen_base = kwargs["struct_param_gen_base"] if "struct_param_gen_base" in kwargs else [
                    [60, "Simple_Layer", {}],
                    [60, "Simple_Layer", {}],
                    [60, "Simple_Layer", {}],
            ]
            activation_generative = kwargs["activation_generative"] if "activation_generative" in kwargs else "leakyRelu"
            activation_model = kwargs["activation_model"] if "activation_model" in kwargs else "leakyRelu"
            num_context_neurons = kwargs["num_context_neurons"] if "num_context_neurons" in kwargs else 0
            layer_type = "Simple_Layer"

            struct_param_weight1 = struct_param_gen_base + [[(input_size, 20), layer_type, {"activation": "linear"}]]
            struct_param_weight2 = struct_param_gen_base + [[(20, 20), layer_type, {"activation": "linear"}]]
            struct_param_weight3 = struct_param_gen_base + [[(20, self.output_size), layer_type, {"activation": "linear"}]]
            struct_param_bias1 = struct_param_gen_base + [[20, layer_type, {"activation": "linear"}]]
            struct_param_bias2 = struct_param_gen_base + [[20, layer_type, {"activation": "linear"}]]
            struct_param_bias3 = struct_param_gen_base + [[self.output_size, layer_type,  {"activation": "linear"}]]

            self.master_model = Generative_Net(input_size = statistics_output_neurons,
                                                num_context_neurons = num_context_neurons,
                                                W_struct_param_list = [struct_param_weight1, struct_param_weight2, struct_param_weight3],
                                                b_struct_param_list = [struct_param_bias1, struct_param_bias2, struct_param_bias3],
                                                settings_generative = {"activation": activation_generative},
                                                settings_model = {"activation": activation_model},
                                                learnable_latent_param = False,
                                                is_cuda = self.is_cuda,
                                               )
        else:
            raise Exception("mode {0} not recognized!".format(master_model_type))
        return self.master_model
    

    def get_parameters(self, targets):
        params = []
        for target in targets:
            if target == "master_model":
                params.append(self.master_model.parameters())
            elif target == "statistics_Net":
                params.append(self.statistics_Net.parameters())
            elif target == "classifier":
                params.append(self.classifier.parameters())
            elif target == "latent_param":
                if self.latent_param is not None:
                    params.append([self.latent_param])
            else:
                raise Exception("target {0} not recognized!".format(target))
        return itertools.chain(*params)


    def set_latent_param(self, latent_param):
        assert isinstance(latent_param, Variable), "The latent_param must be a Variable!"
        if self.latent_param is not None:
            self.latent_param.data.copy_(latent_param.data)
        else:
            self.latent_param = nn.Parameter(latent_param.data)


    def calculate_latent_param(self, X, y):
        return self.statistics_Net(torch.cat([X, y], 1))


    def propose_theory_model_from_latent_param(self, latent_param):
        if self.master_model_type == "symbolic":
            latent_param = {0: {param_name: latent_param[0, k] for k, param_name in enumerate(self.param_name_list)}}
        elif self.master_model_type == "regulated-Net":
            latent_param = get_regulated_latent_param(self.master_model, latent_param)
        theory_model = deepcopy(self.master_model)
        theory_model.init_with_p_dict(latent_param)
        return theory_model


    def propose_theory_model_from_data(self, X, y):
        # Select data and predict the latent variable:
        latent_param_list = []
        thresholds = np.linspace(0.99, 0.5, 99)
        minimum_positive = max(len(y) * 0.005, 100)
        count = 0
        for threshold in thresholds:
            u_pred = (nn.Softmax(dim = 1)(self.classifier(X))[:, 1] > threshold).long()
            if to_np_array(u_pred.sum()) > minimum_positive:
                X_chosen = torch.masked_select(X, u_pred.byte().unsqueeze(1).detach()).view(-1, X.size(1))
                y_chosen = torch.masked_select(y, u_pred.byte().unsqueeze(1).detach()).view(-1, y.size(1))
                latent_param = self.calculate_latent_param(X_chosen, y_chosen)
                latent_param_list.append(latent_param)
                count += 1
                if count > 2:
                    break
        if count > 0:
            latent_param_mean = torch.stack(latent_param_list, -1).mean(-1)

            # Propose theory_model:
            theory_model = self.propose_theory_model_from_latent_param(latent_param_mean)
            return theory_model
        else:
            return None


class Theory_Hub(object):
    """The theory_hub stores theories and master theories, and contains methods for symbolic unification,
    adding theories to hub, proposing new theories based on new data, and propose network master theories 
    based on a collection of theories.
    """
    def __init__(self, is_cuda = False):
        self.theory_collection = OrderedDict()
        self.master_theory_collection = OrderedDict()
        self.is_cuda = is_cuda

    @property
    def model_dict(self):
        model_dict = {"type": "Theory_Hub", "theory_collection": OrderedDict(), "master_theory_collection": OrderedDict()}
        for name, theory_tuple in self.theory_collection.items():
            model_dict["theory_collection"][name] = theory_tuple.model_dict
        for name, master_theory_tuple in self.master_theory_collection.items():
            model_dict["master_theory_collection"][name] = master_theory_tuple.model_dict
        return model_dict

    def load_model_dict(self, model_dict):
        new_theory_hub = load_model_dict_theory_hub(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_theory_hub.__dict__)

    @property
    def theory(self):
        return self.theory_collection

    @property
    def master_theory(self):
        return self.master_theory_collection


    def add_theories(
        self,
        name, 
        pred_nets,
        domain_net,
        dataset,
        verbose = True,
        threshold = 1e-3,
        is_Lagrangian = False,
        ):
        if "{0}_0".format(name) in self.theory_collection:
            print("Name {0} collision! Please change to a different name!".format("{0}_0".format(name)))
            added_theory_info = {}
        else:
            added_theory_info = {}
            if domain_net.__class__.__name__ == "MLP":
                domain_net = domain_net.split_to_model_ensemble(mode = "standardize")
            else:
                domain_net.standardize(mode = "b_mean_zero")
            num_models = domain_net.num_models

            assert pred_nets.num_models == num_models, "pred_nets must have the same num_models as the domain_net!"
            ((X_train, y_train), (X_test, y_test), (reflect_train, reflect_test)), info = dataset
            valid_train = to_one_hot(domain_net(X_train).max(1)[1], num_models).byte()
            valid_test = to_one_hot(domain_net(X_test).max(1)[1], num_models).byte()
            dataset_all = {}

            for i in range(num_models):
                theory_name = "{0}_{1}".format(name, i)
                X_train_split = torch.masked_select(X_train, valid_train[:,i:i+1]).view(-1, X_train.size(1))
                y_train_split = torch.masked_select(y_train, valid_train[:,i:i+1]).view(-1, y_train.size(1))
                if reflect_train is not None:
                    reflect_train_split = torch.masked_select(reflect_train, valid_train[:,i])
                else:
                    reflect_train_split = None

                X_test_split = torch.masked_select(X_test, valid_test[:,i:i+1]).view(-1, X_test.size(1))
                y_test_split = torch.masked_select(y_test, valid_test[:,i:i+1]).view(-1, y_test.size(1))
                if reflect_test is not None:
                    reflect_test_split = torch.masked_select(reflect_test, valid_test[:,i])
                else:
                    reflect_test_split = None

                dataset_split = (((X_train_split, y_train_split), (X_test_split, y_test_split), (reflect_train_split, reflect_test_split)), deepcopy(info))
                theory_tuple = Theory_Tuple(pred_net = getattr(pred_nets, "model_{0}".format(i)),
                                            domain_net = getattr(domain_net, "model_{0}".format(i)),
                                            dataset = dataset_split,
                                            is_Lagrangian = is_Lagrangian,
                                            is_cuda = self.is_cuda,
                                           )
                mse_train, mse_test = theory_tuple.get_loss()
                if mse_train is None or mse_test is None:
                    if verbose:
                        print("theory {0} NOT added because its domain_net does not classify any data for the model.".format(theory_name))
                elif mse_train > threshold or mse_test > threshold:
                    if verbose:
                        print("theory {0} NOT added! mse_train = {1:.9}\tmse_test = {2:.9}".format(theory_name, mse_train, mse_test))
                else:
                    self.theory_collection[theory_name] = theory_tuple
                    added_theory_info[theory_name] = {"mse_train": mse_train, "mse_test": mse_test}
                    if verbose:
                        print("theory {0} added! mse_train = {1:.9}\tmse_test = {2:.9}".format(theory_name, mse_train, mse_test))    
        return added_theory_info


    def add_theories_from_info_dict(
        self,
        name,
        info_dict,
        pred_nets_target = "pred_nets_simplified",
        domain_net_target = "domain_net_simplified_final",
        dataset_target = "dataset",
        ):
        pred_nets = load_model_dict_model_ensemble(info_dict[pred_nets_target], is_cuda = self.is_cuda)
        domain_net = load_model_dict_net(info_dict[domain_net_target], is_cuda = self.is_cuda)
        dataset = info_dict["dataset"]
        self.add_theories(name, pred_nets, domain_net, dataset)
    
    
    def add_master_theory_group_list(
        self,
        group_list,
        is_replace = False,
        ):
        for master_theory_dict, theory_dict in group_list:
            assert len(master_theory_dict) == 1
            name = list(master_theory_dict.keys())[0]
            master_theory = master_theory_dict[name]
            self.add_master_theory(name, master_theory, theory_dict, is_replace = is_replace)      


    def add_master_theory(
        self,
        name,
        master_theory,
        theory_tuples,
        is_replace = False,
        ):
        assert isinstance(theory_tuples, dict), "theory_tuples must be a dictionary of theory_tuples!"
        if is_replace:
            master_theory_tuple = Master_Theory_Tuple(master_theory, theory_tuples, is_cuda = self.is_cuda)
            self.master_theory_collection[name] = master_theory_tuple
        else:
            if name in self.master_theory:
                print("Name {0} collision! Please change to a different name!".format(name))
            else:
                master_theory_tuple = Master_Theory_Tuple(master_theory, theory_tuples, is_cuda = self.is_cuda)
                self.master_theory_collection[name] = master_theory_tuple


    def remove_theories(self, names = None, threshold = None, verbose = True):
        if not isinstance(names, list):
            names = [names]
        popped_theories = OrderedDict()
        if names is not None:
            assert threshold is None
            for name in names:
                popped_theories[name] = self.theory_collection.pop(name)
                if verbose:
                    print("Theory {0} poped!".format(name))
        elif threshold is not None:
            mse_dict = self.get_loss()
            for name, (mse_train, mse_test) in mse_dict.items():
                if mse_train > threshold or mse_test > threshold:
                    popped_theories[name] = self.theory_collection.pop(name)
                    if verbose:
                        print("Theory {0}'s mse_train = {1:.9f}, mse_test = {2:.9f}, larger than threshold, popped!".format(name, mse_train, mse_test))
        else:
            raise
        return popped_theories


    def remove_master_theory(self, names, verbose = True):
        if not isinstance(names, list):
            names = [names]
        popped_master_theories = OrderedDict()
        for name in names:
            popped_master_theories[name] = self.master_theory_collection.pop(name)
            if verbose:
                print("Master theory {0} poped!".format(name))
        return popped_master_theories
    
    
    def get_theory_tuples(self, input_size = None):
        if input_size is None:
            return self.theory
        theory_tuples = OrderedDict()
        for key, theory_tuple in self.theory.items():
            if theory_tuple.pred_net.input_size == input_size:
                theory_tuples[key] = theory_tuple
        return theory_tuples


    def get_all_models(self):
        all_models = OrderedDict()
        for name, theory_tuple in self.theory_collection.items():
            all_models[name] = theory_tuple.pred_net
        return all_models


    def get_pred_nets(self, input_size = None):
        if input_size is None:
            all_models = [theory_tuple.pred_net for theory_tuple in self.theory.values()]
        else:
            all_models = [theory_tuple.pred_net for theory_tuple in self.theory.values() if theory_tuple.pred_net.input_size == input_size]
        return construct_model_ensemble_from_nets(all_models)


    def get_loss(self):
        mse_dict = OrderedDict()
        for theory_name, theory_tuple in self.theory_collection.items():
            mse_train, mse_test = theory_tuple.get_loss()
            print("Theory {0}: mse_train = {1:.9}\tmse_test = {2:.9}".format(theory_name, mse_train, mse_test))
            mse_dict[theory_name] = [mse_train, mse_test]
        return mse_dict
    
    
    def plot_theory(self, DL_rank = None, theory_name = None, **kwargs):
        "Plot simplified theory based on rank or theory_name"
        if not hasattr(self, "df_sympy"):
            df = self.get_df_sympy()
        else:
            df = self.df_sympy
        if DL_rank is not None:
            item = df.iloc[DL_rank]
            theory_name = item["theory_name"]
        else:
            assert theory_name is not None
            df = df.set_index("theory_name")
            item = df.loc[theory_name]
        print("pred_net_DL: {0}".format(item["pred_net_DL"]))
        print("numerical_expression: {0}".format(item["numerical_expression"]))
        print("mse_train: {0}\tmse_test: {1}".format(item["mse_train"], item["mse_test"]))
        theory = self.theory[theory_name]
        theory.plot(**kwargs)


    def plot(self, target = "theory", keys = None, is_train = True):
        if target == "theory":
            for name, theory in self.theory.items():
                if keys is None or (keys is not None and name in keys):
                    if "simplified" in name:
                        theory.pred_net.get_sympy_expression()
                    mse_train, mse_test = theory.get_loss()
                    print("{0}\tmse_train: {1:.9f}\tmse_test: {2:.9f}".format(name, mse_train, mse_test))
                    theory.plot(is_train = is_train)
        elif target == "master_theory":
            for name, master_theory in self.master_theory.items():
                if keys is None or (keys is not None and name in keys):
                    print("{0}:".format(name))
                    master_theory.plot(is_train = is_train)
        else:
            raise Exception("target {0} not recognized!".format(target))

    
    def combine_pred_domain_nets(self, theory_tuples):
        if isinstance(theory_tuples, list):
            theory_tuples = {name: self.theory[name] for name in theory_tuples}
        elif isinstance(theory_tuples, str):
            if theory_tuples == "all":
                theory_tuples = self.theory_collection
        
        pred_net_list = []
        domain_net_list = []
        for name, theory_tuple in theory_tuples.items():
            pred_net_list.append(theory_tuple.pred_net)
            domain_net_list.append(theory_tuple.domain_net)
        pred_nets = construct_model_ensemble_from_nets(pred_net_list)
        domain_net = construct_model_ensemble_from_nets(domain_net_list)
        return pred_nets, domain_net


    def combine_datasets(self, theory_tuples):
        if isinstance(theory_tuples, list):
            theory_tuples = {name: self.theory[name] for name in theory_tuples}
        elif isinstance(theory_tuples, str):
            if theory_tuples == "all":
                theory_tuples = self.theory_collection
        
        X_train_list = []
        X_test_list = []
        y_train_list = []
        y_test_list = []
        reflect_train_list = []
        reflect_test_list = []
        for name, theory_tuple in theory_tuples.items():
            ((X_train, y_train), (X_test, y_test), (reflect_train, reflect_test)), info = theory_tuple.dataset
            X_train_list.append(X_train)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)
            if reflect_train is not None:
                reflect_train_list.append(reflect_train)
            if reflect_test is not None:
                reflect_test_list.append(reflect_test)
        
        X_train_combined = torch.cat(X_train_list, 0)
        X_test_combined = torch.cat(X_test_list, 0)
        y_train_combined = torch.cat(y_train_list, 0)
        y_test_combined = torch.cat(y_test_list, 0)
        if len(reflect_train_list) > 0:
            reflect_train_combined = torch.cat(reflect_train_list, 0)
        else:
            reflect_train_combined = None
        if len(reflect_test_list) > 0:
            reflect_test_combined = torch.cat(reflect_test_list, 0)
        else:
            reflect_test_combined = None
        dataset = ((X_train_combined, y_train_combined), (X_test_combined, y_test_combined), (reflect_train_combined, reflect_test_combined)), info
        return dataset


    def propose_theory_models(self, X, y, source = ["master_theory", "theory"], types = ["neural", "simplified"], fraction_threshold = 1e-4, max_num_models = None, loss_core = "mse", is_Lagrangian = False, isplot = False, verbose = True):
        if len(self.theory) == 0 and len(self.master_theory) == 0:
            return {}, {}
        theory_models = OrderedDict()
        proposed_theory_models = OrderedDict()
        input_size = X.size(1)

        # Add candidate theories from master_theories:
        if "master_theory" in source:
            for name, master_theory_tuple in self.master_theory.items():
                theory_model = master_theory_tuple.master_theory.propose_theory_model_from_data(X, y)
                if theory_model is not None:
                    theory_models[name] = theory_model

        # Add candidate theories from theory collections:
        if "theory" in source:
            for name, theory_tuple in self.theory.items():
                if "simplified" not in types:
                    if "simplified" in name:
                        continue
                if theory_tuple.pred_net.input_size == input_size:
                    theory_models[name] = theory_tuple.pred_net
        
        if len(theory_models) == 0:
            return {}, {}

        # Propose theories models whose fraction_best exceeds the fraction_threshold:
        fraction_best_list = []
        theory_name_list = []
        pred_nets_combined = construct_model_ensemble_from_nets(list(theory_models.values()))
        if is_Lagrangian:
            preds = get_Lagrangian_loss(pred_nets_combined, X)
        else:
            preds = pred_nets_combined(X)
        loss_fun_cumu = Loss_Fun_Cumu(core = loss_core, cumu_mode = "mean")
        loss_indi = loss_fun_cumu(preds, y, cumu_mode = "original", neglect_threshold_on = False, is_mean = False)
        best_id = to_one_hot(loss_indi.min(1)[1], loss_indi.size(1))
        fraction_best = best_id.sum(0).float() / float(loss_indi.size(0))
        if fraction_best.is_cuda:
            fraction_best = fraction_best.cpu()
        fraction_best = fraction_best.data.numpy()
        for i, name in enumerate(list(theory_models.keys())):
            if fraction_best[i] > fraction_threshold:
                loss_mean = to_np_array(torch.masked_select(loss_indi[:,i], best_id[:,i].byte()).mean())
                proposed_theory_models[name] = {"theory_model": theory_models[name], "fraction_best": fraction_best.data[i], "loss_mean": loss_mean}
                theory_name_list.append(name)
                fraction_best_list.append(fraction_best[i])
        fraction_best_list, theory_name_list = sort_two_lists(fraction_best_list, theory_name_list, reverse = True)

        k = 0
        proposed_theory_models_sorted = OrderedDict()
        for name in theory_name_list:
            if max_num_models is not None and k >= max_num_models:
                break
            else:
                proposed_theory_models_sorted[name] = proposed_theory_models[name]
                k += 1

        if len(proposed_theory_models_sorted) == 0:
            return {}, {}
        proposed_pred_nets = construct_model_ensemble_from_nets([theory_info["theory_model"] for theory_info in proposed_theory_models_sorted.values()])
        if is_Lagrangian:
            preds = get_Lagrangian_loss(proposed_pred_nets, X)
        else:
            preds = proposed_pred_nets(X)
        loss_indi = loss_fun_cumu(preds, y, cumu_mode = "original", neglect_threshold_on = False, is_mean = False)
        best_id = to_one_hot(loss_indi.min(1)[1], loss_indi.size(1))
        evaluation = {"loss_best": to_np_array(loss_fun_cumu(preds, y, cumu_mode = "min")),
                      "pred-based_generalized-mean_-1": to_np_array(loss_fun_cumu(preds, y, cumu_mode = "harmonic"))}
        if verbose:
            print("loss_best: {0:.9f}\tharmonic loss: {1:.9f}".format(evaluation["loss_best"], evaluation["pred-based_generalized-mean_-1"]))
            for key, item in proposed_theory_models_sorted.items():
                print("{0}:\tfraction = {1:.5f}\tloss_mean = {2:.9f}".format(key, item["fraction_best"], item["loss_mean"]))
        if isplot:
            print("target:")
            plot3D(X, y)
            print("proposed best_prediction:")
            plot3D(X, preds, best_id)

        return proposed_theory_models_sorted, evaluation


    def propose_master_theory(
        self,
        input_size,
        output_size,
        statistics_output_neurons,
        master_model_type="regulated-Net",
        symbolic_expression=None,
        **kwargs
        ):
        pre_pooling_neurons = kwargs["pre_pooling_neurons"] if "pre_pooling_neurons" in kwargs else 100
        struct_param_pre = [
            [64, "Simple_Layer", {}],
            [64, "Simple_Layer", {}],
            [64, "Simple_Layer", {}],
            [pre_pooling_neurons, "Simple_Layer", {"activation": "linear"}],
        ]
        struct_param_post = [
            [64, "Simple_Layer", {}],
            [64, "Simple_Layer", {}],
            [statistics_output_neurons, "Simple_Layer", {"activation": "linear"}],
        ]
        struct_param_statistics_Net = [struct_param_pre, struct_param_post]
        struct_param_classifier = [[128, "Simple_Layer", {}],
                                   [128, "Simple_Layer", {}],
                                   [128, "Simple_Layer", {}],
                                   [2,  "Simple_Layer", {"activation": "linear"}],
                                  ]
        master_theory = Master_Theory(input_size=input_size,
                                      output_size=output_size,
                                      pre_pooling_neurons=pre_pooling_neurons,
                                      struct_param_statistics_Net=struct_param_statistics_Net,
                                      struct_param_classifier=struct_param_classifier,
                                      is_cuda=self.is_cuda,
                                     )
        master_theory.propose_master_model(theory_collection=self.theory_collection,
                                           input_size=input_size,
                                           statistics_output_neurons=statistics_output_neurons,
                                           master_model_type=master_model_type,
                                           symbolic_expression=symbolic_expression,
                                           **kwargs
                                          )
        return master_theory


    def propose_master_theories(
        self,
        num_master_theories,
        input_size,
        output_size,
        statistics_output_neurons,
        master_model_type="regulated-Net",
        **kwargs
        ):
        master_theory_dict = OrderedDict()
        if master_model_type in ["regulated-Net", "generative_Net"]:
            for i in range(num_master_theories):
                master_theory_dict["master_{0}".format(i)] = self.propose_master_theory(
                    input_size = input_size,
                    output_size = output_size,
                    statistics_output_neurons= statistics_output_neurons,
                    master_model_type = master_model_type,
                    **kwargs
                   )
        elif master_model_type == "symbolic":
            df_sympy, exprs_unified_list = unification_symbolic(
                self.theory_collection,
                num_clusters=num_master_theories,
            )
            for i in range(num_master_theories):
                master_theory_dict["master_{0}".format(i)] = self.propose_master_theory(
                    input_size=input_size,
                    output_size=output_size,
                    statistics_output_neurons=max(1, len(get_param_name_list(exprs_unified_list[i]))), # The statistics net outputs the parameters for the symbolic network
                    master_model_type=master_model_type,
                    symbolic_expression=exprs_unified_list[i],
                    **kwargs
                   )
        else:
            raise Exception("master_model_type {} is not valid!".format(master_model_type))
        return master_theory_dict


    def fit_master_theory(
        self,
        master_theory_dict,
        theory_dict,
        theory_dict_test = None,
        optim_type = ("adam", 1e-4),
        reg_dict = {"master_model": {"weight": 1e-6, "bias": 1e-6},
                    "statistics_Net": {"weight": 1e-6, "bias": 1e-6}},
        loss_core = "huber",
        loss_mode = "harmonic",
        loss_combine_mode = "on-loss",
        num_iter = 5000,
        inspect_interval = 50,
        patience = 20,
        isplot = False,
        filename = None,
        **kwargs
        ):
        if not isinstance(master_theory_dict, dict):
            master_theory_dict = {"master_0": master_theory_dict}
        self.optim_type = optim_type
        self.reg_dict = reg_dict
        self.loss_core = loss_core
        self.loss_mode = loss_mode
        params = itertools.chain(*[master_theory.get_parameters(targets = ["master_model", "statistics_Net"])                                                        for master_theory in master_theory_dict.values()])

        self.master_loss_fun = Master_Loss_Fun(core = self.loss_core, cumu_mode = self.loss_mode, loss_combine_mode = loss_combine_mode)
        if self.optim_type[0] == "LBFGS":
            self.optimizer = torch.optim.LBFGS(params, lr = self.optim_type[1])
        else:
            if self.optim_type[0] == "adam":
                self.optimizer = torch.optim.Adam(params, lr = self.optim_type[1])
            elif self.optim_type[0] == "RMSprop":
                self.optimizer = torch.optim.RMSprop(params, lr = self.optim_type[1])
            else:
                raise Exception("optim_type {0} not recognized!".format(self.optim_type[0]))

        if patience is not None:
            early_stopping = Early_Stopping(patience = patience)
        to_stop = False
        self.data_record = {}
        images_per_row = int(max(1, int(5 / float(max(1, (len(theory_dict) + len(theory_dict)) / float(16))))))
        theory_dict_to_test = theory_dict_test if theory_dict_test is not None else theory_dict
        
        print("Each m x n matrix shows the loss of m master_theories fitting to n theories.")
        for i in range(num_iter + 1):
            self.optimizer.zero_grad()
            loss_train = self.master_loss_fun(master_theory_dict, theory_dict)
            reg = torch.cat([get_reg(master_theory, reg_dict, mode = "L1") for master_theory in master_theory_dict.values()], 0).sum()
            loss = loss_train + reg
            loss.backward()
            self.optimizer.step()

            if np.isnan(to_np_array(loss)):
                raise Exception("NaN encountered!")
            if i % inspect_interval == 0:
                loss_train = self.master_loss_fun(master_theory_dict, theory_dict)
                loss_matrix_train = self.master_loss_fun.loss_matrix
                loss_test = self.master_loss_fun(master_theory_dict, theory_dict_to_test, use_train = False)
                loss_matrix_test = self.master_loss_fun.loss_matrix
                record_data(self.data_record, [to_np_array(loss_train), to_np_array(loss_test), to_np_array(loss_matrix_train), to_np_array(loss_matrix_test)], 
                                              ["loss_train", "loss_test", "loss_matrix_train", "loss_matrix_test"])
                print("iter {0}  \tloss_train: {1:.9f} \tloss_test: {2:.9f} \treg: {3:.9f}".format(i, to_np_array(loss_train), to_np_array(loss_test), to_np_array(reg)))
                if patience is not None:
                    to_stop = early_stopping.monitor(to_np_array(loss_test))
                
                if loss_matrix_train.size(0) > 1:
                    train_best = to_np_array(loss_matrix_train.min(0)[1]).tolist()
                    print("train best:", train_best)
                else:
                    train_best = None
                if isplot:
                    x_axis_list_core = "loss_train: {0:.4f}   loss_test: {1:.4f}   reg: {2:.5f}".format(to_np_array(loss_train), to_np_array(loss_test), to_np_array(reg))
                    x_axis_list = x_axis_list_core + "\n{0}".format(train_best) if train_best is not None and len(train_best) < 10 else x_axis_list_core
                    self.master_loss_fun.plot_loss_matrix(images_per_row = images_per_row, x_axis_list = [x_axis_list],
                                                          filename = filename + "_{0}_train.png".format(i) if filename is not None else None)
                if loss_matrix_test.size(0) > 1:
                    test_best = loss_matrix_train.min(0)[1].cpu().data.numpy().tolist()
                    print("test best:", loss_matrix_test.min(0)[1].cpu().data.numpy().tolist())
                else:
                    test_best = None
                if isplot:
                    x_axis_list = x_axis_list_core + "\n{0}".format(test_best) if test_best is not None and len(test_best) < 10 else x_axis_list_core  
                    self.master_loss_fun.plot_loss_matrix(images_per_row = images_per_row, x_axis_list = [x_axis_list],
                                                          filename = filename + "_{0}_test.png".format(i) if filename is not None else None)
                try:
                    sys.stdout.flush()
                except:
                    pass
                if to_stop:
                    print("Early stopping at iteration {0}".format(i))
                    break
        return deepcopy(self.data_record)
    
    
    def assign_master_theories_to_theories(self, master_theory_dict, theory_dict):
        # Assigned master_theories to theory:
        loss_matrix = self.master_loss_fun.get_loss_matrix(master_theory_dict, theory_dict, loss_combine_mode = "on-loss")
        best_master_theory_id = loss_matrix.min(0)[1]
        group_dict = OrderedDict()
        for i in range(best_master_theory_id.size(0)):
            best_id = int(best_master_theory_id.data[i])
            theory_name = list(theory_dict.keys())[i]
            master_theory_name = list(master_theory_dict.keys())[best_id]
            if master_theory_name not in group_dict:
                group_dict[master_theory_name] = {"master_theory": master_theory_dict[master_theory_name], "assigned_theory_dict": OrderedDict()}
            group_dict[master_theory_name]["assigned_theory_dict"][theory_name] = theory_dict[theory_name]
        # Make the format compatible with self.fit_master_theory():
        group_list = []
        for name, item in group_dict.items():
            assigned_master_theory_dict = {name: item["master_theory"]}
            assigned_theory_dict = item["assigned_theory_dict"]
            group_list.append([assigned_master_theory_dict, assigned_theory_dict])
        return group_list


    def fit_master_classifier_multi(
        self,
        group_list,
        optim_type_classifier = ("adam", 5e-5),
        reg_dict_classifier = {"classifier": {"weight": 1e-3, "bias": 1e-3}},
        patience = 20,
        ):
#         for master_theory_dict, theory_dict in group_list:
#             master_theory = master_theory_dict[list(master_theory_dict.keys())[0]]
#             datasets = self.combine_datasets(theory_dict)
        pass


    def fit_master_classifier(
        self,
        master_theory,
        theory_dict,
        optim_type_classifier = ("adam", 5e-5),
        reg_dict_classifier = {"classifier": {"weight": 1e-3, "bias": 1e-3}},
        patience = 20,
        ):
        self.optim_type_classifier = optim_type_classifier
        self.reg_dict_classifier = reg_dict_classifier
        input_size = master_theory.input_size

        dataset_chosen = self.combine_datasets(theory_dict)
        dataset_excluded = self.combine_datasets({name: theory_tuple for name, theory_tuple in self.theory.items() if name not in theory_dict and theory_tuple.pred_net.input_size == input_size})
        ((X_train_chosen, y_train_chosen), (X_test_chosen, y_test_chosen), _), _ = dataset_chosen
        ((X_train_excluded, y_train_excluded), (X_test_excluded, y_test_excluded), _), _ = dataset_excluded
        X_train_combined = torch.cat([X_train_chosen, X_train_excluded], 0)
        X_test_combined = torch.cat([X_test_chosen, X_test_excluded], 0)
        u_train_combined = Variable(torch.cat([torch.ones(X_train_chosen.size(0)), torch.zeros(X_train_excluded.size(0))], 0).long().unsqueeze(1), requires_grad = False)
        u_test_combined = Variable(torch.cat([torch.ones(X_test_chosen.size(0)), torch.zeros(X_test_excluded.size(0))], 0).long().unsqueeze(1), requires_grad = False)
        batch_gen = Batch_Generator(X_train_combined, u_train_combined, batch_size = 128)
        batch_gen_test = Batch_Generator(X_test_combined, u_test_combined, batch_size = 2000)

        params = master_theory.classifier.parameters()
        if self.optim_type_classifier[0] == "LBFGS":
            self.optimizer_classifier = torch.optim.LBFGS(params, lr = self.optim_type_classifier[1])
            num_iter = 5000
            inspect_interval = 50
        else:
            num_iter = 30000
            inspect_interval = 50
            if self.optim_type_classifier[0] == "adam":
                self.optimizer_classifier = torch.optim.Adam(params, lr = self.optim_type_classifier[1])
            elif self.optim_type_classifier[0] == "RMSprop":
                self.optimizer_classifier = torch.optim.RMSprop(params, lr = self.optim_type_classifier[1])
            else:
                raise Exception("optim_type {0} not recognized!".format(self.optim_type_classifier[0]))

        early_stopping = Early_Stopping(patience = patience)
        to_stop = False
        self.data_record_classifier = {}
        ratio = X_train_chosen.size(0) / float(X_train_chosen.size(0) + X_train_excluded.size(0))
        weight = torch.FloatTensor(np.array([ratio, 1 - ratio]))
        if self.is_cuda:
            weight = weight.cuda()

        for i in range(num_iter + 1):
            X_batch, u_batch = batch_gen.next_batch(isTorch = True, is_cuda = self.is_cuda)
            self.optimizer_classifier.zero_grad()
            pred_batch = master_theory.classifier(X_batch)
            loss_train = nn.CrossEntropyLoss(weight = weight)(pred_batch, u_batch.long().view(-1))
            reg = get_reg(master_theory, self.reg_dict_classifier, "L1")
            loss = loss_train + reg
            loss.backward()
            self.optimizer_classifier.step()

            X_batch_test, u_batch_test = batch_gen_test.next_batch(isTorch = True, is_cuda = self.is_cuda)
            pred_batch_test = master_theory.classifier(X_batch_test)
            loss_test = nn.CrossEntropyLoss(weight = weight)(pred_batch_test, u_batch_test.long().view(-1))
            if np.isnan(to_np_array(loss)):
                raise Exception("NaN encountered!")
            if i % inspect_interval == 0:
                to_stop = early_stopping.monitor(to_np_array(loss_test))
                record_data(self.data_record_classifier, [to_np_array(loss_train), to_np_array(loss_test)], ["loss_train", "loss_test"])
                print("Classifier iter {0}  \tloss_train: {1:.9f} \tloss_test: {2:.9f} \treg: {3:.9f}".format(i, to_np_array(loss_train), to_np_array(loss_test), to_np_array(reg)))
                if to_stop:
                    print("Early stopping at iteration {0}".format(i))
                    break


def get_regulated_latent_param(master_model, latent_param):
    assert len(latent_param.view(-1)) == len(master_model.struct_param) * 2 or len(latent_param.view(-1)) == len(master_model.struct_param)
    if len(latent_param.view(-1)) == len(master_model.struct_param) * 2:
        latent_param = {i: latent_param.view(-1)[2*i: 2*i+2] for i in range(len(master_model.struct_param))}
    else:
        latent_param = {i: latent_param.view(-1)[i:i+1] for i in range(len(master_model.struct_param))}
    return latent_param


def get_reg(master_theory, reg_dict, mode = "L1"):   
    reg = Variable(torch.FloatTensor([0]), requires_grad = False)
    if master_theory.is_cuda:
        reg = reg.cuda()
    for net_target, reg_setting in reg_dict.items():
        for source_target, reg_amp in reg_setting.items():
            reg = reg + master_theory.get_regularization(targets = [net_target], source = [source_target], mode = mode) * reg_amp
    return reg


# The following functions load theories, master theories and theory hub from file:
def load_model_dict_at_theory_hub(model_dict, is_cuda = False):
    if model_dict["type"] == "Theory_Tuple":
        return load_model_dict_theory_tuple(model_dict, is_cuda = is_cuda)
    elif model_dict["type"] == "Master_Theory_Tuple":
        return load_model_dict_master_theory_tuple(model_dict, is_cuda = is_cuda)
    elif model_dict["type"] == "Master_Theory":
        return load_model_dict_master_theory(model_dict, is_cuda = is_cuda)
    elif model_dict["type"] == "Theory_Hub":
        return load_model_dict_theory_hub(model_dict, is_cuda = is_cuda)
    else:
        raise Exception("type {0} not recognized!".format(model_dict["type"]))


def load_model_dict_theory_tuple(model_dict, is_cuda = False):
    return Theory_Tuple(pred_net = model_dict["pred_net"],
                        domain_net = model_dict["domain_net"],
                        dataset = model_dict["dataset"],
                        is_Lagrangian = model_dict["is_Lagrangian"] if "is_Lagrangian" in model_dict else False,
                        is_cuda = is_cuda,
                       )

def load_model_dict_master_theory_tuple(model_dict, is_cuda = False):
    theory_tuples = {name: load_model_dict_theory_tuple(theory_tuple, is_cuda = is_cuda) for name, theory_tuple in model_dict["theory_tuples"].items()}
    return Master_Theory_Tuple(master_theory = load_model_dict_master_theory(model_dict["master_theory"], is_cuda = is_cuda),
                               theory_tuples = theory_tuples,
                               is_cuda = is_cuda,
                              )

def load_model_dict_master_theory(model_dict, is_cuda = False):
    master_theory = Master_Theory(input_size = model_dict["input_size"],
                                  pre_pooling_neurons = model_dict["statistics_Net"]["pre_pooling_neurons"],
                                  struct_param_statistics_Net = [model_dict["statistics_Net"]["struct_param_pre"],
                                                                 model_dict["statistics_Net"]["struct_param_post"]],
                                  struct_param_classifier = model_dict["classifier"]["struct_param"],
                                  is_cuda = is_cuda,
                                 )
    if model_dict["master_model"] is not None:
        master_theory.master_model = load_model_dict_net(model_dict["master_model"], is_cuda = is_cuda)
    if "master_model_type" in model_dict:
        master_theory.master_model_type = model_dict["master_model_type"]
    master_theory.statistics_Net.load_model_dict(model_dict["statistics_Net"])
    master_theory.classifier.load_model_dict(model_dict["classifier"])
    return master_theory

def load_model_dict_theory_hub(model_dict, is_cuda = False):
    theory_hub = Theory_Hub(is_cuda = is_cuda)
    for name, theory_tuple in model_dict["theory_collection"].items():
        theory_hub.theory_collection[name] = load_model_dict_theory_tuple(theory_tuple, is_cuda = is_cuda)
    for name, master_theory_tuple in model_dict["master_theory_collection"].items():
        theory_hub.master_theory_collection[name] = load_model_dict_master_theory_tuple(master_theory_tuple, is_cuda = is_cuda)
    return theory_hub


def select_explained_data(model, X, y, threshold = 1e-4):
    chosen_id = ((model(X) - y) ** 2 < threshold)
    X_chosen = torch.masked_select(X, chosen_id.detach()).view(-1, X.size(1))
    y_chosen = torch.masked_select(y, chosen_id.detach()).view(-1, y.size(1))
    X_others = torch.masked_select(X, ~chosen_id.detach()).view(-1, X.size(1))
    y_others = torch.masked_select(y, ~chosen_id.detach()).view(-1, y.size(1))
    return (X_chosen, y_chosen), (X_others, y_others)


# In[3]:


## Loss function for unifying multiple theory into a network master theory:
class Induce_Loss_Fun(nn.Module):
    def __init__(self, core = "mse", loss_combine_mode = "on-loss", cumu_mode = "harmonic"):
        super(Induce_Loss_Fun, self).__init__()
        self.core = core
        self.loss_combine_mode = loss_combine_mode
        if self.loss_combine_mode == "on-data":
            self.loss_fun_cumu = Loss_Fun_Cumu(core = core, cumu_mode = cumu_mode)
        self.loss_fun = Loss_Fun(core = core)


    def forward(self, master_theory_dict, theory_model, X, y = None, loss_combine_mode = None):
        target = theory_model(X) if y is None else y
        if loss_combine_mode is None:
            loss_combine_mode = self.loss_combine_mode
        if loss_combine_mode == "on-loss":
            loss_column = []
            for _, master_theory in master_theory_dict.items():
                latent_param = master_theory.statistics_Net(torch.cat([X, target], 1))
                if master_theory.master_model_type == "symbolic":
                    latent_param = {0: {param_name: latent_param[0, k] for k, param_name in enumerate(master_theory.param_name_list)}}
                elif master_theory.master_model_type == "regulated-Net":
                    latent_param = get_regulated_latent_param(master_theory.master_model, latent_param)
                master_pred = master_theory.master_model(X, latent_param)
                loss = self.loss_fun(master_pred, target)
                loss_column.append(loss)
            loss_column = torch.stack(loss_column, 0)
        elif loss_combine_mode == "on-data":
            loss_column = []
            master_pred_list = []
            for _, master_theory in master_theory_dict.items():
                latent_param = master_theory.statistics_Net(torch.cat([X, target], 1))
                if master_theory.master_model_type == "symbolic":
                    latent_param = {0: {param_name: latent_param[0, k] for k, param_name in enumerate(master_theory.param_name_list)}}
                elif master_theory.master_model_type == "regulated-Net":
                    latent_param = get_regulated_latent_param(master_theory.master_model, latent_param)
                master_pred = master_theory.master_model(X, latent_param)
                master_pred_list.append(master_pred)
            master_pred_list = torch.stack(master_pred_list, 1)
            loss_column = self.loss_fun_cumu(master_pred_list, target)
        else:
            raise Exception("loss_combine_mode {0} not recognized!".format(loss_combine_mode))
        return loss_column


class Master_Loss_Fun(nn.Module):
    def __init__(self, core = "mse", cumu_mode = "harmonic", loss_combine_mode = "on-loss", epsilon = 1e-10):
        super(Master_Loss_Fun, self).__init__()
        self.epsilon = epsilon
        self.cumu_mode = cumu_mode
        self.induce_loss_fun = Induce_Loss_Fun(core = core, loss_combine_mode = loss_combine_mode, cumu_mode = cumu_mode)


    def get_loss_matrix(self, master_theory_dict, theory_dict, loss_combine_mode = None, use_train = True, use_target = True):
        loss_matrix = []
        for theory_name, theory_tuple in theory_dict.items():
            theory_model = theory_tuple.pred_net
            ((X_train, y_train), (X_test, y_test), _), _ = theory_tuple.dataset
            if use_train:
                X = X_train
                y = y_train if use_target else None
            else:
                X = X_test
                y = y_test if use_target else None
            loss_matrix.append(self.induce_loss_fun(master_theory_dict, theory_model, X, y, loss_combine_mode = loss_combine_mode))
        self.loss_matrix = torch.stack(loss_matrix, 1)
        return self.loss_matrix


    def forward(self, master_theory_dict, theory_dict, use_train = True, use_target = True):
        if not isinstance(master_theory_dict, dict):
            master_theory_dict = {"master_0": master_theory_dict}
        self.get_loss_matrix(master_theory_dict, theory_dict, use_train = use_train, use_target = use_target)
        if self.loss_matrix.size(0) == 1:
            loss = self.loss_matrix.sum()
        else:
            if self.cumu_mode == "harmonic":
                loss = (self.loss_matrix.size(0) / (1 / (self.loss_matrix + self.epsilon)).sum(0)).sum()
            elif self.cumu_mode == "min":
                loss = self.loss_matrix.min(0)[0].sum()
            elif self.cumu_mode[0] == "generalized-mean":
                order = self.cumu_mode[1]
                loss = ((((self.loss_matrix + self.epsilon) ** order).mean(0)) ** (1 / float(order))).sum()
            else:
                raise Exception("mode {0} not recognized!".format(self.cumu_mode))
        return loss


    def plot_loss_matrix(self, master_theory_dict = None, theory_dict = None, loss_combine_mode = None, use_train = True, use_target = True, filename = None, **kwargs):
        if master_theory_dict is not None or theory_dict is not None or loss_combine_mode is not None:
            self.get_loss_matrix(master_theory_dict, theory_dict, loss_combine_mode = loss_combine_mode, use_train = use_train, use_target = use_target)
        loss_matrix = self.loss_matrix
        if loss_matrix.is_cuda:
            loss_matrix = loss_matrix.cpu()
        plot_matrices([np.log10(loss_matrix.data.numpy())], filename = filename, **kwargs)

