import numpy as np
from copy import deepcopy
import time
import pickle
import pprint as pp
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_physicist.pytorch_net.util import to_np_array, to_Variable, filter_filename, standardize_symbolic_expression, get_coeffs, substitute
from AI_physicist.pytorch_net.net import load_model_dict
from AI_physicist.settings.global_param import PrecisionFloorLoss, COLOR_LIST, Dt
from AI_physicist.settings.filepath import theory_PATH
from sklearn.model_selection import train_test_split


def logplus(x):
    return torch.clamp(torch.log(torch.clamp(x, 1e-9)) / np.log(2), 0)
    

def forward(model, X, **kwargs):
    autoencoder = kwargs["autoencoder"] if "autoencoder" in kwargs else None
    is_Lagrangian = kwargs["is_Lagrangian"] if "is_Lagrangian" in kwargs else False
    output = X
    if not is_Lagrangian:
        if isinstance(model, list) or isinstance(model, tuple):
            for model_ele in model:
                output = model_ele(output)
        else:
            output = model(output)
    else:
        if isinstance(model, list) or isinstance(model, tuple):
            for i, model_ele in enumerate(model):
                if i != len(model) - 1:
                    output = model_ele(output)
                else:
                    output = get_Lagrangian_loss(model_ele, output)
        else:
            output = get_Lagrangian_loss(model, output)
    if autoencoder is not None:
        output = autoencoder.decode(output)
    return output
    

def compare_same(dict1, dict2, keys, threshold, verbose = False):
    is_same = True
    diff_dict = {}
    diff_dict["diff_keys"] = []
    for ii, key in enumerate(keys):
        threshold_ele = threshold[ii] if isinstance(threshold, list) else threshold
        diff_relative = abs((dict1[key] - dict2[key]) / float(dict2[key] + 1e-20))
        diff_dict[key] = (dict1[key], dict2[key], diff_relative)
        if diff_relative > threshold_ele:
            is_same = False
            diff_dict["diff_keys"].append(key)
    if verbose:
        print("diff_dict:")
        pp.pprint(diff_dict)
    return is_same, diff_dict


def get_loss_item(data_record, i, loss_key = "mse_with_domain_big_domain", bg = None):
    item = data_record["all_losses_dict"][i]
    event = data_record["event"][i]
    if loss_key not in item:
        if "metrics_big_domain" in item:
            item_ele = item["metrics_big_domain"][loss_key]
        else:
            # load model to inspect:
            if bg is not None:
                print("Load theories to inspect!")
                T = bg["T"]
                T.set_net("pred_nets", load_model_dict(data_record["pred_nets_model_dict"][i]))
                T.set_net("domain_net", load_model_dict(data_record["domain_net_model_dict"][i]))
                loss_dict = T.get_losses(bg["X_test"], bg["y_test"], true_domain_test = bg["true_domain_test"], big_domain_ids = bg["big_domain_ids"])
                is_same, diff_keys = compare_same(loss_dict, item, ["DL_pred_nets", "DL_domain_net", "loss_with_domain", "mse_with_domain"], threshold = 0.0001)
                if not is_same:
                    print("Recorded loss_dict:")
                    pp.pprint(item)
                    print("\nReloaded loss_dict:")
                    pp.pprint(loss_dict)
                    raise Exception("Inconsistency in {0} for recorded and reloaded loss_dict!".format(diff_keys))
                if loss_key not in item:
                    item_ele = loss_dict["metrics_big_domain"][loss_key]
                else:
                    item_ele = loss_dict[loss_key]
            else:
                item_ele = None
    else:
        item_ele = item[loss_key]
    return item_ele, event


def append_loss_to_cumu(loss_cumu_dict, data_record, loss_key = "mse_with_domain_big_domain", event = None, consider_add_theories = False, bg = None, load_add_theory = True):
    if bg is not None:
        if 'loss_precision_floor' in data_record:
            bg["T"].set_loss_core("DLs", data_record['loss_precision_floor'])
        else:
            bg["T"].set_loss_core("DLs", 10)

    for i in range(len(data_record["all_losses_dict"])):
        if load_add_theory is True and data_record["event"][i] is not None and data_record["event"][i][0] == 'add_theories':
            print("load adding theories:")
            data_record_add = data_record["event"][i][1]
            for key_add, item_add in data_record_add.items():
                if "data_record" not in item_add:
                    continue
                for data_record_add_ele in item_add["data_record"]:
                    for j in range(len(data_record_add_ele["all_losses_dict"])):
                        if bg is not None:
                            if 'loss_precision_floor' in data_record:
                                bg["T"].set_loss_core("DLs", data_record_add_ele['loss_precision_floor'])
                            else:
                                bg["T"].set_loss_core("DLs", 10)
                        item_ele, event = get_loss_item(data_record_add_ele, j, loss_key = "mse_with_domain_big_domain", bg = bg)                
                        loss_cumu_dict[loss_key].append(item_ele)
                        loss_cumu_dict["event"].append("{0}_{1}_{2}".format(event, "add", key_add))
        else:
            item_ele, event = get_loss_item(data_record, i, loss_key = "mse_with_domain_big_domain", bg = bg)
        loss_cumu_dict[loss_key].append(item_ele)
        loss_cumu_dict["event"].append(event)
        
        
def get_loss_cumu_dict(info_dict_single, loss_key = "mse_with_domain_big_domain", bg = None, load_add_theory = True):
    loss_cumu_dict = {loss_key: [], "event": []}
    data_record_1 = info_dict_single["data_record_1"]
    if bg is not None:
        bg["T"].domain_net_on = False
    append_loss_to_cumu(loss_cumu_dict, data_record_1, loss_key, event = "mse", bg = bg, load_add_theory = load_add_theory)
    if 'data_record_MDL1_1' in info_dict_single:
        print("load data_record_MDL1_1")
        data_record_MDL1 = info_dict_single['data_record_MDL1_1']
        for data_record_MDL1_ele in data_record_MDL1:
            append_loss_to_cumu(loss_cumu_dict, data_record_MDL1_ele, loss_key, event = "DL-harmonic", bg = bg, load_add_theory = load_add_theory)
    if 'data_record_MDL2_1' in info_dict_single:
        print("load data_record_MDL2_1")
        if bg is not None:
            bg["T"].domain_net_on = True
        data_record_MDL2 = info_dict_single['data_record_MDL2_1']
        for data_record_MDL2_ele in data_record_MDL2:
            append_loss_to_cumu(loss_cumu_dict, data_record_MDL2_ele, loss_key, event = "DL-indi", bg = bg, load_add_theory = load_add_theory)
    return loss_cumu_dict


class Loss_Decay_Scheduler(object):
    def __init__(self, loss_types, lambda_loss_decay):
        self.init_decay_record(deepcopy(loss_types))
        self.lambda_loss_decay = lambda_loss_decay
        self.steps = 0
    
    
    def init_decay_record(self, loss_types):
        self.loss_order_init = {}
        for loss_mode, loss_setting in loss_types.items():
            if "decay_on" in loss_setting and loss_setting["decay_on"] is True and loss_mode.split("_")[1] == "generalized-mean":
                self.loss_order_init[loss_mode] = float(loss_mode.split("_")[2])

    def step(self):
        loss_order_current = {loss_mode: self.lambda_loss_decay(self.steps) + loss_order for loss_mode, loss_order in self.loss_order_init.items()}
        self.steps += 1
        return loss_order_current


def get_epochs_to_loss(loss_cumu_dict, loss_level, record_interval, loss_key = "mse_with_domain_big_domain"):
    epoch_to_loss = np.Inf
    for i, loss in enumerate(loss_cumu_dict[loss_key]):
        if loss is None:
            epoch_to_loss = np.NaN
            break
        if loss <= loss_level:
            epoch_to_loss = i * record_interval
            break
    return epoch_to_loss


def load_theory_training(dirname, filename, env_name, info_dict = None, is_pred_net_simplified = False, is_Lagrangian = False, isplot = True, is_cuda = False):
    import random
    from AI_physicist.theory_learning.theory_model import Theory_Training

    filename_split = filename.split("_")
    mse_amp = float(filename_split[filename_split.index("mse") + 1])
    loss_order = float(filename_split[filename_split.index("order") + 1])
    loss_core = filename_split[filename_split.index("core") + 1]
    loss_balance_model_influence = eval(filename_split[filename_split.index("infl") + 1])
    seed = int(filename_split[filename_split.index("id") + 1])
    domain_net_neurons = int(filename_split[filename_split.index("dom") + 1])
    num_theories_init = int(filename_split[filename_split.index("num") + 1])
    MDL_mode = filename_split[filename_split.index("MDL") + 1]
    num_output_dims = eval(filename_split[filename_split.index("MDL") + 2][0])

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if info_dict is None:
        info_dict = pickle.load(open(dirname + filename, "rb"))
    struct_param_pred = [
        [num_output_dims, "Simple_Layer", {"activation": "linear"}],
    ] 
    struct_param_domain = [
        [domain_net_neurons, "Simple_Layer", {}],
        [domain_net_neurons, "Simple_Layer", {}],
        [num_theories_init, "Simple_Layer", {"activation": "linear"}],
    ]
    loss_types = {
                        "pred-based_generalized-mean_{0}".format(loss_order): {"amp": 1., "decay_on": True},
                        "pred-based_generalized-mean_1": {"amp": mse_amp, "decay_on": False}, 
    }
    T = Theory_Training(num_theories = num_theories_init,
                        proposed_theory_models = {},
                        input_size = 6,
                        struct_param_pred = struct_param_pred,
                        struct_param_domain = struct_param_domain,
                        loss_types = loss_types,
                        loss_core = loss_core,
                        loss_order = loss_order,
                        loss_balance_model_influence = loss_balance_model_influence,
                        reg_multiplier = None,
                        is_Lagrangian = is_Lagrangian,
                        is_cuda = is_cuda,
                       )
    if isplot:
        import matplotlib.pylab as plt
        try:
            plt.figure(figsize = (20, 12))
            _ = process_loss(info_dict[env_name]["info_dict_single"][-1], loss_core = loss_core, is_rmse = False, vertical_minor_linestyle = "-",
                             indi_linestyle = "-", main_linestyle = "-", plt = plt, is_plot_best = False, main_label = "lifelong", is_only_main = False, ylabel = "MSE")
            plt.show()
        except Exception as e:
            print(e)
    if is_pred_net_simplified:
        T.set_net("pred_nets", load_model_dict(info_dict[env_name]['pred_nets_simplified'], is_cuda = is_cuda))
    else:
        T.set_net("pred_nets", load_model_dict(info_dict[env_name]['pred_nets'], is_cuda = is_cuda))
    T.set_net("domain_net", load_model_dict(info_dict[env_name]['domain_net'], is_cuda = is_cuda))
    info_dict_single = info_dict[env_name]["info_dict_single"][-1]
    if "data_record_MDL2_1" not in info_dict_single or len(info_dict_single["data_record_MDL2_1"]) == 0:
        print("data_record_MDL2_1 has 0 length or does not exist!") 
        if "data_record_MDL1_1" not in info_dict_single or len(info_dict_single["data_record_MDL1_1"]) == 0:
            print("data_record_MDL1_1 has 0 length or does not exist!") 
            data_record = info_dict_single["data_record_1"]
            loss_precision_floor = 10
        else:
            loss_precision_floor = info_dict_single["data_record_MDL1_1"][-1]['loss_precision_floor']
    else:
        loss_precision_floor = info_dict_single["data_record_MDL2_1"][-1]['loss_precision_floor']
    print("loss_precision_floor: {0}".format(loss_precision_floor))
    T.set_loss_core("DLs", loss_precision_floor)
    T.input_size = T.pred_nets.input_size
    T.domain_net_on = True
    T.num_theories = T.pred_nets.num_models
    return T, info_dict


def get_dataset(
    filename,
    num_input_steps,
    num_output_dims,
    is_classified,
    forward_steps=1,
    num_examples=None,
    is_Lagrangian=False,
    is_cuda=False,
):
    from sklearn.model_selection import train_test_split
    matrix = np.genfromtxt(filename, delimiter=',')
    if is_Lagrangian:
        matrix = transform_data_to_phase_space(matrix, Dt)
    num = matrix.shape[0]  # Total number of examples

    # Construct X, y and true_domain:
    if is_classified:  # If true domains are provided for evaluation (not for training)
        true_domain = matrix[:num - (forward_steps - 1), -1:].astype(int)
        matrix = matrix[:, :-1]
        
        if not ((np.abs(true_domain - np.round(true_domain)) < 1e-10).all() and (true_domain >= 0).all()):
            raise Exception("If 'is_classified' is True, the last column in the {0} should be true_domain id (non-negative integers) for evaluation. "
                            "Set is_classified=False or add a column to the {0} providing true_domain id.".format(filename)
                           )
        
    if not is_Lagrangian:
        X = matrix[: num - (forward_steps - 1), -(num_output_dims + 1) * num_input_steps: -num_output_dims]
        y = matrix[forward_steps - 1:, -num_output_dims:]
    else:
        X = matrix[: num - (forward_steps - 1), :]
        y = np.zeros((num, num_output_dims))

    # Split into train and test:
    if is_classified:
        X_train, X_test, y_train, y_test, true_domain_train, true_domain_test = train_test_split(X, y, true_domain, test_size = 0.2)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    if num_examples is not None:
        if num_examples < X_train.shape[0]:
            chosen = np.random.choice(len(X_train), size = num_examples, replace = False)
            X_train = X_train[chosen]
            y_train = y_train[chosen]
    num_examples = X_train.shape[0]
    X_train, X_test, y_train, y_test = to_Variable(X_train, X_test, y_train, y_test, is_cuda = is_cuda)
    if is_classified:
        true_domain_train = torch.LongTensor(true_domain_train)
        true_domain_test = torch.LongTensor(true_domain_test)
        if is_cuda:
            true_domain_train = true_domain_train.cuda()
            true_domain_test = true_domain_test.cuda()

    # Record information:
    input_size = tuple(X_train.size()[1:]) if not is_Lagrangian else (8,)
    if len(input_size) == 1:
        input_size = input_size[0]
    info = {"input_size": input_size, "file_source": filename}
    if is_classified:
        info["true_domain_train"] = true_domain_train
        info["true_domain_test"] = true_domain_test
    return ((X_train, y_train), (X_test, y_test), (None, None)), info


def get_df(dirname, write_combine_table = False):
    import pandas as pd
    df_dict = {}
    for filename in os.listdir(dirname):
        if "mys" in filename:
            spreadsheet_fns = filter_filename(dirname  +  "/" + filename, include = ["spreadsheet", ".csv"])
            if len(spreadsheet_fns) == 0:
                print("no csv found in {0}!".format(filename))
                continue
            else:
                for element in spreadsheet_fns:
                    if element != "spreadsheet_combined.csv":
                        break
                spreadsheet_fn = element
            df_ele = pd.read_csv(dirname  +  "/" + filename + "/" + spreadsheet_fn)
            try:
                loss_table = pd.read_csv(dirname  +  "/" + filename + "/loss_table.csv")
                df_ele = pd.merge(df_ele, loss_table, on = "Name")
                if write_combine_table:
                    df_ele.to_csv(dirname  +  "/" + filename + "/" + "spreadsheet_combined.csv")
            except Exception as e:
                print(filename)
                print(e)
                print()
            parse_dict = parse_filename(filename, is_tuple = False)
            if parse_dict["exp_mode"] == "base":
                parse_dict["exp_mode"] = "baseline"
            if parse_dict["exp_mode"] == "newb":
                parse_dict["exp_mode"] = "newborn"
            if parse_dict["exp_mode"] == "continuous":
                parse_dict["exp_mode"] = "zcontinuous"
            df_ele["filename"] = filename
            for key, item in parse_dict.items():
                df_ele[key] = item
                df_ele["filename"] = filename
            df_dict[filename] = df_ele
    df = pd.concat(list(df_dict.values()))
    df["RMSE"] = np.sqrt(df["MSE"])
    df["RMSE_big_domain"] = np.sqrt(df["mse_with_domain_big_domain"])
    df[pd.isnull(df["RMSE"])] = np.NaN
    df[df == "Indeterminate"] = np.NaN
    df["precision"] = df["precision"].astype(float)
    df["f1"] = df["f1"].astype(float)
    df["recall"] = df["recall"].astype(float)
    df["IoU"] = df["IoU"].astype(float)
    return df


def get_mystery(series_base_list, type_range = range(4, 11), mystery_range = range(1, 6), is_classified = False):
    if not isinstance(series_base_list, list):
        series_base_list = [series_base_list]
    mysteries = []
    for series_base in series_base_list:
        for type_name in type_range:
            for indi_name in mystery_range:
                mystery_name = series_base + type_name * 100 + indi_name
                mysteries.append("2D{0}mysteryt{1}".format("classified" if is_classified else "", mystery_name))
    return mysteries


def process_loss(
    info_dict,
    loss_keys = "mse_with_domain",
    loss_core = "mse",
    isplot = True,
    is_show = True,
    filename = None,
    plot_reject = False,
    is_rmse = False,
    plt = None,
    **kwargs
    ):
    """Used for theory2.0"""
    from AI_physicist.settings.global_param import COLOR_LIST
    info_dict = deepcopy(info_dict)
    iter_end_list = []
    if not isinstance(loss_keys, list):
        loss_keys = [loss_keys]
    loss_cumu_dict = {}
    iter_cumu_list = np.array([])
    num_theories = len(info_dict["data_record_1"]["all_losses_dict"][0]['loss_indi_theory'])
    for i in range(num_theories):
        loss_cumu_dict["{0}_theory_{1}".format(loss_core, i)] = []

    # Setting up data_records to accumulate:
    num_rounds_max = 10
    data_record_names = []
    for j in range(1, num_rounds_max):
        data_record_names.append("data_record_{0}".format(j))
        data_record_names.append("data_record_domain_{0}".format(j))
    data_record_names.append("data_record_{0}".format(num_rounds_max))
    simplicication_start = None
    
    for key in data_record_names:
        if key not in info_dict:
            continue
        all_losses_dict_list = info_dict[key]["all_losses_dict"]
        for i, all_losses_dict in enumerate(all_losses_dict_list):
            # Accumulate loss types:
            for criterion_name in all_losses_dict:
                if criterion_name == 'loss_dict':
                    for criterion_ele in all_losses_dict['loss_dict']:
                        if criterion_ele not in loss_cumu_dict:
                            loss_cumu_dict[criterion_ele] = []
                        loss_cumu_dict[criterion_ele] = all_losses_dict['loss_dict'][criterion_ele]
                elif criterion_name == "loss_indi_theory":
                    for i in range(num_theories):
                        if i in all_losses_dict["loss_indi_theory"]:
                            loss_cumu_dict["{0}_theory_{1}".format(loss_core, i)].append(all_losses_dict["loss_indi_theory"][i])
                else:
                    if criterion_name in loss_keys or criterion_name == "loss_best":
                        if criterion_name not in loss_cumu_dict:
                            loss_cumu_dict[criterion_name] = []
                        loss_cumu_dict[criterion_name].append(all_losses_dict[criterion_name])
        last_epoch = iter_cumu_list[-1] if len(iter_cumu_list) else 0
        if "iter" in info_dict[key]:
            iter_list = np.array(info_dict[key]["iter"]) + last_epoch + 1
        else:
            iter_list = np.arange(len(all_losses_dict_list)) + last_epoch + 1
        iter_cumu_list = np.concatenate((iter_cumu_list, iter_list))

        iter_end_list.append(iter_cumu_list[-1])

    loss_processed = {"loss_cumu_dict": loss_cumu_dict,
                      "iter_cumu_list": iter_cumu_list,
                      "iter_end_list": iter_end_list,
                     }
    if isplot:
        if not is_show:
            import matplotlib
            matplotlib.use('Agg')
        if plt is None:
            import matplotlib.pyplot as plt
            plt.figure(figsize = kwargs["figsize"] if "figsize" in kwargs else (20, 12))
        fontsize = kwargs["fontsize"] if "fontsize" in kwargs else 16
        main_linestyle = kwargs["main_linestyle"] if "main_linestyle" in kwargs else "-"
        indi_linestyle = kwargs["indi_linestyle"] if "indi_linestyle" in kwargs else "-"
        vertical_major_linestyle = kwargs["vertical_major_linestyle"] if "vertical_major_linestyle" in kwargs else "-"
        vertical_minor_linestyle = kwargs["vertical_minor_linestyle"] if "vertical_minor_linestyle" in kwargs else "--"
        main_color = kwargs["main_color"] if "main_color" in kwargs else "k"
        main_linewidth = kwargs["main_linewidth"] if "main_linewidth" in kwargs else 3
        ylabel = kwargs["ylabel"] if "ylabel" in kwargs else "loss"
        linewidth = kwargs["linewidth"] if "linewidth" in kwargs else 1
        title = kwargs['title'] if 'title' in kwargs else None
        main_label = kwargs["main_label"] if "main_label" in kwargs else None
        is_plot_best = kwargs["is_plot_best"] if "is_plot_best" in kwargs else True
        is_only_main = kwargs["is_only_main"] if "is_only_main" in kwargs else False
        # Main model:
        for loss_key in loss_keys:
            if not is_rmse:
                plt.semilogy(iter_cumu_list, loss_cumu_dict[loss_key], color = main_color, linestyle = main_linestyle, linewidth = main_linewidth, label = main_label if main_label is not None else loss_key)
            else:
                plt.semilogy(iter_cumu_list, np.sqrt(loss_cumu_dict[loss_key]), color = main_color, linestyle = main_linestyle, linewidth = main_linewidth, label = main_label if main_label is not None else loss_key)
        if not is_only_main:
            # Loss best:
            if is_plot_best:
                if not is_rmse:
                    plt.semilogy(iter_cumu_list, loss_cumu_dict["loss_best"], color = main_color, linestyle = "--", label = "loss_best")
                else:
                    plt.semilogy(iter_cumu_list, np.sqrt(loss_cumu_dict["loss_best"]), color = main_color, linestyle = "--", label = "loss_best")
            # Individual theory:
            for i in range(num_theories):
                if not is_rmse:
                    plt.semilogy(iter_cumu_list[:len(loss_cumu_dict["{0}_theory_{1}".format(loss_core, i)])], loss_cumu_dict["{0}_theory_{1}".format(loss_core, i)], color = COLOR_LIST[i], linestyle = indi_linestyle, label = "theory_{0}".format(i))
                else:
                    plt.semilogy(iter_cumu_list[:len(loss_cumu_dict["{0}_theory_{1}".format(loss_core, i)])], np.sqrt(loss_cumu_dict["{0}_theory_{1}".format(loss_core, i)]), color = COLOR_LIST[i], linestyle = indi_linestyle, label = "theory_{0}".format(i))
            # Vertical lines:
            for iter_end in iter_end_list:
                plt.axvline(iter_end, color='k', linestyle = vertical_minor_linestyle, linewidth = linewidth)
        plt.xlabel("epochs", fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        if 'data_record_simplification-model' in info_dict:
            data_record = info_dict['data_record_simplification-model']
            iter_last_end = loss_processed["iter_cumu_list"][-1]
            last_idx = plot_loss_record(data_record,
                                        num_models = len(data_record['all_losses_dict'][0]['loss_indi_theory']),
                                        plt = plt,
                                        iter_last_end = iter_last_end,
                                        plot_reject = plot_reject,
                                        is_rmse = is_rmse,
                                        **kwargs
                                       )
            if not is_only_main:
                plt.axvspan(iter_last_end, last_idx, alpha = 0.15, color = 'gray')
        plt.legend(bbox_to_anchor = kwargs["bbox_to_anchor"] if "bbox_to_anchor" in kwargs else None, fontsize = fontsize)
        if title is not None:
            plt.title(title, fontsize = fontsize)
        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])
        ax = plt.gca()
        ax.tick_params(axis='both', labelsize =fontsize)
        if filename is not None:
            plt.savefig(filename, format = kwargs["format"] if "format" in kwargs else "pdf", transparent=True)
        if is_show and plt is None:
            plt.show()
        if plt is None:
            plt.clf()
            plt.close()
    return loss_processed


def plot_loss_record(data_record, num_models, plt = None, iter_last_end = 0, plot_reject = False, is_rmse = False, **kwargs):
    from AI_physicist.settings.global_param import COLOR_LIST
    def get_cumu(List, start_id = -1):
        array = np.array(List)
        array_cumu = np.cumsum(array) + start_id
        return array_cumu

    def get_last_pivot_model(is_accept_whole):
        if len(is_accept_whole) == 0:
            return None
        pivot_idx = -1
        for i in range(len(is_accept_whole) - 1, -1, -1):
            if is_accept_whole[i] is True:
                pivot_idx = i
                break
        return pivot_idx

    indi_linestyle = kwargs["indi_linestyle"] if "indi_linestyle" in kwargs else "-"
    main_linestyle = kwargs["main_linestyle"] if "main_linestyle" in kwargs else "-"
    fontsize = kwargs["fontsize"] if "fontsize" in kwargs else 16
    vertical_major_linestyle = kwargs["vertical_major_linestyle"] if "vertical_major_linestyle" in kwargs else "-"
    vertical_minor_linestyle = kwargs["vertical_minor_linestyle"] if "vertical_minor_linestyle" in kwargs else "--"
    reject_linestyle = kwargs["reject_linestyle"] if "reject_linestyle" in kwargs else "--"
    main_color = kwargs["main_color"] if "main_color" in kwargs else "k"
    main_linewidth = kwargs["main_linewidth"] if "main_linewidth" in kwargs else 3
    linewidth = kwargs["linewidth"] if "linewidth" in kwargs else 1
    is_only_main = kwargs["is_only_main"] if "is_only_main" in kwargs else False
    ylabel = kwargs["ylabel"] if "ylabel" in kwargs else None
    model_keys = ["model_{0}".format(k) for k in range(num_models)]
    model_last_mse = {}

    if plt is None:
        import matplotlib.pylab as plt
        plt.figure(figsize = (20,9))
    start_id = iter_last_end
    fraction_list_domain = data_record['all_losses_dict'][0]['fraction_list_domain']
    for j, loss_record in enumerate(data_record["loss_record"]):
        mse_dict_all = {}
        for k, model_key in enumerate(model_keys):
            start_id_initial = start_id
            if model_key not in loss_record:
                continue
            loss_dict_model = loss_record[model_key]
            for mode_ele, loss_dict_mode_ele in loss_dict_model.items():
                mse_record_whole = loss_dict_mode_ele["mse_record_whole"]
                iter_cumu = get_cumu(loss_dict_mode_ele["iter_end_whole"])
                is_accept_whole = loss_dict_mode_ele["is_accept_whole"]
                pivot_idx = get_last_pivot_model(is_accept_whole)
                if len(mse_record_whole) == 0:
                    continue
                if pivot_idx is None:
                    x_axis = np.arange(len(mse_record_whole)) + start_id

                    # Record the indi theory's loss for future summation:
                    if mode_ele not in mse_dict_all:
                        mse_dict_all[mode_ele] = {}
                    mse_dict_all[mode_ele][k] = {"x_axis": x_axis, "mse_indi": mse_record_whole}

                    label = model_key if j == 0 else None
                    if not is_only_main:
                        if not is_rmse:
                            plt.semilogy(x_axis, mse_record_whole, color = COLOR_LIST[k], linestyle = indi_linestyle, label = label if plt is None else None)
                        else:
                            plt.semilogy(x_axis, np.sqrt(mse_record_whole), color = COLOR_LIST[k], linestyle = indi_linestyle, label = label if plt is None else None)
                        for i in range(len(loss_dict_mode_ele["iter_end_whole"])):
                            plt.axvline(iter_cumu[i] + start_id, color = "k", linestyle = vertical_minor_linestyle, linewidth = linewidth, alpha = 0.4)
                    start_id += iter_cumu[-1]
                    latest_mse = mse_record_whole[-1]
                    if not is_only_main:
                        plt.axvline(start_id, color = "k", linestyle = vertical_major_linestyle, linewidth = linewidth, alpha = 1)
                    model_last_mse[model_key] = latest_mse
                    mse_dict_all[mode_ele][k]["last_mse"] = latest_mse
                else:
                    # Plot loss for accepted models:
                    accepted_idx = iter_cumu[pivot_idx + 1]
                    x_axis = np.arange(accepted_idx) + start_id

                    # Record the indi theory's loss for future summation:
                    if mode_ele not in mse_dict_all:
                        mse_dict_all[mode_ele] = {}
                    mse_dict_all[mode_ele][k] = {"x_axis": x_axis, "mse_indi": mse_record_whole[:accepted_idx]}

                    label = model_key if j == 0 else None
                    if not is_only_main:
                        if not is_rmse:
                            plt.semilogy(x_axis, mse_record_whole[:accepted_idx], color = COLOR_LIST[k], linestyle = indi_linestyle, label = label if plt is None else None)
                        else:
                            plt.semilogy(x_axis, np.sqrt(mse_record_whole[:accepted_idx]), color = COLOR_LIST[k], linestyle = indi_linestyle, label = label if plt is None else None)
                        for i in range(pivot_idx + 2):
                            plt.axvline(iter_cumu[i] + start_id, color = "k", linestyle = vertical_minor_linestyle, linewidth = linewidth, alpha = 0.4)  

                    # Plot loss for tentative but rejected models:
                    if plot_reject:
                        rejected_idx = iter_cumu[-1]
                        if rejected_idx > accepted_idx:
                            x_axis = np.arange(accepted_idx, rejected_idx + 1) + start_id
                            label = model_key if j == 0 else None
                            if not is_only_main:
                                if not is_rmse:
                                    plt.semilogy(x_axis, mse_record_whole[accepted_idx:], color = COLOR_LIST[k], linestyle = reject_linestyle, label = label if plt is None else None)
                                else:
                                    plt.semilogy(x_axis, np.sqrt(mse_record_whole[accepted_idx:]), color = COLOR_LIST[k], linestyle = reject_linestyle, label = label if plt is None else None)
                                for i in range(pivot_idx + 2, len(loss_dict_mode_ele["iter_end_whole"])):
                                    plt.axvline(iter_cumu[i] + start_id, color = "k", linestyle = vertical_minor_linestyle, linewidth = linewidth, alpha = 0.4)
                                if not is_rmse:
                                    plt.plot([start_id + accepted_idx, start_id + rejected_idx + 1], [mse_record_whole[accepted_idx], mse_record_whole[accepted_idx]], color = COLOR_LIST[k], linestyle = indi_linestyle)
                                else:
                                    plt.plot([start_id + accepted_idx, start_id + rejected_idx + 1], [np.sqrt(mse_record_whole[accepted_idx]), np.sqrt(mse_record_whole[accepted_idx])], color = COLOR_LIST[k], linestyle = indi_linestyle)

                        start_id += iter_cumu[-1]
                        latest_mse = mse_record_whole[accepted_idx]
                        if not is_only_main:
                            plt.axvline(start_id, color = "k", linestyle = vertical_major_linestyle, linewidth = linewidth, alpha = 1)
                        model_last_mse[model_key] = latest_mse
                        mse_dict_all[mode_ele][k]["last_mse"] = latest_mse
                    else:
                        start_id += iter_cumu[pivot_idx + 1]
                        latest_mse = mse_record_whole[accepted_idx]
                        if not is_only_main:
                            plt.axvline(start_id, color = "k", linestyle = vertical_major_linestyle, linewidth = linewidth, alpha = 1)
                        model_last_mse[model_key] = latest_mse
                        mse_dict_all[mode_ele][k]["last_mse"] = latest_mse

            for model_key_other in model_keys:
                if model_key_other != model_key:
                    if model_key_other in model_last_mse:
                        if not is_only_main:
                            if not is_rmse:
                                plt.plot([start_id_initial, start_id], [model_last_mse[model_key_other], model_last_mse[model_key_other]], color = COLOR_LIST[model_keys.index(model_key_other)], linestyle = indi_linestyle)
                            else:
                                plt.plot([start_id_initial, start_id], [np.sqrt(model_last_mse[model_key_other]), np.sqrt(model_last_mse[model_key_other])], color = COLOR_LIST[model_keys.index(model_key_other)], linestyle = indi_linestyle)

        for mode_ele, item in mse_dict_all.items():
            for k in range(num_models):
                mse_indi_dict = {}
                if k not in item:
                    continue
                axis_indi = item[k]["x_axis"]
                mse_indi_dict[k] = item[k]["mse_indi"]
                for kk in range(k):
                    mse_indi_dict[kk] = mse_dict_all[mode_ele][kk]["last_mse"] * np.ones(len(axis_indi))
                for kk in range(k + 1, num_models):
                    mse_indi_dict[kk] = data_record["all_losses_dict"][j]['loss_indi_theory'][kk] * np.ones(len(axis_indi))
                mse_indi_list = [np.array(mse_indi_dict[kk]) for kk in range(num_models)]
                mse_whole = np.dot(np.array(mse_indi_list).T, fraction_list_domain)
                if not is_rmse:
                    plt.semilogy(axis_indi, mse_whole, color = main_color, linestyle = main_linestyle, linewidth = main_linewidth, label = label if plt is None else None)
                else:
                    plt.semilogy(axis_indi, np.sqrt(mse_whole), color = main_color, linestyle = main_linestyle, linewidth = main_linewidth, label = label if plt is None else None)

    plt.legend()
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize =fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize = fontsize)
    if plt is None:
        plt.show()
    return start_id


def to_one_hot(idx, num):
    if len(idx.size()) == 1:
        idx = idx.unsqueeze(-1)
    if not isinstance(idx, Variable):
        if isinstance(idx, np.ndarray):
            idx = torch.LongTensor(idx)
        idx = Variable(idx, requires_grad = False)
    onehot = Variable(torch.zeros(idx.size(0), num), requires_grad = False)
    if idx.is_cuda:
        onehot = onehot.cuda()
    onehot.scatter_(1, idx, 1)
    return onehot.long()


def load_info_dict(exp_id, datetime, filename, exp_mode = None):
    if exp_mode is not None:
        dirname = theory_PATH + "/{0}_{1}_{2}/".format(exp_id, exp_mode, datetime)
    else:
        dirname = theory_PATH + "/{0}_{1}/".format(exp_id, datetime)
    return pickle.load(open(dirname + filename, "rb"))
    

def get_piecewise_dataset(
    input_size,
    num_pieces,
    num_boundaries = 3,
    func_types = ["linear"],
    x_range = (-10, 10),
    num_examples = 10000,
    isTorch = True,
    is_cuda = False,
    isplot = True,
    **kwargs
    ):
    class Boundary(object):
        def __init__(self, X):
            input_size = X.shape[1]
            min_ratio = 0.1
            for i in range(10):
                self.slope = np.random.rand(input_size)
                self.point = random.choice(X) + np.random.randn(input_size)
                result = (np.dot(X, self.slope) - np.dot(self.point, self.slope)) >= 0
                if min_ratio <= result.sum() / float(len(result)) <= 1 - min_ratio:
                    break

        def __call__(self, X):
            return ((np.dot(X, self.slope) - np.dot(self.point, self.slope)) >= 0).astype(int)

    if isinstance(x_range, list):
        assert len(x_range) == input_size
    else:
        x_range = [x_range for i in range(input_size)]

    X = np.random.rand(num_examples, input_size)
    for i in range(input_size):
        X[:, i] = X[:, i] * (x_range[i][1] - x_range[i][0]) + x_range[i][0]

    boundary_list = []
    for i in range(num_boundaries):
        boundary_list.append(Boundary(X))

    func_list = []
    for i in range(num_pieces):
        func_type = random.choice(func_types)
        func_list.append({"function": get_function(func_type, input_size = input_size), "func_type": func_type})

    region_dict = {np.binary_repr(i, width = num_boundaries): random.choice(range(num_pieces)) for i in range(2 ** num_boundaries)}
    regions = np.stack([boundary_list[i](X) for i in range(num_boundaries)],1)
    regions = ["".join([str(ele) for ele in element]) for element in regions]

    y = np.stack([func_list[region_dict[region]]["function"](x) for x, region in zip(X, regions)])
    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.2)

    if isTorch:
        X_train = Variable(torch.FloatTensor(X_train), requires_grad = False)
        y_train = Variable(torch.FloatTensor(y_train), requires_grad = False)
        X_test = Variable(torch.FloatTensor(X_test), requires_grad = False)
        y_test = Variable(torch.FloatTensor(y_test), requires_grad = False)
        if is_cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()
    info = {"input_size": input_size}
    if isplot:
        import matplotlib.pylab as plt
        from mpl_toolkits.mplot3d import Axes3D
        if input_size == 1:
            plt.plot(X[:, 0], y[:,0], ".")
        else:
            fig = plt.figure(figsize = (7 * (input_size - 1),6))
            view_init = kwargs["view_init"] if "view_init" in kwargs else None
            for i in range(input_size - 1):
                ax = fig.add_subplot(1, input_size - 1, i + 1, projection='3d')
                ax.scatter(X[:,i], X[:,i+1], y, ".", s = 1)
                if view_init is not None:
                    ax.view_init(view_init[0], view_init[1])
            plt.show()
    return ((X_train, y_train), (X_test, y_test), (None, None)), info


def prepare_data_from_file(filename, time_steps = 3, output_dims = 0, is_flatten = True, is_cuda = False):
    from sklearn.model_selection import train_test_split
    trajectory = np.genfromtxt(filename, delimiter=',')
    X = []
    y = []
    is_nan = []
    for i in range(len(trajectory) - time_steps):
        X.append(trajectory[i: i + time_steps])
        y.append(trajectory[i + time_steps: i + time_steps + 1])
        is_nan.append(np.any(np.isnan(trajectory[i: i + time_steps + 1])))
    X = np.array(X)
    y = np.array(y)
    valid = ~np.array(is_nan)
    X = X[valid]
    y = y[valid]
    if output_dims is not None:
        if not isinstance(output_dims, list):
            output_dims = [output_dims]
        y = y[..., np.array(output_dims)]

    if is_flatten:
        X = X.reshape(X.shape[0], -1)
        y = y.reshape(y.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_train = Variable(torch.FloatTensor(X_train), requires_grad = False)
    y_train = Variable(torch.FloatTensor(y_train), requires_grad = False)
    X_test = Variable(torch.FloatTensor(X_test), requires_grad = False)
    y_test = Variable(torch.FloatTensor(y_test), requires_grad = False)
    if is_cuda:
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()
    input_size = tuple(X_train.size()[1:])
    if len(input_size) == 1:
        input_size = input_size[0]
    return ((X_train, y_train), (X_test, y_test), (None, None)), {"input_size": input_size, "file_source": filename}


def transform_data_to_phase_space(matrix, dt = Dt):
    """qx1, qdotx1, qy1, qdoty1, qx2, qdotx2, qy2, qdoty2"""
    assert len(matrix.shape) == 2
    assert matrix.shape[-1] == 8 or matrix.shape[-1] == 9
    qdotx1 = (matrix[:,4] - matrix[:, 2]) / dt
    qdotx2 = (matrix[:,6] - matrix[:, 4]) / dt
    qdoty1 = (matrix[:,5] - matrix[:, 3]) / dt
    qdoty2 = (matrix[:,7] - matrix[:, 5]) / dt
    qx1 = (matrix[:, 2] + matrix[:, 4]) / 2
    qx2 = (matrix[:, 4] + matrix[:, 6]) / 2
    qy1 = (matrix[:, 3] + matrix[:, 5]) / 2
    qy2 = (matrix[:, 5] + matrix[:, 7]) / 2
    if isinstance(matrix, np.ndarray):
        data = np.stack([qx1, qdotx1, qy1, qdoty1, qx2, qdotx2, qy2, qdoty2], 1)
        if matrix.shape[-1] == 9:
            data = np.concatenate([data, matrix[:, -1:]], 1)
    else:
        data = torch.stack([qx1, qdotx1, qy1, qdoty1, qx2, qdotx2, qy2, qdoty2], 1)
        if matrix.shape[-1] == 9:
            data = torch.cat([data, matrix[:, -1:]], 1)
    return data


def get_epochs(batch_size, exp_mode):
    if batch_size is None:
        epochs = 15000
    elif batch_size < 300:
        epochs = 2000
    elif 300 <= batch_size < 1000:
        epochs = 3000
    elif 1000 <= batch_size < 3000:
        epochs = 5000
    elif 3000 <= batch_size < 10000:
        epochs = 8000
    elif 10000 <= batch_size < 15000:
        epochs = 10000
    else:
        epochs = 15000
    if "base" in exp_mode:
        epochs = epochs * 5
    print("epochs: {0}".format(epochs))
    return epochs


def get_theory_losses(theory, X, y, criterion, mode = "indi", is_cuda = False):
    pred, valid = theory(X)
    if valid.sum().data[0] == 0:
        return np.NaN, np.NaN

    if mode == "indi":
        y_valid = y[valid].view(-1, y.size(1))
        pred_valid = pred[valid].view(-1, pred.size(1))[:,:1]
        if pred.size(1) > 1:
            log_std_valid = pred[valid].view(-1, pred.size(1))[:,1:]
    elif mode == "whole":
        y_valid = y
        if criterion.name == "Loss_with_uncertainty":
            pred_valid = pred[:,:1]
            log_std_valid = pred[:,1:]
        elif criterion.name == "Loss_Fun_Cumu":
            pred_valid = pred
    else:
        raise Exception("mode {0} not recognized!".format(mode))


    if criterion.name == "Loss_with_uncertainty":
        loss = criterion(pred = pred_valid, target = y_valid, log_std = log_std_valid)
    elif criterion.name == "Loss_Fun_Cumu":
        loss = criterion(pred = pred_valid, target = y_valid)
    else:
        raise Exception("criterion's name {0} not valid!".format(criterion.name))

    if mode == "whole" and criterion.name == "Loss_Fun_Cumu":
        mse = Loss_Fun_Cumu(core = "mse", cumu_mode = criterion.cumu_mode)(pred = pred_valid, target = y_valid)
    else:
        mse = nn.MSELoss(size_average = True)(input = pred_valid, target = y_valid)
    if is_cuda:
        loss = loss.cpu()
        mse = mse.cpu()
    return loss.data.numpy()[0], mse.data.numpy()[0]


def plot3D_data(X, view_init = (0,195), inline = True, is_cuda = False, alpha = 0.5, figsize = (8, 7), is_show = True):
    if not is_show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if not inline:
        get_ipython().run_line_magic('matplotlib', 'qt')
    X = X.data.numpy()
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c='b', marker = ".", s = 3, alpha = alpha)
    ax.view_init(view_init[0], view_init[1])
    if is_show:
        plt.show()
    plt.clf()
    plt.close()
    return ax


def plot3D(X, y, idx = None, view_init = None, axis_lim = None, plot_type = "scatter", axis_title = None, inline = True, figsize = (7, 6), is_cuda = False, is_show = True, filename = None):
    if not is_show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if not inline:
        get_ipython().run_line_magic('matplotlib', 'qt')
    if not is_show:
        plt.ioff()
    X, y = to_np_array(X, y)
    if idx is not None:
        idx = to_np_array(idx).astype(bool)
        if y.shape[1] > 1 and idx.shape[1] == 1:
            idx = np.repeat(idx, y.shape[1], axis = 1)
    else:
        idx = np.ones((y.shape[0], y.shape[1])).astype(bool)

    dim = X.shape[1]
    fig = plt.figure(figsize = (figsize[0] * dim, figsize[1]))
    if axis_lim is None:
        obtain_new_axis = True
        axis_lim = {}
    else:
        obtain_new_axis = False

    for i in range(dim - 1):
        ax = fig.add_subplot(1, dim, i + 1, projection='3d')
        if obtain_new_axis:
            xlim = (np.floor(X[:,i].min()) - 0.5, np.ceil(X[:,i].max()) + 0.5)
            ylim = (np.floor(X[:,i+1].min()) - 0.5, np.ceil(X[:,i+1].max()) + 0.5)
            zlim = (np.floor(y.min()) - 1, np.ceil(y.max()) + 1)
            axis_lim[i] = [xlim, ylim, zlim]
        else:
            xlim = axis_lim[i][0]
            ylim = axis_lim[i][1]
            zlim = axis_lim[i][2]
        for j in range(y.shape[1]):
            if plot_type == "scatter":
                ax.scatter(X[idx[:,j],i], X[idx[:,j], i+1], y[idx[:,j], j], c = COLOR_LIST[j % len(COLOR_LIST)], marker = ".", s = 3, alpha = 0.5, label = str(j))
            elif plot_type == "surface":
                ax.plot_trisurf(X[idx[:,j],i], X[idx[:,j], i+1], y[idx[:,j], j], color = COLOR_LIST[j % len(COLOR_LIST)], alpha = 0.5, label = str(j))
            else:
                raise Exception("plot_type {0} not recognized!".format(plot_type))
        ax.legend(bbox_to_anchor = (0.85, 0.6, 0.2,0.2))
        if view_init is not None:
            ax.view_init(view_init[0], view_init[1])
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_zlim(zlim[0], zlim[1])
        if axis_title is not None:
            if i < len(axis_title):
                ax.set_title(axis_title[i])
    if filename is not None:
        plt.savefig(filename)
    if is_show:
        plt.show()
    plt.clf()
    plt.close()
    return axis_lim


def plot_indi_domain(X, domain, max_plots = None, is_show = True, filename = None, images_per_row = 4, **kwargs):
    """Plot the domains assuming that X represents trajetories on 2D"""
    if not is_show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pylab as plt
    X = to_np_array(X)
    if len(X.shape) < 3:
        X = X.reshape(X.shape[0], -1, 2)
    if "xlim" in kwargs and kwargs["xlim"] is not None:
        xlim = kwargs["xlim"]
    else:
        xlim = (X[:,:,0].min(), X[:,:,0].max())
    if "ylim" in kwargs and kwargs["ylim"] is not None:
        ylim = kwargs["ylim"]
    else:
        ylim = (X[:,:,1].min(), X[:,:,1].max())
    domain = to_np_array(domain.squeeze())
    unique_domains = np.unique(domain)
    num_plots = len(unique_domains) if max_plots is None else min(max_plots, len(unique_domains))
    rows = int(np.ceil(num_plots / float(images_per_row)))
    row_height = kwargs["row_height"] if "row_height" in kwargs else 4.5
    row_width = kwargs["row_width"] if "row_width" in kwargs else 20
    plt.figure(figsize = (row_width, rows * row_height))
    for j, domain_id in enumerate(unique_domains):
        if j >= num_plots:
            break
        plt.subplot(rows, images_per_row, j + 1)
        plt.xlim(xlim)
        plt.ylim(ylim)
        num_example_domain = (domain == domain_id).sum()
        for i in range(X.shape[0]):
            if domain[i] == domain_id:
                plt.plot(X[i][:,0], X[i][:,1], ".-", color = COLOR_LIST[domain[i] % len(COLOR_LIST)], alpha = 0.5, markersize = 1, linewidth = 1)
        plt.title("domain {0}: #={1:.0f},   {2:.4f}%".format(domain_id, num_example_domain, num_example_domain / float(len(domain)) * 100))
    if filename is not None:
        plt.savefig(filename)
    if is_show:
        plt.show()
    plt.clf()
    plt.close()


def plot_domain(domain_model, axis_lim = None, X = None, threshold = 0.01, plot_type = "scatter", inline = True, view_init = (0,195), is_cuda = False, is_show = True):
    if not is_show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if not inline:
        get_ipython().run_line_magic('matplotlib', 'qt')
    if is_cuda:
        X = X.cpu()
    if axis_lim is not None:
        xlim, ylim, zlim = axis_lim
    else:
        assert X is not None
        xlim = (np.floor(X[:,0].data.min()) - 1, np.ceil(X[:,0].data.max()) + 1)
        ylim = (np.floor(X[:,1].data.min()) - 1, np.ceil(X[:,1].data.max()) + 1)
        zlim = (np.floor(X[:,2].data.min()) - 1, np.ceil(X[:,2].data.max()) + 1)
    xlist = np.linspace(xlim[0], xlim[1], 200)
    ylist = np.linspace(ylim[0], ylim[1], 200)
    zlist = np.linspace(zlim[0], zlim[1], 200)
    XX, YY, ZZ = np.meshgrid(xlist, ylist, zlist)
    Input = np.stack([XX, YY, ZZ], 3).reshape(-1,3)
    Input_torch = Variable(torch.FloatTensor(Input), requires_grad = False)
    if is_cuda:
        Input_torch = Input_torch.cuda()
    pred = nn.Softmax(dim = 1)(domain_model(Input_torch))[:,1]
    if is_cuda:
        pred = pred.cpu()
    boundary_idx = (torch.abs(pred.data - 0.5) <= threshold).numpy().astype(bool)
    if boundary_idx.sum() >= 10:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        boundary = Input[boundary_idx]
        if plot_type == "scatter":
            ax.scatter(boundary[:,0], boundary[:,1], boundary[:,2], c='g', marker = ".", alpha = 0.4, s = 2)
        elif plot_type == "surface":
            ax.plot_trisurf(boundary[:,0], boundary[:,1], boundary[:,2], color='g', alpha = 0.4)
        else:
            raise Exception("plot_type {0} not recognized!".format(plot_type))
        if X is not None:
            if is_cuda:
                X = X.cpu()
            X = X.data.numpy()
            ax.scatter(X[:,0], X[:,1], X[:,2], c='b', marker = ".", s = 3, alpha = 1)
        ax.view_init(view_init[0], view_init[1])
        if is_show:
            plt.show()
        plt.clf()
        plt.close()
    return ax


def plot_pred(X, target, model, idx, enable_std = True, is_logstd = True, is_show = True, color = "r", label = "theory_{0}", markersize = 3, ylim = None, alpha = 0.6, is_cuda = False):
    if not is_show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if is_cuda:
        target = target.cpu()
        idx = idx.cpu()
    if isinstance(target, Variable):
        target = target.data
    if isinstance(idx, Variable):
        idx = idx.data
    preds, valid = model(X)
    if is_cuda:
        X = X.cpu()
        preds = preds.cpu()
        valid = valid.cpu()
    preds = preds.data
    valid = valid.squeeze().data
    if len(valid.size()) > 1: # The model is the theory ensemble, and all data are valid:
        chosen_idx = idx
    else: # The model is individual theory
        chosen_idx = valid & idx

    ax = plt.gca()
    ylim = (np.floor(target.min()) - 5, np.ceil(target.max()) + 5) if ylim is None else ylim
    t_target = target[chosen_idx]
    t_pred = preds[:,0][chosen_idx]
    if enable_std:
        t_std = preds[:,1][chosen_idx]
        if (t_std < 0).any():
            is_logstd = True
        if is_logstd:
            t_std = torch.exp(t_std)
        ax.errorbar(t_target.numpy(), t_pred.numpy(), yerr= t_std.numpy(), fmt='{0}o'.format(color), label = label, markersize= markersize, alpha = alpha)
    else:
        ax.plot(t_target.numpy(), t_pred.numpy(), '{0}.'.format(color), label = label, markersize= markersize, alpha = alpha)
    plt.ylim(ylim)
    if is_show:
        plt.show()
    plt.clf()
    plt.close()


def plot_theories(X, y, idx, theory, enable_std = True, markersize = 3, ylim = None, alpha = 0.6, is_cuda = False, is_show = True):
    if not is_show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if idx is None:
        idx = torch.ones(X.size(0)).byte()
    colors = ["b", "r", "y", "c", "m", "g"]
    for i in range(theory.num_theories):
        plot_pred(X, y, getattr(theory, "theory_{0}".format(i)), idx, enable_std = enable_std, is_show = False, color = colors[i % len(colors)], label = "theory_{0}".format(i), markersize = markersize, ylim = ylim, alpha = alpha, is_cuda = is_cuda)
    if is_cuda:
        plt.plot(y.cpu().data.numpy(), y.cpu().data.numpy())
    else:
        plt.plot(y.data.numpy(),y.data.numpy())
    if enable_std:
        plot_pred(X, y, theory, idx, enable_std = enable_std, is_show = False, color = 'k', label = "combined", markersize = markersize, ylim = ylim, alpha = 0.3, is_cuda = is_cuda)
    plt.legend()
    if is_show:
        plt.show()
    plt.clf()
    plt.close()
    

def split_data(data, model, criteria = {"loss_with_uncertainty": -0.5}, criterion = None, no_data = False):
    if len(data) == 3:
        X, y, reflected = data
    elif len(data) == 2:
        X, y = data
    selected_list = []
    for criterion_name, threshold in criteria.items():
        if criterion_name == "std":
            std = torch.exp(model(X)[:,1:])
            selected_indi = (std < threshold).data
        elif criterion_name == "loss_with_uncertainty":
            assert criterion.name == "Loss_with_uncertainty"
            pred, valid = model(X)
            loss_with_uncertainty = criterion(pred = pred[:,:1], target = y, log_std = pred[:,1:], is_mean = False)
            selected_indi = (loss_with_uncertainty < threshold).data
        elif criterion_name == "pred-based":
            assert criterion.name == "Loss_Fun_Cumu"
            pred, valid = model(X)
            loss_cumu = criterion(pred = pred[:,:1], target = y, is_mean = False)
            selected_indi = (loss_cumu < threshold).data
        else:
            raise Exception("criterion {0} not recognized!".format(criterion_name))
        selected_list.append(selected_indi)
    selected = (torch.cat(selected_list,1).sum(1) > 0).unsqueeze(1)
    
    if no_data:
        return None, None, selected
    else:
        X_fitted = X[selected].view(-1, X.size(1))
        y_fitted = y[selected].view(-1, y.size(1))
        X_tofit = X[~selected].view(-1, X.size(1))
        y_tofit = y[~selected].view(-1, y.size(1))
        if len(data) == 3:
            reflected_fitted = reflected[selected[:,0]]
            reflected_tofit = reflected[~selected[:,0]]
            return (X_fitted, y_fitted, reflected_fitted), (X_tofit, y_tofit, reflected_tofit), selected
        else:
            return (X_fitted, y_fitted), (X_tofit, y_tofit), selected


def get_domain_ratio(model, X, mode = "hard"):
    if mode == "hard":
        valid = model.domain_model.predict(X)
    elif mode == "soft":
        valid = model.domain_model.predict_proba(X)
    else:
        raise Exception("mode {0} not recognized!".format(mode))

    domain_ratio = valid.data.sum() / len(valid)
    return domain_ratio


def sort_multiple_lists(matrix):
    from operator import itemgetter
    return [list(x) for x in zip(*sorted(zip(matrix[:,0], matrix[:,1], matrix[:,2]), key = itemgetter(0)))]


def get_remaining_ids(excluded_ids, num_theories, isTorch = False):
    if not isinstance(excluded_ids, list):
        excluded_ids = [excluded_ids]
    remaining_ids = np.array([i for i in range(num_theories) if i not in excluded_ids])
    if isTorch:
        remaining_ids = torch.LongTensor(remaining_ids)
    return remaining_ids


def export_csv_with_domain_prediction(
    env_name,
    domain_net=None,
    info_dict=None,
    num_output_dims=2,
    num_input_steps=3,
    csv_dirname="../datasets/MYSTERIES/",
    write_dirname="",
    is_Lagrangian=False,
    is_cuda=False,
):
    matrix = np.genfromtxt(csv_dirname + env_name + ".csv", delimiter=',')
    if is_Lagrangian:
        X = matrix[:,:8]
        y = np.zeros((len(X), num_output_dims))
    else:
        X = matrix[:,-(num_output_dims + 1) * num_input_steps: -num_output_dims]
        y = matrix[:, -num_output_dims:]
    if domain_net is None:
        domain_net = load_model_dict(info_dict[env_name]['domain_net'])
    domain_pred = to_np_array(domain_net(to_Variable(X, is_cuda = is_cuda)).max(1)[1].unsqueeze(1))
    combined = np.concatenate((X, y, domain_pred), 1)
    np.savetxt(write_dirname + env_name + "_domain.csv", combined, delimiter=',')
    print("\nData with domain prediction saved to " + csv_dirname + write_dirname + env_name + "_domain.csv" + "\n") 
    

def extract_target_expression(target_csv, csv_dirname = "../../mystery_data/"):
    target_dict = {}
    with open(csv_dirname + target_csv, "r") as f:
        for line in f:
            if len(line.split('"')) > 1:
                mystery_name = "mystery{0}".format(line.split('"')[0].split(",")[0])
                target = eval(line.split('"')[1])
                target_standardize = []
                for target_ele in target:
                    target_ele_standardize = "+".join(["{0}*x{1}".format(value, idx) for idx, value in enumerate(target_ele)][:-1])
                    target_ele_standardize += "+{0}".format(target_ele[-1])
                    target_standardize.append(standardize_symbolic_expression(target_ele_standardize))
                target_dict[mystery_name] = target_standardize
    return target_dict


def get_coeff_learned_list(pred_nets):
    expressions = pred_nets.get_sympy_expression()
    coeff_learned_list = []
    is_snapped_list = []
    for k in range(pred_nets.num_models):
        expression = expressions["model_{0}".format(k)][0]
        if expression is not None:
            symbolic_expression = expression["symbolic_expression"]
            for j in range(len(symbolic_expression)):
                coeff_learned_list_ele = []
                is_snapped_list_ele = []
                coeffs, variables = get_coeffs(symbolic_expression[j])
                coeffs_numerical, has_param = substitute(coeffs, expression['param_dict'])

                # Get weights:
                for i in range(pred_nets.input_size):
                    variable_name = "x{0}".format(i)
                    if variable_name in variables:
                        idx = variables.index(variable_name)
                        coeff_learned = coeffs_numerical[idx]
                        if has_param[idx] is False:
                            is_snapped_list_ele.append(1)
                            if abs(coeff_learned - int(coeff_learned)) < 2 ** (-32):
                                coeff_learned = int(coeff_learned)
                        else:
                            is_snapped_list_ele.append(0)
                    else:
                        coeff_learned = 0
                        is_snapped_list_ele.append(1)
                    coeff_learned_list_ele.append(coeff_learned)

                # Get bias:
                if len(coeffs) > len(variables):
                    bias = coeffs_numerical[-1]
                    if has_param[-1] is False:
                        is_snapped_list_ele.append(1)
                        if abs(bias - int(bias)) < 2 ** (-32):
                            bias = int(bias)
                    else:
                        is_snapped_list_ele.append(0)
                else:
                    is_snapped_list_ele.append(1)
                    bias = 0
                coeff_learned_list_ele.append(bias)
                coeff_learned_list.append(coeff_learned_list_ele)
                is_snapped_list.append(is_snapped_list_ele)
    return coeff_learned_list, is_snapped_list


def get_tuple(filename):
    filename_split = filename.split("-")
    List = []
    for element in filename_split:
        try:
            element = eval(element)
        except:
            pass
        List.append(element)
    return tuple(List)


def parse_filename(filename, is_tuple = False):
    import pandas as pd
    filename_split = filename.split("/")[-1].split("_")
    parse_dict = {}
    parse_dict["env_name"] = filename_split[0]
    parse_dict["exp_mode"] = filename_split[1]
    parse_dict["num_theories_init"] = int(filename_split[filename_split.index("num") + 1])
    parse_dict["pred_nets_neurons"] = int(filename_split[filename_split.index("pred") + 1])
    parse_dict["pred_nets_activation"] = filename_split[filename_split.index("pred") + 2]
    parse_dict["domain_net_neurons"] = int(filename_split[filename_split.index("dom") + 1])
    parse_dict["domain_pred_mode"] = filename_split[filename_split.index("dom") + 2]
    parse_dict["mse_amp"] = float(filename_split[filename_split.index("mse") + 1])
    
    if is_tuple:
        parse_dict["simplify_criteria"] = get_tuple(filename_split[filename_split.index("sim") + 1])
        if "sched" in filename_split:
            parse_dict["scheduler_settings"] = get_tuple(filename_split[filename_split.index("sched") + 1])
        parse_dict["optim_type"] = get_tuple(filename_split[filename_split.index("optim") + 1])
        parse_dict["optim_domain_type"] = get_tuple(filename_split[filename_split.index("optim") + 2])
    else:
        parse_dict["simplify_criteria"] = filename_split[filename_split.index("sim") + 1]
        if "sched" in filename_split:
            parse_dict["scheduler_settings"] = filename_split[filename_split.index("sched") + 1]
        parse_dict["optim_type"] = filename_split[filename_split.index("optim") + 1]
        parse_dict["optim_domain_type"] = filename_split[filename_split.index("optim") + 2]
    parse_dict["reg_amp"] = float(filename_split[filename_split.index("reg") + 1])
    parse_dict["reg_domain_amp"] = float(filename_split[filename_split.index("reg") + 2])
    parse_dict["batch_size"] = int(filename_split[filename_split.index("batch") + 1])
    parse_dict["loss_core"] = filename_split[filename_split.index("core") + 1]
    if "order" in filename_split:
        parse_dict["loss_order"] = float(filename_split[filename_split.index("order") + 1])
    else:
        parse_dict["loss_order"] = 1
    if "lossd" in filename_split:
        parse_dict["loss_decay_scale"] = filename_split[filename_split.index("lossd") + 1]
        parse_dict["is_mse_decay"] = eval(filename_split[filename_split.index("lossd") + 2])
    else:
        parse_dict["loss_decay_scale"] = "None"
    if "phase" in filename_split:
        parse_dict["num_phases"] = int(filename_split[filename_split.index("phase") + 1])
    if "initp" in filename_split:
        parse_dict["initial_precision_floor"] = float(filename_split[filename_split.index("initp") + 1])
        parse_dict["is_mse_decay"] = eval(filename_split[filename_split.index("initp") + 2])
    parse_dict["loss_balance_model_influence"] = eval(filename_split[filename_split.index("infl") + 1])
    parse_dict["num_examples"] = int(filename_split[filename_split.index("#") + 1])
    parse_dict["iter_to_saturation"] = int(filename_split[filename_split.index("mul") + 1])
    parse_dict["MDL_mode"] = filename_split[filename_split.index("MDL") + 1]
    if "D" in filename_split[filename_split.index("MDL") + 2]:
        parse_dict["num_output_dims"] = int(filename_split[filename_split.index("MDL") + 2][:-1])
    if "L" in filename_split[filename_split.index("MDL") + 3]:
        parse_dict["num_layers"] = int(filename_split[filename_split.index("MDL") + 3][:-1])
    parse_dict["seed"] = int(filename_split[filename_split.index("id") + 1])
    return parse_dict


def parse_filename_unification(filename_unification):
    filename_split = filename_unification.split("/")[-1].split("_")
    parse_dict = {}
    parse_dict["num_master_theories"] = int(filename_split[filename_split.index("num")+ 1])
    parse_dict["master_model_type"] = filename_split[filename_split.index("type")+ 1]
    parse_dict["statistics_output_neurons"] = int(filename_split[filename_split.index("statistics")+ 1])
    parse_dict["master_loss_combine_mode"] = filename_split[filename_split.index("statistics")+ 2]
    parse_dict["master_loss_core"] = filename_split[filename_split.index("core")+ 1]
    parse_dict["master_loss_mode"] = filename_split[filename_split.index("mode")+ 1]
    parse_dict["master_model_num_neurons"] = filename_split[filename_split.index("neurons") + 1]
    parse_dict["master_optim_type"] = tuple(filename_split[filename_split.index("optim") + 1].split("-"))
    parse_dict["master_optim_type_classifier"] = tuple(filename_split[filename_split.index("optim") + 2].split("-"))
    parse_dict["master_reg_amp"] = float(filename_split[filename_split.index("reg") + 1])
    parse_dict["master_reg_amp_classifier"] = float(filename_split[filename_split.index("reg") + 2])
    parse_dict["theory_remove_threshold"] = float(filename_split[filename_split.index("thresh") + 1])
    parse_dict["array_id"] = filename_split[filename_split.index("id") + 1]
    return parse_dict


def check_consistent(
    T,
    info_dict_single,
    dataset,
    compare_keys = ["DL_pred_nets", "DL_domain_net", "DL_data", "mse_with_domain", "loss_with_domain"],
    threshold = [1e-6, 1e-6, 1e-3, 1e-5, 1e-2],
    verbose = False,
    DL_mode = "DL",
    ):
    ((X_train, y_train), (X_test, y_test), _), info = dataset
    num = 3
    U = deepcopy(T)
    U.domain_net_on = False
    data_record = info_dict_single["data_record_1"]
    num_records = len(data_record["all_losses_dict"])
    
    if verbose:
        print("\nfirst stage:")
    for i in range(num):
        ii = int(np.random.randint(num_records))
        U.set_net("pred_nets", load_model_dict(data_record['pred_nets_model_dict'][ii]))
        U.set_net("domain_net", load_model_dict(data_record['domain_net_model_dict'][ii]))
        U.set_loss_core("DLs", 10)
        new_losses_dict = U.get_losses(X_test, y_test, DL_mode = DL_mode)
        record_lossed_dict = data_record["all_losses_dict"][ii]
        is_same, diff_keys = compare_same(new_losses_dict, record_lossed_dict, compare_keys, threshold = threshold, verbose = verbose)
        if not is_same:
            raise Exception("keys not the same: {0} duing first stage.".format(diff_keys))    
    
    if verbose:
        print("\nMDL1:")
    for j, data_record in enumerate(info_dict_single["data_record_MDL1_1"]):
        num_records = len(data_record["all_losses_dict"])
        for i in range(num):
            ii = int(np.random.randint(num_records))
            U.set_net("pred_nets", load_model_dict(data_record['pred_nets_model_dict'][ii]))
            U.set_net("domain_net", load_model_dict(data_record['domain_net_model_dict'][ii]))
            U.set_loss_core("DLs", data_record['loss_precision_floor'])
            new_losses_dict = U.get_losses(X_test, y_test, DL_mode = DL_mode)
            record_lossed_dict = data_record["all_losses_dict"][ii]
            is_same, diff_keys = compare_same(new_losses_dict, record_lossed_dict, compare_keys, threshold = threshold, verbose = verbose)
            if not is_same:
                raise Exception("keys not the same: {0} duing MDL1 phase {1}.".format(diff_keys, j))
    
    if verbose:
        print("\nMDL2:")
    U.domain_net_on = True
    for j, data_record in enumerate(info_dict_single["data_record_MDL2_1"]):
        num_records = len(data_record["all_losses_dict"])
        for i in range(num):
            ii = int(np.random.randint(num_records))
            U.set_net("pred_nets", load_model_dict(data_record['pred_nets_model_dict'][ii]))
            U.set_net("domain_net", load_model_dict(data_record['domain_net_model_dict'][ii]))
            U.set_loss_core("DLs", data_record['loss_precision_floor'])
            new_losses_dict = U.get_losses(X_test, y_test, DL_mode = DL_mode)
            record_lossed_dict = data_record["all_losses_dict"][ii]
            is_same, diff_keys = compare_same(new_losses_dict, record_lossed_dict, compare_keys, threshold = threshold, verbose = verbose)
            if not is_same:
                raise Exception("keys not the same: {0} duing MDL2 phase {1}.".format(diff_keys, j))
    print("Passed checking!")


def get_interior_idx(x, y, height, width, radius, shape):
    """Get the interior index when transforming cordinate into pixel figures"""
    if shape == "square":
        x_range = range(max(0, np.ceil(x - radius).astype(int)), int(min(width, np.floor(x + radius + 1))))
        y_range = range(max(0, np.ceil(y - radius).astype(int)), int(min(height, np.floor(y + radius + 1))))
        idx = np.array(np.meshgrid(y_range, x_range)).transpose([2,1,0]).reshape(-1, 2)
    elif shape == "circle":
        x_range = range(max(0, np.ceil(x - radius).astype(int)), int(min(width, np.floor(x + radius + 1))))
        y_range = range(max(0, np.ceil(y - radius).astype(int)), int(min(height, np.floor(y + radius + 1))))
        idx = np.array(np.meshgrid(y_range, x_range)).transpose([2,1,0]).reshape(-1, 2)
        center = np.array([np.floor(y) + 0.5, np.floor(x) + 0.5]) 
        distance = ((idx - center) ** 2).sum(1)
        idx = idx[distance <= max(radius ** 2 - 1, 1)]
    elif shape == "square_hollow":
        x_min = max(0, np.ceil(x - radius).astype(int))
        x_max = int(min(width, np.floor(x + radius)))
        y_min = max(0, np.ceil(y - radius).astype(int))
        y_max = int(min(height, np.floor(y + radius)))
        x_range = np.array([x_min, x_max])
        y_range = np.array([y_min, y_max])
        x_range_all = range(x_min, x_max + 1)
        y_range_all = range(y_min, y_max + 1)
        idx_v = np.array(np.meshgrid(y_range, x_range_all)).transpose([2,1,0]).reshape(-1, 2)
        idx_h = np.array(np.meshgrid(y_range_all, x_range)).transpose([2,1,0]).reshape(-1, 2)
        idx = np.concatenate([idx_h, idx_v])
    elif shape == "plus":
        center_x = np.round(x)
        center_y = np.round(y)
        horizontal = range(int(max(0, center_x - radius + 1)), int(min(width, center_x + radius)))
        vertical = range(int(max(0, center_y - radius + 1)), int(min(height, center_y + radius)))
        idx_h = np.stack([np.ones(len(horizontal)) * center_y, horizontal], 1)
        if center_x < width:
            idx_v = np.stack([vertical, np.ones(len(vertical)) * center_x], 1)
            idx = np.concatenate([idx_h, idx_v])
        else:
            idx = idx_h
    else:
        raise Exception("ball's shape {0} not recognized!".format(shape))
    return idx


def to_pixels(*args, **kwargs):
    """Transform the trajectory dataset into pixel format"""
    height, width = kwargs["size"]
    radius = kwargs["radius"]
    shape = kwargs["shape"]
    is_cuda = kwargs["is_cuda"] if "is_cuda" in kwargs else False
    tensor_list = []
    for tensor in args:
        tensor_reshape = deepcopy(to_np_array(tensor.view(tensor.size(0), -1, 2)))
        if "xlim" in kwargs:
            min_x, max_x = kwargs["xlim"]
        else:
            min_x, max_x = tensor_reshape[...,0].min(), tensor_reshape[...,0].max()
        if "ylim" in kwargs:
            min_y, max_y = kwargs["ylim"]
        else:
            min_y, max_y = tensor_reshape[...,1].min(), tensor_reshape[...,1].max()

        tensor_reshape[...,0] = (tensor_reshape[...,0] - min_x) / (max_x - min_x) * (width - 2 * radius) + radius
        tensor_reshape[...,1] = (tensor_reshape[...,1] - min_y) / (max_y - min_y) * (height - 2 * radius) + radius

        time_steps = tensor_reshape.shape[-2]
        obs = np.zeros((len(tensor_reshape), time_steps, height, width))
        for t in range(time_steps):
            for i in range(len(tensor_reshape)):
                x, y = tensor_reshape[i, t, :]
                object_region = get_interior_idx(x, y, height, width, radius, shape)
                obs[i, t, object_region[:,0], object_region[:, 1]] = 1
        tensor_list.append(to_Variable(obs, is_cuda = is_cuda))
    if len(args) == 1:
        tensor_list = tensor_list[0]
    return tensor_list


def check_matching(expression, target_expression, param_dict, tolerance = 1e-5, snapped_tolerance = 1e-9, verbose = True):
    """Check if the expression with param_dict match the target_expression within the specified tolerance."""
    coeffs, variable_names = get_coeffs(expression)
    new_coeffs, has_param_list = substitute(coeffs, param_dict)
    target_coeffs, target_variable_names = get_coeffs(target_expression)
    is_match = True
    if verbose:
        print("Comparing expression {0} given {1} with target_expression {2}:".format(expression, param_dict, target_expression))
    if set(variable_names) != set(target_variable_names):
        is_match = False
        print("    The variables {0} of the expression {1} does not fully match the variables {2} of the target_expression {3}!".format(variable_names, expression, target_variable_names, target_expression))
        return is_match
    for new_coeff, target_coeff, has_param in zip(new_coeffs, target_coeffs, has_param_list):
        if has_param:
            if abs(new_coeff - target_coeff) > tolerance:
                is_match = False
                if verbose:
                    print("    The difference  {0} between the coeff {1} and corresponding coeff {2} in target_expression is larger than the tolerance {3}.".format(abs(new_coeff - target_coeff), new_coeff, target_coeff, tolerance))
        else:
            if abs(new_coeff - target_coeff) > snapped_tolerance:
                is_match = False
                if verbose:
                    print("    The difference {0} between the snapped coefficient {1} and corresponding coefficient {2} in target_expression is larger than the snapped_tolerance {3}.".format(abs(new_coeff - target_coeff), new_coeff, target_coeff, snapped_tolerance))
    if is_match and verbose:
        print("    The expression {0} given {1} matches the target_expression {2} within the specified tolerances".format(expression, param_dict, target_expression))
    return is_match


def check_expression_matching(expressions, target_expressions, tolerance = 1e-4, snapped_tolerance = 1e-9, verbose = True):
    """Check if all the expressions discovered by the theories are within the target_expressions"""
    is_matching_whole = True
    num_matches = 0
    for model_name, model_expression in expressions.items():
        assert len(model_expression) == 1, "Each pred_net must have only one layer for comparison!"
        if model_expression[0] is None:
            is_matching_whole = False
            continue
        expression = model_expression[0]["symbolic_expression"][0]
        param_dict = model_expression[0]["param_dict"]
        is_matching = False
        match_indices = []
        for i, target_expression in enumerate(target_expressions):
            # Go through the whole list to see if the expression match any element in the target_expression:
            is_matching_ele = check_matching(expression, target_expression[0], param_dict, tolerance = tolerance, snapped_tolerance = snapped_tolerance, verbose = verbose)
            if is_matching_ele is True:
                is_matching = True
                match_indices.append(i)
        if not is_matching:
            print("\n" + "*"* 35 + "\n{0} given {1} does not match any element in the target_expressions {2} within the specified tolerance = {3}, snapped_tolerance = {4}.\n".format([expression], param_dict, target_expressions, tolerance, snapped_tolerance) + "*"* 35, end = "\n\n")
            is_matching_whole = False
        else:
            matching_elements = [target_expressions[i] for i in match_indices]
            num_matches += 1
            print("\n" + "*"* 35 + "\n{0} given {1} matches {2} in the target_expressions {3} within the specified tolerance = {4}, snapped_tolerance = {5}.\n".format([expression], param_dict, matching_elements, target_expressions, tolerance, snapped_tolerance) + "*"* 35, end = "\n\n")
    return is_matching_whole, num_matches



# The following 5 functions computes the association of domains and minimum misclassified points:
def mismatches(domain1, domain2):
    return (domain1 != domain2).sum()


def associate(domain, current_ids, new_ids):
    new_domain = deepcopy(domain)
    entry_list = []
    for current_id in current_ids:
        entry = (domain == current_id)
        entry_list.append(entry)
    for i, entry in enumerate(entry_list):
        new_domain[entry] = new_ids[i]
    return new_domain


def compute_class_correspondence(predicted_domain, true_domain, verbose = False):
    import itertools
    assert isinstance(predicted_domain, np.ndarray)
    assert isinstance(true_domain, np.ndarray)
    predicted_domain = predicted_domain.flatten().astype(int)
    true_domain = true_domain.flatten().astype(int)
    predicted_ids = np.unique(predicted_domain)
    true_ids = np.unique(true_domain)

    union_ids = np.sort(list(set(predicted_ids).union(set(true_ids))))
    if len(union_ids) > 7:
        if verbose:
            print("num_domains = {0}, too large!".format(len(union_ids)))
        return None, None, None
    if verbose:
        print("union_ids: {0}".format(union_ids))
    min_num_mismatches = np.inf
    argmin_permute = None
    predicted_domain_argmin = None

    for union_ids_permute in itertools.permutations(union_ids):
        predicted_domain_permute = associate(predicted_domain, union_ids, union_ids_permute)
        num_mismatches = mismatches(predicted_domain_permute, true_domain)
        if num_mismatches < min_num_mismatches:
            min_num_mismatches = num_mismatches
            argmin_permute = union_ids_permute
            predicted_domain_argmin = predicted_domain_permute
        if min_num_mismatches == 0:
            break
    return predicted_domain_argmin, min_num_mismatches, list(zip(union_ids, argmin_permute))



def count_metrics(predicted_domain, true_domain, big_domain_ids, verbose = True):
    assert len(predicted_domain) == len(true_domain)
    true_domain_ids = np.unique(true_domain)
    for id in big_domain_ids:
        assert id in true_domain_ids, "The big_domain_ids must be in the scope of true_domain_ids = {0}!".format(true_domain_ids)
    predicted_domain_argmin = compute_class_correspondence(predicted_domain, true_domain, verbose = verbose)[0]
    if predicted_domain_argmin is None:
        return (None, None, None, None), None
    
    def get_counts(domain):
        unique, counts = np.unique(domain, return_counts=True)
        counts, unique = sort_two_lists(counts, unique, reverse = True)
        return list(zip(unique, counts))
    
    if verbose:
        print("pred: {0}".format(get_counts(predicted_domain_argmin)))
        print("true: {0}".format(get_counts(true_domain)))
    
    union = 0
    predicted_big_domains = 0
    true_big_domains = 0
    intersection = 0
    for i in range(len(true_domain)):
        predicted_id = predicted_domain_argmin[i]
        true_id = true_domain[i]
        if predicted_id in big_domain_ids or true_id in big_domain_ids:
            union += 1
            if predicted_id in big_domain_ids:
                predicted_big_domains += 1
            if true_id in big_domain_ids:
                true_big_domains += 1
            if predicted_id == true_id:
                intersection += 1

    return (union, predicted_big_domains, true_big_domains, intersection), predicted_domain_argmin



def count_metrics_pytorch(predicted_domain, true_domain, big_domain_ids, verbose = False):
    def flatten_out(domain):
        domain = to_np_array(domain)
        if len(domain.shape) > 1:
            assert len(domain.shape) == 2
            assert domain.shape[1] == 1
            domain = domain.flatten()
        return domain
    return count_metrics(flatten_out(predicted_domain),
                         flatten_out(true_domain),
                         big_domain_ids = big_domain_ids,
                         verbose = verbose,
                        )


def get_group_norm(tensor, indi_order = 1, combine_order = 2, is_combine = True):
    indi_all = (tensor.abs() ** indi_order).sum(-1)
    dim = len(tensor.shape)
    if dim >= 4:
        if indi_all.size(-1) > 1:
            indi_all = ((indi_all ** combine_order).mean(-1)) ** (1. / combine_order)
        else:
            indi_all = indi_all.squeeze(-1)
    if is_combine:
        return ((indi_all ** combine_order + 1e-10).mean(0)) ** (1. / combine_order)
    else:
        return indi_all