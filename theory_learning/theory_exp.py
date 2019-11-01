
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import confusion_matrix
import pickle
import random
from copy import deepcopy
import itertools
import datetime
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from sympy import symbols
import pprint as pp

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_physicist.theory_learning.models import Loss_Fun_Cumu
from AI_physicist.theory_learning.theory_model import Theory_Training, get_loss, get_best_model_idx, get_preds_valid, get_valid_prob
from AI_physicist.theory_learning.theory_hub import Theory_Hub, load_model_dict_at_theory_hub
from AI_physicist.theory_learning.util_theory import plot_theories, plot3D, process_loss, plot_loss_record, to_one_hot, plot_indi_domain, get_mystery, to_pixels, standardize_symbolic_expression, check_expression_matching
from AI_physicist.theory_learning.util_theory import prepare_data_from_matrix_file, get_epochs, export_csv_with_domain_prediction, extract_target_expression, get_coeff_learned_list, load_theory_training, get_dataset_from_file, get_piecewise_dataset
from AI_physicist.settings.filepath import theory_PATH
from AI_physicist.settings.global_param import COLOR_LIST
from AI_physicist.pytorch_net.util import Loss_Fun, make_dir, Loss_with_uncertainty, Early_Stopping, record_data, plot_matrices, get_args, serialize, to_string, filter_filename, to_np_array, to_Variable
from AI_physicist.pytorch_net.net import Conv_Autoencoder, load_model_dict, train_simple

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    isplot = True
except:
    import matplotlib
    matplotlib.use('Agg')
    isplot = False
is_cuda = torch.cuda.is_available()
standardize = standardize_symbolic_expression


# ## Loading data from file:

# In[2]:


forward_steps = 1            # Number of forward steps to predict
num_output_dims = 2          # It sets the dimension of output
num_input_steps = 2          # It sets the number of steps for the input
exp_mode = "continuous"      # Choose from "continuous" (full AI Physicist), "newb" (newborn) and "base" (baseline)
data_format = "states"       # Choose from "states" or "images"
num_theories_init = 4        # Number of theories to start with
pred_nets_activation = "linear" # Activation for the prediction function f. Choose from "linear", "leakyRelu"
num_layers = 3               # Number of layers for the prediction function f.
# scheduler_settings = ("LambdaLR", "exp", 2, False)    # Settings for the learning rate scheduler
scheduler_settings = ("ReduceLROnPlateau", 40, 0.1)   # Settings for the learning rate scheduler
matching_numerical_tolerance = 2e-4 # The tolerance below which you regard the numerical coefficient matches.
matching_snapped_tolerance = 1e-9   # The tolerance below which you regard the snapped coefficient matches.
add_theory_loss_threshold = 2e-6
add_theory_criteria = ("loss_with_domain", 0)
add_theory_quota = 1
add_theory_limit = None     # maximum allowed number of theories. If None, do not set limit
MDL_mode = "both"
show_3D_plot = False
show_vs = False

is_Lagrangian = False       # If True, learn the Lagrangian. If False, learn Equation of Motion (EOM)
loss_balance_model_influence = False
loss_success_threshold = 1e-4   # MSE level you regard as success
theory_add_mse_threshold = 0.05 # MSE level below which you will add to the theory hub
theory_remove_fraction_threshold = 0.005  # Fraction threshold below which you will remove a theory after each stage of training.
load_previous = True        # Whether to load previously trained instances on
max_trial_times = 1         # Maximum number of trial times before going on to next target (DEFAULT=1)
is_simplify_model = True    # Whether to perform simplification of theory models
is_simplify_domain = False  # Whether to perform simplification of theory domains
record_mode = 2             # Record data mode. Choose from 0 (minimal recording), 1, 2 (record everything)

csv_filename_list = get_mystery(50000, range(4,7), range(1, 6)) + get_mystery(50000, [20], range(1, 6)) + get_mystery(50000, range(7, 11), range(1, 6))


# ## Settings:

# In[3]:


exp_id = "exp1.0"
env_names = "file"
# env_names = "envs-piece"
pred_nets_neurons = 8
domain_net_neurons = 8
domain_pred_mode = "onehot"
mse_amp = 1e-7
optim_type = ("adam", 5e-3)
optim_domain_type = ("adam", 1e-3)
optim_autoencoder_type = ("adam", 1e-5, 1e-1) # optim_type, lr, loss_scale
reg_mode = "L1"
reg_amp = 1e-8
reg_smooth = None
reg_domain_mode = "L1"
reg_domain_amp = 1e-5
batch_size = 10000
loss_core = "DLs"
loss_order = -1
loss_decay_scale = None
is_mse_decay = False
num_examples = 20000
epochs = 10000
iter_to_saturation = int(epochs / 2)
date_time = "{0}-{1}".format(datetime.datetime.now().month, datetime.datetime.now().day)
seed = 0
array_id = "Lag"
simplify_criteria = ("DLs", 0, 3, "relative") # The (criteria type, threshold, patience, compare_mode) upon which not satisfied, we break the current simplification and continue to the next layer/model

big_domain_dict = [(key, [1, 2]) for key in get_mystery([20000, 30000, 40000, 50000], range(4, 7), range(11))] +                   [(key, [1, 2]) for key in get_mystery([40000, 50000], [20], range(11))] +                   [(key, [1, 2, 3]) for key in get_mystery([20000, 30000, 40000, 50000], range(7, 10), range(11))] +                   [(key, [1, 2, 3, 4]) for key in get_mystery([20000, 30000, 40000, 50000], [10], range(11))]
big_domain_dict = {key: item for key, item in big_domain_dict}
########################################
# Settings for double pendulum:
is_pendulum = False
is_pendulum = get_args(is_pendulum, 27, "bool")
if is_pendulum:  
    num_layers = 5
    pred_nets_neurons = 160
    domain_net_neurons = 16
    reg_domain_amp = 1e-7
    show_3D_plot = False
    pred_nets_activation = "tanh"
#     csv_filename_list = ['mysteryt_px_50', 'mysteryt_px_51', 'mysteryt_px_52', 'mysteryt_px_53', 'mysteryt_px_60', 'mysteryt_px_61', 'mysteryt_px_62', 'mysteryt_px_70', 'mysteryt_px_71', 'mysteryt_px_72']
    csv_filename_list = ['mysteryt_px_10', 'mysteryt_px_11', 'mysteryt_px_12', 'mysteryt_px_13', 'mysteryt_px_20', 'mysteryt_px_21', 'mysteryt_px_23']
    is_simplify_model = False
    is_simplify_domain = False
    add_theory_loss_threshold = 1e-4
    num_theories_init = 2
    add_theory_limit = 2
    is_xv = True
    record_mode = 1
#     reg_smooth = (0.1, 2, 10, 1e-2, 1)
    if is_xv:
#         csv_filename_list = ['mysteryt_pxv_50', 'mysteryt_pxv_51', 'mysteryt_pxv_52', 'mysteryt_pxv_53', 'mysteryt_pxv_60', 'mysteryt_pxv_61', 'mysteryt_pxv_62', 'mysteryt_pxv_70', 'mysteryt_pxv_71', 'mysteryt_pxv_72']
        csv_filename_list = ['mysteryt_pxv_10', 'mysteryt_pxv_11', 'mysteryt_pxv_12', 'mysteryt_pxv_13', 'mysteryt_pxv_20', 'mysteryt_pxv_21', 'mysteryt_pxv_23']
        num_output_dims = 4
        num_input_steps = 1
        add_theory_loss_threshold = 3e-4
    for key in csv_filename_list:
        big_domain_dict[key] = [0, 1]
########################################

########################################
# Settings for data_format = "images":
if data_format == "images":
    batch_size = 100
    epochs = 10000
    loss_core = "mse"
    add_theory_quota = 0
    is_simplify_model = False
    is_simplify_domain = False
########################################

########################################
# Settings for Lagrangian:
if is_Lagrangian:
    num_input_steps = 3
    is_simplify_model = False
########################################

exp_id = get_args(exp_id, 1)
env_names = get_args(env_names, 2)
exp_mode = get_args(exp_mode, 3)
num_theories_init = get_args(num_theories_init, 4, "int")
pred_nets_neurons = get_args(pred_nets_neurons, 5, "int")
pred_nets_activation = get_args(pred_nets_activation, 6)
domain_net_neurons = get_args(domain_net_neurons, 7, "int")
domain_pred_mode = get_args(domain_pred_mode, 8)
mse_amp = get_args(mse_amp, 9, "float")
simplify_criteria = get_args(simplify_criteria, 10, "tuple")
scheduler_settings = get_args(scheduler_settings, 11, "tuple")
optim_type = get_args(optim_type, 12, "tuple")
optim_domain_type = get_args(optim_domain_type, 13, "tuple")
reg_amp = get_args(reg_amp, 14, "float")
reg_domain_amp = get_args(reg_domain_amp, 15, "float")
batch_size = get_args(batch_size, 16, "int")
loss_core = get_args(loss_core, 17)
loss_order = get_args(loss_order, 18)
loss_decay_scale = get_args(loss_decay_scale, 19, "eval")
is_mse_decay = get_args(is_mse_decay, 20, "bool")
loss_balance_model_influence = get_args(loss_balance_model_influence, 21, "bool")
num_examples = get_args(num_examples, 22, "int")
iter_to_saturation = get_args(iter_to_saturation, 23, "eval")
MDL_mode = get_args(MDL_mode, 24)
num_output_dims = get_args(num_output_dims, 25, "int")
num_layers = get_args(num_layers, 26, "int")
date_time = get_args(date_time, 28)
seed = get_args(seed, 29, "int")
array_id = get_args(array_id, 30, "int")
is_batch = False
array_id_core = array_id
if isinstance(env_names, str) and "mystery" in env_names:
    csv_filename_list = [env_names]
    env_names = "file"
    array_id = seed
    load_previous = False
    is_batch = True

if exp_mode == "continuous":
    is_propose_models = True
elif exp_mode == "newb":
    is_propose_models = False
elif exp_mode == "base":
    is_propose_models = False
    num_theories_init = 1
    pred_nets_activation = "leakyRelu" if pred_nets_activation == "linear" else pred_nets_activation
    pred_nets_neurons *= 2
    epochs *= 5
    add_theory_quota = 0
    is_simplify_model = False
    is_simplify_domain = False
    loss_core = "mse"
    MDL_mode = "None"
else:
    raise
batch_size = min(batch_size, num_examples)
save_image = True
is_domain_fit_setting = True
render = False
loss_floor = 1e-12
csv_dirname = "../datasets/"
simplify_lr = 1e-6
simplify_epochs = 40000
simplify_patience = 200
# ramping_L1_list = np.logspace(-7, 0, 40) # The list of L1 amplitudes you will ramp up in the "ramping-L1" simplification
# ramping_mse_threshold = 1e-3    # The MSE threshold above which you stop the ramping
# ramping_final_multiplier = 0.01 # The multiplier you apply on the final L1 amplitude you will use (PERHAPS USE 1e-2 INSTEAD)
# reg_scale_factor_dict = {key: [1, 0.1, 0.01, 0.001] for key in csv_filename_list}
# target_symbolic_expressions = extract_target_expression("spreadsheet.csv", csv_dirname = csv_dirname)
target_symbolic_expressions = {}
view_init = (10, 190)           # Angle you want to view the 3D plots


if iter_to_saturation is not None:
    iter_to_saturation = int(iter_to_saturation)
    reg_multiplier = np.linspace(0, 1, iter_to_saturation) ** 2
else:
    reg_multiplier = None

if epochs <= 3000:
    change_interval = 2
    record_interval = 5
else:
    change_interval = 10
    record_interval = 5

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

dirname = theory_PATH + "/{0}_{1}/".format(exp_id, date_time)
filename = dirname + "{0}_{1}_num_{2}_pred_{3}_{4}_dom_{5}_{6}_mse_{7}_sim_{8}_optim_{10}_{11}_reg_{12}_{13}_batch_{14}_core_{15}_order_{16}_lossd_{17}_{18}_infl_{19}_#_{20}_mul_{21}_MDL_{22}_{23}D_{24}L_id_{25}_{26}.p".format(
            env_names, exp_mode, num_theories_init, pred_nets_neurons, pred_nets_activation, domain_net_neurons, domain_pred_mode, mse_amp,
            to_string(simplify_criteria), to_string(scheduler_settings, num_strings = 5), to_string(optim_type), to_string(optim_domain_type), reg_amp, reg_domain_amp, batch_size, loss_core, 
            loss_order, loss_decay_scale, is_mse_decay, loss_balance_model_influence, num_examples, iter_to_saturation, MDL_mode, num_output_dims, num_layers, seed, array_id,
)
if is_batch:
    filename_batch = filename[:-2] + "_model/" + "{0}_{1}_num_{2}_pred_{3}_{4}_dom_{5}_{6}_mse_{7}_sim_{8}_optim_{10}_{11}_reg_{12}_{13}_batch_{14}_core_{15}_order_{16}_lossd_{17}_{18}_infl_{19}_#_{20}_mul_{21}_MDL_{22}_{23}D_{24}L_id_{25}_{26}.p".format(
            csv_filename_list[0], exp_mode, num_theories_init, pred_nets_neurons, pred_nets_activation, domain_net_neurons, domain_pred_mode, mse_amp,
            to_string(simplify_criteria), to_string(scheduler_settings, num_strings = 5), to_string(optim_type), to_string(optim_domain_type), reg_amp, reg_domain_amp, batch_size, loss_core, 
            loss_order, loss_decay_scale, is_mse_decay, loss_balance_model_influence, num_examples, iter_to_saturation, MDL_mode, num_output_dims, num_layers, seed, array_id_core,
    )
    make_dir(filename_batch)
make_dir(filename)
make_dir(filename[:-2] + "_mys/file")
if save_image:
    make_dir(filename[:-2] + "/images")

# Load theory_hub:
filename_hub = filename[:-2] + "_hub.p"
print(filename)

# Load or initialize theory hub:
load_augmenting_hub = False
if load_augmenting_hub:
    is_propose_models = True
    dirname_augmenting_hub = theory_PATH + "/{0}_{1}/".format("transfer-hit5.5", "10-9")
    hub_candi1 = "{0}_{1}_num_{2}_pred_{3}_{4}_dom_{5}_{6}_mse_{7}_sim_{8}_optim_{10}_{11}_reg_{12}_{13}".format(
                env_names, "continuous", num_theories_init, pred_nets_neurons, pred_nets_activation, domain_net_neurons, domain_pred_mode, mse_amp,
                to_string(simplify_criteria), to_string(scheduler_settings, num_strings = 5), to_string(optim_type), to_string(optim_domain_type), reg_amp, reg_domain_amp, 
                
    )
    hub_candi2 = "core_{0}_order_{1}_lossd_{2}_{3}_infl_{4}".format(loss_core, loss_order, loss_decay_scale, is_mse_decay, loss_balance_model_influence)
    hub_candi3 = "mul_{0}_MDL_{1}_{2}D_{3}L_id_{4}".format(iter_to_saturation, MDL_mode, num_output_dims, num_layers, seed, array_id)
    filename_augmenting_hub = dirname_augmenting_hub + filter_filename(dirname_augmenting_hub, include = [hub_candi1, hub_candi2, hub_candi3, "hub.p"])[0]
    theory_hub = load_model_dict_at_theory_hub(pickle.load(open(filename_augmenting_hub, "rb")), is_cuda = is_cuda)
    pickle.dump(theory_hub.model_dict, open(filename_hub, "wb"))
    print("load augmenting hub succeed: {0}".format(filename_augmenting_hub))
else:
    if load_previous:
        try:
            theory_hub = load_model_dict_at_theory_hub(pickle.load(open(filename_hub, "rb")), is_cuda = is_cuda)
        except:
            theory_hub = Theory_Hub()
            pickle.dump(theory_hub.model_dict, open(filename_hub, "wb"))
    else:
        theory_hub = Theory_Hub()
        pickle.dump(theory_hub.model_dict, open(filename_hub, "wb"))


def save_to_hub(pred_nets, domain_net, theory_hub, theory_type, theory_add_threshold, is_Lagrangian):
    """Save qualified theories to theory_hub"""
    if load_previous:
        theory_hub = load_model_dict_at_theory_hub(pickle.load(open(filename_hub, "rb")))
    added_theory_info = theory_hub.add_theories(name = hub_theory_name if theory_type == "neural" else hub_theory_name + "_simplified",
                                                pred_nets = pred_nets,
                                                domain_net = domain_net,
                                                dataset = dataset,
                                                threshold = theory_add_threshold,
                                                is_Lagrangian = is_Lagrangian,
                                               )
    if theory_type == "neural":
        name = "theory"
    elif theory_type == "simplified":
        name = "simplified_theory"
    else:
        raise
    info_dict[env_name]["added_{0}_info".format(name)] = added_theory_info
    pickle.dump(theory_hub.model_dict, open(filename_hub, "wb"))

def load_info_dict(info_dict, filename, trial_times = 1):
    load_succeed = False
    for i in range(trial_times):
        try:
            info_dict_updated = pickle.load(open(filename, "rb"))
            print("Succesfully loaded {0}".format(filename))
            info_dict_updated.update(info_dict)
            info_dict = info_dict_updated
            load_succeed = True
            break
        except Exception as e:
            print("Load trial {0}: ".format(i), e)
            time.sleep(2 ** i * (0.8 + 0.4 * np.random.rand()))
    return info_dict, load_succeed


# In[ ]:


info_dict = {}
info_dict["array_id"] = array_id
info_dict["reg_smooth"] = reg_smooth
if load_previous:
    info_dict, load_succeed = load_info_dict(info_dict, filename)
    if not load_succeed:
        pickle.dump(info_dict, open(filename, "wb"))

###########################################################################################################################
# Level I. Iterate over environments:
###########################################################################################################################
for env_name in csv_filename_list:
    big_domain_ids = big_domain_dict[env_name] if env_name in big_domain_dict else None 
    # Dealing with specific environments:
    if "mystery" in env_name and num_output_dims == 2:
        env_name = "2D" + env_name
    if "mystery" in env_name and num_output_dims == 4:
        env_name = "4D" + env_name
    if "N" in env_name.split("_"): # Lorenz
        env_name_split = env_name.split("_")
        num_output_dims = int(env_name_split[env_name_split.index("N") + 1])
        pred_nets_activation = "sigmoid"
    print("\n" + "=" * 150 + "\nnew environment\n" + "{0}\n".format(env_name) + "=" * 150 + "\n\n")
    
    # Check if the environment has been run. If so, skip:
    if is_simplify_domain is True:
        assert is_simplify_model is True
        key_to_check = "domain_net_simplified"
    elif is_simplify_model is True:
        key_to_check = "pred_nets_simplified"
    else:
        key_to_check = "pred_nets"
    if env_name not in info_dict:
        info_dict[env_name] = {}
    if key_to_check in info_dict[env_name]:
        print("{0} already ran. Go to the next environment".format(env_name))
        continue
    hub_theory_name = "{0}_id_{1}".format(env_name, array_id)
    
    # Load or generate dataset:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Get dataset from file:
    dataset = get_dataset_from_file(env_name, num_output_dims = num_output_dims, num_input_steps = num_input_steps, csv_dirname = csv_dirname + "GROUND_TRUTH/", 
                                    forward_steps = forward_steps, num_examples = num_examples,
                                    is_Lagrangian = is_Lagrangian,
                                    is_cuda = is_cuda,
                                   )
    if is_Lagrangian:
        num_output_dims = 4
        num_input_steps = 2
    
    ((X_train, y_train), (X_test, y_test), _), info = dataset
    if num_output_dims <= 2 and "true_domain_train" in info:
        plot_indi_domain(X_train, domain = info["true_domain_train"], is_show = isplot, filename = filename[:-2] + "/{0}_indi-domains.png".format(env_name) if save_image else None)

    # Setting up network and training hyperparameters:
    # Structure for theory model:
    if is_Lagrangian:
        struct_param_pred = [[1, "Symbolic_Layer", {"symbolic_expression": "g0 * x0 + g1 * x2"}]]
    else:
        struct_param_pred = [[pred_nets_neurons, "Simple_Layer", {}]] * (num_layers - 1) + [[y_train.size(1), "Simple_Layer", {"activation": "linear"}]]
    settings_pred = {"activation": pred_nets_activation}   # Default activation if not stipulated
    
    if data_format == "images":
        X_train, y_train, X_test, y_test = to_pixels(X_train, y_train, X_test, y_test, size = (39, 39), xlim = (-2, 2), ylim = (-2, 2), radius = 3, shape = "circle", is_cuda = is_cuda)
    
    # Structure for theory domain:
    struct_param_domain = [
        [domain_net_neurons, "Simple_Layer", {}],
        [domain_net_neurons, "Simple_Layer", {}],
        [num_theories_init, "Simple_Layer", {"activation": "linear"}],
    ]
#     if is_pendulum:
#         struct_param_domain = [
#             [domain_net_neurons, "Simple_Layer", {}],
#             [domain_net_neurons, "Simple_Layer", {}],
#             [domain_net_neurons, "Simple_Layer", {}],
#             [num_theories_init, "Simple_Layer", {"activation": "linear"}],
#         ]
    settings_domain = {"activation": "leakyRelu"} # Default activation if not stipulated
    loss_types = {
                    "pred-based_generalized-mean_{0}".format(loss_order): {"amp": 1., "decay_on": True},
                    "pred-based_generalized-mean_1": {"amp": mse_amp, "decay_on": is_mse_decay}, 
    }
    
    # Setting up regularization:
    reg_dict = {"pred_nets": {"weight": reg_amp, "bias": reg_amp}}
    reg_domain_dict = {"domain_net": {"weight": reg_domain_amp, "bias": reg_domain_amp}}
    
    # Setting up training model and domain at the same time at the first stage of training:
    if is_domain_fit_setting:
        domain_fit_setting = {"optim_domain_type": optim_domain_type, "reg_domain_dict": reg_domain_dict, "reg_domain_mode": reg_domain_mode}
    else:
        domain_fit_setting = None 
    
    
    if data_format == "images":
        enc_fully_size = 512
        latent_size = 2
        num_channels = 32
        struct_param_encoder = [
            [num_channels, "Conv2d", {"kernel_size": 3, "stride": 2}],
            [num_channels, "Conv2d", {"kernel_size": 3, "stride": 2}],
            [num_channels, "Conv2d", {"kernel_size": 3, "stride": 2}],
            [latent_size, "Simple_Layer", {"layer_input_size": enc_fully_size}],
        ]
        struct_param_decoder = [
            [enc_fully_size, "Simple_Layer", {"layer_input_size": latent_size}],
            [num_channels, "ConvTranspose2d", {"kernel_size": 3, "stride": 2, "layer_input_size": (num_channels, 4, 4)}],
            [num_channels, "ConvTranspose2d", {"kernel_size": 3, "stride": 2}],
            [1, "ConvTranspose2d", {"kernel_size": 3, "stride": 2}],
        ]
        autoencoder = Conv_Autoencoder(
            input_channels_encoder = 1,
            input_channels_decoder = num_channels,
            struct_param_encoder = struct_param_encoder,
            struct_param_decoder = struct_param_decoder,
            share_model_among_steps = True,
            settings = {"activation": "leakyReluFlat"},
            is_cuda = is_cuda,
        )
        print("Pre-training autoencoder:")
        train_simple(autoencoder, X_train, X_train, loss_type = "mse")
        print("Pre-training autoencoder completes.")
        if isplot:
            X_train_recons = autoencoder(X_train)
            for i in np.random.randint(len(X_train), size = 4):
                plot_matrices(torch.cat([X_train[i], X_train_recons[i]], 0), images_per_row = 4)
        
    else:
        autoencoder = None

    #########################################################################################################
    # Level II.a. Iterate over number of trials:
    #########################################################################################################
    event_name = "hit_until_loss_threshold"
    for trial_times in range(max_trial_times):
        print("=" * 120 + "\nTrial {0}\n".format(trial_times) + "=" * 120 + "\n\n")
        # Each trial start with a different seed:
        torch.manual_seed(seed + trial_times) 
        info_dict_single = {}

        # Propose theory from theory_hub:
        if load_previous:
            theory_hub = load_model_dict_at_theory_hub(pickle.load(open(filename_hub, "rb")), is_cuda = is_cuda)
        if is_propose_models:
            print("From theory_hub:")
            proposed_theory_models, propose_evaluation = theory_hub.propose_theory_models(X_train, y_train, max_num_models = max(2, int(num_theories_init * 0.7)), types = ["neural"], is_Lagrangian = is_Lagrangian)
        else:
            proposed_theory_models = {}
            propose_evaluation = {}
        info_dict_proposed_models = {}
        for name, theory_info in proposed_theory_models.items():
            info_dict_proposed_models[name] = {}
            for key, item in theory_info.items():
                if key == "theory_model":
                    info_dict_proposed_models[name][key] = item.model_dict
                else:
                    info_dict_proposed_models[name][key] = item
        info_dict_single["proposed_theory_models"] = info_dict_proposed_models
        info_dict_single["proposed_theory_models_evaluation"] = propose_evaluation


        # Initialize Theory_Training:
        T = Theory_Training(
            num_theories = num_theories_init,
            proposed_theory_models = proposed_theory_models,
            input_size = info["input_size"],
            struct_param_pred = struct_param_pred,
            struct_param_domain = struct_param_domain,
            settings_pred = settings_pred,
            settings_domain = settings_domain,
            autoencoder = autoencoder,
            loss_types = loss_types,
            loss_core = loss_core,
            loss_order = loss_order,
            loss_balance_model_influence = loss_balance_model_influence,
            is_Lagrangian = is_Lagrangian,
            reg_multiplier = reg_multiplier,
            is_cuda = is_cuda,
        )

        #################################################################################################
        # Level III.a Iterative training of models and domains
        #################################################################################################
        print("\n" + "=" * 80)
        print("{0}, trial times {1}, training model:".format(env_name, trial_times))
        print("=" * 80 + "\n")
        loss_precision_floor_init = 10
        T.set_loss_core(loss_core, loss_precision_floor_init)
        data_record = T.fit_model(
            X_train,
            y_train,
            validation_data = (X_test, y_test),
            optim_type = optim_type,
            optim_autoencoder_type = optim_autoencoder_type,
            reg_dict = reg_dict,
            reg_mode = reg_mode,
            reg_smooth = reg_smooth,
            domain_fit_setting = domain_fit_setting,
            forward_steps = forward_steps,
            domain_pred_mode = domain_pred_mode,
            scheduler_settings = scheduler_settings,
            loss_order_decay = (lambda epoch: - epoch / float(loss_decay_scale)) if loss_decay_scale is not None else None,
            view_init = view_init,
            epochs = epochs,
            patience = int(epochs / 50) if exp_mode != "base" else None,
            batch_size = batch_size,
            inspect_interval = int(epochs / 100) if is_pendulum else int(epochs / 10),
            change_interval = change_interval,
            record_interval = record_interval,
            record_mode = record_mode,
            isplot = isplot,
            filename = filename[:-2] + "/{0}_training-model".format(env_name) if save_image else None,
            raise_nan = False,
            add_theory_quota = add_theory_quota,
            add_theory_limit = add_theory_limit,
            add_theory_criteria = add_theory_criteria,
            add_theory_loss_threshold = add_theory_loss_threshold,
            loss_floor = loss_floor,
            true_domain_test = info["true_domain_test"] if "true_domain_test" in info else None,
            num_output_dims = num_output_dims,
            prefix = "{0}_train_model:".format(env_name),
            show_3D_plot = show_3D_plot,
            show_vs = show_vs,
            big_domain_ids = big_domain_ids,
            fix_adaptive_precision_floor = True,
        )
        data_record["loss_precision_floor"] = loss_precision_floor_init
        info_dict_single["data_record"] = deepcopy(data_record)
        info_dict_single["pred_nets"] = deepcopy(T.pred_nets.model_dict)
        info_dict_single["removed_theories_0"] = T.remove_theories_based_on_data(X_test, y_test, threshold = theory_remove_fraction_threshold)

        # Perform MDL training after MSE training if stipulated:
        if MDL_mode == "both":
            if loss_core == "mse":
                T.set_loss_core("DLs")
            if hasattr(T, "optimizer"):
                delattr(T, "optimizer")
            data_record_MDL1 = T.fit_model_schedule(
                X_train,
                y_train,
                validation_data = (X_test, y_test),
                optim_type = ("adam", 1e-4),
                reg_dict = reg_dict,
                reg_mode = reg_mode,
                reg_smooth = reg_smooth,
                domain_fit_setting = domain_fit_setting,
                forward_steps = forward_steps,
                domain_pred_mode = domain_pred_mode,
                scheduler_settings = scheduler_settings,
                loss_order_decay = (lambda epoch: - epoch / float(loss_decay_scale)) if loss_decay_scale is not None else None,
                view_init = view_init,
                epochs = int(epochs / 2),
                patience = int(epochs / 50) if exp_mode != "base" else None,
                batch_size = batch_size,
                inspect_interval = int(epochs / 100) if is_pendulum else int(epochs / 10),
                change_interval = change_interval,
                record_interval = record_interval,
                record_mode = record_mode,
                isplot = isplot,
                filename = filename[:-2] + "/{0}_training-modelDL1".format(env_name) if save_image else None,
                raise_nan = False,
                add_theory_quota = add_theory_quota,
                add_theory_limit = add_theory_limit,
                add_theory_criteria = add_theory_criteria,
                add_theory_loss_threshold = add_theory_loss_threshold,
                theory_remove_fraction_threshold = theory_remove_fraction_threshold,
                loss_floor = loss_floor,
                true_domain_test = info["true_domain_test"],
                num_output_dims = num_output_dims,
                prefix = "{0}_train_modelDL1:".format(env_name),
                show_3D_plot = show_3D_plot,
                show_vs = show_vs,
                big_domain_ids = big_domain_ids,
                num_phases = 4,
            )
            info_dict_single["data_record_MDL1"] = deepcopy(data_record_MDL1)

            T.domain_net_on = True
            print("\ndomain_net turned on\n")
            data_record_MDL2 = T.fit_model_schedule(
                X_train,
                y_train,
                validation_data = (X_test, y_test),
                optim_type = ("adam", 1e-4),
                reg_dict = reg_dict,
                reg_mode = reg_mode,
                reg_smooth = reg_smooth,
                domain_fit_setting = domain_fit_setting,
                forward_steps = forward_steps,
                domain_pred_mode = domain_pred_mode,
                scheduler_settings = scheduler_settings,
                loss_order_decay = (lambda epoch: - epoch / float(loss_decay_scale)) if (loss_decay_scale is not None) else None,
                view_init = view_init,
                epochs = int(epochs / 2),
                patience = int(epochs / 50) if exp_mode != "base" else None,
                batch_size = batch_size,
                inspect_interval = int(epochs / 100) if is_pendulum else int(epochs / 10),
                change_interval = change_interval,
                record_interval = record_interval,
                record_mode = record_mode,
                isplot = isplot,
                filename = filename[:-2] + "/{0}_training-modelDL2".format(env_name) if save_image else None,
                raise_nan = False,
                add_theory_quota = add_theory_quota,
                add_theory_limit = add_theory_limit,
                add_theory_criteria = add_theory_criteria,
                add_theory_loss_threshold = add_theory_loss_threshold,
                theory_remove_fraction_threshold = theory_remove_fraction_threshold,
                loss_floor = loss_floor,
                true_domain_test = info["true_domain_test"],
                num_output_dims = num_output_dims,
                prefix = "{0}_train_modelDL2:".format(env_name),
                show_3D_plot = show_3D_plot,
                show_vs = show_vs,
                big_domain_ids = big_domain_ids,
                num_phases = 2,
            )
            info_dict_single["data_record_MDL2"] = deepcopy(data_record_MDL2)
            if hasattr(T, "optimizer"):
                delattr(T, "optimizer")

        # Remove theories whose fraction is below the remove_theshold:
        fraction_list = T.get_fraction_list(X_test, y_test)
        removed_theories = T.remove_theories_based_on_data(X_test, y_test, threshold = theory_remove_fraction_threshold)
        info_dict_single["fraction_list"] = fraction_list
        info_dict_single["removed_theories"] = removed_theories
        info_dict_single["domain_net"] = deepcopy(T.domain_net.model_dict)
        
        # Record pred_nets and domain_nets before simplification
        pred_nets = deepcopy(T.pred_nets)
        domain_net = deepcopy(T.domain_net)

        #########################################################################################################
        # Level III.b. Simplify theory models and/or domains:
        #########################################################################################################
        kwargs = {"true_domain_test": info["true_domain_test"], "big_domain_ids": big_domain_ids, "is_Lagrangian": is_Lagrangian}
        if is_simplify_model:
            print("\n" + "=" * 150)
            print("{0}, {1}".format(env_name, "simplifying model:"))
            print("=" * 150 + "\n")
            info_dict_single["data_record_simplification-model"] = {}
            record_data(info_dict_single["data_record_simplification-model"], [T.get_losses(X_test, y_test, **kwargs), T.pred_nets.model_dict, "before_simplification"], ["all_losses_dict", "pred_nets_model_dict", "event"])

            # Get prediction by theory domains:
            valid_onehot = to_one_hot(T.domain_net_forward(X_train).max(1)[1], T.num_theories)
            valid_onehot_test = to_one_hot(T.domain_net_forward(X_test).max(1)[1], T.num_theories)
            validation_pred_nets = (X_test, y_test, valid_onehot_test)

            # Collapse Layers:
            loss_record = T.pred_nets.simplify(X_train, y_train, valid_onehot, "collapse_layers", simplify_criteria = simplify_criteria, validation_data = validation_pred_nets, is_Lagrangian = is_Lagrangian)
            T.pred_nets.get_weights_bias(W_source = "core", b_source = "core", verbose = True)
            record_data(info_dict_single["data_record_simplification-model"], [T.get_losses(X_test, y_test, **kwargs), loss_record, T.pred_nets.model_dict, "after_collapsing_layer"], ["all_losses_dict", "loss_record", "pred_nets_model_dict", "event"])

            # Local snapping:
            loss_record = T.pred_nets.simplify(X_train, y_train, valid_onehot, "local", loss_type = "DLs", loss_precision_floor = T.loss_precision_floor, simplify_criteria = simplify_criteria, validation_data = validation_pred_nets, patience = simplify_patience, lr = simplify_lr, epochs = simplify_epochs, is_Lagrangian = is_Lagrangian, verbose = 2)
            record_data(info_dict_single["data_record_simplification-model"], [T.get_losses(X_test, y_test, **kwargs), loss_record, T.pred_nets.model_dict, "after_local_snapping"], ["all_losses_dict", "loss_record", "pred_nets_model_dict", "event"])

#             # Ramping L1:
#             loss_record = T.pred_nets.simplify(X_train, y_train, valid_onehot, "ramping-L1", 
#                                                reg_scale_factor = reg_scale_factor_dict[env_name], 
#                                                ramping_L1_list = ramping_L1_list,
#                                                ramping_mse_threshold = ramping_mse_threshold,
#                                                ramping_final_multiplier = ramping_final_multiplier,
#                                                validation_data = validation_pred_nets,
#                                                is_Lagrangian = is_Lagrangian,
#                                                isplot = isplot,
#                                               )
#             T.pred_nets.get_weights_bias(W_source = "core", b_source = "core", verbose = True)
#             record_data(info_dict_single["data_record_simplification-model"], [T.get_losses(X_test, y_test, **kwargs), loss_record, T.pred_nets.model_dict, "after_ramping-L1"], ["all_losses_dict", "loss_record", "pred_nets_model_dict", "event"])

            # Integer snapping:
            loss_record = T.pred_nets.simplify(X_train, y_train, valid_onehot, "snap", snap_mode = "integer", loss_type = "DLs", loss_precision_floor = T.loss_precision_floor, simplify_criteria = simplify_criteria, validation_data = validation_pred_nets, patience = simplify_patience, lr = simplify_lr, epochs = simplify_epochs, is_Lagrangian = is_Lagrangian, verbose = 2)
            record_data(info_dict_single["data_record_simplification-model"], [T.get_losses(X_test, y_test, **kwargs), loss_record, T.pred_nets.model_dict, "after_integer_snapping"], ["all_losses_dict", "loss_record", "pred_nets_model_dict", "event"])

            # Rational snapping:
            loss_record = T.pred_nets.simplify(X_train, y_train, valid_onehot, "snap", snap_mode = "rational", loss_type = "DLs", loss_precision_floor = T.loss_precision_floor, simplify_criteria = simplify_criteria, validation_data = validation_pred_nets, patience = simplify_patience, lr = simplify_lr, epochs = simplify_epochs, is_Lagrangian = is_Lagrangian, verbose = 2)
            record_data(info_dict_single["data_record_simplification-model"], [T.get_losses(X_test, y_test, **kwargs), loss_record, T.pred_nets.model_dict, "after_rational_snapping"], ["all_losses_dict", "loss_record", "pred_nets_model_dict", "event"])
            
            # To symbolic:
            loss_record = T.pred_nets.simplify(X_train, y_train, valid_onehot, "to_symbolic", simplify_criteria = simplify_criteria, validation_data = validation_pred_nets, is_Lagrangian = is_Lagrangian)
            T.pred_nets.get_sympy_expression()
            record_data(info_dict_single["data_record_simplification-model"], [T.get_losses(X_test, y_test, **kwargs), loss_record, T.pred_nets.model_dict, "after_to_symbolic_and_snapping"], ["all_losses_dict", "loss_record", "pred_nets_model_dict", "event"])

            # Saving and show expression:
            info_dict_single["pred_nets_simplified"] = deepcopy(T.pred_nets.model_dict)
            print()
            print("MSE curve for simplification of the theory_model:")
            if isplot:
                try:
                    plot_loss_record(info_dict_single['data_record_simplification-model'], T.pred_nets.num_models)
                except:
                    pass
            if save_image:
                T.plot(X_test, y_test, true_domain = info["true_domain_test"], view_init = view_init, figsize = (10, 8), filename = filename[:-2] + "/{0}_model-simplified".format(env_name) if save_image else None, is_show = isplot)

            # Simplifying domain:
            if is_simplify_domain:
                print("\n" + "=" * 150)
                print("{0}, {1}".format(env_name, "simplifying domain:"))
                print("=" * 150 + "\n")
                info_dict_single["data_record_simplification-domain-model"] = {}
                record_data(info_dict_single["data_record_simplification-domain-model"], [T.get_losses(X_test, y_test, **kwargs), T.domain_net.model_dict, "before_domain_simplification"], ["all_losses_dict", "domain_net_model_dict", "event"])

                best_theory_idx = get_best_model_idx(T.net_dict, X_train, y_train, loss_fun_cumu = T.loss_fun_cumu)
                loss_record = T.domain_net.simplify(X_train, best_theory_idx, mode = "to_symbolic", loss_type = "cross-entropy", simplify_criteria = simplify_criteria)
                record_data(info_dict_single["data_record_simplification-domain-model"], [T.get_losses(X_test, y_test, **kwargs), loss_record, T.domain_net.model_dict, "after_to_symbolic"], ["all_losses_dict", "loss_record", "domain_net_model_dict", "event"])

                loss_record = T.domain_net.simplify(X_train, best_theory_idx, mode = "pair_snap", loss_type = "cross-entropy", snap_mode = "integer", simplify_criteria = simplify_criteria)
                record_data(info_dict_single["data_record_simplification-domain-model"], [T.get_losses(X_test, y_test, **kwargs), loss_record, T.domain_net.model_dict, "after_pair_snap"], ["all_losses_dict", "loss_record", "domain_net_model_dict", "event"])
                info_dict_single["domain_net_simplified_pair_snap"] = deepcopy(T.domain_net.model_dict)

                loss_record = T.domain_net.simplify(X_train, best_theory_idx, mode = "activation_snap", loss_type = "cross-entropy", activation_source = "sigmoid", 
                                                    activation_target = "heaviside", epochs = 10000, simplify_criteria = simplify_criteria)
                record_data(info_dict_single["data_record_simplification-domain-model"], [T.get_losses(X_test, y_test, **kwargs), loss_record, T.domain_net.model_dict, "after_activation_snap"], ["all_losses_dict", "loss_record", "domain_net_model_dict", "event"])
                info_dict_single["domain_net_simplified_activation_snap"] = deepcopy(T.domain_net.model_dict)

                loss_record = T.domain_net.simplify(X_train, best_theory_idx, mode = "pair_snap", loss_type = "cross-entropy", snap_mode = "integer", simplify_criteria = simplify_criteria)
                record_data(info_dict_single["data_record_simplification-domain-model"], [T.get_losses(X_test, y_test, **kwargs), loss_record, T.domain_net.model_dict, "after_second_pair_snap"], ["all_losses_dict", "loss_record", "domain_net_model_dict", "event"])
                info_dict_single["domain_net_simplified_final"] = deepcopy(T.domain_net.model_dict)
                print("\nSimplified domain expressions:")
                T.domain_net.show_expression()
                print()
                if save_image:
                    T.plot(X_test, y_test, true_domain = info["true_domain_test"], view_init = view_init, figsize = (10, 8), filename = filename[:-2] + "/{0}_domain-simplified".format(env_name) if save_image else None, is_show = isplot)

            print("\n\n")

        if isplot:
            try:
                process_loss(info_dict_single, loss_core = loss_core)
            except:
                pass

        # Record and determine whether to run another trial:
        loss_dict = T.get_losses(X_test, y_test)
        loss_evaluation = loss_dict["mse_with_domain"] if exp_mode != "base" else loss_dict["mse_without_domain"]
        record_data(info_dict[env_name], [loss_evaluation, info_dict_single, event_name], ["loss_evaluation", "info_dict_single", "event"])
#         info_dict, _ = load_info_dict(info_dict, filename)
        print("#" * 70 + "\nEvaluation for current trial for {0}:\n".format(env_name) + "#" * 70)
        if loss_evaluation < loss_success_threshold:
            if exp_mode in ["continuous", "newb"]:
                event_name = "hit_until_symbolic_matching"
                if env_name not in target_symbolic_expressions or target_symbolic_expressions[env_name] is None or is_simplify_model is False:
                    print("When the target_symbolic_expression for the corresponding env_name is None or 'is_simplify_model' is False, cannot compare symbolic matching.")
                    print("\nThe loss is below the success threshold {0}. Break.\n".format(loss_success_threshold))
                    break
                else:
                    is_symbolic_matching, num_matches = check_expression_matching(T.pred_nets.get_sympy_expression(), 
                                                                                  target_symbolic_expressions[env_name] if env_name in target_symbolic_expressions else [standardize("0")], 
                                                                                  tolerance = matching_numerical_tolerance, 
                                                                                  snapped_tolerance = matching_snapped_tolerance)
                    if is_symbolic_matching:
                        print("\nThe symbolic expressions fully match the target expressions. Break.\n")
                        break
                    else:
                        print("\n{0} out of {1} expressions match the target expression {2}. Do another trial.\n".format(num_matches, len(T.pred_nets.get_sympy_expression()), target_symbolic_expressions[env_name]))
            elif exp_mode == "base":
                print("\nThe loss is below the success threshold {0}. Break.\n".format(loss_success_threshold))
                break
            else:
                raise Exception("exp_mode {0} not recognized!".format(exp_mode))
        else:
            print("\nThe loss does not get down to {0}, Do another trial.".format(loss_success_threshold))
        print("#" * 70 + "\n")

    # Record the last trial (either succeed or at the max_trial):
    export_csv_with_domain_prediction(env_name = env_name, domain_net = T.domain_net, num_output_dims = num_output_dims, num_input_steps = num_input_steps, total_provided_steps = 3 if num_output_dims != 4 else 1,
                                      csv_dirname = csv_dirname + "MYSTERIES", write_dirname = filename[:-2] + "_mys/", is_Lagrangian = is_Lagrangian, is_cuda = is_cuda)
    info_dict[env_name]["pred_nets"] = pred_nets.model_dict
    info_dict[env_name]["domain_net"] = domain_net.model_dict
    if hasattr(T, "autoencoder"):
        info_dict[env_name]["autoencoder"] = T.autoencoder.model_dict
    if is_simplify_model:
        info_dict[env_name]["pred_nets_simplified"] = T.pred_nets.model_dict
    if is_simplify_domain:
        info_dict[env_name]["domain_net_simplified"] = T.domain_net.model_dict
    # Add theory to the theory hub:
    save_to_hub(pred_nets, domain_net, theory_hub, theory_type = "neural", theory_add_threshold = theory_add_mse_threshold, is_Lagrangian = is_Lagrangian)
    save_to_hub(T.pred_nets, T.domain_net, theory_hub, theory_type = "simplified", theory_add_threshold = theory_add_mse_threshold, is_Lagrangian = is_Lagrangian)

    # Save files:
    if not is_batch:
        pickle.dump(info_dict, open(filename, "wb"))
    else:
        pickle.dump(info_dict, open(filename_batch, "wb"))

    # Write the learned expression to csv_file:
    coeff_learned_list, is_snapped_list = get_coeff_learned_list(T.pred_nets)
    open(filename[:-2] + "_mys/" + env_name + "_learned.csv", 'w').close()
    open(filename[:-2] + "_mys/" + env_name + "_snapped.csv", 'w').close()
    with open(filename[:-2] + "_mys/" + env_name + "_learned.csv", "a") as f:
        for coeff in coeff_learned_list:
            if len(coeff) > 0:
                line = ",".join([str(element) for element in coeff]) + '\n'
                f.write(line)
    with open(filename[:-2] + "_mys/" + env_name + "_snapped.csv", "a") as f:
        for is_snapped in is_snapped_list:
            if len(is_snapped) > 0:
                line = ",".join([str(element) for element in is_snapped]) + '\n'
                f.write(line)

    if isplot:
        try:
            process_loss(info_dict[env_name]["info_dict_single"][-1], loss_core = loss_core)
        except Exception as e:
            print(e)

