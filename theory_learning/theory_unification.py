
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from copy import deepcopy
import itertools
import datetime
import torch
import pprint as pp

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_physicist.theory_learning.models import Loss_Fun_Cumu
from AI_physicist.theory_learning.theory_model import Theory_Training, get_loss, get_best_model_idx, get_preds_valid, get_valid_prob
from AI_physicist.theory_learning.theory_hub import Theory_Hub, load_model_dict_at_theory_hub, unification_symbolic
from AI_physicist.theory_learning.util_theory import plot_theories, plot3D, process_loss, plot_loss_record, to_one_hot, plot_indi_domain, get_mystery, to_pixels, standardize_symbolic_expression, check_expression_matching
from AI_physicist.theory_learning.util_theory import prepare_data_from_matrix_file, get_epochs, export_csv_with_domain_prediction, extract_target_expression, get_coeff_learned_list, load_theory_training, get_dataset_from_file, get_piecewise_dataset
from AI_physicist.settings.filepath import theory_PATH
from AI_physicist.settings.global_param import COLOR_LIST
from AI_physicist.pytorch_net.util import Loss_Fun, make_dir, Loss_with_uncertainty, Early_Stopping, record_data, plot_matrices, get_args, serialize, to_string, filter_filename, to_np_array, to_Variable, get_variable_name_list, get_param_name_list
from AI_physicist.pytorch_net.net import Conv_Autoencoder, load_model_dict, train_simple
import pandas as pd
pd.set_option('max_colwidth', 400)

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    isplot = True
except:
    import matplotlib
    matplotlib.use('Agg')
    isplot = False
is_cuda = torch.cuda.is_available()


# ## Settings for which unification methods to turn on:

# In[2]:


is_symbolic_unification = True
is_network_unification = True


# ## Load theory hub:

# In[3]:


"""Load the theory hub. Should point to the directory that the theory hub lives."""
exp_id = "test"
date_time = "10-15"
dirname = theory_PATH + "/{0}_{1}/".format(exp_id, date_time)
filename_hub = dirname + "file_continuous_num_4_pred_8_linear_dom_8_onehot_mse_1e-07_sim_DLs-0-3-relative_optim_adam-0.005_adam-0.001_reg_1e-08_1e-05_batch_2000_core_DLs_order_-1_lossd_None_False_infl_False_#_20000_mul_5000_MDL_both_2D_3L_id_210_7_hub.p"
theory_hub = load_model_dict_at_theory_hub(pickle.load(open(filename_hub, "rb")), is_cuda = is_cuda)


# ## Symbolic unification:

# In[ ]:


"""Implementing the symbolic unification algorithm (Alg. 4) in (Wu and Tegmark, 2019):"""
if is_symbolic_unification:
    num_clusters = 3  # The number of clusters for clustering the theories into
    df, exprs_unified_list = unification_symbolic(theory_hub.theory, num_clusters=num_clusters, verbose=True)


# ## Learning network unification:

# In[ ]:


"""Directly learn multiple master theories that unifies subsets of theories in the theory hub (new feature).
Here the unification is using generalized-mean loss as introduced in (Wu and Tegmark, 2019). Instead of operating
on data points, this generalized-mean loss is operating on theories, s.t. for each master theories, it only 
specilizes to subset of theories.
"""
if is_network_unification:  
    # Settings on theory:
    array_id = 0
    input_size = 4
    output_size = 2

    num_master_theories = 5
    master_model_type = "regulated-Net"
    statistics_output_neurons = 6
    master_loss_combine_mode = "on-loss"
    master_loss_core = "mse"
    master_loss_mode = "harmonic"
    master_model_num_neurons = 8
    master_optim_type = ("adam", 1e-3)
    master_optim_type_classifier = ("adam", 5e-5)
    master_reg_amp = 1e-3
    master_reg_amp_classifier = 1e-3
    theory_remove_threshold = 5e-3
    target_array_id = "7"
    filter_words = []

    exp_id = get_args(exp_id, 1)
    num_master_theories = get_args(num_master_theories, 2, "int")
    master_model_type = get_args(master_model_type, 3)
    statistics_output_neurons = get_args(statistics_output_neurons, 4, "int")
    master_loss_combine_mode = get_args(master_loss_combine_mode, 5)
    master_loss_core = get_args(master_loss_core, 6)
    master_loss_mode = get_args(master_loss_mode, 7)
    master_model_num_neurons = get_args(master_model_num_neurons, 8, "int")
    master_optim_type = get_args(master_optim_type, 9, "tuple")
    master_optim_type_classifier = get_args(master_optim_type_classifier, 10, "tuple")
    master_reg_amp = get_args(master_reg_amp, 11, "float")
    master_reg_amp_classifier = get_args(master_reg_amp_classifier, 12, "float")
    theory_remove_threshold = get_args(theory_remove_threshold, 13, "float")
    target_array_id = get_args(target_array_id, 14, "int")
    filter_words = get_args(filter_words, 15, "tuple")
    date_time = get_args(date_time, 16)
    array_id = get_args(array_id, 17, "int")

    # Setting up dirname and filename:
    load_previous = True
    filename_hub_cand = filter_filename(dirname, include = [target_array_id, "hub.p", *filter_words])
    filename_hub = dirname + filename_hub_cand[0]
    print("filename_hub: {0}".format(filename_hub))

    filename = filename_hub[:-6] + ".p"
    make_dir(filename)
    print("filename: {0}\n".format(filename))

    filename_unification = filename[:-2] + "/unification/num_{0}_type_{1}_statistics_{2}_{3}_core_{4}_mode_{5}_neurons_{6}_optim_{7}_{8}_reg_{9}_{10}_thresh_{11}_id_{12}".format(
        num_master_theories, master_model_type, statistics_output_neurons, master_loss_combine_mode, master_loss_core, master_loss_mode, master_model_num_neurons,
        to_string(master_optim_type), to_string(master_optim_type_classifier), master_reg_amp, master_reg_amp_classifier, theory_remove_threshold, array_id,
    )
    make_dir(filename_unification)
    print("filename_unification: {0}\n".format(filename_unification))

    # Initialize certain parameters:
    master_reg_dict = {"master_model": {"weight": master_reg_amp, "bias": master_reg_amp},
                       "statistics_Net": {"weight": master_reg_amp, "bias": master_reg_amp},
                      }
    master_reg_dict_classifier = {"classifier": {"weight": master_reg_amp_classifier, "bias": master_reg_amp_classifier}}
    struct_param_regulated_Net = [
            [master_model_num_neurons, "Simple_Layer", {}],
            [master_model_num_neurons, "Simple_Layer", {}],
            [output_size, "Simple_Layer", {"activation": "linear"}],
    ]
    
    # Load theory_hub:
    wait_time = 1
    wait_time_exponent = 1.2
    while True:
        theory_hub = load_model_dict_at_theory_hub(pickle.load(open(filename_hub, "rb")), is_cuda = is_cuda)
        if len(theory_hub.theory) == 0:
            wait_time *= wait_time_exponent
            print("No theory exists in the theory_hub. Wait for {0:.1f} seconds...".format(wait_time))
            time.sleep(wait_time)
        else:
            print("Succesfully load theory_hub {0} with non-empty theory_collections!".format(filename_hub))
            break

    info_dict = {}
    # Propose master_theories:
    theory_dict = theory_hub.get_theory_tuples(input_size = input_size)
    master_theory_dict = theory_hub.propose_master_theories(num_master_theories = num_master_theories,
                                                            input_size = input_size,
                                                            output_size = output_size,
                                                            statistics_output_neurons = statistics_output_neurons,
                                                            master_model_type = master_model_type,
                                                            struct_param_regulated_Net = struct_param_regulated_Net,
                                                           )

    # Fit the master_theories to all the theories:
    data_record = theory_hub.fit_master_theory(
        master_theory_dict = master_theory_dict,
        theory_dict = theory_dict,
        optim_type = master_optim_type,
        reg_dict = master_reg_dict,
        loss_core = master_loss_core,
        loss_mode = master_loss_mode,
        loss_combine_mode = master_loss_combine_mode,
        num_iter = 1000,
        patience = 10,
        inspect_interval = 10,
        isplot = isplot,
        filename = filename_unification,
    )
    info_dict["data_record_whole"] = deepcopy(data_record)
    info_dict["master_theory_whole"] = deepcopy({name: master_theory.model_dict for name, master_theory in master_theory_dict.items()})
    pickle.dump(info_dict, open(filename_unification + ".p", "wb"))

    # Assign master theory to theories:
    group_list = theory_hub.assign_master_theories_to_theories(master_theory_dict, theory_dict)
    print("=" * 150 + "\nMaster_theory assignment:")
    for assigned_master_theory_dict, assigned_theory_dict in group_list:
        print("master_theory: {0}".format(list(assigned_master_theory_dict.keys())[0]))
        print("assigned_theories: {0}\n".format(list(assigned_theory_dict.keys())))

    # Train each master_theory individually:
    for i, (assigned_master_theory_dict, assigned_theory_dict) in enumerate(group_list):
        print("=" * 150)
        print("Fitting {0}th assigned group:".format(i))
        print("master_theory: {0}".format(list(assigned_master_theory_dict.keys())[0]))
        print("assigned_theories: {0}\n".format(list(assigned_theory_dict.keys())) + "=" * 150 + "\n")
        master_theory_name = list(assigned_master_theory_dict.keys())[0]
        master_theory = assigned_master_theory_dict[master_theory_name]

        # Train the master_model:
        data_record = theory_hub.fit_master_theory(
            master_theory_dict = assigned_master_theory_dict,
            theory_dict = assigned_theory_dict,
            optim_type = master_optim_type,
            reg_dict = master_reg_dict,
            loss_core = master_loss_core,
            loss_mode = master_loss_mode,
            loss_combine_mode = master_loss_combine_mode,
            num_iter = 10000,
            patience = 20,
            inspect_interval = 50,
            isplot = isplot,
            filename = filename_unification,
        )
        info_dict["data_record_{0}".format(i)] = deepcopy(data_record)

        # Train the master classifier:
        data_record_classifier = theory_hub.fit_master_classifier(
            master_theory = master_theory,
            theory_dict = assigned_theory_dict,
            optim_type_classifier = master_optim_type_classifier,
            reg_dict_classifier = master_reg_dict_classifier,
        )
        info_dict["data_record_classifier_{0}".format(i)] = deepcopy(data_record_classifier)

        # Add master_theory_tuple:
        theory_hub = load_model_dict_at_theory_hub(pickle.load(open(filename_hub, "rb")), is_cuda = is_cuda)
        theory_hub.add_master_theory(name = master_theory_name,
                                     master_theory = assigned_master_theory_dict[master_theory_name],
                                     theory_tuples = assigned_theory_dict,
                                     is_replace = True,
                                    )
        master_theory_tuple = Master_Theory_Tuple(master_theory = assigned_master_theory_dict[master_theory_name], theory_tuples = assigned_theory_dict)
        info_dict["master_theory_tuple_{0}".format(i)] = deepcopy(master_theory_tuple.model_dict)

        # Removed passed theory (whose loss with the master_theory is less than the theory_remove_threshold):
        master_loss_fun = Master_Loss_Fun(core = master_loss_core, cumu_mode = master_loss_mode)
        loss_matrix = master_loss_fun.get_loss_matrix(assigned_master_theory_dict, assigned_theory_dict, use_train = False)
        passed_theory = (loss_matrix < theory_remove_threshold).data.long()[0]
        passed_theory_names = []
        for j in range(len(passed_theory)):
            is_pass = passed_theory[j]
            if is_pass == 1:
                passed_theory_names.append(list(assigned_theory_dict.keys())[j])
    #     if experiment_mode != "on-unification":
    #         popped_theories = theory_hub.remove_theories(passed_theory_names)
    #         pickle.dump(theory_hub.model_dict, open(filename_hub, "wb"))
        info_dict["popped_theories"] = passed_theory_names
        pickle.dump(info_dict, open(filename_unification + ".p", "wb"))

