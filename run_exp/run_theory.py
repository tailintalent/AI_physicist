import os, sys

exp_id=[
"exp1.0",
]

env_names=[
"file",
]

exp_mode = [
"continuous",
#"newb",
#"base",
]

num_theories_init=[
4,
]

pred_nets_neurons=[
8,
]

pred_nets_activation=[
"linear",
# "leakyRelu",
]

domain_net_neurons=[
8,
]

domain_pred_mode=[
"onehot",
]

mse_amp=[
1e-7,
]

simplify_criteria=[
'\("DLs",0,3,"relative"\)',
]

scheduler_settings=[
'\("ReduceLROnPlateau",40,0.1\)',
]

optim_type=[
'\("adam",5e-3\)',
]

optim_domain_type=[
'\("adam",1e-3\)',
]

reg_amp=[
1e-8,
]

reg_domain_amp = [
1e-5,
]

batch_size = [
2000,
]

loss_core = [
"DLs",
]

loss_order = [
-1,
]

loss_decay_scale = [
"None",
]

is_mse_decay = [
False,
]

loss_balance_model_influence = [
False,
]

num_examples = [
20000,
]

iter_to_saturation = [
5000,
]

MDL_mode = [
"both",
]

num_output_dims = [
2,
]

num_layers = [
3,
]

is_pendulum = [
False,
]

date_time = [
"10-9",
]

seed = [
0,
30,
60,
90,
120,
150,
180,
210,
240,
270,
]


def assign_array_id(array_id, param_list):
    if len(param_list) == 0:
        print("redundancy: {0}".format(array_id))
        return []
    else:
        param_bottom = param_list[-1]
        length = len(param_bottom)
        current_param = param_bottom[array_id % length]
        return assign_array_id(int(array_id / length), param_list[:-1]) + [current_param]

array_id = int(sys.argv[1])
param_list = [exp_id,
				env_names,
				exp_mode,
				num_theories_init,
				pred_nets_neurons,
				pred_nets_activation,
				domain_net_neurons,
				domain_pred_mode,
				mse_amp,
				simplify_criteria,
				scheduler_settings,
				optim_type,
				optim_domain_type,
				reg_amp,
				reg_domain_amp,
				batch_size,
				loss_core,
				loss_order,
				loss_decay_scale,
				is_mse_decay,
				loss_balance_model_influence,
				num_examples,
				iter_to_saturation,
				MDL_mode,
				num_output_dims,
				num_layers,
				is_pendulum,
				date_time,
				seed,
]
param_chosen = assign_array_id(array_id, param_list)
exec_str = "python ../theory_learning/theory_exp.py"
for param in param_chosen:
	exec_str += " {0}".format(param)
exec_str += " {0}".format(array_id)
print(param_chosen)
print(exec_str)

from shutil import copyfile
current_PATH = os.path.dirname(os.path.realpath(__file__))
def make_dir(filename):
    import os
    import errno
    if not os.path.exists(os.path.dirname(filename)):
        print("directory {0} does not exist, created.".format(os.path.dirname(filename)))
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print(exc)
            raise

filename = "../data/" + "{0}_{1}/".format(param_chosen[0], param_chosen[-2])
make_dir(filename)
fc = "run_theory.py"
if not os.path.isfile(filename + fc):
        copyfile(current_PATH + "/" + fc, filename + fc)


os.system(exec_str)
