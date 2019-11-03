# AI_physicist

``AI_physicist'' provides a paradigm and algorithms for learning and manipulation of ***theories***, i.e. small specialized models together with the domain that they are accurate. It contains algorithms for 
- **Differentiable divide-and-conquer (DDAC)** for simultaneous learning of the theories and their domains
- **Simplification** of theories by Occam's razor with MDL
- **Unification** of theories into master theories that can generate a continuum of theories
- **Lifelong learning** by storing of theories in theory hub and proposing theories for novel environments.

More details are provided in the paper "Toward an AI Physicist for unsupervised learning", Tailin Wu and Max Tegmark (2019) \[[*Physical Review E*](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.033311)\]\[[arXiv](https://arxiv.org/abs/1810.10525)\]. It is under active development and more functionalities are continuously added to the agent.

## Installation
First clone the directory. Since the model construction and training of this repository utilize [pytorch_net](https://github.com/tailintalent/pytorch_net) as a submodule, then run the following command:
```
git submodule init; git submodule update
```

The PyTorch requirement is >=0.4.1. Other requirements are in [requirements.txt](https://github.com/tailintalent/AI_physicist/blob/master/requirements.txt)

## Datasets
The datasets used for the paper is provided in [here](https://space.mit.edu/home/tegmark/aiphysicist.html). Put the unzipped "MYSTERIES", "GROUND_TRUTH" directly under datasets/. To use your own dataset, put your csv file under datasets/MYSTERIES/. For each dataset, the first num_output_dims * num_input_dims columns are used as input X, the next num_output_dims columns are used as target Y, and if is_classified = True, the last column should provide true domain ID for evaluation. Take num_output_dims = 2, num_input_dims = 2, is_classified = True as an example, the dataset should look like:

x1, y1, x2, y2, x3, y3, domain_id1

x2, y2, x3, y3, x4, y4, domain_id2

x3, y3, x4, y4, x5, y5, domain_id3

...


If using files in "GROUND_TRUTH" (where true domain ids are provided as evaluation), set csv_dirname = "../datasets/GROUND_TRUTH/" and is_classified = True; If using files in "MYSTERIES", set csv_dirname = "../datasets/MYSTERIES/" and is_classified = False;

## Usage
The main experiment file is [theory_learning/theory_exp.ipynb](https://github.com/tailintalent/AI_physicist/blob/master/theory_learning/theory_exp.ipynb) (or the [theory_exp.py](https://github.com/tailintalent/AI_physicist/blob/master/theory_learning/theory_exp.py) counterpart for terminal. Both of which can be directly run), which contains the DDAC, simplification and lifelong learning algorithms for the AI Physicist.

Before running the experiment, first set up the path and correct settings for the datasets in line 61-83 of [theory_exp.py](https://github.com/tailintalent/AI_physicist/blob/master/theory_learning/theory_exp.py) (or the corresponding [theory_exp.ipynb](https://github.com/tailintalent/AI_physicist/blob/master/theory_learning/theory_exp.ipynb) file). Particularly, important settings are:
- csv_dirname: path to the dataset file
- csv_filename_list: list of dataset files to run, so that each dataset's path is csv_dirname + env_name + ".csv", where env_name is the element in csv_filename_list
- is_classified: whether the csv files provide the true_domain id for evaluation
- num_output_dims: dimension of states at each time step
- num_input_dims: number of time steps in the past used as X
- Other important settings, e.g. num_theories_init (number of random theories to start with) and add_theory_limit (maximum allowed number of theories).

To run batch experiments in a cluster, set up the hyperparameters in run_exp/run_theory.py, and run
```
python run_exp/run_theory.py JOB_ID
```
where the ``JOB_ID'' is a number (between 0 to TOTAL # of hyperparameter combinations - 1) will map to a specific hyperparameter combination.

[theory_learning/theory_unification.ipynb](https://github.com/tailintalent/AI_physicist/blob/master/theory_learning/theory_unification.ipynb) in addition contains the unification algorithm.

The AI Physicist uses [pytorch_net](https://github.com/tailintalent/pytorch_net) for flexible construction of PyTorch neural networks with different types of layers, including Simple_Layer (dense layer), Symbolic_Layer (a symbolic layer using sympy with learnable parameters), and methods for training, simplification and conversion between different types of layers. See its [tutorial](https://github.com/tailintalent/pytorch_net/blob/master/Tutorial.ipynb) for how to use it.

## New features added to the AI Physicist:
Features are continuously added to the AI physicist. These features may or may not work, and can be turned on and off in the hyperparameter settings in [theory_learning/theory_exp.ipynb](https://github.com/tailintalent/AI_physicist/blob/master/theory_learning/theory_exp.ipynb) or in [run_exp/run_theory.py](https://github.com/tailintalent/AI_physicist/blob/master/run_exp/run_theory.py). Some features added compared to the original paper (Wu and Tegmark, 2019) are:
- Autoencoder
- Learning Lagrangian instead of the Equation of Motion
- Annealing of order for the generalized-mean loss
- Unification of theories with neural network (instead of symbolic)

## Citation
If you compare with, build on, or use aspects of the AI Physicist work, please cite the following:

```
@article{wu2019toward,
    title={Toward an artificial intelligence physicist for unsupervised learning},
    author={Wu, Tailin and Tegmark, Max},
    journal={Physical Review E},
    volume={100},
    number={3},
    pages={033311},
    year={2019},
    publisher={APS}
}
```
