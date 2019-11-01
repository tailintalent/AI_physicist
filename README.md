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

The datasets used for the paper is provided in [here](https://space.mit.edu/home/tegmark/aiphysicist.html). Put the unzipped "MYSTERIES", "GROUND_TRUTH" and "filenames.csv" directly under datasets/.

## Usage
The main experiment file is [theory_learning/theory_exp.ipynb](https://github.com/tailintalent/AI_physicist/blob/master/theory_learning/theory_exp.ipynb) (or the .py counterpart for terminal), which contains the DDAC, simplification and lifelong learning algorithms for the AI Physicist.
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
