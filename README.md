# CS F425 Deep Learning Assignemnt


## Pre Requisites
1. Install miniconda
2. Test that miniconda is working `conda --help`

# Setting up the environment

```sh
conda env create python=3.7 --file env.yml
conda activate DL
```

## Running the code and tracking 

Edit the main file to set the desired architecture

```sh
python DNN/main.py
```

To view tensorboard logs, in another terminal run

```
conda activate DL
tensorboard --logdir=runs
```