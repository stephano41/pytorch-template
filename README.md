# PyTorch Template Project

Simple project base template for PyTorch deep Learning project.

<!-- TOC depthFrom:1 depthTo:6 orderedList:false -->

- [PyTorch Template Project](#pytorch-template-project)
    - [Installation](#installation)
        - [Requirements](#requirements)
        - [Features](#features)
        - [Folder Structure](#folder-structure)
    - [Usage](#usage)
        - [Hierarchical configurations with Hydra](#hierarchical-configurations-with-hydra)
        - [Using config files](#using-config-files)
        - [Checkpoints](#checkpoints)
        - [Resuming from checkpoints](#resuming-from-checkpoints)
        - [Using Multiple GPU](#using-multiple-gpu)
    - [Customization](#customization)
        - [Project initialization](#project-initialization)
        - [Data Loader](#data-loader)
        - [Trainer](#trainer)
        - [Model](#model)
        - [Loss](#loss)
        - [Metrics](#metrics)
        - [Additional logging](#additional-logging)
        - [Testing](#testing)
        - [Validation data](#validation-data)
        - [Checkpoints](#checkpoints-1)
        - [Tensorboard Visualization](#tensorboard-visualization)
    - [Contribution](#contribution)
    - [TODOs](#todos)
    - [License](#license)

<!-- /TOC -->

## Installation

### Requirements

* Python >= 3.6
* PyTorch >= 1.2
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))
* tqdm
* hydra-core >= 1.2
* optuna
* ray
* scikit-learn

### Features

* Simple and clear directory structure, suitable for most of deep learning projects.
* Hierarchical management of project configurations with [Hydra](https://hydra.cc/docs/intro).
* Advanced logging and monitoring for validation metrics. Automatic handling of model checkpoints.
* Hyperparameter search implementation with ray tune

### Folder Structure

```yaml
  pytorch-template/
  ├── train.py                  # script to start training.
  ├── evaluate.py               # script to evaluate trained model on testset.
  ├── tune.py                   # script to start hyperparameter tuning
  ├── sk_trainer.py             # script to train/tune sk_train models 
  ├── conf # config files. explained in separated section below.
  │   └── ...
  ├── src # source code.
  │   ├── data_loader           # data loading, preprocessing
  │   │   ├── __init__.py
  │   │   └── mnist_data_loaders.py
  │   ├── models
  │   │   ├── __init__.py
  │   │   └── images.py
  │   ├── loss
  │   │   ├── __init__.py
  │   │   └── loss.py
  │   ├── metrics
  │   │   ├── __init__.py
  │   │   ├── confusion_matrix.py
  │   │   └── metric.py
  │   ├── utils
  │   │   ├── __init__.py
  │   │   ├── files.py
  │   │   ├── tune.py
  │   │   ├── util.py
  │   ├── trainer
  │   │   ├── __init__.py
  │   │   └── tune_trainer.py
  │   ├── main_worker.py        # customized class managing training process
  │   ├── logger.py             # tensorboard, train / validation metric logging
  ├── new_project.py            # script to initialize new project
  ├── requirements.txt
  ├── README.md
  └── LICENSE
```

## Usage

This template itself is an working example project which trains a simple model(LeNet)
on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Try `python train.py` to run training.

### Hierarchical configurations with Hydra

This repository is designed to be used with [Hydra](https://hydra.cc/) framework, which has useful key features as
following.

- Hierarchical configuration composable from multiple sources
- Configuration can be specified or overridden from the command line
- Dynamic command line tab completion
- Run your application locally or launch it to run remotely
- Run multiple jobs with different arguments with a single command

Check [Hydra documentation](https://hydra.cc/), for more information.

`conf/` directory contains `.yaml`config files which are structured into multiple **config groups**.

```yaml
  conf/ # hierarchical, structured config files to be used with 'Hydra' framework
  ├── train.yaml                # main config file used for train.py
  ├── evaluate.yaml             # main config file used for evaluate.py
  ├── tune.yaml                 # main config file used for tune.py
  ├── sk_train.yaml             # main config file used for sk_trainer.py
  ├── data_loader
  │   ├── mnist_test.yaml
  │   └── mnist_train.yaml
  ├── model                     # select NN architecture to train
  │   ├── svm.yaml              # for sk_trainer.py
  │   └── mnist_lenet.yaml
  ├── search_alg                     
  │   ├── optuna.yaml             
  │   └── sk_grid_search.yaml
  ├── search_space                   
  │   ├── lenet_search.yaml              
  │   └── svm_search.yaml
  ├── status                    # set train/debug mode.
  │   ├── debug.yaml
  │   ├── tune.yaml             #   tune mode is default with full logging
  │   └── train.yaml            #   train mode is default with full logging
  ├── trainer                            
  │   └── torch_trainer.yaml
  ├── tune_scheduler                            
  │   └── ASHAScheduler.yaml
  │
  └── hydra                     # configure hydra framework
      ├── job_logging           #   config for python logging module
      │   └── custom.yaml
      └── run/dir               #   setup working directory
          └── job_timestamp.yaml
```

### Using config files

Modify the configurations in `.yaml` files in `conf/` dir, then run:

  ```
  python train.py
  ```

At runtime, one file from each config group is selected and combined to be used as one global config.

```yaml
name: MnistLeNet # experiment name.

save_dir: models/
log_dir: ${name}/
resume:

# configuration for data loading.
data_loader:
  _target_: src.data_loader.data_loaders.get_data_loaders
  data_dir: data/
  batch_size: ${batch_size}
  shuffle: true
  validation_split: 0.1
  num_workers: ${n_cpu}

arch:
  _target_: src.model.model.MnistModel
  num_classes: 10
loss:
  _target_: src.model.loss.nll_loss
optimizer:
  _target_: torch.optim.Adam
  lr: ${learning_rate}
  weight_decay: ${weight_decay}
  amsgrad: true
lr_scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: ${scheduler_step_size}
  gamma: ${scheduler_gamma}

metrics:
- _target_: src.model.metric.accuracy
- _target_: src.model.metric.top_k_acc

n_gpu: 1
n_cpu: 8
trainer:
  epochs: 20
  logging_step: 100
  verbosity: 2
  monitor: min loss/valid
  early_stop: 10
  tensorboard: true
```

Add addional configurations if you need.

Those config items containing `_target_` are designed to be used with `instantiate` function of Hydra. For example, When
your config looks like

```yaml
# @package _global_
classitem:
  _target_: location.to.class.definition
  arg1: 123
  arg2: 'example'
```

then usage of instantiate as

```python
example_object = instantiate(config.classitem)
```

is equivalent to

```python
from location.to.class import definition

example_object = definition(arg1=1, arg2='example')
```

This feature is especially useful, when you switch between multiple models with same interface(input, output), like
choosing ResNet or MobileNet for CNN backbone of detection model. You can change architecture by simply using different
config file, not even needing to importing both in code.

### Checkpoints

```yaml
# new directory with timestamp will be created automatically.
# if you enable debug mode by status=debug either in command line or main config,
# checkpoints will be saved under separate directory `outputs/debug`.
outputs/train-!{name}/2020-07-29-12-44-37/
├── basic-variant-state-2020-07-29-12-44-37.json # ray tune files
├── experiment_state-2020-07-29-12-44-37.json # ray tune files
├── .hydra
│   ├── config.yaml # composed config file
│   ├── hydra.yaml
│   └── overrides.yaml 
├── train_func_{trial_id}
│   ├── checkpoint_000000
│   ├── checkpoint_000001
│   ├── ...
│   ├── checkpoint_000014 # last checkpoint
│   ├── config.yaml # contains final config used to build trial after ray selects hyperparamters 
│   ├── # tensorboard log file
│   ├── params.json # used for ray tune hyperparameter tuning 
│   ├── params.pkl
│   ├── progress.csv
│   ├── result.json
│   └── model_latest.pth # checkpoint which is saved last
└── train.log
```

### Resuming from checkpoints

You can resume from a previously saved checkpoint by:

  ```
  python train.py resume=output/train/path/to/checkpoint.pth
  ```

### Using Multiple GPU

You can enable multi-GPU training(with DataParallel) by setting `n_gpu` argument of the config file to larger number. If
configured to use smaller number of gpu than available, first n devices will be used by default. When you want to run
multiple instances of training on larger maching, specify indices of available GPUs by cuda environmental variable.

  ```bash
  # assume running on a machine with 4 GPUs.
  python train.py n_gpu=2 # This will use first two GPU, which are on index 0 and 1
  CUDA_VISIBLE_DEVICES=2,3 python train.py n_gpu=2 # This will use remaining 2 GPUs on index 2 and 3
  ```

## Customization

### Project initialization

Use the `new_project.py` script to make your new project directory with template files.
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made. This script will filter
out unneccessary files like cache, git files or readme file.

### Data Loader

* **Writing your own data loader**

Please refer to `data_loader/data_loaders.py` for an MNIST data loading example.

### Trainer

* **Writing your own trainer**

See ray tune [trainable documentation](https://docs.ray.io/en/latest/tune/api_docs/trainable.html)

### Model

* **Writing your own model**

Write it just like any other pytorch model:

1. **Inherit `nn.Module`**

2. **Implementing abstract methods**

   Implement the foward pass method `forward()`

* **Example**

  Please refer to `models/images.py` for a LeNet example.

### Loss

Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config
file, to corresponding name.

### Metrics

Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file trainer.run.metric, e.g.:

  ```yaml
run:
    metric: [accuracy, top_k_acc],
  ```

### Testing

You can test trained model by running `test.py` passing path to the trained checkpoint by `--checkpoint_dir` and `--checkpoint_name` argument.

### Validation data

To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader
for validation of size specified in your config file. The `validation_split` is the ratio of validation set per total
data(0.0 <= float < 1.0).

**Note**: the `split_validation()` method will modify the original data loader

### Checkpoints

You can specify the name of the training session in config files:

  ```yaml
  name: MNIST_LeNet,
  ```

The checkpoints will be saved in `save_dir/state-name/timestamp/train_func_trial_id/checkpoint_epoch_n`, with timestamp in Y-m-d-H-M-S format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:

  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config': arch_cfg # full config file to create this trial
  }
  ```

### Tensorboard Visualization

This template supports Tensorboard visualization with ray tune

1. **Open Tensorboard server**

   Type `tensorboard --logdir outputs/train/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file and model parameters will be logged. 

## Contribution

Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs

- [ ] Support DistributedDataParallel
- [x] Option to keep top-k checkpoints only
- [ ] Simple unittest code for `nn.Module` and others

## License

This project is licensed under the MIT License. See LICENSE for more details
