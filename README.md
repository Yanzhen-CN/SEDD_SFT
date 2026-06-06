# SEDD SFT on s1K-1.1

This fork adds `sft_pipeline/` for matched QA/QAR supervised fine-tuning experiments on `simplescaling/s1K-1.1`, while keeping the original SEDD training code intact.

## SFT Pipeline

Edit the config:

```bash
vim sft_pipeline/sft_config.yaml
```

Run data processing and all selected SFT jobs from one entry point:

```bash
python run_sft.py
```

On a server, a typical background command is:

```bash
nohup python run_sft.py > sft_run.log 2>&1 &
tail -f sft_run.log
```

Preview without launching training:

```bash
python run_sft.py --dry-run
```

If the server is offline, set `data.arrow_path` in `sft_config.yaml` to a local Hugging Face Arrow cache file.

### Config fields

The config file is a thin wrapper around the original SEDD Hydra arguments.

```yaml
data:
  build: true          # run data_process.py before training
  arrow_path: null     # optional local s1K-1.1 Arrow file for offline servers
  valid_ratio: 0.05    # validation split ratio for QA/QAR
  seed: 42             # split seed; QA and QAR share the same split

run:
  selected: all        # "all", "QA", "QAR", or ["QA", "QAR"]
  parallel: true       # launch selected jobs at the same time
  execute: true        # false only prints the official SEDD commands

defaults:
  steps: 200           # maps to training.n_iters
  length: 256          # maps to model.length
  batch_size: 2        # maps to training.batch_size and eval.batch_size
  log_freq: 10         # maps to training.log_freq
  eval_freq: 50        # maps to training.eval_freq
  snapshot_freq: 200   # maps to training.snapshot_freq

runs:
  QA:
    dataset: QA
    length: 256
    batch_size: 2
    cuda_visible_devices: "0"

  QAR:
    dataset: QAR
    length: 512
    batch_size: 1
    cuda_visible_devices: "1"

sedd:
  ngpus: 1             # maps to ngpus
  model: small         # maps to model=small / model=medium
  graph: absorb        # maps to graph.type
  noise: loglinear     # maps to noise.type
  accum: 1             # maps to training.accum
  snapshot_sampling: false
  perplexity: false
  cache_dir: sft_pipeline/cache
```

For QAR, use a larger `length` and usually a smaller `batch_size` because reasoning traces are much longer:

```yaml
runs:
  QAR:
    dataset: QAR
    length: 512
    batch_size: 1
    cuda_visible_devices: "1"
```

If the server has only one GPU, set:

```yaml
run:
  selected: all
  parallel: false
```

This runs QA first, then QAR.

The main comparison is QA vs QAR under the same train/validation split. The primary evaluation signal is validation Score Entropy loss and generation stability, not mathematical answer accuracy.

### Outputs and monitoring

`run_sft.py` calls the original SEDD `train.py`. Outputs are created by the original SEDD code under:

```text
exp_local/sft_QA/DATE/TIME/
exp_local/sft_QAR/DATE/TIME/
```

Each run contains:

```text
.hydra/config.yaml       # final Hydra config used by SEDD
logs                     # plain text terminal-style log
checkpoints/             # periodic checkpoints
checkpoints-meta/        # resume checkpoint
samples/                 # only if snapshot_sampling=true
```

The terminal prints logs in real time, and the same messages are appended to the `logs` file. The original SEDD trainer logs lines such as:

```text
step: 50, training_loss: ...
step: 100, evaluation_loss: ...
```

This setup does not currently write a CSV or implement automatic early stopping. It follows the original SEDD training loop, which runs for `training.n_iters` steps and logs/checkpoints at the configured frequencies. For this take-home, we manually compare QA/QAR using the terminal output and `logs` file, then decide whether to adjust `steps`, `length`, or `batch_size`.

---

# Original Score Entropy Discrete Diffusion
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repo contains a PyTorch implementation for the paper [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
](https://arxiv.org/abs/2310.16834) by [Aaron Lou](https://aaronlou.com), [Chenlin Meng](https://cs.stanford.edu/~chenlin/) and [Stefano Ermon](https://cs.stanford.edu/~ermon/).

![cover](assets/main.gif)

## Design Choices

This codebase is built modularly to promote future research (as opposed to a more compact framework, which would be better for applications). The primary files are 

1. ```noise_lib.py```: the noise schedule
2. ```graph_lib```: the forward diffusion process
3. ```sampling.py```: the sampling strategies
4. ```model/```: the model architecture

## Installation

Simply run

```
conda env create -f environment.yml
```

which will create a ```sedd``` environment with packages installed. Note that this installs with CUDA 11.8, and different CUDA versions must be installed manually. The biggest factor is making sure that the ```torch``` and ```flash-attn``` packages use the same CUDA version (more found [here](https://github.com/Dao-AILab/flash-attention)).

## Working with Pretrained Models

### Download Models

Our pretrained models are hosted on huggingface ([small](https://huggingface.co/louaaron/sedd-small), [medium](https://huggingface.co/louaaron/sedd-medium)). However, models can also be loaded in locally (say after training). All functionality is found in ```load_model.py```.

```
# load in a pretrained model
pretrained_small_model, graph, noise = load_model("louaaron/sedd-small")
pretrained_medium_model, graph, noise = load_model("louaaron/sedd-medium")
# load in a local experiment
local_model, graph, noise = load_model("exp_local/experiment)
```

This loading gives the model, as well as the graph and noise (which are used for the loss/sampling setup).

### Run Sampling

We can run sampling using a command 

```
python run_sample.py --model_path MODEL_PATH --steps STEPS
```

We can also sample conditionally using

```
python run_sample_cond.py --model_path MODEL_PATH --step STEPS --prefix PREFIX --suffix SUFFIX
```

## Training New Models

### Run Training

We provide training code, which can be run with the command
```
python run_train.py
```
This creates a new directory `direc=exp_local/DATE/TIME` with the following structure (compatible with running sampling experiments locally)
```
鈹溾攢鈹€ direc
鈹?  鈹溾攢鈹€ .hydra
鈹?  鈹?  鈹溾攢鈹€ config.yaml
鈹?  鈹?  鈹溾攢鈹€ ...
鈹?  鈹溾攢鈹€ checkpoints
鈹?  鈹?  鈹溾攢鈹€ checkpoint_*.pth
鈹?  鈹溾攢鈹€ checkpoints-meta
鈹?  鈹?  鈹溾攢鈹€ checkpoint.pth
鈹?  鈹溾攢鈹€ samples
鈹?  鈹?  鈹溾攢鈹€ iter_*
鈹?  鈹?  鈹?  鈹溾攢鈹€ sample_*.txt
鈹?  鈹溾攢鈹€ logs
```
Here, `checkpoints-meta` is used for reloading the run following interruptions, `samples` contains generated images as the run progresses, and `logs` contains the run output. Arguments can be added with `ARG_NAME=ARG_VALUE`, with important ones being:
```
ngpus                     the number of gpus to use in training (using pytorch DDP)
training.accum            number of accumulation steps, set to 1 for small and 2 for medium (assuming an 8x80GB node)
noise.type                one of geometric, loglinear 
graph.type                one of uniform, absorb
model                     one of small, medium
model.scale_by_sigma      set to False if graph.type=uniform (not yet configured)
```
Some example commands include
```
# training hyperparameters for SEDD absorb
python train.py noise_lib=loglinear graph.type=absorb model=medium training.accum=2
# training hyperparameters for SEDD uniform
python train.py noise_lib=geometric graph.type=uniform model=small model.scale_by_sigma=False
```

## Other Features

### SLURM compatibility

To train on slurm, simply run 
```
python train.py -m args
```

## Citation
```
@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```
## Acknowledgements

This repository builds heavily off of [score sde](https://github.com/yang-song/score_sde_pytorch), [plaid](https://github.com/igul222/plaid), and [DiT](https://github.com/facebookresearch/DiT).

