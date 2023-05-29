# Controllable-Multi-Objective-Reranking

Controllable-Multi-Objective-Reranking is modified on [LibRerank](https://github.com/LibRerank-Community/LibRerank) 

## Requirements

+ Ubuntu 20.04 or later (64-bit)
+ GPU support requires a CUDA®-enabled card
+ For NVIDIA GPUs, the r455 driver must be installed

For wheel installation:
+ Python 3.8
+ pip 19.0 or later

## Quick Started

Our experimental environment is Ubuntu20.04(necessary)+Python3.8(necessary)+CUDA11.4+TensorFlow1.15.5.

#### Create virtual environment(optional)

```
pip install --user virtualenv
~/.local/bin/virtualenv -p python3 ./venv
source venv/bin/activate
```

#### Install CMR from source

```
git clone https://github.com/lyingCS/Controllable-Multi-Objective-Reranking.git
cd Controllable-Multi-Objective-Reranking
pip config set global.extra-index-url https://pypi.ngc.nvidia.com    # optional
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple    # optional
pip config set global.trusted-host mirrors.aliyun.com\npypi.ngc.nvidia.com    # optional
make init 
```

#### Decompress evaluator checkpoint

For facilitate the training of the generator, we provide a  version of the checkpoints of CMR_evaluator that have been pretrained. We first need to decompress it.

```
tar -xzvf ./model/save_model_ad/10/*.tar.gz -C ./model/save_model_ad/10/
```

#### Run example

Run re-ranker

```
python run_reranker.py
```

Model parameters can be set by using a config file, and specify its file path at `--setting_path`, e.g., `python run_ranker.py --setting_path config`. The config files for the different models can be found in `example/config`. Moreover, model parameters can also be directly set from the command line.

**For more information please refer to [LibRerank_README.md](./LibRerank_README.md)**
