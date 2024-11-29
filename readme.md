# GNN Recruitment (gnnr)
Recruitment process enhanced by Graph Neural Networks (GNNs)
 
### Prerequirements:
- Install [uv](https://docs.astral.sh/uv/)

Clone the repo and change directory:
```shell
git clone https://github.com/dartt0n/gnn-recruitment.git && cd gnn-recruitment
```

### Notebooks
Different python notebooks are available in the `experiments` directory.
### Premade notebooks

| Notebook        | Colab                                                                                                                                                                  |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 001-gin-shared  | [![001-gin-shared](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ihrwer306qKIWNzgXSYmfplnRV3_q_jD?usp=sharing])  |
| 002-sage-shared | [![002-sage-shared](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17DeQqtKRhH7ECYM7sNqlH-rJpyaZ_7JV?usp=sharing]) |
| 003-gat-shared  | [![003-gat-shared](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15uV6UfSsuji_1ZPCKOYJGD_veiUDzL6h?usp=sharing])  |
| 004-rgnn-shared | [![004-rgnn-shared](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rKQ9JLH5sH3udmIwQPvoMwy-qp9y6E67?usp=sharing]) |
| 005-cgnn-shared | [![005-cgnn-shared](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_m8FTw4bVhCXcgaMoINZEjBXcnbGktxE?usp=sharing]) |


### Manual
To run them, you need to install the dependencies. The best way to do it is to use [uv](https://docs.astral.sh/uv).
```shell
git clone https://github.com/dartt0n/gnn-recruitment.git && cd gnn-recruitment
uv sync # install base dependencies
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cpu.html # for cpu
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html # for gpu

uv sync --extra cli # if you want to use CLI
uv sync --extra jupyter # if you want to use Jupyter
uv sync --extra cli --extra jupyter # for both
```

However, if you are using Google Colab, Kaggle, or InnoDataHub you can just add the following command at the top of the notebook:
1. Google Colab or Kaggle
```shell
!pip install datasets lightning wandb sentence-transformers polars torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```
1. InnoDataHub
```shell
!pip install datasets lightning wandb sentence-transformers polars torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.1+cu119.html
```

### Create Python environment
```shell
uv sync # install base dependencies
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cpu.html # for cpu
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html # for gpu

uv sync --extra cli # if you want to use CLI
uv sync --extra jupyter # if you want to use Jupyter
uv sync --extra cli --extra jupyter # for both
```

### Generate data
```shell
# access help
uv run gnnr --help
uv run gnnr data --help
uv run gnnr data generate --help

# generate data with synthetic data
uv run gnnr data generate --output ./data
# or
uv run gnnr data generate --output ./data --no-synthetic

# also generate synthetic data
uv run gnnr data generate --output ./data --synthetic

# specify model to use for embeddings
uv run gnnr data generate --output ./data \\
    --synthetic \\
    --model sentence-transformers/all-MiniLM-L12-v2

# only synthetic data
uv run gnnr data synthetic \
    --model sentence-transformers/all-MiniLM-L12-v2 \
    --output ./data
```

## Format code
```shell
uv run ruff format .
```
