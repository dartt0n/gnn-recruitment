# GNN Recruitment (gnnr)
Recruitment process enhanced by Graph Neural Networks (GNNs)

## Development
### Prerequirements:
- Install [uv](https://docs.astral.sh/uv/)

Clone the repo and change directory:
```shell
git clone https://github.com/dartt0n/gnn-recruitment.git && cd gnn-recruitment
```

### Create Python environment
```shell
uv sync
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cpu.html # for cpu
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html # for gpu

uv sync --extra cli # if you want to use CLI
uv sync --extra jupyter # if you want to use Jupyter
```

### Generate data
```shell
uv run gnnr data generate --output ./data
```

## Format code
```shell
uv run ruff format .
```

## Usage
