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
