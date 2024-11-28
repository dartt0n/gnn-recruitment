from pathlib import Path
from typing import Annotated

import typer as t
from rich.console import Console

from gnnr.scripts import dataset_generation

data_cmd = t.Typer()
app = t.Typer(name="gnnr")
app.add_typer(data_cmd, name="data")


@data_cmd.command(name="generate")
def data_generate(
    output: Annotated[Path, t.Option(help="The path to directory where to store generated dataset files.")] = Path("output"),
):
    console = Console()

    output.mkdir(parents=True, exist_ok=True)

    with console.status("generating dataset part from Stereotypes in LLMs..."):
        dataset_generation.from_HRSLLM(output / "_dataset_00.json")


@data_cmd.command(name="synthetic")
def data_synthetic(
    candidates_data: Annotated[Path, t.Option(help="The path to csv file with CVs.")] = Path("input.csv"),
    vacancies_data: Annotated[Path, t.Option(help="The path to csv file with job descriptions.")] = Path("input.csv"),
    output: Annotated[Path, t.Option(help="The path to directory where to store generated dataset files.")] = Path("output"),
    model: Annotated[str, t.Option(help="The model to use for generating synthetic data.")] = "qwen2.5:3b",
):
    output.mkdir(parents=True, exist_ok=True)

    dataset_generation.synthetic(candidates_data, vacancies_data, output, model)


if __name__ == "__main__":
    app()
