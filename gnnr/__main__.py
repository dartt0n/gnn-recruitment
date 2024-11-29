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
    model: Annotated[str, t.Option(help="The model to use for embeddings.")] = "sentence-transformers/all-MiniLM-L12-v2",
    synthetic: Annotated[bool, t.Option(help="Whether to generate synthetic dataset.")] = False,
):
    console = Console()

    output.mkdir(parents=True, exist_ok=True)
    generated_files = []

    with console.status("generating dataset part from Stereotypes in LLMs..."):
        dataset_generation.from_HRSLLM(output / "_part00")
        generated_files.append(output / "_part00")

    if synthetic:
        with console.status("generating syntetic dataset part..."):
            dataset_generation.synthetic(model, output / "_part01")
            generated_files.append(output / "_part01")

    with console.status("merging dataset parts..."):
        dataset_generation.merge_jsons(generated_files, output / "dataset")


@data_cmd.command(name="synthetic")
def data_synthetic(
    model: Annotated[str, t.Option(help="The model to use for embeddings.")] = "sentence-transformers/all-MiniLM-L12-v2",
    output: Annotated[Path, t.Option(help="The path to directory where to store generated dataset files.")] = Path("output"),
):
    output.mkdir(parents=True, exist_ok=True)
    dataset_generation.synthetic(model, output / "_dataset_01.json")


if __name__ == "__main__":
    app()
