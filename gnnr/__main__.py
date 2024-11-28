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
        dataset_generation.from_stereotypes_in_llms(output / "_dataset_00.json")


if __name__ == "__main__":
    app()
