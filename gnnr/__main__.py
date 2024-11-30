import networkx as nx
from pathlib import Path
from typing import Annotated

import polars as pl
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


@data_cmd.command(name="export")
def data_export(
    format: Annotated[str, t.Argument(help="The format to export the dataset.")],
    input: Annotated[Path, t.Option(help="The path to file to read edge data from")] = Path("edges.json"),
    output: Annotated[Path, t.Option(help="The path to output file where to store the dataset.")] = Path("output.csv"),
):
    console = Console()
    output.parent.mkdir(parents=True, exist_ok=True)

    with console.status("Generating graph object..."):
        G = nx.from_pandas_edgelist(
            pl.read_json(input)
            .select(
                pl.col("candidate_id").alias("source"),
                pl.col("job_id").alias("target"),
                pl.col("decision").alias("weight"),
            )
            .to_pandas(),
            "source",
            "target",
            edge_attr=["weight"],
        )

    if format == "edgelist":
        with console.status("Writing edgelist..."):
            nx.write_edgelist(G, output)
    elif format == "graphml":
        with console.status("Writing graphml..."):
            nx.write_graphml(G, output)
    elif format == "gexf":
        with console.status("Writing gexf..."):
            nx.write_gexf(G, output)
    elif format == "dot":
        with console.status("Writing dot..."):
            nx.drawing.nx_pydot.write_dot(G, output)
    else:
        raise ValueError(f"Unknown format: {format}")


if __name__ == "__main__":
    app()
