from pathlib import Path

import polars as pl


def from_stereotypes_in_llms(write_location: str | Path):
    (
        pl.read_parquet("hf://datasets/Stereotypes-in-LLMs/hiring-analyses-optimized_parameters-en/data")
        .lazy()
        .unique(["candidate_id", "job_id"])
        .drop(["lang", "protected_group", "protected_attr", "group_id"])
        .collect()
        .write_json(write_location)
    )
