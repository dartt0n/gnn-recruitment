import re
from itertools import product
from pathlib import Path
from uuid import uuid4

import polars as pl
from ollama import ChatResponse, chat
from tqdm import tqdm


def from_HRSLLM(write_location: str | Path):
    """Generates dataset from Stereotypes-in-LLMs/hiring-analyses-optimized_parameters-en"""
    (
        pl.read_parquet("hf://datasets/Stereotypes-in-LLMs/hiring-analyses-optimized_parameters-en/data")
        .lazy()
        .unique(["candidate_id", "job_id"])
        .drop(["lang", "protected_group", "protected_attr", "group_id"])
        .collect()
        .write_json(write_location)
    )


def synthetic(
    candidate_data_path: str | Path,
    vacancy_data_path: str | Path,
    write_dir: str | Path,
    model: str,
):
    if isinstance(candidate_data_path, str):
        candidate_data_path = Path(candidate_data_path)

    if isinstance(vacancy_data_path, str):
        vacancy_data_path = Path(vacancy_data_path)

    if isinstance(write_dir, str):
        write_dir = Path(write_dir)

    def invoke(candidate_cv: str, job_description: str) -> bool:
        response: ChatResponse = chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are hiring assistant. Your task to analyse candidate CV and job description and decide, whether to
                    hire the candidate or not. You should return only 1 number: 1 if you think the candidate is suitable
                    for the job, 0 otherwise.
                    """.replace("\n", " "),
                },
                {
                    "role": "user",
                    "content": f"Candidate CV: {candidate_cv}\nJob description: {job_description}",
                },
            ],
            options={"temperature": 0.2, "num_ctx": 8192},
        )

        return re.sub(r"\D", "", response.message.content) == "1"

    candidate_df = pl.read_csv(candidate_data_path).with_columns(
        pl.col("CV").map_elements(lambda x: str(uuid4()), return_dtype=pl.String).alias("candidate_id"),
    )
    candidate_df.write_csv(write_dir / "tmp_candidate.csv")

    vacancy_df = pl.read_csv(vacancy_data_path).with_columns(
        pl.col("Description").map_elements(lambda x: str(uuid4()), return_dtype=pl.String).alias("job_id"),
    )
    vacancy_df.write_csv(write_dir / "tmp_vacancy.csv")

    with open(write_dir / "tmp_hire_decision.csv", "a") as f:
        for (cv, cid), (job, jid) in tqdm(
            product(
                candidate_df.select(["CV", "candidate_id"]).iter_rows(),
                vacancy_df.select(["Description", "job_id"]).iter_rows(),
            ),
            desc="rating candidates using LLM",
            total=len(candidate_df) * len(vacancy_df),
        ):
            decision = invoke(cv, job)

            f.write(f"{cid},{jid},{int(decision)}\n")
