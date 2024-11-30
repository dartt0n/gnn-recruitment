from itertools import product
from pathlib import Path
from uuid import uuid4

import polars as pl
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


def from_HRSLLM(write_location: Path):
    write_location.mkdir(parents=True, exist_ok=True)
    df = (
        pl.read_parquet("hf://datasets/Stereotypes-in-LLMs/hiring-analyses-optimized_parameters-en/data")
        .unique(["candidate_id", "job_id"])
        .drop(["lang", "protected_group", "protected_attr", "group_id"])
    )

    df.select(
        pl.col("candidate_id"),
        pl.col("CV").alias("candidate_text"),
    ).unique("candidate_id").write_json(write_location / "candidates.json")

    df.select(
        pl.col("job_id"),
        pl.col("Job Description").alias("job_text"),
    ).unique("job_id").write_json(write_location / "vacancies.json")

    df.select(
        pl.col("candidate_id"),
        pl.col("job_id"),
        pl.col("decision").map_elements(lambda x: 1 if x == "hire" else 0, return_dtype=pl.Int32),
    ).write_json(write_location / "edges.json")


def merge_jsons(jsons: list[Path], write_location: Path):
    write_location.mkdir(parents=True, exist_ok=True)
    pl.concat([pl.read_json(json / "candidates.json").select("candidate_id", "candidate_text") for json in jsons]).write_json(
        write_location / "candidates.json"
    )
    pl.concat([pl.read_json(json / "vacancies.json").select("job_id", "job_text") for json in jsons]).write_json(
        write_location / "vacancies.json"
    )
    pl.concat([pl.read_json(json / "edges.json").select("candidate_id", "job_id", "decision") for json in jsons]).write_json(
        write_location / "edges.json"
    )


def synthetic(
    model: str,
    write_location: Path,
):
    write_location.mkdir(parents=True, exist_ok=True)
    sentence_encoder = SentenceTransformer(model)

    candidate_df = pl.read_csv("hf://datasets/sankar12345/Resume-Dataset/Resume.csv").select(
        pl.col("Resume_str").alias("candidate_text"),
        pl.col("Resume_str").map_elements(lambda _: str(uuid4()), return_dtype=pl.String).alias("candidate_id"),
    )
    candidate_df.write_json(write_location / "candidates.json")

    vacancy_df = pl.read_csv("hf://datasets/nakamoto-yama/job-descriptions-public/selected_job_descriptions.csv").select(
        pl.col("Description").alias("job_text"),
        pl.col("Description").map_elements(lambda _: str(uuid4()), return_dtype=pl.String).alias("job_id"),
    )
    vacancy_df.write_json(write_location / "vacancies.json")

    candidate_features = {
        candidate_id: sentence_encoder.encode(cv_text)
        for candidate_id, cv_text in candidate_df.select(pl.col("candidate_id"), pl.col("candidate_text")).iter_rows()
    }

    vacancy_features = {
        vacancy_id: sentence_encoder.encode(vacancy_text)
        for vacancy_id, vacancy_text in vacancy_df.select(pl.col("job_id"), pl.col("job_text")).iter_rows()
    }

    data = []

    for candidate, vacancy in product(candidate_df["candidate_id"].to_list(), vacancy_df["job_id"].to_list()):
        cosine_sim = cosine(candidate_features[candidate], vacancy_features[vacancy])
        decision = cosine_sim > 0.95

        data.append({"candidate_id": candidate, "job_id": vacancy, "decision": int(decision)})

    edge_data = pl.DataFrame(data)
    edge_data.write_json(write_location / "edges.json")
