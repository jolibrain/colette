import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from pydantic import BaseModel
from tqdm import tqdm

col_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src")
sys.path.append(col_dir)

from httpjsonapi import app # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Result(BaseModel):
    collection_name: str
    collection_id: int
    document_name: str
    document_id: int
    document_metadata: dict
    page_number: int
    raw_score: float
    normalized_score: float
    img_base64: str

class QueryResults(BaseModel):
    query: str
    results: list[Result]


PAYLOAD = {
  "app": {
    "repository": "",
    "verbose": "info",
    "models_repository": "models"
  },
  "parameters": {
    "input": {
      "lib": "hf",
      "data_output_type": "img",
      "preprocessing": {
        "files": [
          "all"
        ],
        "save_output": False,
        "filters": [
          "/~[^/]*$"
        ],
        "dpi": 300
      },
      "rag": {
        "indexdb_lib": "chromadb",
        "embedding_lib": "huggingface",
        "embedding_model": "MrLight/dse-qwen2-2b-mrl-v1",
        "gpu_id": 0,
        "top_k": 5,
        "reindex": True,
        "index_protection": False,
        "ragm": {
          "layout_detection": False,
          "image_width": 512,
          "image_height": 512
        }
      },
      "template": {},
      "data": [
      ]
    },
    "llm": {
      "source": "Qwen/Qwen2-VL-7B-Instruct",
      "gpu_ids": [
        0
      ],
      "image_width": 320,
      "image_height": 480,
      "inference": {
        "lib": "huggingface"
      }
    }
  }
}


COLLECTION_NAMES = [
    "arxivqa_test_subsampled",
    # "docvqa_test_subsampled",
    # "infovqa_test_subsampled",
    # "shiftproject_test",
    # "syntheticDocQA_artificial_intelligence_test",
    # "syntheticDocQA_energy_test",
    # "syntheticDocQA_government_reports_test",
    # "syntheticDocQA_healthcare_industry_test",
    # "tabfquad_test_subsampled",
    # "tatdqa_test",
]


def pretty_print_response(response: dict):
    # Create a copy of the response with truncated 'content' values
    truncated_response = response.copy()

    # Truncate each 'content' in the 'context' list to the first 25 characters
    if truncated_response.get("sources") and truncated_response["sources"].get(
        "context"
    ):
        for item in truncated_response["sources"]["context"]:
            if (
                "content" in item
                and isinstance(item["content"], str)
                and item["content"].startswith("data:image/")
            ):
                item["content"] = item["content"][:45]

    # Pretty print the modified response
    pprint(truncated_response)


def dcg(scores: list[float]) -> float:
    """
    Calculate the Discounted Cumulative Gain (DCG) for a list of relevance scores.

    Args:
        scores (List[float]): List of relevance scores.

    Returns:
        float: The DCG score.
    """
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(scores))


def ndcg_at_k(results: list[Any], true_doc_id: int, k: int = 5) -> float:
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) at rank k.

    Args:
        results (list[Any]): List of search results.
        true_doc_id (int): The ID of the true document.
        k (int, optional): The rank position to evaluate. Defaults to 5.

    Returns:
        float: The NDCG score at rank k.
    """
    relevance_scores = [
        result.raw_score
        if result.document_metadata["image_file_name"] == str(true_doc_id)
        else 0
        for result in results[:k]
    ]
    dcg_score = dcg(relevance_scores)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg_score = dcg(ideal_relevance) if ideal_relevance else 1
    return dcg_score / idcg_score if idcg_score else 0


def get_search_results(client:TestClient, query:str, collection_name:str, top_k:int):
    payload = {
        "app": {
            "verbose": "debug",
            "models_repository": "/path/to/colette/models/"
        },
        "parameters": {
            "input": {
                "message": query,
            },
            "rag": {
                "top_k": top_k
            },
        },
        "llm": {
            "source": "Qwen/Qwen2-VL-2B-Instruct",
            "gpu_ids": [
                0
            ],
            "image_width": 320,
            "image_height": 480,
            "inference": {
                "lib": "huggingface"
            }
        }
    }

    response = client.post(
        f"/v2/predict/{collection_name}",
        json=payload,
    )
    response.raise_for_status()
    # pretty_print_response(response.json())

    list_results = []
    for source in response.json()["sources"]["context"]:
        last_two_parts = '/'.join(source["source"].split("/")[-2:])

        list_results.append(Result(
            collection_name=collection_name,
            collection_id=0,
            document_metadata=dict(image_file_name=last_two_parts),
            document_name=source["source"],
            document_id=0,
            page_number=0,
            raw_score=source["distance"],
            normalized_score=source["distance"],
            img_base64=""
        ))

    qr = QueryResults(query=query, results=list_results)
    
    return qr


def evaluate_retriever(queries_df: pd.DataFrame, client: TestClient, collection_name:str, top_k:int) -> tuple[float, list[float], float]:
    ndcg_scores = []
    latencies = []
    for _, query_data in tqdm(
        queries_df.iterrows(), total=len(queries_df), desc="Evaluating"
    ):
        query_text = query_data["query"]
        true_doc_id = query_data["image_filename"]

        try:
            start = time.time()
            results = get_search_results(
                client, query_text, collection_name, top_k=top_k
            )
            end = time.time()
            latencies.append(end - start)
            ndcg_score = ndcg_at_k(results.results, true_doc_id, k=top_k)
        except Exception as e:
            print(f"Failed to retrieve results for query '{query_text}': {e}")
            ndcg_score = 0  # Assign a score of 0 if retrieval fails after retries

        ndcg_scores.append(ndcg_score)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    mean_ndcg_score = np.mean(ndcg_scores)

    return mean_ndcg_score, ndcg_scores, avg_latency


def main(data_dir: str, nrows: int = None, nqueries: int = None, top_k: int = 5):
    client = TestClient(app)

    avg_ndcg_scores_list = []
    ndcg_scores_dict = {}

    for collection in COLLECTION_NAMES:
        logger.info(f"Processing collection {collection}")
        # Load collection
        try:
            df = pd.read_pickle(f"{data_dir}/full/{collection}.pkl").reset_index().rename(columns={"index": "id"})
            df = df.sample(n=nrows) if nrows is not None else df
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {data_dir}/full/{collection}.pkl was not found.")
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")
                
        tmpdir = f"test_{collection}"
        os.makedirs(f"{tmpdir}", exist_ok=True)
        with open(f"{tmpdir}/{collection}.log", "w") as f:
            os.makedirs(f"{tmpdir}/data/images", exist_ok=True)
            for i in tqdm(range(0, len(df)), desc="Writing documents"):
                pil_image = df.iloc[i]["image"]
                # pil_image.save(f"{tmpdir}/data/{i:04d}.png")
                pil_image.save(f"{tmpdir}/data/{df.iloc[i]['image_filename']}")

            img_path = Path(f"{tmpdir}/data/images/")
            num_documents = sum(1 for f in img_path.rglob("**/*") if f.is_file())
            print(img_path)
            logger.info(f"{img_path} has {num_documents} documents")
            f.write(f"Collection {collection} has {num_documents} documents\n")
            print(f"Collection {collection} has {num_documents} documents")

            PAYLOAD["app"]["repository"] = tmpdir
            PAYLOAD["parameters"]["input"]["data"] = [f"{tmpdir}/data"]

            response = client.post(f"/v2/app/{collection}", json=PAYLOAD)
            response.raise_for_status()

            logger.info(f"Collection {collection} initialized with {len(df)} documents")

            # Testing the app
            # queries_df: pd.DataFrame = pd.read_pickle(f"tools/data/queries/{collection}_queries.pkl")

            queries_df: pd.DataFrame = pd.read_csv(f"{data_dir}/queries/{collection}_queries.csv")
            queries_df.dropna(subset=["query"], inplace=True)
            if nqueries is not None:
                queries_df = queries_df.head(nqueries).copy()  # Create a copy to avoid warnings

            avg_ndcg_score, ndcg_scores, avg_latency = evaluate_retriever(queries_df, client, collection, top_k=top_k)
            print(f"Average NDCG@{top_k} Score for {collection}: {avg_ndcg_score:.4f}")

            avg_ndcg_scores_list.append(
                {
                    "filename": collection,
                    "avg_ndcg_score": avg_ndcg_score,
                    "avg_latency": avg_latency,
                    "num_docs": num_documents,
                }
            )
            # Store results for ndcg_scores DataFrame
            ndcg_scores_dict[collection] = ndcg_scores

            print(f"Average NDCG@{top_k} Score for {collection}: {avg_ndcg_score:.4f}")

        shutil.rmtree(tmpdir, ignore_errors=True)
    os.makedirs("out", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # DataFrame for avg_ndcg_score
    avg_ndcg_df = pd.DataFrame(avg_ndcg_scores_list)
    avg_ndcg_df.to_pickle(f"out/avg_ndcg_scores_{timestamp}.pkl")

    # DataFrame for ndcg_scores with NaN padding for different lengths
    ndcg_scores_df = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in ndcg_scores_dict.items()])
    )
    ndcg_scores_df.to_pickle(f"out/ndcg_scores_{timestamp}.pkl")

    with open(f"out/payload_{timestamp}.json", "w") as f:
        json.dump(PAYLOAD, f, indent=4)

    print("Average NDCG scores saved to out/avg_ndcg_scores.pkl")
    print("Detailed NDCG scores saved to out/ndcg_scores.pkl")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark retriever")
    parser.add_argument("--data_dir", type=str, default="colivara_data", help="Data directory")
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to process")
    parser.add_argument("--nqueries", type=int, default=None, help="Number of queries to process")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")

    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        raise FileNotFoundError(f"Data directory {args.data_dir} not found")

    main(args.data_dir, args.nrows, args.nqueries, args.top_k)
