import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import polars as pd
from bert_score import score as bert_scorer
from dotenv import load_dotenv
from httpx import Client
from pydantic import BaseModel
from rouge_score import rouge_scorer
from tabulate import tabulate
from tqdm import tqdm

col_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src")
sys.path.append(col_dir)

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

class Reference(BaseModel):
    file: str
    pages: Optional[list[str]] = None


class QAPair(BaseModel):
    id: int
    question: str
    answer: str
    short_answer: Optional[list[str]] = None
    references: Optional[list[Reference]] = None
    notes: Optional[str] = None
    lang: str = "en"


# Load .env if it exists
load_dotenv()

def get_prediction(
    client: Any, app_name: str, prompt: str, query: str, local: bool
):
    """
    Bot response based on the last message in the chat history.
    """
    try:
        payload = {
            "app": {"verbose": "debug"},
            "parameters": {
                "input": {
                    "message": query,
                },
                "template": {
                    "template_prompt": prompt,
                    "template_prompt_variables": ["context", "question"],
                },
            },
        }

        response = client.post("/v2/predict/" + app_name, json=payload, timeout=120.0)
        bot_response = response.json()
        assert response.status_code == 200, f"Error: {response.text}"
        
        sources = bot_response.get("sources", [])

        retrieved_documents = []
        if sources["context"] is not None:
            for source in sources["context"]:
                pattern = re.compile(
                    r'^(?P<uuid>[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})_'
                    r'(?P<page>\d+)'
                    r'(?:_(?P<kind>[a-z]+)(?:_(?P<number>\d+))?)?$'
                )

                if "key" in source:
                    # new format
                    uuid = os.path.splitext(os.path.basename(source["source"]))[0]

                    match = pattern.match(source["key"])
                    if match:
                        page = match.group('page')
                        kind = match.group('kind') or 'image'
                        number = match.group('number') or -1
                    else:
                        content = "no content found"
                        logger.debug(f"No match found for: {s}")

                else:
                    # old format
                    s = source["source"]

                    # Extract folder_name, page, and crop from the key
                    match = pattern.match(s)
                    if match is not None:
                        uuid = match.group('uuid') or 'no_uuid'
                        page = match.group('page') or -1
                        kind = match.group('kind') or 'txt'
                        number = match.group('number') or -1
                    else:
                        logger.debug(f"No match found for: {s}")
                        print(f"No match found for: {s}")
                        uuid = 'no_uuid'
                        page = -1
                        kind = 'txt'

                retrieved_documents.append(
                    {
                        "name": uuid,
                        "page": int(page),
                        "kind": kind,
                    }
                )

        return bot_response["output"], retrieved_documents, []
    except Exception as e:
        return "Error!!", [], [f"{str(e)}"]

def truncate_text(text: str, max_length: int = 15) -> str:
    """
    Truncate the text to the specified max_length. Append '...' if truncated.
    """
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def compute_metrics(
    expected_answer: str,
    generated_answer: str,
    short_answer: Optional[list[str]],
    scorer,
    lang="en",
):
    """
    Compute ROUGE scores and precision given expected and generated answers.

    Args:
        expected_answer (str): The expected answer text.
        generated_answer (str): The generated answer text.
        short_answer (Optional[list[str]]): list of acceptable short answers.
        scorer: The ROUGE scorer object.

    Returns:
        dict: A dictionary containing ROUGE scores and precision.
    """
    # Compute ROUGE scores
    rouge_scores = scorer.score(expected_answer, generated_answer)
    _, _, bert_score = bert_scorer([generated_answer], [expected_answer], lang=lang)
    scores = {

        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rouge2": rouge_scores["rouge2"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
        "BERTScore": bert_score.mean().item(),
    }

    # Compute precision based on short_answer
    if short_answer:
        precision = any(
            acceptable_answer.lower() in generated_answer.lower()
            for acceptable_answer in short_answer
        )
        precision = 1.0 if precision else 0.0
    else:
        precision = "N/A"

    return {**scores, "precision": precision}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate answers for ADS POC")
    parser.add_argument("--app-dir", type=str, help="Application directory")
    parser.add_argument("--host", type=str, help="Host address of the server")
    parser.add_argument("--port", type=int, help="Port number of the server")
    parser.add_argument("--app-name", type=str, help="Application name")
    parser.add_argument("--qa", type=str, required=True, help="Path to the QA pairs JSON file")
    parser.add_argument("--evaluate-chunks", action="store_true", help="Evaluate chunks instead of full answers")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose mode (print metrics for each chunk)")
    parser.add_argument("--topk", type=int, help="Number of topk results to return")
    parser.add_argument("--vllm", type=str, help="VLLM model name")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    # Check if the arguments are valid
    if args.app_dir and (args.host or args.port or args.app_name):
        parser.error("Cannot specify both app_dir and host/port/app_name")

    if any([args.host, args.port, args.app_name]) and not all([args.host, args.port, args.app_name]):
            parser.error("The arguments --host, --port and --app_name must be provided together")

    return args

def run_evaluation(args):
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    client = None

    if args.app_dir:
        local = True

        from fastapi.testclient import TestClient

        from httpjsonapi import app

        client = TestClient(app)

        # Load the configuration file
        with open(args.app_dir + "/config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        response = client.get("/v2/info")
        response.raise_for_status()

        if "lib" in  config["parameters"]["input"]["preprocessing"]:
            del config["parameters"]["input"]["preprocessing"]["lib"]

        if "embedding_model_path" in config["parameters"]["input"]["rag"]:
            del config["parameters"]["input"]["rag"]["embedding_model_path"]

        if "llm" not in config["parameters"] or config["parameters"]["llm"] is None:
            logger.info("LLM parameters is not set")
            if args.txt:
                logger.info("RAG mode TXT")
                # DeepSeek-R1-Distill-Llama-8B
                config["parameters"]["llm"] = {
                    "source": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                    "context_size": 8192,
                    "inference": {
                        "lib": "vllm"
                    }
                }
            else:
                logger.info("RAG mode IMG")
                config["parameters"]["llm"] = {
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

        if config["parameters"]["input"]["rag"]["reindex"]:
            logger.info("Setting reindex to False")
            config["parameters"]["input"]["rag"]["reindex"] = False
            config["parameters"]["input"]["rag"]["index_protection"] = True

        app_name = Path(args.app_dir).stem

        # create service
        if app_name not in response.json()["info"]["services"]:
            logger.debug(json.dumps(config, indent=2))
            response = client.put("/v2/app/" + app_name, json=config)
            if response.status_code == 422:
                logger.info("Check your `models_repository` setting")
            assert response.status_code == 200

        logger.info(f"Using app {app_name} with FastAPI client")
    else:
        print("Using HTTPX client")
        local = False

        app_name = args.app_name

        if args.vllm is not None:
            raise ValueError("VLLM model is not supported with HTTPX client")

        # TODO: set proxies and/or headers if needed (CORS?)
        client = Client(base_url=f"http://{args.host}:{args.port}")

        response = client.get("/v2/info")
        response.raise_for_status()

        if app_name not in response.json()["info"]["services"]:
            raise ValueError(f"App {app_name} not found")

        logger.info(f"Using app {app_name} at http://{args.host}:{args.port} with HTTPX client")


    # Load and parse QA pairs using Pydantic
    with open(args.qa, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    qa_pairs = [QAPair(**item) for item in qa_data]

    lang_dict = {"en": 0, "fr": 0}

    data_columns = {
        "ID": pd.Int64,
        "Question": pd.Utf8,
        "Answer": pd.Utf8,
        "ROUGE-1": pd.Float64,
        "ROUGE-2": pd.Float64,
        "ROUGE-L": pd.Float64,
        "BERTScore": pd.Float64,
        "Num Chunks Retrieved": pd.Int64,
        "Unique Documents": pd.List(pd.Utf8),
        "Page": pd.List(pd.Int64)
    }

    results_df = pd.DataFrame(schema=data_columns)

    retriever_columns = {
        "ID": pd.Int64,
        "Precision": pd.Float64,
        "Recall": pd.Float64,
        "F1 Score": pd.Float64,
    }
    retriever_df = pd.DataFrame(schema=retriever_columns)

    logger.info("Evaluating answers...")

    # Process each QA pair
    for qa_pair in tqdm(qa_pairs, desc="Processing QA pairs", unit="pair"):
        qa_id = qa_pair.id
        question = qa_pair.question
        expected_answer = qa_pair.answer

        # Prepare the prompt #TODO: Add support for French/English in the config itself
        if qa_pair.lang == "en":
            prompt = ""
        elif qa_pair.lang == "fr":
            prompt = ""
        else:
            raise ValueError(f"Unsupported language: {qa_pair.lang}")
        lang_dict[qa_pair.lang] += 1

        # Get the generated answer from your model
        generated_answer, retrieved_documents, errors = get_prediction(
            client, app_name, prompt, question, local
        )

        if errors:
            raise ValueError(f"Error for QA pair {qa_id}: {errors}")

        metrics = compute_metrics(
            expected_answer,
            generated_answer,
            qa_pair.short_answer,
            scorer,
            lang=qa_pair.lang,
        )

        # Extract additional information for each answer
        rouge1, rouge2, rougeL, bertscore = (
            metrics["rouge1"],
            metrics["rouge2"],
            metrics["rougeL"],
            metrics["BERTScore"],
        )

        num_chunks_retrieved = len(retrieved_documents)
        logger.debug(f"Retrieved {retrieved_documents} documents")

        # Add data to the DataFrame
        row_data = {
            "ID": qa_id,
            "Question": question,
            "Answer": generated_answer,
            "ROUGE-1": float(rouge1),
            "ROUGE-2": float(rouge2),
            "ROUGE-L": float(rougeL),
            "BERTScore": float(bertscore),
            "Num Chunks Retrieved": int(num_chunks_retrieved),
            "Unique Documents": list(set([doc["name"] for doc in retrieved_documents])),
            "Page": [doc["page"] for doc in retrieved_documents]
        }
        new_row_df = pd.DataFrame([row_data])

        results_df = results_df.vstack(new_row_df)

        # change retrieved documents to a set
        retrieved_documents_set = set([doc["name"] for doc in retrieved_documents])

        # Compute retriever metrics
        if qa_pair.references and retrieved_documents:
            expected_documents = {
                reference.file for reference in qa_pair.references or []
            }

            logger.debug(f"Expected documents: {expected_documents}")
            logger.debug(f"Retrieved documents: {retrieved_documents_set}")

            # Calculate Precision, Recall, and F1
            tp = len(retrieved_documents_set.intersection(expected_documents))
            fp = len(retrieved_documents_set.difference(expected_documents))
            fn = len(expected_documents.difference(retrieved_documents_set))

            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

            # Add data to the DataFrame
            row_data = {
                "ID": qa_id,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            }
            retriever_df = retriever_df.vstack(pd.DataFrame(row_data))

    return config, results_df, retriever_df, lang_dict

if __name__ == "__main__":
    args = parse_arguments()
    config, results_df, retriever_df, lang_dict = run_evaluation(args)

    date_time_obj = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    # Store the results in csv files
    Path(f"{date_time_obj}").mkdir(exist_ok=True)

    with open(f"{date_time_obj}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    app_name = Path(args.app_dir).stem

    with open(f"{date_time_obj}/{app_name}_llm.json", "w") as f:
        json.dump(results_df.to_dicts(), f, indent=4)

    with open(f"{date_time_obj}/{app_name}_retriever.csv", "w") as f:
        f.write(retriever_df.write_csv())

    # Store the results
    with open(f"{date_time_obj}/{app_name}_results.txt", "w") as f:
        # Write language dictionary as a table
        f.write(
            tabulate(
                [list(lang_dict.values())],
                headers=list(lang_dict.keys()),
                tablefmt="pretty",
            )
        )
        f.write("\n\n")

        # Write average values for results_df
        results_df = results_df.drop(["ID", "Question", "Answer", "Unique Documents"])
        results = results_df.mean().to_numpy().flatten().tolist()
        f.write(tabulate([results], headers=results_df.columns, tablefmt="pretty"))
        f.write("\n\n")

        # Write average values for retriever_df
        retriever_df = retriever_df.drop(["ID"])
        retriever = retriever_df.mean().to_numpy().flatten().tolist()
        f.write(tabulate([retriever], headers=retriever_df.columns, tablefmt="pretty"))
        f.write("\n\n")