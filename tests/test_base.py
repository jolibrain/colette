import copy
import os
import sys

from fastapi.testclient import TestClient

from colette.httpjsonapi import app  # noqa

# messages

json_create = {
    "app": {"repository": "colette_test"},
    "parameters": {
        "input": {
            "lib": "langchain",
            "preprocessing": {"files": ["all"], "lib": "unstructured"},
            "rag": {
                "indexdb_lib": "chromadb",
                "embedding_lib": "gpt4all",
                "embedding_model": "all-MiniLM-L6-v2.gguf2.f16.gguf",
                "reindex": True,
                "index_protection": False,
            },
            "template": {
                "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Contexte: {context} Réponse: ",
                "template_prompt_variables": ["context", "question"],
            },
            "data": ["tests/data"],
        },
        "llm": {
            "source": "qwen2.5:0.5B-Instruct",
            # "source": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            # "filename": "qwen2.5-0.5b-instruct-q8_0.gguf",
            "inference": {
                "lib": "ollama"
                # "lib": "vllm",
                # "lib": "llamacpp"
            },
        },
    },
}

json_predict = {
    "app": {"repository": "colette_test/"},
    "parameters": {
        "input": {"message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"}
    },
}

json_predict_prompt = {
    "app": {"repository": "colette_test/"},
    "parameters": {
        "input": {
            "template": {
                "template_prompt": "Répond en une seule phrase. Question: {question} Contexte: {context} Réponse: "
            },
            "message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?",
        }
    },
}

# testing
client = TestClient(app)


def test_info():
    response = client.get("/v1/info")
    assert response.status_code == 200
    assert "commit" in response.json()["version"]


# def test_service_create_ollama():
#     response = client.put("/v1/app/test", json=json_create)
#     assert response.status_code == 200
#     assert response.json()["service_name"] == "test"


# def test_service_create_llamacpp():
#     json_create_llamacpp = copy.deepcopy(json_create)
#     json_create_llamacpp["parameters"]["llm"]["source"] = (
#         "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
#     )
#     json_create_llamacpp["parameters"]["llm"]["filename"] = (
#         "qwen2.5-0.5b-instruct-q8_0.gguf"
#     )
#     json_create_llamacpp["parameters"]["llm"]["inference"]["lib"] = "llamacpp"

#     response = client.put("/v1/app/test", json=json_create_llamacpp)
#     assert response.status_code == 200


# def test_service_delete():
#     response = client.delete("/v1/app/test")
#     print(response.json())


# def test_service_predict():
#     response = client.post("/v1/predict/test", json=json_predict)
#     print(response.json())


# def test_service_create_norag():
#     json_create_norag = copy.deepcopy(json_create)
#     del json_create_norag["parameters"]["input"]["rag"]
#     del json_create_norag["parameters"]["input"]["data"]
#     template = json_create_norag["parameters"]["input"]["template"]
#     template["template_prompt"] = "Question: {question}"
#     template["template_prompt_variables"] = ["question"]
#     response = client.put("/v1/app/test", json=json_create_norag)
#     assert response.status_code == 200
#     print(response.json())


# def test_service_predict_norag():
#     response = client.post("/v1/predict/test", json=json_predict)
#     assert response.status_code == 200
#     print(response.json())


# def test_service_session():
#     # create a service with history in the prompt
#     json_create_session = copy.deepcopy(json_create)
#     del json_create_session["parameters"]["input"]["rag"]
#     del json_create_session["parameters"]["input"]["data"]
#     json_create_session["parameters"]["llm"]["conversational"] = True
#     template = json_create_session["parameters"]["input"]["template"]
#     template["template_prompt"] = "Tu es un assistant de réponse à des questions."
#     template["template_prompt_variables"] = ["question"]
#     response = client.put("/v1/app/test", json=json_create_session)
#     assert response.status_code == 200
#     # simulate 2 sessions
#     sessions = {"a": "Alice", "b": "Bob"}
#     # both state something
#     for session, name in sessions.items():
#         json_predict_session = copy.deepcopy(json_predict)
#         json_predict_session["parameters"]["input"]["session_id"] = session
#         json_predict_session["parameters"]["input"]["message"] = f"Je suis {name}."
#         response = client.post("/v1/predict/test", json=json_predict_session)
#         assert response.status_code == 200
#     # both ask for statement
#     for session in sessions.keys():
#         json_predict_session = copy.deepcopy(json_predict)
#         json_predict_session["parameters"]["input"]["session_id"] = session
#         json_predict_session["parameters"]["input"]["message"] = "Quel est mon nom ?"
#         response = client.post("/v1/predict/test", json=json_predict_session)
#         assert response.status_code == 200
#         print(response.json())


# if __name__ == "__main__":
#    response = client.get("/v1/info")
#    assert "commit" in response.json()["version"]


class TestRagLangchainTxt:
    def test_ollama_gpt4all(self):
        ##############################################
        # build the service with ollama and no rag
        json_create_ollama_gpt4all_all = {
            "app": {"repository": "test_ollama_norag"},
            "parameters": {
                "input": {
                    "lib": "langchain",
                    "preprocessing": {
                        "files": ["all"],
                        "lib": "unstructured",
                        "save_output": True,
                        "strategy": "fast",
                    },
                    "template": {
                        "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Réponse: ",
                        "template_prompt_variables": ["question"],
                    },
                },
                "llm": {"source": "qwen2.5:0.5b", "inference": {"lib": "ollama"}},
            },
        }
        response = client.put(
            "/v1/app/test_ollama_norag",
            json=json_create_ollama_gpt4all_all,
        )
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_ollama_norag"

        response = client.get("/v1/info")
        assert "test_ollama_norag" in response.json()["info"]["services"]

        json_predict_norag = {
            "parameters": {"input": {"message": "Quel est la capitale de la France ?"}},
        }
        response = client.post("/v1/predict/test_ollama_norag", json=json_predict_norag)
        print("response=", response.json())
        assert "Paris" in response.json()["output"]

        # delete the service
        response = client.delete("/v1/app/test_ollama_norag")
        assert response.status_code == 200

    def test_ollama_gpt4all_rag(self):
        ##############################################
        # build the service with ollama
        json_create_ollama_gpt4all_all = {
            "app": {"repository": "test_ollama_gpt4all_all-MiniLM-L6-v2"},
            "parameters": {
                "input": {
                    "lib": "langchain",
                    "preprocessing": {
                        "files": ["all"],
                        "lib": "unstructured",
                        "save_output": True,
                        "strategy": "fast",
                    },
                    "rag": {
                        "indexdb_lib": "chromadb",
                        "embedding_lib": "gpt4all",
                        "embedding_model": "all-MiniLM-L6-v2.gguf2.f16.gguf",
                        "search": True,
                        "reindex": True,
                        "index_protection": False,
                        "top_k": 1,
                    },
                    "template": {
                        "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Contexte: {context} Réponse: ",
                        "template_prompt_variables": ["context", "question"],
                    },
                    "data": ["tests/data"],
                },
                "llm": {"source": "deepseek-r1:1.5b", "inference": {"lib": "ollama"}},
            },
        }
        response = client.put(
            "/v1/app/test_ollama_gpt4all_all-MiniLM-L6-v2",
            json=json_create_ollama_gpt4all_all,
        )
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_ollama_gpt4all_all-MiniLM-L6-v2"

        response = client.get("/v1/info")
        assert (
            "test_ollama_gpt4all_all-MiniLM-L6-v2"
            in response.json()["info"]["services"]
        )

        response = client.post(
            "/v1/predict/test_ollama_gpt4all_all-MiniLM-L6-v2", json=json_predict
        )
        print("response=", response.json())
        # assert "36500" in response.json()['full_response']['content']

        response = client.post(
            "/v1/predict/test_ollama_gpt4all_all-MiniLM-L6-v2", json=json_predict_prompt
        )
        print("response=", response.json())

        # delete the service
        response = client.delete("/v1/app/test_ollama_gpt4all_all-MiniLM-L6-v2")
        assert response.status_code == 200

    def test_llamacpp_gpt4all(self):
        ##############################################
        # build the service with llamacpp and gpt4all embeddings
        json_create_llamacpp_gpt4all_all = {
            "app": {"repository": "test_llamacpp_gpt4all_all-MiniLM-L6-v2"},
            "parameters": {
                "input": {
                    "lib": "langchain",
                    "preprocessing": {
                        "files": ["all"],
                        "lib": "unstructured",
                        "strategy": "fast",
                        "cleaning": False,
                    },
                    "rag": {
                        "indexdb_lib": "chromadb",
                        "embedding_lib": "gpt4all",
                        "embedding_model": "all-MiniLM-L6-v2.gguf2.f16.gguf",
                        "search": False,
                        "reindex": True,
                        "index_protection": False,
                    },
                    "template": {
                        "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Contexte: {context} Réponse: ",
                        "template_prompt_variables": ["context", "question"],
                    },
                    "data": ["tests/data"],
                },
                "llm": {
                    "source": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                    "filename": "qwen2.5-0.5b-instruct-q8_0.gguf",
                    "inference": {"lib": "llamacpp"},
                },
            },
        }

        response = client.put(
            "/v1/app/test_llamacpp_gpt4all_all-MiniLM-L6-v2",
            json=json_create_llamacpp_gpt4all_all,
        )
        assert response.status_code == 200
        assert (
            response.json()["service_name"] == "test_llamacpp_gpt4all_all-MiniLM-L6-v2"
        )

        # predict with the service
        json_predict = {
            "app": {"repository": "test_llamacpp_gpt4all_all-MiniLM-L6-v2"},
            "parameters": {
                "input": {
                    "message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"
                }
            },
        }
        response = client.post(
            "/v1/predict/test_llamacpp_gpt4all_all-MiniLM-L6-v2", json=json_predict
        )
        print("response status code=", response.status_code)
        print("response=", response.json())
        assert response.status_code == 200
        # assert "36,500" in response.json()["output"]

        resp_1 = response.json()["output"]

        # delete the service
        response = client.delete("/v1/app/test_llamacpp_gpt4all_all-MiniLM-L6-v2")
        assert response.status_code == 200

    def test_llamacpp_hf(self):
        ##############################################
        # build the service with llamacpp and huggingface embeddings
        json_create_llamacpp_hf_all = {
            "app": {"repository": "test_llamacpp_hf_all-MiniLM-L6-v2"},
            "parameters": {
                "input": {
                    "lib": "langchain",
                    "preprocessing": {
                        "files": ["all"],
                        "lib": "unstructured",
                        "strategy": "fast",
                    },
                    "rag": {
                        "indexdb_lib": "chromadb",
                        "embedding_lib": "huggingface",
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "search": False,
                        "reindex": True,
                        "index_protection": False,
                    },
                    "template": {
                        "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Contexte: {context} Réponse: ",
                        "template_prompt_variables": ["context", "question"],
                    },
                    "data": ["tests/data"],
                },
                "llm": {
                    "source": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
                    "filename": "Qwen2.5-0.5B-Instruct-Q8_0.gguf",
                    "inference": {"lib": "llamacpp"},
                },
            },
        }

        response = client.put(
            "/v1/app/test_llamacpp_hf_all-MiniLM-L6-v2",
            json=json_create_llamacpp_hf_all,
        )
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_llamacpp_hf_all-MiniLM-L6-v2"

        # predict with the service
        json_predict = {
            "app": {"repository": "test_llamacpp_hf_all-MiniLM-L6-v2"},
            "parameters": {
                "input": {
                    "message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"
                }
            },
        }
        response = client.post(
            "/v1/predict/test_llamacpp_hf_all-MiniLM-L6-v2", json=json_predict
        )
        assert response.status_code == 200
        print(response.json())
        # assert "36500" in response.json()["output"]
        resp_2 = response.json()["output"]

    def test_llamacpp_e5(self):
        ##############################################
        # build a new service with same embeddings but different lib i.e. huggingface
        json_create_llamacpp_e5 = {
            "app": {"repository": "test_llamacpp_hf_e5"},
            "parameters": {
                "input": {
                    "lib": "langchain",
                    "preprocessing": {
                        "files": ["all"],
                        "lib": "unstructured",
                        "strategy": "fast",
                    },
                    "rag": {
                        "indexdb_lib": "chromadb",
                        "embedding_lib": "huggingface",
                        "embedding_model": "intfloat/multilingual-e5-small",
                        "reindex": True,
                        "index_protection": False,
                    },
                    "template": {
                        "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Contexte: {context} Réponse: ",
                        "template_prompt_variables": ["context", "question"],
                    },
                    "data": ["tests/data"],
                },
                "llm": {
                    "source": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
                    "filename": "Qwen2.5-0.5B-Instruct-Q8_0.gguf",
                    "inference": {"lib": "llamacpp"},
                },
            },
        }
        response = client.put("/v1/app/test_llamacpp_e5", json=json_create_llamacpp_e5)
        assert response.status_code == 200

        # predict with the service
        json_predict = {
            "app": {"repository": "test_llamacpp_hf_e5"},
            "parameters": {
                "input": {
                    "message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"
                }
            },
        }
        response = client.post("/v1/predict/test_llamacpp_e5", json=json_predict)
        assert response.status_code == 200

        resp_3 = response.json()["output"]

        print(f"{resp_1}\n\n\n{resp_2}\n\n\n{resp_3}")

    def test_vllm(self):
        ##############################################
        # build a service with vllm
        json_create_vllm = {
            "app": {"repository": "test_vllm"},
            "parameters": {
                "input": {
                    "lib": "langchain",
                    "preprocessing": {
                        "files": ["all"],
                        "lib": "unstructured",
                        "strategy": "fast",
                    },
                    "rag": {
                        "indexdb_lib": "chromadb",
                        "embedding_lib": "huggingface",
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "search": False,
                        "reindex": True,
                        "index_protection": True,
                        "gpu_id": -1,
                    },
                    "template": {
                        "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Contexte: {context} Réponse: ",
                        "template_prompt_variables": ["context", "question"],
                    },
                    "data": ["tests/data"],
                },
                "llm": {
                    "source": "Qwen/Qwen2.5-0.5B",
                    "context_size": 2048,
                    "memory_utilization": 0.3,
                    "dtype": "float32",
                    "inference": {"lib": "vllm"},
                },
            },
        }

        # test index protection
        response = client.put("/v1/app/test_vllm", json=json_create_vllm)
        assert response.status_code == 400

        json_create_vllm["parameters"]["input"]["rag"]["index_protection"] = False
        response = client.put("/v1/app/test_vllm", json=json_create_vllm)
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_vllm"

        # predict with the service
        json_predict = {
            "app": {"repository": "test_vllm"},
            "parameters": {
                "input": {
                    "message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"
                }
            },
        }
        response = client.post("/v1/predict/test_vllm", json=json_predict)
        assert response.status_code == 200
        # assert "36500" in response.json()["output"]
        print(response.json()["output"])

        # delete the service
        response = client.delete("/v1/app/test_vllm")
        assert response.status_code == 200
