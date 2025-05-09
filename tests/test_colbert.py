import os
import shutil
import time

import pytest
from pydantic import DirectoryPath

from colette.apidata import (
    APIData,
    AppObj,
    InputConnectorObj,
    LLMInferenceObj,
    LLMModelObj,
    ParametersObj,
    PreprocessingObj,
    RAGObj,
    TemplatePromptObj,
)
from colette.jsonapi import JSONApi


@pytest.mark.asyncio
async def test_colbert():
    api = JSONApi()

    ad = APIData()
    ad_index = APIData()
    ad.app = AppObj()
    ad_index.app = AppObj()
    ad.app.repository = DirectoryPath("colette_colbert")
    ad.parameters = ParametersObj()
    ad_index.parameters = ParametersObj()
    ad.parameters.input = InputConnectorObj()
    ad_index.parameters.input = InputConnectorObj()
    ad.parameters.input.lib = "langchain"
    ad_index.parameters.input.preprocessing = PreprocessingObj()
    ad_index.parameters.input.preprocessing.files = ["all"]
    ad_index.parameters.input.preprocessing.lib = "unstructured"
    ad_index.parameters.input.preprocessing.save_output = True
    ad_index.parameters.input.preprocessing.strategy = "fast"
    ad.parameters.input.rag = RAGObj()
    ad_index.parameters.input.rag = RAGObj()
    ad.parameters.input.rag.indexdb_lib = "coldb"
    ad.parameters.input.rag.embedding_lib = "colbert"
    ad.parameters.input.rag.embedding_model = "colbertv2.0"
    ad_index.parameters.input.rag.gpu_id = 0
    ad.parameters.input.rag.num_partitions = 2
    ad.parameters.input.template = TemplatePromptObj()
    ad.parameters.input.template.template_prompt = (
        "Tu es un assistant de réponse à des questions. Question: {question} Contexte: {context} Réponse: "
    )
    ad.parameters.input.template.template_prompt_variables = ["context", "question"]
    ad_index.parameters.input.data = [DirectoryPath("tests/data")]
    ad.parameters.llm = LLMModelObj()
    ad.parameters.llm.source = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
    ad.parameters.llm.filename = "Qwen2.5-0.5B-Instruct-Q8_0.gguf"
    ad.parameters.llm.inference = LLMInferenceObj()
    ad.parameters.llm.inference.lib = "llamacpp"
    ad.parameters.llm.context_size = 3000

    try:
        api.service_create("test_colbert", ad)

        response = await api.service_index("test_colbert", ad_index)
        while "finished" not in response.message:
            time.sleep(0.5)
            response = await api.service_index_status("test_colbert")

        assert os.path.exists("colette_colbert/models/colbertv2.0")
        assert os.path.exists("colette_colbert/index/mydb/plan.json")
        assert os.path.exists("colette_colbert/index/mydb/centroids.pt")

        ad.parameters.input.message = "Quel est le nombre d'objets spatiaux de plus de 10cm ?"

        response = await api.service_predict("test_colbert", ad)
        print(f"\npredict response:\n{response.output}\n")

        await api.service_delete("test_colbert")

        # reread index test
        await api.service_create("test_colbert", ad)
        response2 = await api.service_predict("test_colbert", ad)
        print(f"\npredict response:\n{response2.output}\n")
        assert response2.output != ""

        # BM25 test
        ad.parameters.input.rag.search = True
        await api.service_create("test_colbert_bm25", ad)
        assert os.path.exists("colette_colbert/models/colbertv2.0")
        assert os.path.exists("colette_colbert/index/mydb/plan.json")
        assert os.path.exists("colette_colbert/index/mydb/centroids.pt")
        response = await api.service_predict("test_colbert_bm25", ad)
        print(f"\npredict response:\n{response.output}\n")
    finally:
        # cleanup
        shutil.rmtree("colette_colbert")
