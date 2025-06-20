import asyncio
import json
import os
import subprocess
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, FastAPI, Form, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from .apidata import (
    APIData,
    APIResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChatMessage,
)
from .jsonapi import (
    JSONApi,
    colette_bad_request_400,
)

VERSION = "v1"
NAME_PREFIX = "Colette -"

git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode("ascii").strip()
print(f"Colette {VERSION} server - commit={git_hash}", flush=True)
version_str = f"*commit:* [{git_hash}]\n\n"

description = version_str + f"This is the Colette {VERSION} API server"

TAGS = [
    {"name": "info", "description": "Service and version information"},
    {"name": "app", "description": "Service lifecycle (create, delete)"},
    {"name": "predict", "description": "Model prediction"},
    {"name": "streaming", "description": "Streaming model prediction"},
    {"name": "index", "description": "Indexing documents"},
]


def set_response_http_status(response: Response, output: APIResponse):
    if "status" in output.model_dump() and output.status is not None:
        response.status_code = output.status.code


class HTTPJsonApi(JSONApi):
    def __init__(self, version: str = VERSION) -> None:
        super().__init__()
        self.router = APIRouter(prefix=f"/{version}", tags=[f"{version}"])
        self.openwebui_router = APIRouter(tags=[f"{version}"])

        self.router.add_api_route(
            "/info",
            self.info,
            methods=["GET"],
            description="",
            response_model=APIResponse,
        )

        self.router.add_api_route(
            "/app/{sname}",
            self.create_service,
            methods=["PUT", "POST"],
            description="",
            response_model=APIResponse,
        )

        self.router.add_api_route(
            "/app/{sname}",
            self.delete_service,
            methods=["DELETE"],
            description="",
            response_model=APIResponse,
        )

        self.router.add_api_route(
            "/predict/{sname}",
            self.predict_service,
            methods=["POST"],
            description="",
            response_model=APIResponse,
        )

        self.router.add_api_route(
            "/streaming/{sname}",
            self.streaming_service,
            methods=["POST"],
            description="",
        )

        self.router.add_api_route(
            "/index/{sname}",
            self.index_service,
            methods=["PUT"],
            description="",
            response_model=APIResponse,
        )

        self.router.add_api_route(
            "/upload/{sname}",
            self.upload_service,
            methods=["PUT"],
            description="",
            response_model=APIResponse,
        )

        self.router.add_api_route(
            "/index/{sname}/status",
            self.index_status,
            methods=["GET"],
            response_model=APIResponse,
        )

        self.openwebui_router.add_api_route(
            "/models",
            self.get_models,
            methods=["GET"],
            description="Get all available models",
        )

        self.openwebui_router.add_api_route(
            "/chat/completions",
            self.chat_completion,
            methods=["POST"],
            description="Chat completions endpoint",
        )

    def info(self) -> APIResponse:
        response = self.service_info()
        response.version = version_str
        return response
    
    async def get_models(self) :
        """
        Get all available models
        """
        
        services = self.list_services()
        return {
            "data": [{"id": NAME_PREFIX + service, "object": "model", "created": 0, "owned_by": "colette"} for service in services],
            "object": "list",
        }
    
    async def chat_completion(self, request : ChatCompletionRequest, response: Response) :
        service = request.model[len(NAME_PREFIX):]
        self.logger_api.debug(f"Service name : {service}")
        query = request.messages[-1].content
        with open('/home/ad/bodint/data/colette/src/colette/config/message_example.json', 'r') as file:
            message_template = json.load(file)
        message_template["parameters"]["input"]["message"] = query
        api_data_query = APIData(**message_template)
        response_service = await self.predict_service(service, api_data_query, response)
        
        sources = ""
        context_items = response_service.sources["context"]
        for item in sorted(context_items, key=lambda x: x.get("distance", 0), reverse=True):
            d = item.get("distance", 0)
            path = item.get("source", "Unknown path")
            path = path.replace("/home/ad/bodint/data", "")
            text = f"Similarity {d*100:.2f} %   (`{path}`)"
            image_url = item.get("content", "")
            link = f"[See source]({image_url})"
            sources += f"- {text} \n {link}\n"
        
        answer = f"**Answer** : {response_service.output}\n #### Sources -> \n{sources}"

        return ChatCompletionResponse(
            id = "1337",
            object = "chat.completion",
            created = int(time.time()),
            model = request.model,
            choices = [Choice(message=ChatMessage(role="assistant", content=answer))]
        )

    async def delete_service(self, sname: str) -> APIResponse:
        response = await self.service_delete(sname)
        return response

    async def predict_service(self, sname: str, ad: APIData, response: Response) -> APIResponse:
        output = self.service_predict(sname, ad)
        set_response_http_status(response, output)
        return output

    def streaming_service(self, sname: str, ad: APIData):
        streamer = self.service_streaming(sname, ad)
        return StreamingResponse(streamer)

    async def index_service(self, sname: str, ad: APIData, response: Response) -> APIResponse:
        output = await self.service_index_async(sname, ad)
        set_response_http_status(response, output)
        return output

    async def upload_service(
        self, sname: str, ad: str = Form(...), files: list[UploadFile] | None = None, response: Response = None
    ) -> APIResponse:
        try:
            ad_data = json.loads(ad)
            api_data = APIData(**ad_data)
        except json.JSONDecodeError as e:
            output = colette_bad_request_400(str(e))
            set_response_http_status(response, output)
            return output
        except ValidationError as ve:
            output = colette_bad_request_400(str(ve))
            set_response_http_status(response, output)
            return output

        output = await self.service_upload(sname, api_data, files)
        set_response_http_status(response, output)
        return output

    async def create_service(self, sname: str, ad: APIData, response: Response = None) -> APIResponse:
        """
        Create a new service
        """
        # try:
        #     ad_data = json.loads(ad)
        #     api_data = APIData(**ad_data)
        # except json.JSONDecodeError:
        #     output = APIResponse(status=400, error="Invalid JSON")
        #     set_response_http_status(response, output)
        #     return output
        # except ValidationError as ve:
        #     output = APIResponse(status=400, error=f"Invalid APIData: {ve}")
        #     set_response_http_status(response, output)
        #     return output

        output = self.service_create(sname, ad)
        set_response_http_status(response, output)
        return output

    async def index_status(self, sname: str) -> APIResponse:
        return await self.service_index_status(sname)


http_json_api = HTTPJsonApi()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: start the indexing loop
    http_json_api.start_indexing_loop()
    yield
    # Shutdown: you could cancel the indexing task if desired:
    http_json_api._indexing_task.cancel()
    try:
        await http_json_api._indexing_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Colette {VERSION} server", description=description, version=VERSION, openapi_tags=TAGS, lifespan=lifespan
)
app.include_router(http_json_api.router)
app.include_router(http_json_api.openwebui_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1873)
