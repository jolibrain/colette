import json
import time
import uvicorn
import subprocess
import os
import asyncio

from contextlib import asynccontextmanager

from .httpjsonapi import HTTPJsonApi, VERSION, NAME_PREFIX, TAGS
from fastapi import APIRouter, FastAPI, Response

from .apidata import(
    APIData,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChatMessage,
)

git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode("ascii").strip()
version_str = f"*commit:* [{git_hash}]\n\n"
description = version_str + f"This is the Colette {VERSION} API server"


class OpenWebUIApi(HTTPJsonApi) :
    def __init__(self, version: str = VERSION):
        super().__init__(version)
        self.openwebui_router = APIRouter(tags=[f"{version}"])

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
        """
        Chat completions endpoint
        """

        service = request.model[len(NAME_PREFIX):]
        self.logger_api.debug(f"Service name : {service}")
        query = request.messages[-1].content
        with open('./src/colette/config/message_example.json', 'r') as file:
            message_template = json.load(file)
        message_template["parameters"]["input"]["message"] = query
        api_data_query = APIData(**message_template)
        response_service = await self.predict_service(service, api_data_query, response)
        
        sources = ""
        context_items = response_service.sources["context"]
        for item in sorted(context_items, key=lambda x: x.get("distance", 0), reverse=True):
            d = item.get("distance", 0)
            path = item.get("source", "Unknown path")
            # path = path.replace("/home/xxx/xxx", "")  # To remove the start of the full path
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
    
openwebuiapi = OpenWebUIApi()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: start the indexing loop
    openwebuiapi.start_indexing_loop()
    yield
    # Shutdown: you could cancel the indexing task if desired:
    openwebuiapi._indexing_task.cancel()
    try:
        await openwebuiapi._indexing_task
    except asyncio.CancelledError:
        pass
    
app = FastAPI(
    title="Colette {VERSION} server", description=description, version=VERSION, openapi_tags=TAGS, lifespan=lifespan
)
app.include_router(openwebuiapi.router)
app.include_router(openwebuiapi.openwebui_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1873)