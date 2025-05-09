# Diffuser backend for Colette

import base64
import uuid
from io import BytesIO

from colette.apidata import APIData, APIResponse
from colette.inputconnector import (
    InputConnector,
)
from colette.llmlib import LLMLib
from colette.llmmodel import LLMModel
from colette.outputconnector import OutputConnector


class DiffusrLib(LLMLib):
    def __init__(self, inputc: InputConnector, outputc: OutputConnector, llmmodel: LLMModel):
        super().__init__(inputc, outputc, llmmodel)

    def init(self, ad: APIData, kvstore):
        self.images_repository = self.app_repository / "images"
        self.images_repository.mkdir(parents=True, exist_ok=True)

    def predict(self, ad: APIData) -> APIResponse:
        # get prompt
        prompt = ad.parameters.input.message
        self.logger.debug(f"prompt: {prompt}")

        # generate image
        image = self.llmmodel.generate(prompt)

        # Save image to a file
        image_name = prompt.replace(" ", "_")
        image_path = self.images_repository / f"{image_name}_{uuid.uuid4()}.png"
        image.save(image_path, format="PNG")
        self.logger.debug(f"Saved image_path: {image_path}")

        # Encode the image to Base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        image_bytes = buffered.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Construct the data URL
        data_url = f"data:image/png;base64,{image_base64}"

        response = APIResponse(message=prompt, output=data_url)
        return response
