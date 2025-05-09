import base64
from io import BytesIO

from colette.apidata import OutputConnectorObj
from colette.outputconnector import OutputConnector


def transform_pil_image_to_base64(image):
    #
    # Transform PIL image to base64
    # @param image: PIL image
    # @return: str
    #
    buffered = BytesIO()
    # extract image format from image
    image_type = image.format
    image.save(buffered, format=image_type)
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{image_type.lower()};base64,{encoded_string}"


class HFOutputConn(OutputConnector):
    def __init__(self):
        super().__init__()

    def init(self, ad: OutputConnectorObj):
        super().init(ad)
        self.base64 = ad.base64
        self.num_tokens = ad.num_tokens

    def add_base64_to_source(self, source, img):
        if self.base64:
            source["content"] = transform_pil_image_to_base64(img)
        return source
