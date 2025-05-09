import os
from urllib.parse import urlparse

import numpy as np
import requests
import torch


class LayoutDetector:
    def __init__(
        self, model_path, resize_width, resize_height, models_repository, logger, device
    ):
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.models_repository = models_repository
        self.logger = logger
        self.device = device
        self.model = self.load_model(model_path)
        self.label_map = {
            1: "text",
            2: "figure",
            3: "table",
        }

    def load_model(self, model_path):
        if urlparse(model_path).scheme in ("http", "https"):
            local_model_path = self.models_repository / os.path.basename(
                urlparse(model_path).path
            )
            self.download_model(model_path, str(local_model_path))
            model_path = local_model_path
        model = torch.jit.load(model_path).to(self.device)
        model.eval()
        return model

    def unload_model(self):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

    def download_model(self, url, save_path):
        if not os.path.exists(save_path):
            self.logger.info("Downloading layout model from %s to %s", url, save_path)
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
            else:
                raise Exception(
                    f"Failed to download the model from {url}. Status code: {response.status_code}",
                )
            self.logger.info("Downloaded layout model to %s", save_path)
        else:
            self.logger.info(f"Layout model already exists at {save_path}")

    def preprocess_image(self, original_image):
        rgb_image = original_image.convert("RGB")
        resized_image = rgb_image.resize((self.resize_width, self.resize_height))
        image_tensor = (
            torch.from_numpy(np.array(resized_image)).permute(2, 0, 1).float()
        )
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return original_image, resized_image, image_tensor

    def run_inference(self, image_tensor):
        with torch.no_grad():
            predictions = self.model(image_tensor)
        return predictions

    def postprocess(self, predictions, confidence_threshold=0.5):
        boxes, scores, labels = [], [], []
        for idx, score in enumerate(predictions[1][0]["scores"]):
            if score > confidence_threshold:
                boxes.append(predictions[1][0]["boxes"][idx].cpu().numpy())
                scores.append(score.item())
                labels.append(predictions[1][0]["labels"][idx].cpu().item())
        return boxes, scores, labels

    def crop_boxes(self, original_image, boxes, labels, scores):
        height_ratio = original_image.height / self.resize_height
        width_ratio = original_image.width / self.resize_width
        crops = []
        for _, (box, label, score) in enumerate(zip(boxes, labels, scores, strict=False)):
            x1, y1, x2, y2 = map(int, box)
            x1 = int(x1 * width_ratio)
            y1 = int(y1 * height_ratio)
            x2 = int(x2 * width_ratio)
            y2 = int(y2 * height_ratio)

            # skip bad boxes, if any
            if (
                x1 < 0
                or y1 < 0
                or x2 > original_image.width
                or y2 > original_image.height
            ):
                continue
            if (x2 - x1) * (y2 - y1) < 10:
                continue

            # crop
            crop = original_image.crop((x1, y1, x2, y2))

            # label to str
            label = self.label_map[label]
            crops.append([crop, label, score])
        return crops

    def detect(self, image_path, confidence_threshold=0.2):
        original_image, resized_image, image_tensor = self.preprocess_image(image_path)
        predictions = self.run_inference(image_tensor)
        boxes, scores, labels = self.postprocess(predictions, confidence_threshold)
        return self.crop_boxes(original_image, boxes, labels, scores)
