import numpy as np
import torch
from doctr.models import detection_predictor
from doctr.utils.geometry import detach_scores


class WordDetector:
    def __init__(self, min_font_size, logger, device):
        self.min_font_size = min_font_size
        self.logger = logger
        self.model = detection_predictor(arch="fast_base", pretrained=True).to("cuda:" + str(device))
        self.logger.info("Successfully loaded word detector model")

    def unload_model(self):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

    def detect_words(self, image):
        doc = [np.asarray(image)]
        result = self.model(doc)
        words = result[0]["words"]
        nwords = len(words)
        # print('words: ', words)
        self.logger.debug("Number of words detected: " + str(nwords))

        if nwords == 0:
            return []

        # Extract bounding boxes and scores from Doctr's result
        boxes = detach_scores([words])[0][0]
        # print('boxes: ', boxes)
        # print('len=', len(boxes))

        # for box in boxes:
        #    print(box)

        img_width, img_height = doc[0].shape[1], doc[0].shape[0]
        pixel_boxes = [
            (
                int(box[0] * img_width),
                int(box[1] * img_height),
                int(box[2] * img_width),
                int(box[3] * img_height),
            )
            for box in boxes
        ]
        return pixel_boxes

    def word_sizes(self, word_boxes):
        word_heights = [y2 - y1 for (_, y1, _, y2) in word_boxes] if word_boxes else []
        min_word_height = min(word_heights) if word_boxes else 0
        avg_word_height = sum(word_heights) / len(word_heights) if word_heights else 0
        max_word_height = max(word_heights) if word_heights else 0
        return min_word_height, avg_word_height, max_word_height

    def resize_img_for_font(self, image, min_word_height, avg_word_height, max_word_height):
        # resize image based on detected font size
        font_size_ratio = self.min_font_size / avg_word_height
        self.logger.debug(f"Detected font size_ratio {font_size_ratio}")
        # image_height, image_width = image.shape[:2]
        image_width, image_height = image.size
        # resize image with font size ratio if > 1
        if font_size_ratio > 1:
            # resize with PIL
            image = image.resize(
                (
                    int(image_width * font_size_ratio),
                    int(image_height * font_size_ratio),
                )
            )
            self.logger.debug(
                f"Upsized image from {(image_width, image_height)} to {image.size} based on font ratio"
            )
        elif font_size_ratio < 0.9:
            # resize with PIL
            image = image.resize(
                (
                    int(image_width * (font_size_ratio + 0.1)),
                    int(image_height * (font_size_ratio + 0.1)),
                )
            )
            self.logger.debug(
                f"Downsized image from {(image_width, image_height)} to {image.size} based on font ratio"
            )
        # else keep the image unchanged
        return image

    def detect_and_resize(self, image):
        word_boxes = self.detect_words(image)
        if word_boxes == []:
            return image
        min_word_height, avg_word_height, max_word_height = self.word_sizes(word_boxes)
        image = self.resize_img_for_font(image, min_word_height, avg_word_height, max_word_height)
        return image
