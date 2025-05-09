# Example usage:
## python3 calibrate_vllm_ocr.py --image /path/to/STANDARD-V1/IADC/IADC-010101/3c644e69-3937-4c59-928d-6711eb6ad27b-08_crop_1.jpg --ground_truth 3c644e69-3937-4c59-928d-6711eb6ad27b-08_crop_1.txt

import torch
import cv2
import argparse
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import detection_predictor
from doctr.utils.geometry import detach_scores
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from difflib import SequenceMatcher

# Argument parser
parser = argparse.ArgumentParser(description="OCR Calibration Script")
parser.add_argument("--image", type=str, required=True, help="Path to the input image")
parser.add_argument(
    "--ground_truth",
    type=str,
    required=True,
    help="Path to the text file containing ground truth text",
)
args = parser.parse_args()

# Load models
ocr_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
ocr_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)
word_detection_model = detection_predictor(pretrained=True)


# Image preprocessing function
def preprocess_image(image, resolution):
    return cv2.resize(image, (resolution, resolution))


# Word detection function using Doctr
def detect_words(model, image_path):
    doc = DocumentFile.from_images(image_path)
    result = model(doc)
    words = result[0]["words"]

    # Extract bounding boxes and scores from Doctr's result
    boxes = detach_scores([words])[0][0]

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


# OCR function using Qwen2-VL-2B
def recognize_text(processor, model, image):
    # print(type(image))
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {
                    "type": "text",
                    "text": "Please read the text content from this document accurately.",
                },
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


# Function to calculate similarity between recognized text and ground truth
def calculate_similarity(recognized_text, ground_truth):
    return SequenceMatcher(None, recognized_text, ground_truth).ratio()


# Load ground truth text
ground_truth = ""
with open(args.ground_truth, "r", encoding="utf-8") as f:
    ground_truth = f.read().strip()

# Iterate over different resolutions
resolutions = [256, 320, 384, 448, 512, 640]
image_path = args.image

results = []

for res in resolutions:
    # Preprocess the image to the desired resolution
    resized_image_path = f"resized_{res}.jpg"
    image = cv2.imread(image_path)
    resized_image = preprocess_image(image, res)
    cv2.imwrite(resized_image_path, resized_image)

    # Detect words in the resized image
    word_boxes = detect_words(word_detection_model, resized_image_path)

    # Get minimum word height from the bounding boxes
    word_heights = [y2 - y1 for (_, y1, _, y2) in word_boxes] if word_boxes else []
    min_word_height = min(word_heights) if word_boxes else 0
    avg_word_height = sum(word_heights) / len(word_heights) if word_heights else 0
    max_word_height = max(word_heights) if word_heights else 0

    # Recognize text using the OCR model
    recognized_text = recognize_text(ocr_processor, ocr_model, resized_image)

    # Calculate similarity with ground truth
    similarity = calculate_similarity(recognized_text, ground_truth)

    # Store the results
    results.append(
        {
            "resolution": res,
            "recognized_text": recognized_text,
            "min_word_height": min_word_height,
            "avg_word_height": avg_word_height,
            "max_word_height": max_word_height,
            "similarity": similarity,
        }
    )

# Find the resolution with the highest similarity
best_result = max(results, key=lambda x: x["similarity"])

# Calculate the resolution to min word height ratio for the best result
ratio = (
    best_result["resolution"] / best_result["avg_word_height"]
    if best_result["avg_word_height"] > 0
    else float("inf")
)

# Print the results for analysis
for result in results:
    print(
        f"Resolution: {result['resolution']}, Min Word Height: {result['min_word_height']}, Avg Word Height: {result['avg_word_height']}, Max Word Height: {result['max_word_height']}, Similarity: {result['similarity']:.4f}, OCR Output: {result['recognized_text']}"
    )

print("\nBest Resolution:")
print(
    f"Resolution: {best_result['resolution']}, Similarity: {best_result['similarity']:.4f}, OCR Output: {best_result['recognized_text']}"
)
print(f"Resolution to Min Word Height Ratio: {ratio:.4f}")
