from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import sys
import json

# from pydantic import DirectoryPath
import argparse
import glob
import tqdm

col_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src")
sys.path.append(col_dir)

# from backends.hf.preprocessing import DocumentProcessor, ImageProcessor
# from backends.hf.layout_detector import LayoutDetector
# from logger import get_colette_logger

# dpi = 300
# chunk_num = 1
# chunk_overlap = 0
# index_overview = False
# logger = get_colette_logger("qwenlogger")
# sorted_documents = {}
# sorted_documents["pdf"] = [
#     "/path/to/doc.pdf"
# ]


parser = argparse.ArgumentParser()
parser.add_argument("--crops_path", help="path to crops")
args = parser.parse_args()


crops = glob.glob(args.crops_path + "/**/*png", recursive=True)
# docproc = DocumentProcessor(app_repository=".", logger=logger, dpi=dpi)


# layout_detector = None
# layout_detector = LayoutDetector(
#     model_path="https://colette.chat/models/layout/layout_detector_publaynet_merged_6000.pt",
#     resize_width=768,
#     resize_height=1024,
#     app_repository=DirectoryPath("."),
#     logger=logger,
#     device=0,
# )

# image_processor = ImageProcessor(
#     layout_detector,
#     chunk_num,
#     chunk_overlap,
#     index_overview,
#     DirectoryPath("."),
#     logger,
# )

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="cuda:0"
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
)

prompt1 = "Describe this image."
# prompt2 = 'You are an assistant specialized in Multimodal RAG tasks. The task is the following: given an image from a pdf page, you will have to generate questions that can be asked by a user to retrieve information from a large documentary corpus. The question should be relevant to the page, and should not be too specific or too general. The question should be about the subject of the page, and the answer need to be found in the page. Remember that the question is asked by a user to get some information from a large documentary corpus that contains multimodal data. Generate a question that could be asked by a user without knowing the existence and the content of the corpus. Generate as well the answer to the question, which should be found in the page. And the format of the answer should be a list of words answering the question. Generate at most THREE pairs of questions and answers per page in a dictionary with the following format, answer ONLY this dictionary NOTHING ELSE: { "questions": [ { "question": "XXXXXX", "answer": ["YYYYYY"] }, { "question": "XXXXXX", "answer": ["YYYYYY"] }, { "question": "XXXXXX", "answer": ["YYYYYY"] }, ] } where XXXXXX is the question and ["YYYYYY"] is the corresponding list of answers that could be as long as needed. Note: If there are no questions to ask about the page, return an empty list. Focus on making relevant questions concerning the page. Here is the page:'
prompt2 = 'You are an assistant specialized in Multimodal RAG tasks. The task is the followin: given an image from a pdf page, you will have to generate questions that can be asked by a user to retrieve information from a large documentary corpus. The questions should be relevant to the image. The questions should be about some of the image, and the answer need to be found in the image.  Generate as well the answers to the questions, which should be found in the image. Generate  THREE questions along with their answers in a dictionary with the following format, answer only the dictionary and nothing else:  { "questions": [ { "question": "XXXXXX", "answer": "YYYYYY" }, { "question": "XXXXXX", "answer": "YYYYYY" }, { "question": "XXXXXX", "answer": "YYYYYY" }, ] } where XXXXXX is the question and YYYYYY is the corresponding  answer. Do not use double quotes in XXXXXX neither in YYYYYY. Here is the image:'
# or : do not use any double quotes in XXXXXX and YYYYYY
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": prompt2},
        ],
    }
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)


# sorted_images = docproc.transform_documents_to_images(sorted_documents)
# docimg = image_processor.preprocess_image(imgpath)

images = []
generated_queries = {}

error_images = []
error_outs = []

for img in tqdm.tqdm(crops, desc="processing crops"):
    inputs = processor(
        text=[text_prompt],
        images=[Image.open(img)],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:0")

    output_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    out = output_text[0]
    out = out.replace("```json", "").replace("```", "").replace("\n", " ").strip()

    try:
        json_result = json.loads(out)
        print("img", img)
        print("correct out", out)
        imgid = len(images)
        # images.append(os.path.basename(img))
        images.append(img.replace(args.crops_path, ""))
        # print(f"\n\n\nimage  {img} \n\n output_text\n {json_result} ")
        queries = json_result["questions"]
        for query in queries:
            q = query["question"]
            a = query["answer"]
            try:
                generated_queries[q].append((imgid, a))
            except KeyError:
                generated_queries[q] = [(imgid, a)]
    except Exception:
        error_images.append(img)
        error_outs.append(out)
        print(f"\n\n\n Error on image  {img} \n\n output_text\n {out} ")


queries_list = []
for q in generated_queries.keys():
    queries_list.append(
        {
            "query": q,
            "docids": [
                generated_queries[q][i][0] for i in range(len(generated_queries[q]))
            ],
            "answers": [
                generated_queries[q][i][1] for i in range(len(generated_queries[q]))
            ],
        }
    )

dataset = {"images": images, "queries": queries_list}
with open("dataset.json", "w") as f:
    json.dump(dataset, f)
