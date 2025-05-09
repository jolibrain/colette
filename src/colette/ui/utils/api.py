import json
import os
import re
import time
from copy import deepcopy
from pprint import pprint

import requests

from colette.ui.utils.config import IMAGE_404, Config
from colette.ui.utils.logger import logger

# def html_src(harm_level):
#     return f"""
# <div style="display: flex; gap: 5px;">
#   <div style="background-color: {color_map[harm_level]}; padding: 2px; border-radius: 5px;">
#   {harm_level}
#   </div>
# </div>
# """

def pretty_print_response(response: dict):
    truncated_response = deepcopy(response)
    nodes = [truncated_response]

    while nodes:
        current = nodes.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if isinstance(value, str) and value.startswith("data:image/"):
                    current[key] = "data:image/..."
                elif isinstance(value, dict | list):
                    nodes.append(value)
        elif isinstance(current, list):
            for i in range(len(current)):
                item = current[i]
                if isinstance(item, str) and item.startswith("data:image/"):
                    current[i] = "data:image/..."
                elif isinstance(item, dict | list):
                    nodes.append(item)

    pprint(truncated_response)

def fetch_apps(url: str) -> list[str]:
    """Fetch available services from the backend."""
    try:
        response = requests.get(f"{url}/v1/info")
        response.raise_for_status()

        response_json = response.json()
        logger.debug(json.dumps(response_json, indent=2))

        return response_json["info"]["services"]
    except requests.exceptions.RequestException:
        return []

def find_or_create_apps(file_paths: list | None = None) -> list[str]:
    list_of_apps: set[str] = set()

    config = Config()

    urls = set([config.apps[app]["url"] for app in config.apps])
    urls.add(config.upload_url)
    logger.error(f".....URLs: {urls}")
    for url in urls:
        apps = fetch_apps(url)
        logger.error(f".....Apps: {apps}")

        # services available on server
        server_services: set[str] = set(apps)
        logger.error(f".....Server services: {server_services}")

        # adding services to the list of apps
        list_of_apps = list_of_apps.union(server_services)
        logger.error(f".....List of apps: {list_of_apps}")

        apps_for_url = [app for app in config.apps if config.apps[app]["url"] == url]
        logger.error(f".....Apps for URL {url}: {apps_for_url}")

        for app_name in apps_for_url:
            logger.error(f"\n.....Service name: {app_name}")
            # already existing service
            if app_name in server_services:
                logger.error(f"Service {app_name} already up on {url}")
                continue

            # create service
            logger.error(f"Creating service {app_name} on {url} with config {json.dumps(config.apps[app_name]['config'], indent=2)}")
            full_url = f"{url}/v1/app/{app_name}"
            service_config = config.apps[app_name]["config"]
            response = requests.put(full_url, json=service_config)
            if response.status_code != 200:
                logger.info(f"Failed to create service {app_name}. Status code: {response.status_code}")
            else:
                logger.info(f"Service {app_name} created successfully.")
                list_of_apps.add(app_name) # TODO: fix the case when different urls have the same service name

                if file_paths:
                    # now launch the indexing for this app
                    full_index_url = f"{url}/v1/upload/{app_name}"
                    logger.info(f"Launching upload and indexing with\n{json.dumps(service_config, indent=2)}")
                    files = []
                    for path in file_paths:
                        logger.error(f"Processing file {path}")
                        file_name = path.split("/")[-1]
                        files.append(('files', (file_name, open(path, 'rb'))))
                    response = requests.put(full_index_url, data={'ad': json.dumps(service_config)}, files=files)
                    logger.info(response.status_code)
                else:
                    # now launch the indexing for this app
                    full_index_url = f"{url}/v1/index/{app_name}"
                    logger.info(f"Launching indexing with\n{json.dumps(service_config, indent=2)}")
                    response = requests.put(full_index_url, json=service_config)
                    logger.info(response.status_code)


    # check all apps and add missing information if any
    for app_name in list_of_apps:
        if app_name not in config.apps:
            print(f"Service {app_name} not found in the configuration file")
            config.apps[app_name] = dict(
                config = config.upload_config.copy(),
                url = config.upload_url,
                examples = []
            )

    # refetch service list after service creation
    logger.info(f"List of apps: [{','.join(list_of_apps)}]")

    return list(list_of_apps)

def get_prediction(
    url: str, app_name: str, system_prompt: str, session_id: str, query: str
) -> dict:
    """
    Bot response based on the last message in the chat history.
    """
    # response_type = random.choice(["text", "gallery", "image", "video", "audio", "html"])

    # if response_type == "gallery":
    #     history[-1][1] = gr.Gallery(
    #         [
    #             "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
    #             "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
    #         ]
    #     )
    # elif response_type == "image":
    #     history[-1][1] = gr.Image(
    #         "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
    #     )
    # elif response_type == "video":
    #     history[-1][1] = gr.Video(
    #         "https://github.com/gradio-app/gradio/raw/main/demo/video_component/files/world.mp4"
    #     )
    # elif response_type == "audio":
    #     history[-1][1] = gr.Audio(
    #         "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav"
    #     )
    # elif response_type == "html":
    #     history[-1][1] = gr.HTML(
    #         html_src(random.choice(["harmful", "neutral", "beneficial"]))
    #     )
    # else:
    #     history[-1][1] = "Cool!"
    url = url.strip("/")

    try:
        response = requests.get(f"{url}/v1/index/{app_name}/status")
        response.raise_for_status()

        while "running" in response.json()["message"]:
            logger.info(json.dumps(response.json(), indent=2))
            logger.info("Sleeping for 2 seconds")
            time.sleep(2)
            response = requests.get(f"{url}/v1/index/{app_name}/status")
        payload = {
            "app": {"verbose": "info"},
            "parameters": {
                "input": {
                    "template": {
                        "template_prompt": system_prompt,
                        "template_prompt_variables": ["context", "question"],
                    },
                    "message": query,
                    "session_id": session_id,
                }
            },
        }
        response = requests.post(
            f"{url}/v1/predict/{app_name}", json=payload
        )
        response.raise_for_status()

        response_json = response.json()
        logger.debug(json.dumps(pretty_print_response(response_json), indent=2))

        output = {}

        output["answer"] = response_json["output"]
        output["documents"] = []

        if output["answer"].startswith("data:image/"):
            # output from image service
            return output
        else:
            # Output from RAG
            if response_json["sources"] is not None and "context" in response_json["sources"]:
                for source in response_json["sources"]["context"]:

                    pattern = re.compile(
                        r'^(?P<uuid>[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})_'
                        r'(?P<page>\d+)'
                        r'(?:_(?P<kind>[a-z]+)(?:_(?P<number>\d+))?)?$'
                    )

                    if "key" in source:
                        # new format
                        uuid = os.path.basename(source["source"])

                        match = pattern.match(source["key"])
                        if match:
                            page = match.group('page')
                            kind = match.group('kind') or 'image'
                            number = match.group('number') or -1
                            is_image = True
                            content = IMAGE_404
                        else:
                            page = None
                            kind = None
                            number = None
                            is_image = False
                            content = "no content found"
                            logger.debug(f"No match found for: {source['key']}")

                    else:
                        # old format
                        s = source["source"]

                        # Extract folder_name, page, and crop from the key
                        match = pattern.match(s)
                        if match:
                            uuid = match.group('uuid')
                            page = match.group('page')
                            kind = match.group('kind') or 'image'
                            number = match.group('number') or -1
                            is_image = True
                            content = IMAGE_404
                        else:
                            uuid = None
                            page = None
                            kind = None
                            number = None
                            is_image = False
                            content = "no content found"
                            logger.debug(f"No match found for: {s}")

                    content = source["content"] if "content" in source else content

                    output["documents"].append(
                        {
                            "folder_name": uuid,
                            "page": page,
                            "type": kind,
                            "number": number,
                            "distance": source["distance"] if "distance" in source else None,
                            "content": content,
                            "is_image": is_image,
                        }
                    )

            return output
    except requests.exceptions.RequestException as e:
        logger.error(e)
        return {"answer": "", "documents": []}
