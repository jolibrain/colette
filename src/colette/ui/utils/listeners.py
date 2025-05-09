import base64
import json
import re
import shutil
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import gradio as gr
from gradio_i18n import gettext as _
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from PIL import Image

from colette.ui.utils.api import find_or_create_apps, get_prediction
from colette.ui.utils.config import Config
from colette.ui.utils.logger import logger
from colette.ui.utils.namesgenerator import get_random_name, is_random_name


def mask_base64_data(msg: dict) -> dict:
    """Process the base64 image data in the message, replacing it with placeholders for easier logging

    Args:
        msg (dict): A dictionary of messages in OpenAI format

    Returns:
        dict: This is the processed message dictionary with the image data replaced with placeholders
    """
    if not isinstance(msg, dict):
        return msg

    new_msg = msg.copy()
    content = new_msg.get("content")
    img_base64_prefix = "data:image/"

    if isinstance(content, list):
        # Handling multimodal content (like gpt-4v format)
        new_content = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")
                if image_url.startswith(img_base64_prefix):
                    item = item.copy()
                    item["image_url"] = {"url": "<Image base64 data has been omitted>"}
            new_content.append(item)
        new_msg["content"] = new_content
    elif isinstance(content, str) and img_base64_prefix in content:
        # Process plain text messages containing base64 image data
        new_msg["content"] = "<Messages containing image base64 data have been omitted>"
    return new_msg

def get_uuid():
    return str(uuid4()).split("-")[-1]

def generate_name_from_history(history):
    try:
        for message in history:
            print(message)
            if "content" in message and isinstance(message["content"], str):
                tokens = word_tokenize(message["content"])
                filtered_tokens = [word for word in tokens if word.lower() not in stopwords.words("english")]
                tagged_tokens = pos_tag(filtered_tokens)
                keywords = [word for word, tag in tagged_tokens if tag.startswith('NN') or tag.startswith('VB')]
                title = " ".join(keywords)
                logger.debug(f"Title: {title}")
                # returns firt 20 charecters of the title
                return title[-20:]
    except Exception as e:
        logger.error(f"Error generating name from history: {e}")
        return get_random_name()

def log_like_dislike(x: gr.LikeData):
    message = f"User {'liked' if x.liked else 'disliked'} message {x.index}: {x.value}"
    logger.info(message)

def add_message(
        history,
        message,
        sessions,
        current_session,
):
    """
    Add a message to the chat history
    """
    logger.debug("add_message")
    logger.debug(f"history: {history}")
    logger.debug(f"message: {message}")
    logger.debug(f"sessions: {sessions}")
    logger.debug(f"current_session: {current_session}")

    try:
        sessions_dict = sessions.value
    except AttributeError:
        logger.debug("AttributeError")
        sessions_dict = sessions

    config = Config()

    current_app = sessions_dict[current_session]["app_name"]

    if "files" in message and message["files"]:
        # Create a new session
        new_session_id = get_random_name()
        if new_session_id is None:
            logger.error("Failed to generate session ID")
            return history, gr.update(value=None), sessions, gr.update(value=current_session), gr.update(value=current_app)

        history = []
        file_paths = []
        for file_path in message["files"]:
            history.append({ "role": "user", "content": {"path": file_path}})
            file_paths.append(file_path)

        # create new app with newly uploaded filepath
        config.apps[new_session_id] = dict(
            url=config.upload_url,
            config=config.upload_config.copy(),
            examples=[],
            system_promt=""
        )

        current_app = new_session_id

        config.apps[new_session_id]["config"]["app"]["repository"] = current_app
        logger.info(json.dumps(config.apps[new_session_id]["config"], indent=2))

        # update list of apps
        list_of_apps = find_or_create_apps(file_paths=file_paths)

        if new_session_id in list_of_apps:
            logger.info(f"New app created: {new_session_id}")
            sessions_dict[new_session_id] = {"history": [], "app_name": current_app, "name": new_session_id}
            current_session = new_session_id
        else:
            logger.error(f"Error while creating new app: {new_session_id}")
    else:
        # no file provided
        list_of_apps = find_or_create_apps()

    if "text" in message:
        history.append({ "role": "user", "content": message["text"] })
    # outputs:
    # - history
    # - chat_input
    # - sessions,
    # - sessions_dropdown,
    # - apps_dropdown,
    return (
        history,
        gr.update(value=None),
        sessions,
        gr.update(choices=[(v["name"], k) for k, v in sessions_dict.items()], value=current_session),
        gr.update(choices=list_of_apps, value=current_app),
    )

def add_custom_message(evt: gr.SelectData, history: list):
    """
    Add a custom message to the history
    """
    history.append({ "role": "user", "content": evt.value["text"] })
    return history

def highlight_text(text, keywords):
    # Escape keywords for regex
    keywords_escaped = [re.escape(k) for k in keywords if k.strip()]
    if not keywords_escaped:
        return text
    pattern = re.compile(r"(" + "|".join(keywords_escaped) + r")", re.IGNORECASE)
    highlighted_text = pattern.sub(r"<mark>\1</mark>", text)
    return highlighted_text

def colette(
    app: str, session: str, history: list
):
    """
    Get the bot response based on the user input

    Args:
        app (str): The selected app
        session (str): The current session
        history (list): The chat history
    Returns:
    """
    # Get the actual prompt text
    logger.debug("colette")
    logger.debug(f"app: {app}")
    logger.debug(f"session: {session}")
    logger.debug(f"history: {history}")

    system_prompt = ""

    config = Config()

    # Call the get_prediction function with the system prompt
    app_url = config.apps[app]["url"]

    # Get the last user message from history which has type str
    query_txt = ""
    for message in history[::-1]:
        if message["role"] == "user" and isinstance(message["content"], str):
            query_txt = message["content"]
            break

    output = get_prediction(
        app_url, app, system_prompt, session, query_txt
    )

    detected_language = "en" # detect(query_txt)

    # Determine the language for stopwords based on the system prompt key
    if detected_language == "fr":
        language = _("french")
    else:
        language = _("english")

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(query_txt)
    keywords = [
        w for w in word_tokens if w.lower() not in stop_words and w.isalpha()
    ]

    context_html = ""
    if output["documents"]:
        for document in output["documents"]:
            context_html += "<details open>"
            context_html += f"<summary>{document['folder_name']} - page {document['page']}"
            context_html += f" - {document['type']}"
            context_html += "</summary>"

            if document["is_image"]:
                # onclick: open base64 image in a new tab inside Chrome
                context_html += f'''
                <a
                    href="#"
                    onclick="
                    var image = new Image();
                    image.src=\'{document["content"]}\';
                    var _window = window.open(\'\');
                    var document_html = \'<html><div style=\\'height: 100vh;\\' >\';
                    document_html += image.outerHTML.replace(\'<img\', \'<img style=\\'height:100%;\\' \');
                    document_html += \'</div></html>\';
                    _window.document.write(document_html);
                    ">
                    <img src="{document["content"]}">
                </a>'''
            else:
                highlighted_content = highlight_text(document["content"], keywords)
                context_html += "<b>"
                context_html += _("Content")
                context_html += f":</b><br>{highlighted_content}"

            context_html += "</details>"

    history.append({"role": "assistant", "content": ""})
    # To stream the response character by character
    # for character in output["answer"]:
    #     history[-1]["content"] += character
    #     time.sleep(0.02)
    #     yield history, context_html
    if output["answer"].startswith("data:image/"):
        # decode the base64 image
        header, encoded = output["answer"].split(",", 1)
        try:
            decoded = base64.b64decode(encoded)
            image = Image.open(BytesIO(decoded))
            history[-1]["content"] = gr.Image(image, interactive=False)
        except Exception as e:
            history[-1]["content"] = f"Error decoding image: {e}"
    else:
        history[-1]["content"] = output["answer"]

    return history, context_html

def new_session(
    sessions: dict,
    current_session: str,
    chatbot_history: list,
    app: str,
):
    """
    Start a new session with the current app

    Args:
        sessions (gr.State): The existing sessions
        current_session (str: The current session
        history (list): The chat history
        app (str): The selected app

    Returns:
        session state (gr.update): The new session state
        session dropdown (gr.update): The updated session dropdown
        chatbot (gr.update): The updated chatbot
        chat input (gr.update): The updated chat input

    """
    logger.debug("new session")
    logger.debug(f"sessions: {sessions}")
    logger.debug(f"current_session: {current_session}")
    logger.debug(f"chatbot_history: {chatbot_history}")
    logger.debug(f"app: {app}")
    try:
        sessions_dict = sessions.value
    except AttributeError:
        logger.debug("AttributeError")
        sessions_dict = sessions
    logger.debug(f"sessions_dict: {sessions_dict}")

    # if current session is empty, don´t create a new session
    if len(chatbot_history) == 0:
        return (
            gr.update(value=sessions_dict),
            gr.update(choices=[(v["name"], k) for k, v in sessions_dict.items()], value=current_session),
            gr.update(value=[]),
            gr.update(interactive=True)
        )

    sessions_dict[current_session]["history"] = chatbot_history

    # Create a new session
    new_session_id = get_random_name()
    # Store the selected system prompt key in the session
    sessions_dict[new_session_id] = {
        "history": [],
        "app_name": app,
        "name": get_random_name()
    }

    logger.debug(f"{sessions_dict} {new_session_id}")

    return (
        gr.update(value=sessions_dict),
        gr.update(choices=[(v["name"], k) for k, v in sessions_dict.items()], value=new_session_id),
        gr.update(value=[]),
        gr.update(interactive=True)
    )

def change_app(
    sessions: dict,
    current_session: str,
    chatbot_history: list,
    app: str,
):
    """
    Start a new session with the current app

    Args:
        sessions (gr.State): The existing sessions
        current_session (str: The current session
        history (list): The chat history
        app (str): The selected app

    Returns:
        session state (gr.update): The new session state
        session dropdown (gr.update): The updated session dropdown
        chatbot (gr.update): The updated chatbot
        chat input (gr.update): The updated chat input

    """
    logger.debug("change_app")
    try:
        sessions_dict = sessions.value
    except AttributeError:
        logger.debug("AttributeError")
        sessions_dict = sessions

    logger.debug(f"Sessions: {sessions}")
    logger.debug(f"Current session: {current_session}")
    logger.debug(f"Chatbot history: {chatbot_history}")
    logger.debug(f"Selected app: {app}")

    config = Config()

    # Store the current session only if there is something in the history
    if current_session and chatbot_history:
        sessions_dict[current_session]["history"] = chatbot_history

    # Create a new session only if chatbot history is not empty
    if chatbot_history:
        new_session_id = get_random_name()
        # Store the selected system prompt key in the session
        sessions_dict[new_session_id] = {"history": [], "app_name": app}
    else:
        new_session_id = current_session

    list_of_examples = []
    for example in config.apps[app].get("examples", []):
        list_of_examples.append({"text": _(example.get("text", ""))})

    return (
        gr.update(value=sessions_dict),
        gr.update(choices=[(v["name"], k) for k, v in sessions_dict.items()], value=new_session_id),
        gr.update(value=[], examples=list_of_examples),
        gr.update(interactive=True)
    )

def select_session(session_id: str, sessions: dict):
    logger.debug("select_session")
    logger.debug(f"Session ID: {session_id}")
    logger.debug(f"Sessions: {sessions}")

    interactive = False
    history = []

    try:
        sessions_dict = sessions.value
    except AttributeError:
        logger.debug("AttributeError")
        sessions_dict = sessions

    if session_id:
        # Retrieve the session data
        session_data = sessions_dict[session_id]
        history = session_data["history"]
        interactive = True
        app_name = session_data["app_name"]

    return (
        gr.update(interactive=interactive),
        gr.update(value=history),
        gr.update(value=app_name)
    )

def update_session(sessions: dict, session_id: str, history: list):
    """
    Update the session with the chat history

    Args:
        sessions (gr.State): The existing sessions
        session_id (gr.State): The current session
        chatbot_history (list): The chat history

    Returns:
        session state (gr.update): The updated session state
        session dropdown (gr.update): The updated session dropdown

    """
    logger.debug("update_session")
    logger.debug(f"Sessions: {sessions}")
    logger.debug(f"Session ID: {session_id}")
    logger.debug(f"History: {history}")
    try:
        sessions_dict = sessions.value
    except AttributeError:
        logger.debug("AttributeError")
        sessions_dict = sessions

    if session_id:
        logger.debug(session_id, sessions_dict)
        sessions_dict[session_id]["history"] = history
        if is_random_name(sessions_dict[session_id]["name"]):
            sessions_dict[session_id]["name"] = generate_name_from_history(history)
    return (
        sessions_dict,
        gr.MultimodalTextbox(interactive=True)
    )
