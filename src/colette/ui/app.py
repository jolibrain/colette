import argparse
from pathlib import Path

import gradio as gr
from gradio_i18n import Translate
from gradio_i18n import gettext as _
from gradio_log import Log

from colette.ui.utils.api import find_or_create_apps
from colette.ui.utils.config import Config
from colette.ui.utils.listeners import (
    add_custom_message,
    add_message,
    change_app,
    colette,
    log_like_dislike,
    new_session,
    select_session,
    update_session,
)
from colette.ui.utils.logger import log_file, logger
from colette.ui.utils.namesgenerator import get_random_name

ASSETS_DIR = Path(__file__).parent / "assets"

def create_gradio_interface(config_path):
    # Initialize configuration and applications
    print(f"Calling config: {str(config_path)}")
    config = Config()
    if config_path.exists():
        print("Calling load_config")
        config.load_config(config_path)
    else:
        logger.error(f"Configuration file {config_path} not found.")
        exit(1)

    list_of_apps = find_or_create_apps()
    if not list_of_apps:
        logger.error("No apps found in the configuration file")
        exit(1)

    current_app = list_of_apps[0]
    current_ses = get_random_name()

    gr.set_static_paths(paths=[ASSETS_DIR])

    with gr.Blocks(
            title="Colette",
            css_paths=[
                ASSETS_DIR / "css" / "custom.css",
                ASSETS_DIR / "css" / "theme.css",
            ]
        ) as colette_ui:
        with Translate(
                str(ASSETS_DIR / "translation.yaml"),
                placeholder_langs=["fr", "en"]
        ):
            with gr.Row():
                gr.Markdown("# Colette")
                gr.Image(
                    config.logo_path,
                    height=30,
                    width=100,
                    elem_id="logo-image",
                    interactive=False,
                    container=False,
                    show_share_button=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    show_label=False
                )

            sessions = gr.State({})

            if current_ses not in sessions.value.keys():
                sessions.value[current_ses] = dict(
                    history=[],
                    app_name=current_app,
                    name=get_random_name(),
                )

            with gr.Tab(_("Chatbot")):
                with gr.Row():
                    with gr.Column(min_width=100):
                        sessions_dropdown = gr.Dropdown(
                            choices=[(v["name"], k) for k, v in sessions.value.items()],
                            label=_("Sessions"),
                            container=False,
                        )
                        with gr.Row(visible=True):
                            with gr.Row(scale=1):
                                new_session_button = gr.Button(
                                    value=_("New session"),
                                )

                    with gr.Column(min_width=100):
                        apps_dropdown = gr.Dropdown(
                            choices=list_of_apps,
                            label=_("Available Services"),
                            container=False,
                            value=current_app,
                        )

                    with gr.Column(scale=4, elem_id="chatbot-column"):
                        list_of_examples = []
                        for example in config.apps[current_app].get("examples", []):
                            list_of_examples.append({"text": _(example.get("text", ""))})

                        logger.info(f"List of examples: {list_of_examples}")

                        chatbot = gr.Chatbot(
                            elem_id="chatbot",
                            # bubble_full_width=False,
                            type="messages",
                            height=500,
                            examples=list_of_examples,
                        )

                        chat_input = gr.MultimodalTextbox(
                            scale=4,
                            interactive=True,
                            placeholder=_("Enter message..."),
                            show_label=False,
                            file_count="multiple",
                            file_types=["file"],
                            sources=["upload"],
                        )

                    with gr.Column(scale=2):
                        with gr.Accordion(_("Sources"), open=True):
                            context_display = gr.HTML(label=_("Context Chunks"))

            with gr.Tab(_("Logs")):
                Log(log_file, tail=500, height="100%", dark=False)

            with gr.Tab(_("About") ):
                gr.Markdown(value=lambda: _('About content'))

            chatbot.like(log_like_dislike, None, None, like_user_message=True)

            chat_input.submit(
                add_message,
                inputs=[chatbot, chat_input, sessions, sessions_dropdown],
                outputs=[chatbot, chat_input, sessions, sessions_dropdown, apps_dropdown]
            ).then(
                colette,
                inputs=[apps_dropdown, sessions_dropdown, chatbot],
                outputs=[chatbot, context_display],
            ).then(
                update_session,
                inputs=[sessions, sessions_dropdown, chatbot],
                outputs=[sessions, chat_input]
            )

            chatbot.example_select(
                add_custom_message,
                inputs=chatbot,
                outputs=chatbot
            ).then(
                colette,
                inputs=[apps_dropdown, sessions_dropdown, chatbot],
                outputs=[chatbot, context_display],
            ).then(
                update_session,
                inputs=[sessions, sessions_dropdown, chatbot],
                outputs=[sessions, chat_input]
            )

            new_session_button.click(
                new_session,
                inputs=[sessions, sessions_dropdown, chatbot, apps_dropdown],
                outputs=[sessions, sessions_dropdown, chatbot, chat_input]
            )

            apps_dropdown.input(
                change_app,
                inputs=[sessions, sessions_dropdown, chatbot, apps_dropdown],
                outputs=[sessions, sessions_dropdown, chatbot, chat_input]
            )

            sessions_dropdown.change(
                select_session,
                inputs=[sessions_dropdown, sessions],
                outputs=[chat_input, chatbot, apps_dropdown],
            )

    return colette_ui

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Colette UI")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    print(f"Starting Colette UI with config file: {args.config}")
    gradio_app = create_gradio_interface(Path(args.config))
    try:
        gradio_app.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        print(f"Error launching Colette UI: {e}")
