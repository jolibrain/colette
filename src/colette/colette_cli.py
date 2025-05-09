import json
import time
from pathlib import Path

import typer
from fastapi.testclient import TestClient

from colette.httpjsonapi import app

cli = typer.Typer(help="Colette CLI: indexing, chatting, and UI management.")

option_appdir = typer.Option(..., help="Specify the application directory")
option_datadir = typer.Option(..., help="Specify the data directory")
option_modelsdir = typer.Option(None, help="Specify the models directory")
option_configfile = typer.Option(None, help="Specify the config file")
option_indexfile = typer.Option(None, help="Specify the index file")
option_msg = typer.Option(..., help="Specify the user message")
option_host = typer.Option("0.0.0.0", help="Specify the host")
option_port = typer.Option(7860, help="Specify the port")


@cli.command()
def index(
    app_dir: Path = option_appdir,
    data_dir: Path = option_datadir,
    models_dir: Path = option_modelsdir,
    config_file: Path = option_configfile,
    index_file: Path = option_indexfile,
):
    """Run the indexing phase."""
    typer.echo(f"Starting the indexing of: {data_dir}")

    try:
        app_dir = app_dir.absolute()
        data_dir = data_dir.absolute()
        app_name = app_dir.name
        typer.echo(f"App name: {app_name}")

        if not config_file:
            config_file = Path(__file__).parent / "config" / "vrag_default.json"
        else:
            typer.echo(f"Using config file: {config_file}")

        if not index_file:
            index_file = Path(__file__).parent / "config" / "vrag_default_index.json"
        else:
            typer.echo(f"Using index file: {index_file}")

            if not models_dir:
                models_dir = app_dir / "models"

        try:
            with open(config_file) as f:
                json_data = json.load(f)
        except Exception as e:
            typer.echo(f"Failed to load config file: {e}", err=True)
            raise typer.Exit(code=1) from e

        try:
            with open(index_file) as f:
                index_data = json.load(f)
        except Exception as e:
            typer.echo(f"Failed to load index file: {e}", err=True)
            raise typer.Exit(code=1) from e

        json_data["app"]["repository"] = str(app_dir)
        json_data["app"]["models_repository"] = str(models_dir)
        index_data["parameters"]["input"]["data"] = [str(data_dir)]

        # beware : context manager is mandatory in order to trigger lifespan events
        # (here indexing loops)
        # do NOT   client = TestClient(app)
        with TestClient(app) as client:
            # client = TestClient(app)
            response = client.put(f"/v1/app/{app_name}", json=json_data)
            if response.status_code != 200:
                typer.echo(f"Service creation failed: {response.text}", err=True)
                raise typer.Exit(code=1)

            response = client.put(f"/v1/index/{app_name}", json=index_data)
            if response.status_code != 200:
                typer.echo(f"Indexing launch failed: {response.text}", err=True)
                raise typer.Exit(code=1)

            response_index = client.get(f"/v1/index/{app_name}/status")
            while (
                "finished" not in response_index.json()["message"] and "error" not in response_index.json()["message"]
            ):
                time.sleep(0.5)
                response_index = client.get(f"/v1/index/{app_name}/status")
                if response_index.status_code != 200:
                    typer.echo(f"Indexing failed: {response.text}", err=True)
                    raise typer.Exit(code=1)

            response = client.delete(f"/v1/app/{app_name}")
            if response.status_code != 200:
                typer.echo(f"Delete request failed: {response.text}", err=True)
                raise typer.Exit(code=1)

    except Exception as e:
        typer.echo(f"Indexing failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    if "error" in response_index.json()["message"]:
        typer.echo("Indexing failed")
    else:
        typer.echo("Indexing completed.")


@cli.command()
def chat(app_dir: Path = option_appdir, msg: str = option_msg, models_dir: Path = option_modelsdir):
    """Start a chat with the application."""
    typer.echo(f"Starting chat with {app_dir} using message: {msg}")
    try:
        app_dir = app_dir.absolute()
        app_name = app_dir.name

        config_file = app_dir / "config.json"
        if not models_dir:
            models_dir = app_dir / "models"

        try:
            with open(config_file) as f:
                json_data = json.load(f)
        except Exception as e:
            typer.echo(f"Failed to load config file: {e}", err=True)
            raise typer.Exit(code=1) from e

        json_data["app"]["repository"] = str(app_dir)
        json_data["app"]["models_repository"] = str(models_dir)
        json_data["parameters"]["input"]["rag"]["reindex"] = False
        json_data["parameters"]["input"]["rag"]["index_protection"] = True
        json_data["parameters"]["input"]["rag"]["gpu_id"] = 0

        client = TestClient(app)
        response = client.put(f"/v1/app/{app_name}", json=json_data)
        if response.status_code != 200:
            typer.echo(f"Initialization failed: {response.text}", err=True)
            raise typer.Exit(code=1)

        chat_payload = dict(parameters=dict(input=dict(message=msg)))
        response = client.post(f"/v1/predict/{app_name}", json=chat_payload)
        if response.status_code != 200:
            typer.echo(f"Chat request failed: {response.text}", err=True)
            raise typer.Exit(code=1)

        typer.echo(response.json()["output"])

        response = client.delete(f"/v1/app/{app_name}")
        if response.status_code != 200:
            typer.echo(f"Chat request failed: {response.text}", err=True)
            raise typer.Exit(code=1)

    except Exception as e:
        typer.echo(f"Chat failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    typer.echo("Chat completed.")


@cli.command()
def ui(
    host: str = option_host,
    port: int = option_port,
    config: Path = option_configfile,
):
    """Launch the Colette Gradio UI."""
    typer.echo(f"Starting Colette UI with config file: {config}")
    try:
        from colette.ui.app import create_gradio_interface

        gradio_app = create_gradio_interface(config)
        gradio_app.launch(server_name=host, server_port=port)
    except ImportError as ie:
        typer.echo("Please install the Colette UI dependencies.", err=True)
        raise typer.Exit(code=1) from ie
    except Exception as e:
        typer.echo(f"Error launching Colette UI: {e}", err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    cli()
