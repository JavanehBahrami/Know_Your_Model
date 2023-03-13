import logging
import sys
from pathlib import Path

import click
import pandas as pd
import tensorflow.keras as tfk

from .hhd.dashboard import Controller, get_app
from .hhd.utils import memory_limit


@click.group()
def cli():
    print("Know Your Model toolkit.")


@cli.command()
@click.option("--model-meta")
@click.option("--saved-model")
@click.option("--data-registry")
@click.option("--debug", is_flag=True)
@click.option("--port", default=8050, type=int)
def hhd(model_meta: Path, saved_model: Path, data_registry: Path, debug: bool, port: int):
    """Runs dash plotly app on default_port = 8050

    Note: if memory usage exceeds .94 the app will be closed automatically

    Args:
        model_meta: path to model meta data pickle file
        saved_model: path to tfk saved model
        data_registry: path to data registry which contains all datasources and labeling-job masks
        debug: boolean debug mode for dash plotly `app.run`
        port: local port number for running app
    """
    memory_limit(0.94)
    try:
        try:
            model = tfk.models.load_model(saved_model, compile=False)
            print("model loaded successfully.")
        except Exception as e:
            model = None
            print(f"could not load the model: {e.args}")

        try:
            model_meta = pd.read_pickle(model_meta)
        except Exception as e:
            print(f"could not read the model_meta from {model_meta}: {e.args}")
            raise e

        app = get_app(model_meta_=model_meta)
        controller = Controller(model_meta, model, data_registry)
        controller.add_callbacks(app)
        app.run(debug=debug, port=port)
    except MemoryError:
        logging.error("Memory Exception")
        sys.stderr.write("\n\nERROR: Memory Exception\n")
        sys.exit(1)
