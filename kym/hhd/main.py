from pathlib import Path

import click
import pandas as pd
import tensorflow as tf

from .dashboard import Controller, get_app

tfk = tf.keras


@click.command()
@click.option("--model-meta")
@click.option("--saved-model")
@click.option("--data-registry")
@click.option("--debug", is_flag=True)
@click.option("--port", default=8050, type=int)
def main(model_meta: Path, saved_model: Path, data_registry: Path, debug: bool = True, port: int = 8050):
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

    app = get_app(model_meta_=model_meta, model_=model, data_registry_=data_registry)
    controller = Controller(model_meta, model, data_registry)
    controller.add_callbacks(app)
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    main()
