import base64
import resource
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Hashable, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from dash import html
from PIL import Image

from .viz import HHDVisualizer

tfk = tf.keras


def display_predictions(sample: pd.DataFrame, model: tfk.models, data_registry: Path, bottleneck_layer: str) -> list:
    """Displays the sample with overlaid gt mask and model prediction mask.

    Args:
        sample: seledted row of the model_meta dataframe
        model: trained tfk saved model
        data_registry: path to data registry

    Returns:
        children: a list of dash.html elements
    """
    visualizer = HHDVisualizer(data_registry=data_registry, bottleneck_layer=bottleneck_layer)

    overlaid, overlaid_pred, overlaid_xai_hyper, overlaid_xai_hypo = visualizer.get_overlaid_mask_on_slice(sample, model=model)
    image_b64 = _numpy_to_b64(overlaid)
    children = [
        html.Div(
            style={"display": "flex", "flex-direction": "column"},
            children=[
                html.Label(
                    f"{int(sample.NumberOfSlices)} slice series from \
                    {sample.DataSource}",
                ),
                html.Img(src="data:image/png;base64, " + image_b64, style={"height": "33vh", "display": "block", "margin": "auto"}),
            ],
        )
    ]

    if overlaid_pred is not None:
        image_b64 = _numpy_to_b64(overlaid_pred)

        children.extend(
            [
                html.Div(
                    style={"display": "flex", "flex-direction": "column"},
                    children=[
                        html.Label(
                            "Model Predictions",
                        ),
                        html.Img(src="data:image/png;base64, " + image_b64, style={"height": "33vh", "display": "block", "margin": "auto"}),
                    ],
                )
            ]
        )

    if overlaid_xai_hyper is not None:
        image_b64 = _numpy_to_b64(overlaid_xai_hyper)

        children.extend(
            [
                html.Div(
                    style={"display": "flex", "flex-direction": "column"},
                    children=[
                        html.Label(
                            "Hyper Explainability",
                        ),
                        html.Img(src="data:image/png;base64, " + image_b64, style={"height": "33vh", "display": "block", "margin": "auto"}),
                    ],
                )
            ]
        )

    if overlaid_xai_hypo is not None:
        image_b64 = _numpy_to_b64(overlaid_xai_hypo)

        children.extend(
            [
                html.Div(
                    style={"display": "flex", "flex-direction": "column"},
                    children=[
                        html.Label(
                            "Hypo Explainability",
                        ),
                        html.Img(src="data:image/png;base64, " + image_b64, style={"height": "33vh", "display": "block", "margin": "auto"}),
                    ],
                )
            ]
        )
    children = [html.Div(children=children, style={"display": "flex", "flex-direction": "row", "justify-content": "space-evenly"})]
    return children


def infer_false_options(df: pd.DataFrame) -> list:
    """Returns FN and FP columns.

    Args:
        df: data frame to use
    """
    target_options = list()
    for col in df.columns:
        if ("FP" in col) or ("FN" in col):
            target_options.append(col)
    return target_options


def infer_metric(df: pd.DataFrame) -> list:
    """Returns metric columns.

    Args:
        df: data frame to use
    """
    target_options = list()
    for col in df.columns:
        if ("Dice" in col) or ("IoU" in col):
            target_options.append(col)
    return target_options


def infer_target_cohort_options(df: pd.DataFrame, unique_limit: int = 10) -> Tuple[List[Hashable], List[Hashable]]:
    """Decides what columns to consider as target (float values) and cohort (categorical) based on dtype and values.

    Notes:
        - target columns are determined by:
            - float columns with unique values greater than ``unique_limit``
            - object columns that are convertible to float and have unique
              values greater than ``unique_limit``
        - cohort columns are determined by:
            - float columns with unique values smaller than ``unique_limit``
            - int columns with unique values smaller than ``unique_limit``
            - object columns that have unique values smaller than
              ``unique_limit``
            - object columns that are convertible to float and have unique
              values smaller than ``unique_limit``

    Args:
        df: data frame to use
        unique_limit: the limit to decide between being cohort or target, in
        non-float columns

    Returns:
        two lists of hashables which are target and cohort columns
    """
    target_options = list()
    cohort_options = list()
    for i, j in df.dtypes.iteritems():
        if j in (np.float16, np.float32, np.float64):
            if len(np.unique(df[i].dropna().values)) > unique_limit:
                target_options.append(i)
            else:
                cohort_options.append(i)
                vals = df[i]
                df[i] = vals.fillna(0).astype(int)
                df.loc[vals.isna(), i] = None
        else:
            if j == int:
                if len(df[i].unique()) < unique_limit:
                    cohort_options.append(i)
            else:
                col_as_series = df[i]
                if any(col_as_series.dropna()):
                    try:
                        unique_values = len(np.unique(col_as_series.dropna().values))
                    except TypeError:
                        pass
                    else:
                        if unique_values <= unique_limit:
                            cohort_options.append(i)

    return target_options, cohort_options


def get_records_and_columns(df: pd.DataFrame, flag_round: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Digests the data-frame to be able to use in a dash DataTable.

    Returns:
        tuple containing:
            records (list): data-frame rows formatted to use in the dash.DataTable
            cols (list): data-frame columns formatted to use in the dash.DataTable
    """
    if flag_round:
        df = df.round(8)
    records = df.to_dict("records")
    for ind, record in enumerate(records):
        record[df.columns.name] = df.index[ind]

    cols = list()
    for col in df.columns:
        cols.append({"name": str(col), "id": str(col)})
    cols.append({"name": str(df.columns.name), "id": str(df.columns.name)})

    return records, cols


def _numpy_to_b64(image: Image.Image) -> str:
    """Converts the PIL image to base64-encoded image in order to display in the dash app.

    Args:
        image: PIL Image to convert

    Returns:
        base64-encoded string version of the given image
    """
    buff = BytesIO()
    image.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return im_b64


def memory_limit(percentage: float = 0.94):
    """Sets your desired percentage of memory usage limit."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * percentage * 1024, hard))


def get_memory():
    """Returns free memory."""
    with open("/proc/meminfo", "r") as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                free_memory += int(sline[1])
    return free_memory
