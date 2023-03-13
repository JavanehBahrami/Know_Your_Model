import gc
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pydicom
import tensorflow as tf
from dash import Dash, Input, Output, callback_context, dash_table, html
from memory_profiler import profile

from .components import (
    _create_input,
    _create_section_header,
    create_compare_models_section,
    create_error_hist_section,
    create_header,
    create_scatter_plot_section,
    create_wrong_cases_section,
)
from .utils import (
    display_predictions,
    get_records_and_columns,
    infer_metric,
    infer_target_cohort_options,
)

tfk = tf.keras


class ErrorPlotType:
    """Plot types to use in the Error Analysis section of the dashboard."""

    HISTOGRAM: str = "Histogram"
    BOX: str = "Box"


def get_app(model_meta_: pd.DataFrame) -> Dash:
    """Creates the app and adds its layout.

    Examples:
        >>> app = get_app(model_meta)
        >>> controller = Controller(model_meta, model, data_registry)
        >>> controller.add_callbacks(app)
        >>> app.run()

    Notes:
        - the target and cohort columns will be inferred based on data types
          and unique values. see ``utils.infer_target_cohort_options``
        - you have to add the app's callbacks before running the app:
            ``add_callbacks(app)``

    Args:
        model_meta_: model's evaluation results, as a data-frame.

    Returns:
        a dash app that can be run using ``app.run()``
    """
    model_meta = model_meta_

    app = Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
    app.title = "KYM Dashboard"

    metric_options = infer_metric(model_meta)
    # false_target_options = infer_false_options(model_meta)
    target_options, cohort_options = infer_target_cohort_options(model_meta)
    records, cols = get_records_and_columns(model_meta.groupby(cohort_options[0])[target_options[0]].describe().T)

    app.layout = html.Div(
        className="row",
        style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
        children=[
            # Header
            create_header(),
            # Overview
            # get_overview(),
            html.Div(
                children=[
                    html.Div("Note: The value of this field is going to be used as the conv_layer for seg-grad-cam and embedding analysis."),
                    html.Label("Bottleneck layer name"),
                    _create_input(placeholder="Enter The name of the bottleneck layer", id_prefix="bottleneck", type="text"),
                ],
                style={"padding": "35px", "background-color": "#f9f9f9"},
            ),
            # Error Hist/Box
            _create_section_header("Cohort Error Analysis"),
            create_error_hist_section(
                target_options=target_options,
                cohort_options=cohort_options,
                plot_types=[ErrorPlotType.HISTOGRAM, ErrorPlotType.BOX],
                data_table_records=records,
                data_table_cols=cols,
            ),
            # Scatter
            _create_section_header("Scatter Plots"),
            create_scatter_plot_section(target_options=target_options, cohort_options=cohort_options),
            # Wrong Cases
            _create_section_header("Wrong Cases Analysis"),
            create_wrong_cases_section(metric_options=metric_options, data_table_records=None, data_table_cols=None),
            # Embeddings Analysis
            _create_section_header("Embeddings Analysis"),
            # Compare
            _create_section_header("Compare Two Models"),
            create_compare_models_section(metric_options=metric_options),
        ],
    )

    return app


class Controller:
    """Adds the callbacks to make the app interactive.

    ...
    Attributes
    ----------
        model_meta: model's evaluation results, as a data-frame.
        model: trained model, will be used for making visualizations
        data_registry: absolute path to data-registry

    Methods
    -------
        add_callbacks()
        update_error_graph()
        update_scatter_graph()
        update_scatter_click_image()
        update_wrong_cases_graph()
        update_compare_models_graph()
        update_compare_models_radiocontainer()

    """

    def __init__(self, model_meta_: pd.DataFrame, model_: tf.keras.Model, data_registry_: Path):
        self.model_meta = model_meta_
        self.model = model_
        self.data_registry = data_registry_

    def add_callbacks(self, app: Dash):
        """Adds callbacks to different parts of the app
        Args:
            app: Dash app
        """

        @app.callback(
            Output("error-graph", "figure"),
            Output("describe-table", "data"),
            Output("describe-table", "columns"),
            [Input("target-dropdown", "value"), Input("cohort-dropdown", "value"), Input("plot_type-dropdown", "value")],
        )
        def _update_error_graph(target, cohort, plot_type):
            fig, records, cols = self.update_error_graph(target, cohort, plot_type)
            return fig, records, cols

        @app.callback(
            [Output("scatter-graph", "figure"), Output("column-current-values", "data")],
            [Input("scatter_x-dropdown", "value"), Input("scatter_y-dropdown", "value"), Input("scatter_color-dropdown", "value")],
        )
        def _update_scatter_graph(x_col, y_col, color):
            fig, [x_col, y_col] = self.update_scatter_graph(x_col, y_col, color)
            return fig, [x_col, y_col]

        @app.callback(
            Output("div-scatter-click-image", "children"),
            [Input("scatter-graph", "clickData"), Input("column-current-values", "data"), Input("bottleneck-input", "value")],
        )
        def _update_scatter_click_image(click_data, current_columns, bottleneck_layer):
            children = self.update_scatter_click_image(click_data, current_columns, bottleneck_layer)
            return children

        @app.callback(
            [
                Output("wrong-case-image", "children"),
                Output("wrong_next-button", "n_clicks"),
                Output("wrong_back-button", "n_clicks"),
                Output("specifications-table", "data"),
                Output("specifications-table", "columns"),
            ],
            [
                Input("wrong_metric-dropdown", "value"),
                Input("wrong_next-button", "n_clicks"),
                Input("wrong_back-button", "n_clicks"),
                Input("bottleneck-input", "value"),
            ],
        )
        def _update_wrong_cases_graph(metric, n_clicks_next, n_clicks_back, bottleneck_layer):
            children, n_clicks_next, n_clicks_back, records, columns = self.update_wrong_cases_graph(
                metric, n_clicks_next, n_clicks_back, bottleneck_layer
            )
            return children, n_clicks_next, n_clicks_back, records, columns

        @app.callback(
            [
                Output("compare-model-image", "children"),
                Output("compare_next-button", "n_clicks"),
                Output("compare_back-button", "n_clicks"),
                Output("compare_enter-button", "n_clicks"),
            ],
            [
                Input("compare-model-image", "children"),
                Input("compare_enter-button", "n_clicks"),
                Input("second_model-input", "value"),
                Input("compare_model-radiobutton", "value"),
                Input("compare_metric-dropdown", "value"),
                Input("compare_next-button", "n_clicks"),
                Input("compare_back-button", "n_clicks"),
                Input("bottleneck-input", "value"),
                Input("second_bottleneck-input", "value"),
            ],
        )
        def _update_compare_models_graph(
            children, n_clicks_enter, model_path, radio_value, metric, n_clicks_next, n_clicks_back, bottleneck_layer, second_bottleneck
        ):
            children, n_clicks_next, n_clicks_back, n_clicks_enter = self.update_compare_models_graph(
                children, n_clicks_enter, model_path, radio_value, metric, n_clicks_next, n_clicks_back, bottleneck_layer, second_bottleneck
            )
            return children, n_clicks_next, n_clicks_back, n_clicks_enter

        @app.callback(
            [
                Output("second_bottleneck-input", "style"),
            ],
            Input("compare_model-radiobutton", "value"),
        )
        def _update_compare_models_radiocontainer(radio_value):
            style = self.update_compare_models_radiocontainer(radio_value)
            return style

    # @profile
    def update_error_graph(self, target, cohort, plot_type):
        """Updates the error graph whenever one of the drop-downs changes."""
        records, cols = get_records_and_columns(self.model_meta[[target, cohort]].dropna().groupby(cohort)[target].describe().T)
        if plot_type == ErrorPlotType.HISTOGRAM:
            fig = px.histogram(
                self.model_meta[[target, cohort]].dropna().convert_dtypes(),
                x=target,
                color=cohort,
                marginal="rug",
                # log_y=True,
                opacity=0.75,
            )
        elif plot_type == ErrorPlotType.BOX:
            fig = px.box(self.model_meta[[target, cohort]].dropna().convert_dtypes(), x=cohort, y=target, points="all")
        else:
            fig = None
            print(f"{plot_type} is not supported as PlotType.")

        return fig, records, cols

    # @profile
    def update_scatter_graph(self, x_col, y_col, color):
        """Updates the scatter graph whenever one of corresponding drop-downs changes."""
        if len(self.model_meta[color].unique()) < 10:
            meta = self.model_meta.astype({color: np.uint8})
        else:
            meta = self.model_meta
        fig = px.scatter(meta.convert_dtypes(), x=x_col, y=y_col, marginal_x="histogram", marginal_y="histogram", color=color)
        return fig, [x_col, y_col]

    # @profile
    def update_scatter_click_image(self, click_data, current_columns, bottleneck_layer):
        """Displays the slice, mask and the model's predictions whenever one of the data-points has been clicked."""
        x_col, y_col = current_columns
        children = []
        if click_data:
            x = click_data["points"][0]["x"]
            y = click_data["points"][0]["y"]
            sample = self.model_meta.loc[self.model_meta[x_col].eq(x) & self.model_meta[y_col].eq(y)].iloc[0]
            children = display_predictions(sample, self.model, self.data_registry, bottleneck_layer)

        return children

    # @profile
    def update_wrong_cases_graph(self, metric, n_clicks_next, n_clicks_back, bottleneck_layer, _test_mode=False):
        """Displays the slice, mask and the model's predictions.
        Notes:
            - It starts to display images respectively based on chosed metric
            - whenever the user clicks next, the next image with higher metric
              with respect to the previous one will be showed.
            - If the user choose another metric, the n_clicks resets and starts
              from the minimum chosen metric again.
        """
        # del children
        # gc.collect()

        ctx = callback_context

        if not ctx.triggered:
            button_id = "No_clicks"
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if button_id == "wrong_metric-dropdown":
                n_clicks_next = 0
                n_clicks_back = 0
            if button_id == "wrong_back-button":
                n_clicks_next -= 1

        sorted_df = self.model_meta.sort_values(metric)
        sample = sorted_df.iloc[n_clicks_next]
        sample_df = pd.DataFrame(data=[sample], columns=self.model_meta.columns)
        for col in self.model_meta.columns:
            if type(sample_df[col].iloc[0]) == pydicom.multival.MultiValue:
                sample_df = sample_df.drop(col, axis=1)
        records, cols = get_records_and_columns(df=sample_df, flag_round=False)
        children = [
            html.Div(
                f"The next button has been clicked {n_clicks_next}\
                               times and the back button {n_clicks_back} times and\
                               the {metric} for current slice is {sample[metric]}"
            )
        ]
        children.extend(
            html.Div(
                html.Label("Slice specifications"),
            )
        )
        children.extend(dash_table.DataTable(data=records, columns=cols, id="specifications-table", style_table={"overflowY": "scroll"}))
        if not _test_mode:
            children.extend(display_predictions(sample, self.model, self.data_registry, bottleneck_layer))
        return children, n_clicks_next, n_clicks_back, records, cols

    # @profile
    def update_compare_models_graph(
        self,
        children,
        n_clicks_enter,
        model_path,
        radio_value,
        metric,
        n_clicks_next,
        n_clicks_back,
        bottleneck_layer,
        second_bottleneck,
        _test_mode=False,
    ):
        """Displays the slice, mask and the model's predictions for the original model and the model saved in the model_path entered by user.

        Notes:
            - It starts to display images respectively based on chosed metric
            - whenever the user clicks next, the next image with higher metric
              with respect to the previous one will be showed.
            - If the user choose another metric, the n_clicks resets and starts
              from the minimum chosen metric again.
            - seg-grad-cam heatmap also will be showed if the user selects `with explainability` radiobutton
        """
        del children
        gc.collect()

        ctx = callback_context

        if not ctx.triggered:
            button_id = "No clicks yet"
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if button_id == "second_model-input":
                n_clicks_enter = 0
            if n_clicks_enter == 1 and button_id == "compare_enter-button" and not _test_mode:
                global second_model
                try:
                    second_model = tfk.models.load_model(model_path, compile=False)
                    print("second model loaded successfully.")
                except Exception as e:
                    second_model = None
                    print(f"could not load the second model: {e.args}")
            if button_id == "compare_enter-button" or button_id == "compare_next-button" or button_id == "compare_back-button":
                if button_id == "compare_enter-button":
                    n_clicks_next = 0
                    n_clicks_back = 0
                if button_id == "compare_back-button":
                    n_clicks_next -= 1
                sorted_df = self.model_meta.sort_values(metric)
                sample = sorted_df.iloc[n_clicks_next]

                if radio_value == "With Explainability" and second_bottleneck is not None:
                    first_bottleneck = bottleneck_layer
                elif radio_value == "Without Explainability" or second_bottleneck is None:
                    first_bottleneck = None
                    second_bottleneck = None
                else:
                    raise ValueError

                if not _test_mode:
                    children = [
                        html.Div(
                            children=[
                                html.Div(
                                    [
                                        html.H3("First Model: "),
                                        html.Div(f"The next button has been clicked {n_clicks_next} times."),
                                        html.Div(f"The back button has been clicked {n_clicks_back} times."),
                                        html.Div(f"The enter button has been clicked {n_clicks_enter} times."),
                                        html.Div(f"The {metric} for current slice is {sample[metric]}"),
                                    ],
                                    style={"display": "flex", "flex-direction": "column", "justify-content": "flex-start", "max-width": "22%"},
                                ),
                                html.Div(
                                    children=display_predictions(
                                        sample=sample, model=self.model, data_registry=self.data_registry, bottleneck_layer=first_bottleneck
                                    )
                                ),
                                html.Div(
                                    html.H3("Second Model: "),
                                    style={"display": "flex", "flex-direction": "column", "justify-content": "flex-start", "max-width": "22%"},
                                ),
                                html.Div(
                                    children=display_predictions(
                                        sample=sample, model=second_model, data_registry=self.data_registry, bottleneck_layer=second_bottleneck
                                    )
                                ),
                            ],
                            style={"display": "flex", "flex-direction": "column", "justify-content": "space-evenly"},
                        )
                    ]
                else:
                    children = []
            elif button_id == "compare_metric-dropdown":
                n_clicks_next = 0
                n_clicks_back = 0
                n_clicks_enter = 0
                children = []
            else:
                children = []

            return children, n_clicks_next, n_clicks_back, n_clicks_enter

    # @profile
    def update_compare_models_radiocontainer(self, radio_value):
        """Decides whether to display input box for the second model bottelneck layer or not based on user decision for displaying explainability."""
        if radio_value == "With Explainability":
            style = {"display": "inline-block", "padding": "px", "margin-left": "10px", "width": "99%"}
        elif radio_value == "Without Explainability":
            style = {"display": "none"}
        else:
            raise ValueError
        return [style]
