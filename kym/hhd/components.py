from typing import Any, Dict, Hashable, List

from dash import dash_table, dcc, html


def create_header(title: str = "KYM", caption: str = "Know Your Model Dashboard") -> html.Div:
    """Creates the header for the app."""
    header = html.Div(
        className="row header",
        id="app-header",
        style={"background-color": "#f9f9f9"},
        children=[
            html.H1(
                children=title,
                style={
                    "textAlign": "center",
                },
            ),
            html.Div(
                children=caption,
                style={
                    "textAlign": "center",
                },
            ),
        ],
    )
    return header


def _create_section_header(title: str) -> html.Div:
    """Helper to create a section header."""
    return html.Div(
        className="row header",
        id=f"header-{title}",
        style={"background-color": "#f9f9f9"},
        children=[html.H2(children=title)],
    )


def create_error_hist_section(
    target_options: List[Hashable],
    cohort_options: List[Hashable],
    plot_types: List[str],
    data_table_records: List[Dict[str, Any]],
    data_table_cols: List[Dict[str, Any]],
) -> html.Div:
    """Creates the error histogram/box section.

    Args:
        target_options: which columns to consider as target variables (error
        columns in the dataframe)
        cohort_options: which columns to consider as cohort variables
        plot_types: types of supported plots to display in the section
        data_table_records: initial records to show in the summary data-table
        data_table_cols: initial columns to show in the summary data-table

    Returns:
        html.Div to put in the ``app.layout``
    """
    return html.Div(
        className="row background",
        style={"padding": "10px"},
        children=[
            # Dropdowns
            html.Div(
                className="three columns",
                children=[
                    # Target
                    _create_dropdown(target_options, "Target Column", "target"),
                    # Cohort
                    _create_dropdown(cohort_options, "Cohort Column", "cohort"),
                    # Plot type
                    _create_dropdown(plot_types, "Plot Type", "plot_type"),
                    _create_describe_data_table(data_table_records, data_table_cols),
                ],
            ),
            # Plot
            html.Div(
                className="nine columns",
                children=[
                    dcc.Graph(
                        id="error-graph",
                        # figure={'data': [],
                        # 'layout': error_graph_layout},
                        style={"height": "80vh"},
                    )
                ],
            ),
        ],
    )


def create_scatter_plot_section(
    target_options: List[Hashable],
    cohort_options: List[Hashable],
) -> html.Div:
    """Creates the Scatter-plot section in the app's layout.
    Args:
        target_options: which columns to consider as target variables (error
        columns in the dataframe)
        cohort_options: which columns to consider as cohort variables

    Returns:
        html.Div to put in the ``app.layout``
    """
    return html.Div(
        className="row background",
        style={"padding": "10px", "display": "flex", "flex-direction": "column"},
        children=[
            html.Div(
                children=[
                    html.Div(
                        className="three columns",
                        children=[
                            _create_dropdown(options=target_options, label="X-Axis", id_prefix="scatter_x"),
                            _create_dropdown(options=target_options, label="Y-Axis", id_prefix="scatter_y"),
                            _create_dropdown(options=cohort_options + target_options, label="Color", id_prefix="scatter_color"),
                            # html.Div(id='div-scatter-click-image',
                            #          style={'padding': 10, 'flex': 1})
                        ],
                    ),
                    html.Div(className="nine columns", children=[dcc.Graph(id="scatter-graph", style={"height": "100vh"})]),
                ]
            ),
            html.Div(id="div-scatter-click-image", style={"padding": 10, "flex": 1}),
            dcc.Store(id="column-current-values"),
        ],
    )


def create_wrong_cases_section(metric_options: List[Hashable], data_table_records, data_table_cols) -> html.Div:
    """Creates the Wrong cases section in the app's layout.

    Args:
        metric_options: which columns to consider as target variables (metric
        columns in the dataframe)
        data_table_records: records to be displayed inside the data-table
        data_table_cols: columns of the data-table

    Returns:
        html.Div to put in the ``app.layout``
    """
    return html.Div(
        className="row background",
        style={"padding": "10px"},
        children=[
            html.Div(
                className="three columns",
                children=[
                    _create_dropdown(options=metric_options, label="Segmentation Metric", id_prefix="wrong_metric"),
                    html.Button("back", n_clicks=0, id="wrong_back-button", style={"marginLeft": "10px"}),
                    html.Button("next", n_clicks=0, id="wrong_next-button", style={"marginLeft": "10px"}),
                ],
            ),
            html.Div(
                className="nine columns",
                children=[html.Div(children=["This part is not updated yet"], id="wrong-case-image")],
            ),
            html.Div(
                children=[
                    html.Label("Slice specifications"),
                    dash_table.DataTable(
                        data=data_table_records,
                        columns=data_table_cols,
                        id="specifications-table",
                        style_table={"overflowY": "scroll"},
                    ),
                ],
                style={"padding": 10, "flex": 1},
            ),
            # dcc.Store(id='column-current-values')
        ],
    )


def create_compare_models_section(metric_options: List[Hashable]) -> html.Div:
    """Creates the Compare models section in the app's layout.
    Args:
        metric_options: which columns to consider as target variables (metric
        columns in the dataframe)

    Returns:
        html.Div to put in the ``app.layout``
    """
    return html.Div(
        className="row background",
        style={"padding": "10px"},
        children=[
            html.Div(
                className="three columns",
                children=[
                    _create_dropdown(options=metric_options, label="Segmentation Metric", id_prefix="compare_metric"),
                    _create_input(
                        placeholder="Enter second_model\
                                                     path...",
                        id_prefix="second_model",
                        type="text",
                        style={"display": "inline-block", "padding": "px", "margin-left": "10px", "width": "99%"},
                    ),
                    html.Div(
                        children=[
                            dcc.RadioItems(
                                options=["With Explainability", "Without Explainability"],
                                value="Without Explainability",
                                inline=True,
                                id="compare_model-radiobutton",
                                style={"display": "flex", "justify-content": "space-around"},
                            ),
                            _create_input(
                                placeholder="Enter your second model bottleneck name",
                                value=None,
                                id_prefix="second_bottleneck",
                                style={"display": "none"},
                            ),
                        ],
                        id="compare_model-radiobutton-container",
                    ),
                    html.Button("enter", id="compare_enter-button", n_clicks=None, style={"marginLeft": "10px", "width": "100%"}),
                    # _create_dropdown(options=metric_options, label="Segmentation Metric", id_prefix="compare_metric"),
                    html.Div(
                        children=[
                            html.Button("back", id="compare_back-button", n_clicks=0, style={"width": "49%"}),
                            html.Button("next", id="compare_next-button", n_clicks=0, style={"width": "49%"}),
                        ],
                        style={"marginLeft": "10px", "display": "flex", "justify-content": "space-between", "width": "100%"},
                    ),
                ],
            ),
            html.Div(
                className="nine columns",
                children=[
                    html.Div(
                        children=["Enter your second model path to see the comparison between baseline and the imported model"],
                        id="compare-model-image",
                    )
                ],
            ),
            dcc.Store(id="column-current-values"),
        ],
    )


def _create_dropdown(options: List, label: str, id_prefix: str) -> html.Div:
    """Creates a good-looking dropdown.

    Args:
        options: drop-down's options
        label: drop-down's label
        id_prefix: will be used in drop-down's id -> id of the drop-down
        component will be {id_prefix}-dropdown

    Returns:
        html.Div to put in the ``app.layout``
    """
    return html.Div(
        children=[
            html.Label(
                label,
                id=f"{id_prefix}-label",
            ),
            dcc.Dropdown(
                options=options,
                id=f"{id_prefix}-dropdown",
                value=options[0],
                searchable=False,
                clearable=False,
                # placeholder='Select a target column'
            ),
        ],
        style={"padding": 10, "flex": 1},
    )


def _create_input(placeholder: str, id_prefix: str, value: str = "", style: dict = None, type: str = "text", required: bool = True) -> dcc.Input:
    """Creates a good-looking inputbox.

    Args:
        placeholder: input placeholder
        id_prefix: will be used in input's id -> id of the input component
                   will be {id_prefix}-input
        type: type of the input
        required: is this input required or not

    Returns:
        dcc.Input to put in the ``app.layout``
    """
    if style is None:
        style = {"padding": 10, "flex": 1}
    return dcc.Input(placeholder=placeholder, type=type, value=value, id=f"{id_prefix}-input", required=required, style=style)


def _create_describe_data_table(records: List[Dict[str, Any]], cols: List[Dict[str, Any]]) -> html.Div:
    """Create data-table for a described data-frame.

    Args:
        records: the rows of the data-table. you can generate this using
        ``df.to_dict('records')``
        cols: the column names of the data-table.

    Returns:
        html.Div to put in the ``app.layout``
    """
    return html.Div(
        children=[
            html.Label("Described DataFrame", id="describe_table-label"),
            dash_table.DataTable(data=records, columns=cols, id="describe-table", page_current=0, page_size=10, page_action="custom"),
        ],
        style={"padding": 10, "flex": 1},
    )
