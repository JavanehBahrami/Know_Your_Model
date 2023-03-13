from contextvars import copy_context

import pandas as pd
import pytest
import tensorflow as tf
from dash._callback_context import context_value
from dash._utils import AttributeDict

from kym.hhd.dashboard import Controller, ErrorPlotType

tfk = tf.keras


@pytest.fixture(scope="module")
# @pytest.fixture
def model_meta_df():
    columns = [
        "IoU-class2",
        "Dice-class2",
        "TP-1-0.05",
        "TN-1-0.05",
        "FP-1-0.05",
        "FN-1-0.05",
        "TP-2-0.05",
        "TN-2-0.05",
        "FP-2-0.05",
        "FN-2-0.05",
        "TP-1-0.1",
        "TN-1-0.1",
        "FP-1-0.1",
        "FN-1-0.1",
        "TP-2-0.1",
        "TN-2-0.1",
        "FP-2-0.1",
        "FN-2-0.1",
        "TP-1-0.15",
        "TN-1-0.15",
        "FP-1-0.15",
        "FN-1-0.15",
        "TP-2-0.15",
        "TN-2-0.15",
        "FP-2-0.15",
        "FN-2-0.15",
        "TP-1-0.2",
        "TN-1-0.2",
        "FP-1-0.2",
        "FN-1-0.2",
        "TP-2-0.2",
        "TN-2-0.2",
        "FP-2-0.2",
        "FN-2-0.2",
        "TP-1-0.25",
        "TN-1-0.25",
        "FP-1-0.25",
        "FN-1-0.25",
        "TP-2-0.25",
        "TN-2-0.25",
        "FP-2-0.25",
        "FN-2-0.25",
        "TP-1-0.3",
        "TN-1-0.3",
        "FP-1-0.3",
        "FN-1-0.3",
        "TP-2-0.3",
        "TN-2-0.3",
        "FP-2-0.3",
        "FN-2-0.3",
        "TP-1-0.35",
        "TN-1-0.35",
        "FP-1-0.35",
        "FN-1-0.35",
        "TP-2-0.35",
        "TN-2-0.35",
        "FP-2-0.35",
        "FN-2-0.35",
        "TP-1-0.4",
        "TN-1-0.4",
        "FP-1-0.4",
        "FN-1-0.4",
        "TP-2-0.4",
        "TN-2-0.4",
        "FP-2-0.4",
        "FN-2-0.4",
        "TP-1-0.45",
        "TN-1-0.45",
        "FP-1-0.45",
        "FN-1-0.45",
        "TP-2-0.45",
        "TN-2-0.45",
        "FP-2-0.45",
        "FN-2-0.45",
        "TP-1-0.5",
        "TN-1-0.5",
        "FP-1-0.5",
        "FN-1-0.5",
        "TP-2-0.5",
        "TN-2-0.5",
        "FP-2-0.5",
        "FN-2-0.5",
        "TP-1-0.55",
        "TN-1-0.55",
        "FP-1-0.55",
        "FN-1-0.55",
        "TP-2-0.55",
        "TN-2-0.55",
        "FP-2-0.55",
        "FN-2-0.55",
        "TP-1-0.6",
        "TN-1-0.6",
        "FP-1-0.6",
        "FN-1-0.6",
        "TP-2-0.6",
        "TN-2-0.6",
        "FP-2-0.6",
        "FN-2-0.6",
        "TP-1-0.65",
        "TN-1-0.65",
        "FP-1-0.65",
        "FN-1-0.65",
        "TP-2-0.65",
        "TN-2-0.65",
        "FP-2-0.65",
        "FN-2-0.65",
        "TP-1-0.7",
        "TN-1-0.7",
        "FP-1-0.7",
        "FN-1-0.7",
        "TP-2-0.7",
        "TN-2-0.7",
        "FP-2-0.7",
        "FN-2-0.7",
        "TP-1-0.75",
        "TN-1-0.75",
        "FP-1-0.75",
        "FN-1-0.75",
        "TP-2-0.75",
        "TN-2-0.75",
        "FP-2-0.75",
        "FN-2-0.75",
        "TP-1-0.8",
        "TN-1-0.8",
        "FP-1-0.8",
        "FN-1-0.8",
        "TP-2-0.8",
        "TN-2-0.8",
        "FP-2-0.8",
        "FN-2-0.8",
        "TP-1-0.85",
        "TN-1-0.85",
        "FP-1-0.85",
        "FN-1-0.85",
        "TP-2-0.85",
        "TN-2-0.85",
        "FP-2-0.85",
        "FN-2-0.85",
        "TP-1-0.9",
        "TN-1-0.9",
        "FP-1-0.9",
        "FN-1-0.9",
        "TP-2-0.9",
        "TN-2-0.9",
        "FP-2-0.9",
        "FN-2-0.9",
        "TP-1-0.95",
        "TN-1-0.95",
        "FP-1-0.95",
        "FN-1-0.95",
        "TP-2-0.95",
        "TN-2-0.95",
        "FP-2-0.95",
        "FN-2-0.95",
        "SOPInstanceUID",
        "XnatProjectID",
        "XnatSubjectID",
        "XnatExperimentID",
        "StandardSIUID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Modality",
        "BodyPartExamined",
        "Manufacturer",
        "ManufacturerModelName",
        "PatientAge",
        "PatientSex",
        "PatientID",
        "SliceThickness",
        "PixelSpacing",
        "SamplesPerPixel",
        "ImageType",
        "ConvolutionKernel",
        "NumberOfSlices",
        "Labeler",
        "DataSource",
        "SliceIndex",
        "RelativeLocation",
        "Normal",
        "ConflictBetweenNumberOfSlicesAndLabeledSlices",
        "HyperDensity",
        "HypoDensity",
        "MaskName",
        "LabelingJob",
        "Split",
        "TFRecordFileName",
        "hyperdensity",
        "hypodensity",
        "hemorrhage",
    ]
    data = [
        2.7777778e-08,
        2.7777778e-08,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        "1.2.392.200036.9116.2.6.1.48.1211476691.1461116016.14950",
        "CTB-P6-Reviewer1",
        "anonymous-2",
        771626,
        "1.2.392.200036.9116.2.6.1.48.1211476691.1461116004.588820",
        "1.2.392.200036.9116.2.6.1.48.1211476691.1461115693.987894",
        "1.2.392.200036.9116.2.6.1.48.1211476691.1461116004.588820",
        "CT",
        "HEAD",
        "Head 8.0",
        "TOSHIBA",
        "Aquilion",
        64,
        "M",
        771626,
        8,
        [0.468, 0.468],
        1,
        ["ORIGINAL", "PRIMARY", "AXIAL"],
        "FC23",
        16,
        "Reviewer1",
        "jahanbakhshi-p3",
        0,
        0,
        True,
        False,
        False,
        False,
        "1.2.392.200036.9116.2.6.1.48.1211476691.1461116004.588820_0.npy",
        "HHD-CTB-P6-Reviewer1",
        "evaluation",
        "jahanbakhshi-p3_1.2.392.200036.9116.2.6.1.48.1211476691.1461116004.588820_0.tfrecord",
        False,
        False,
        False,
    ]
    #     print("len data = ", len(data))
    #     print("len columns = ", len(columns))
    model_meta = pd.DataFrame(data=[data, data, data, data], columns=columns)
    #     print('len meta_data.index = ', len(model_meta.index))
    return model_meta


@pytest.mark.unit
class TestCallbacks:
    """a set of unit test to check kym package callbacks.

    Methods:
        1. test_update_wrong_cases_graph_callback()
        2. test_update_error_graph_callback()
    """

    @pytest.mark.usefixtures("model_meta_df")
    def test_update_wrong_cases_graph_callback(self, model_meta_df):
        """Testing callbacks for next and back buttons and metric dropdown."""
        controller = Controller(model_meta_=model_meta_df, model_=None, data_registry_=None)

        def run_callback(metric, n_clicks_next, n_clicks_back, triggered_id):
            context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": triggered_id}]}))
            return controller.update_wrong_cases_graph(
                metric=metric, n_clicks_next=n_clicks_next, n_clicks_back=n_clicks_back, bottleneck_layer=None, _test_mode=True
            )

        # 1-Check if the wrong_buttons reset after clicking wrong_dropdown
        ctx = copy_context()
        output = ctx.run(run_callback, metric="Dice-class2", n_clicks_next=3, n_clicks_back=2, triggered_id="wrong_metric-dropdown")
        assert (output[1], output[2]) == (0, 0), "wrong_buttons did not reset after clicking wrong_dropdown"

        # 2-Check if the wrong_buttons reset after clicking compare_dropdown
        ctx = copy_context()
        output = ctx.run(run_callback, metric="Dice-class2", n_clicks_next=3, n_clicks_back=2, triggered_id="compare_metric-dropdown")
        assert (output[1], output[2]) == (3, 2), "wrong_buttons reset after clicking compare_dropdown"

        # 3-Check if the next button updates after clicking wrong_back
        output = ctx.run(run_callback, metric="Dice-class2", n_clicks_next=3, n_clicks_back=2, triggered_id="wrong_back-button")
        assert (output[1], output[2]) == (2, 2), "n_clicks for the next button did not update correctly after clicking next button"

        # 4-Check if the next button updates after clicking compare_next
        output = ctx.run(run_callback, metric="Dice-class2", n_clicks_next=3, n_clicks_back=2, triggered_id="compare_next-button")
        assert (output[1], output[2]) == (3, 2), "n_clicks for the next button updates with `compare` section next button"

        # 5-Check if the back button updates after clicking compare_back
        output = ctx.run(run_callback, metric="Dice-class2", n_clicks_next=3, n_clicks_back=2, triggered_id="compare_back-button")
        assert (output[1], output[2]) == (3, 2), "n_clicks for the back button updates with `compare` section back button"

    @pytest.mark.usefixtures("model_meta_df")
    def test_update_compare_models_graph_callback(self, model_meta_df):
        """Testing callbacks for next and back buttons and metric dropdown."""
        controller = Controller(model_meta_=model_meta_df, model_=None, data_registry_=None)

        def run_callback(n_clicks_enter, metric, radio_value, n_clicks_next, n_clicks_back, triggered_id):
            context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": triggered_id}]}))
            return controller.update_compare_models_graph(
                children=[],
                n_clicks_enter=n_clicks_enter,
                model_path=None,
                radio_value=radio_value,
                metric=metric,
                n_clicks_next=n_clicks_next,
                n_clicks_back=n_clicks_back,
                bottleneck_layer=None,
                second_bottleneck=None,
                _test_mode=True,
            )

        # 1- Check if the compare next/back buttons reset after compare enter button
        ctx = copy_context()
        output = ctx.run(
            run_callback,
            n_clicks_enter=2,
            metric="Dice-class2",
            radio_value="With Explainability",
            n_clicks_next=3,
            n_clicks_back=2,
            triggered_id="compare_enter-button",
        )
        assert (output[1], output[2]) == (0, 0), "compare next/back buttons did not reset after clicking compare enter button"

        # 2- Check if the compare next button correctly updates after clicking compare back button
        ctx = copy_context()
        output = ctx.run(
            run_callback,
            n_clicks_enter=2,
            metric="Dice-class2",
            radio_value="With Explainability",
            n_clicks_next=3,
            n_clicks_back=2,
            triggered_id="compare_back-button",
        )
        assert (output[1], output[2]) == (2, 2), "compare next button did not correctly update after clicking compare back button"

        # 3- Check if the next/back/enter buttons reset after clicking compare dropdown
        output = ctx.run(
            run_callback,
            n_clicks_enter=2,
            metric="Dice-class2",
            radio_value="With Explainability",
            n_clicks_next=3,
            n_clicks_back=2,
            triggered_id="compare_metric-dropdown",
        )
        assert (output[1], output[2], output[3]) == (0, 0, 0), "next/back/enter buttons did not reset after clicking compare dropdown"

        # 4- Check if the buttons did not update after clicking compare_next
        output = ctx.run(
            run_callback,
            n_clicks_enter=2,
            metric="Dice-class2",
            radio_value="With Explainability",
            n_clicks_next=3,
            n_clicks_back=2,
            triggered_id="compare_next-button",
        )
        assert (output[1], output[2], output[3]) == (3, 2, 2), "next/back/enter buttons update after clicking compare next button"

        # 5- Check if compare enter button resets after changing model path
        output = ctx.run(
            run_callback,
            n_clicks_enter=3,
            metric="Dice-class2",
            radio_value="With Explainability",
            n_clicks_next=3,
            n_clicks_back=2,
            triggered_id="second_model-input",
        )
        assert (output[1], output[2], output[3]) == (3, 2, 0), "enter button did not reset after changing model path"

    # todo: review this test later
    @pytest.mark.usefixtures("model_meta_df")
    def test_update_error_graph_callback(self, model_meta_df):
        """Testing callback for updating error_graph table."""
        controller = Controller(model_meta_=model_meta_df, model_=None, data_registry_=None)
        _, returned_records, returned_cols = controller.update_error_graph("IoU-class2", "FP-1-0.05", ErrorPlotType.HISTOGRAM)

        actual_cols = [{"name": "0", "id": "0"}, {"name": "FP-1-0.05", "id": "FP-1-0.05"}]
        actual_records = [
            {0: 4.0, "FP-1-0.05": "count"},
            {0: 3e-08, "FP-1-0.05": "mean"},
            {0: 0.0, "FP-1-0.05": "std"},
            {0: 3e-08, "FP-1-0.05": "min"},
            {0: 3e-08, "FP-1-0.05": "25%"},
            {0: 3e-08, "FP-1-0.05": "50%"},
            {0: 3e-08, "FP-1-0.05": "75%"},
            {0: 3e-08, "FP-1-0.05": "max"},
        ]

        assert actual_cols == returned_cols, "The columns returned for create_describe_data_table are not equal to actual columns."

        assert actual_records == returned_records, "The records returned for create_describe_data_table are not equal to actual records."
