import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from kym.hhd.utils import (
    get_records_and_columns,
    infer_false_options,
    infer_metric,
    infer_target_cohort_options,
)

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
    print("len data = ", len(data))
    print("len columns = ", len(columns))
    model_meta = pd.DataFrame(data=[data], columns=columns)
    return model_meta


@pytest.mark.unit
class TestUtils:
    """a set of unit test to check kym package utils.

    Methods:
        1. test_infer_false_options()
        2. test_infer_metric()
        3. test_infer_target_cohort_options()
        4. test_get_records_and_columns()
        5. test_numpy_to_b64()
    """

    def test_infer_false_options(self, model_meta_df):
        """Testing the type of target_options including FN or FP."""
        target_options = infer_false_options(model_meta_df)
        assert isinstance(target_options, list), f"incorrect output type: {type(target_options)}"

    def test_infer_metric(self, model_meta_df):
        """Testing the type and dtype of target_options including metrics."""
        target_options = infer_metric(model_meta_df)
        # test output type
        assert isinstance(target_options, list), f"incorrect output type: {type(target_options)}"
        # test output dtype
        # target_options = np.ndarray(target_options)
        df_dtypes = []
        for _, j in model_meta_df[target_options].dtypes.iteritems():
            df_dtypes = np.append(df_dtypes, j)
        allowed_formats = [np.float16, np.float32, np.float64]
        if all(item in allowed_formats for item in df_dtypes):
            pass
        else:
            raise ValueError("Some formats in the df are not allowed")

    @pytest.mark.parametrize("unique_limit", [10])
    def test_infer_target_cohort_options(self, model_meta_df, unique_limit):
        """Testing the type and dtype of target_options and cohort_options."""
        target_options, cohort_options = infer_target_cohort_options(model_meta_df)
        # test output type
        assert isinstance(
            target_options, list
        ), f"incorrect target_options type:\
                               {type(target_options)}"
        assert isinstance(
            cohort_options, list
        ), f"incorrect cohort_options type:\
                               {type(cohort_options)}"

        # test lenght of unique values of df[target_options]
        wrong_unique_length = False
        for target_option in target_options:
            if len(np.unique(model_meta_df[target_option].dropna().values)) <= unique_limit:
                wrong_unique_length = True
                break

        if wrong_unique_length:
            raise AssertionError(
                f"target_options unique limit error;\
                                   len(np.unique(model_meta[{target_option}]))\
                                   <= {unique_limit}"
            )

        # test lenght of unique values of df[cohort_options]
        wrong_unique_length = False
        for cohort_option in cohort_options:
            if len(np.unique(model_meta_df[cohort_option].dropna().values)) > unique_limit:
                wrong_unique_length = True
                break
        if wrong_unique_length:
            raise AssertionError(
                f"cohort_option unique limit error;\
                                   len(np.unique(model_meta[{cohort_option}]))\
                                    > {unique_limit}"
            )

    def test_get_records_and_columns(self, model_meta_df):
        """Testing the type and dtype of target_options and cohort_options."""
        records, cols = get_records_and_columns(model_meta_df.groupby("TP-1-0.05")["IoU-class2"].describe().T)
        # test output type
        assert isinstance(records, list), f"incorrect records type: {type(records)}"
        assert isinstance(cols, list), f"incorrect cols type: {type(cols)}"
