import gc
import io
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
import tensorflow.keras as tfk
from aimedic.preprocessing import BasePreprocessorV2
from aimedic.preprocessing.base import PreprocessorInterface
from aimedic.xai.seg_explainers import ClassRoI, RoIBase, SegGradCAM
from PIL import Image, ImageFilter

plt.switch_backend("agg")


class HHDVisualizer:
    """HHD Visualizer

    ...
    Attributes
    ----------
        data_registry_: path to data registry
        hyper_mask_color_: rgb color for visualizing hyper masks
        hypo_mask_color_: rgb color for visualizing hypo masks
        bottleneck_layer: the name of the bottleneck layer, which is the last conv layer in the encoder

    Methods
    -------
        get_overlaid_mask_on_slice()
    """

    def __init__(self, data_registry: Path, bottleneck_layer: str):
        self.data_registry_ = data_registry
        self.hyper_mask_color_ = (255, 0, 0)
        self.hypo_mask_color_ = (0, 255, 0)
        self.bottleneck_layer = bottleneck_layer

    def get_overlaid_mask_on_slice(
        self,
        sample: pd.Series,
        preprocessor: PreprocessorInterface = BasePreprocessorV2(),
        model: Optional[tfk.Model] = None,
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        """Overlays the mask on the preprocessed slice in ``brain`` window.

        Notes:
            - if ``model`` is not None, overlaid predictions will be returned
              as well as the overlaid mask.
            - the ``sample`` is assumed to have these fields:
                - ``DataSource``
                - ``SeriesInstanceUID``
                - ``SliceIndex``
                - ``LabelingJob``
                - ``MaskName``

        Args:
            sample: the slice row in the model-meta data-frame
            preprocessor: which version of the preprocessing to use
            model: trained model

        Returns:
            (overlaid_mask, overlaid_prediction, overlaid_xai_hyper, overlaid_xai_hypo)
                - overlaid mask on the image
                - overlaid predictions on the image
                - overlaid seg-grad-cam for hyper
                - overlaid seg-grad-cam for hypo
        """
        ds_name = sample.DataSource
        siuid = sample.SeriesInstanceUID
        slice_ind = int(sample.SliceIndex)
        ljob = sample.LabelingJob
        mask_name = sample.MaskName

        # series_path = self.data_registry_ / "datasources" / ds_name / siuid
        # mask_path = self.data_registry_ / "tasks" / "HHD" / f"masks-{ljob}" / mask_name

        series_path = os.path.join(self.data_registry_, "datasources", ds_name, siuid)
        mask_path = os.path.join(self.data_registry_, "tasks", "HHD", f"masks-{ljob}", mask_name)

        preprocessed, _, _ = preprocessor.do(series_path)

        slice_npy = preprocessed[slice_ind]
        del preprocessed
        gc.collect()
        slice_image = Image.fromarray(slice_npy[:, :, 0])
        # Add gt masks on the image
        hyper_mask, hypo_mask = self._load_mask(mask_path)
        overlaid = self._overlay_on_pil_image(slice_image, hyper_mask, self.hyper_mask_color_)
        overlaid = self._overlay_on_pil_image(overlaid, hypo_mask, self.hypo_mask_color_)
        overlaid_pred = None
        overlaid_xai_hyper = None
        overlaid_xai_hypo = None
        if model is not None:
            hyper_pred, hypo_pred = self._predict(model, np.expand_dims(slice_npy, axis=0))
            # Overlay predictions
            overlaid_pred = self._overlay_on_pil_image(slice_image, hyper_pred, self.hyper_mask_color_)
            overlaid_pred = self._overlay_on_pil_image(overlaid_pred, hypo_pred, self.hypo_mask_color_)
            overlaid_xai_hyper = self._overlay_xai(slice_npy, model, class_index=1, conv_layer_name=self.bottleneck_layer)
            overlaid_xai_hypo = self._overlay_xai(slice_npy, model, class_index=2, conv_layer_name=self.bottleneck_layer)

        return overlaid, overlaid_pred, overlaid_xai_hyper, overlaid_xai_hypo

    @staticmethod
    def _overlay_on_pil_image(image: Image, mask: np.ndarray, color: tuple = (255, 0, 0)) -> Image:
        """Draw edges of contour 'c' from segmented image 's' onto 'm' in colour 'RGB'.

        Args:
            image: the original PIL Image
            mask: np array with the same shape of image.shape which only
            contains (0, 1)
        """
        img = np.array(image.convert("RGB"))
        msk = Image.fromarray(mask.astype("uint8"))

        # Fill contour "c" with white, make all else black
        this_contour = msk.point(lambda p: p == 1 and 255)
        # DEBUG: thisContour.save(f"interim{c}.png")

        # Find edges of this contour and make into Numpy array
        this_edges = this_contour.filter(ImageFilter.FIND_EDGES)
        this_edges_n = np.array(this_edges)

        # Paint locations of found edges in color "RGB" onto "main"
        img[np.nonzero(this_edges_n)] = color
        return Image.fromarray(img)

    def _overlay_xai(self, slice_npy: np.ndarray, model: tfk.Model, class_index: int, conv_layer_name: str) -> PIL.Image:
        """Draw seg-grad-cam on top of the input image.

        Args:
            slice_npy: np array of shape(512, 512, 3) which is in range(0, 1)
            model: tfk saved model
            class_index: integer number which indicates the required class.
                         in this case 1 indicates hyper and 2 indicates hypo density
            conv_layer_name: the name of the bottleneck layer, which is the last conv layer in the encoder

        Returns:
            pil_im: PIL.Image of image with seg-grad-cam heatmap on it if the user enters any conv_layer_name
                    and None otherwise
        """
        if conv_layer_name is not None:
            clsroi = ClassRoI(model=model, image=slice_npy, class_index=class_index)
            seggradcam = SegGradCAM(model=model, conv_layer_name=conv_layer_name, roi=clsroi, normalize=True, abs_w=False, posit_w=False)
            if class_index == 1:
                title = "hyper_explainer"
            elif class_index == 2:
                title = "hypo_explainer"
            else:
                raise ValueError
            roi = seggradcam.roi
            if len(np.where(roi.roi == 1)[0]) != 0:
                cam = seggradcam.make_heatmap(slice_npy, class_index).astype(np.float16)
                pil_im = self._seggradcam_plot(image=slice_npy, roi=roi, cam=cam, title=title)
            else:
                pil_im = None
        else:
            pil_im = None
        return pil_im

    @staticmethod
    def _predict(model: tf.keras.Model, preprocessed_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns model hyper/hypo predictions.

        Args:
        model: tfk saved model
        preprocessed_input:np array of shape(1, 512, 512, 3) which is in range(0, 1)
        """
        predicted = tf.argmax(model(preprocessed_input, training=False), axis=-1).numpy()
        predicted_hyper = np.zeros_like(predicted)
        predicted_hyper[np.where(predicted == 1)] = 1
        predicted_hypo = np.zeros_like(predicted)
        predicted_hypo[np.where(predicted == 2)] = 1
        return predicted_hyper[0], predicted_hypo[0]

    @staticmethod
    def _load_mask(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Reads raw mask which could contain 0, 1, 2 and returns separate hyper mask and hypo mask
        Args:
            path: path to raw mask

        Returns:
            hyper_mask: hyper ground-truth corresponds to all ones in the mask
            hypo_mask: hypo ground-truth corresponds to all twos in the mask
        """
        mask = np.load(path)

        hyper_mask = np.zeros_like(mask)
        hyper_mask[np.where(mask == 1)] = 1
        #     hyper_mask = tf.squeeze(tf.image.resize(tf.expand_dims(hyper_mask, axis=-1),
        #     (256, 256))).numpy().astype(bool)
        hypo_mask = np.zeros_like(mask)
        hypo_mask[np.where(mask == 2)] = 1
        #     hyper_mask = tf.squeeze(tf.image.resize(tf.expand_dims(hypo_mask, axis=-1),
        #     (256, 256))).numpy().astype(bool)
        return hyper_mask, hypo_mask

    def _seggradcam_plot(self, image: np.ndarray, roi: RoIBase(), cam: SegGradCAM.make_heatmap, title: str) -> PIL.Image:
        """Returns PIL image of the seg-grad-cam heatmap.

        Args:
            image: numpy array of the slice image
            roi: region of interest object
            cam: seg-grad-cam heatmap which takes image and class_index as arguments
            title: plt plot title
        """
        f = plt.figure()
        plt.imshow(image, vmin=0, vmax=1, cmap="gray")

        # class contour
        X, Y = roi.meshgrid()
        roi_contour1 = roi.roi

        plt.contour(X, Y, roi_contour1, colors="pink")

        plt.title(title, fontsize=23)
        plt.imshow(cam, cmap="jet", alpha=0.6)  # vmin=0,vmax=1,
        jet = plt.colorbar(fraction=0.046, pad=0.04, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        jet.set_label(label="Importance", size=23)
        jet.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], size=23)
        figure = plt.gcf()
        image = self._fig2img(figure)
        # To release memory from plt figure
        f.clear()
        plt.close(f)
        return image

    @staticmethod
    def _fig2img(fig: plt.gcf) -> PIL.Image:
        """Converts a Matplotlib figure to a PIL Image
        Args:
            fig: matplotlib figure
        """
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img
