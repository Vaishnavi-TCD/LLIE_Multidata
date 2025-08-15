# my_brisque.py
# Local BRISQUE wrapper using OpenCV's QualityBRISQUE
# Requires opencv-contrib-python and the two model files:
#   - brisque_model_live.yml
#   - brisque_range_live.yml
# Download from:
# https://github.com/opencv/opencv_contrib/blob/master/modules/quality/samples/brisque_model_live.yml
# https://github.com/opencv/opencv_contrib/blob/master/modules/quality/samples/brisque_range_live.yml

import os
import cv2
import numpy as np

class BRISQUEScore:
    def __init__(self,
                 model_file: str = "brisque_model_live.yml",
                 range_file: str = "brisque_range_live.yml"):
        """
        Thin wrapper around OpenCV's QualityBRISQUE.
        model_file and range_file should be placed in the same directory as this script,
        or you can specify full paths.
        """
        # verify files exist
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"BRISQUE model file not found: {model_file}")
        if not os.path.isfile(range_file):
            raise FileNotFoundError(f"BRISQUE range file not found: {range_file}")

        # create the scorer
        self.scorer = cv2.quality.QualityBRISQUE_create(model_file, range_file)

    def score(self, img: np.ndarray) -> float:
        """
        Compute BRISQUE score on an image.

        Args:
            img: HxWx3 uint8 image in RGB or BGR.
        Returns:
            float: lower BRISQUE score indicates better quality.
        """
        # OpenCV expects BGR
        if img.ndim == 3 and img.shape[2] == 3:
            # assume input is RGB, convert to BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img

        # compute returns a scalar or tuple; take first element
        res = self.scorer.compute(img_bgr)
        if isinstance(res, (tuple, list)):
            val = res[0]
        else:
            val = res
        # convert to float
        return float(val)
