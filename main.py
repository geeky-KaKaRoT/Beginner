# imports
import os
import math
import argparse
import subprocess
import sys

try:
    import cv2 as cv
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'opencv-python'])
finally:
    import cv2 as cv

try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'numpy'])
finally:
    import numpy as np


if __name__ == '__main__':
    img_rgb = cv.imread('test_image.png')
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread('template_image.png', 0)
    w, h = template.shape[::-1]

    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv.imwrite('Result_matchtemplate.png', img_rgb)

