#!/usr/bin/env python

"""
Required:
pip install numpy
pip install opencv-python
pip install img2pdf

Optional (speedup):
pip install PyTurboJPEG
sudo apt install libturbojpeg

Inspiration: https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/

Debugging:
pip install matplotlib
pip install line_profiler
    kernprof -l -v crop.py
"""

import os
import math
import argparse
import pathlib
import sys
import shutil

import numpy as np
try:
    import cv2.cv2 as cv2
except ImportError:
    import cv2
import img2pdf

from utils import show_images, time_ctx


try:
    profile
except NameError:
    def profile(f):
        return f


@profile
def resize(img, factor):
    return cv2.resize(img, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def convex_area(contour):
    hull = cv2.convexHull(contour)
    return cv2.contourArea(hull)


@profile
def get_contour_canny(img, ksize_smooth=7, _debug=False):
    scale = 500 / img.shape[0]
    resized = resize(img, scale)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurr = cv2.medianBlur(gray, ksize_smooth)
    opened = cv2.morphologyEx(blurr, cv2.MORPH_OPEN, np.ones((ksize_smooth,)*2, np.uint8))
    edged = cv2.Canny(opened, 40, 200)
    closed_edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((ksize_smooth,)*2, np.uint8))

    contours = cv2.findContours(closed_edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    #hulls = map(cv2.convexHull, contours)
    raw_contour = max(contours, key=convex_area)
    contour = np.round(raw_contour.reshape(-1, 2) / scale).astype(int)

    if _debug:
        show_images([opened, edged, closed_edged, with_contour(img, contour)])

    return contour


def to_mask(contour, img):
    res = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(res, [contour], 0, 255, -1)
    return res


def with_contour(img, contour):
    output = img.copy()
    cv2.drawContours(output, [np.array(contour).astype(int)], 0, (0, 255, 0), 10)
    return output


@profile
def get_cropped_min_area_rect(contour, img):
    rect = cv2.minAreaRect(contour)
    width = rect[1][1]
    height = rect[1][0]

    # The lowest point of the rectangle will always be the first element.
    # All other points will follow in a clockwise-direction.
    # We make sure that the first element is always the top left corner.
    corners = cv2.boxPoints(rect)
    # We rotate until the closest point to the origin is at the front
    while np.argmin(corners.sum(axis=1)) != 0:
        corners = np.roll(corners, -1, axis=0)
        height, width = width, height

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (math.ceil(width), math.ceil(height)), flags=cv2.INTER_CUBIC)


@profile
def crop_page(img, contour):
    mask = to_mask(contour, img)
    page = cv2.bitwise_or(img, img, mask=mask)
    crop = get_cropped_min_area_rect(contour, page)
    return crop


@profile
def correct_brightness(img, threshold=0.05, min_white=150, _debug=False):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # calculate white point
    hist = cv2.calcHist([yuv], channels=[0], mask=None, histSize=[256], ranges=[0,255])[:,0]
    hist /= (img.shape[0] * img.shape[1])
    hist[0] = 0
    cutoff = np.max(hist) * threshold
    white = np.max(np.arange(256)[hist > cutoff])
    white = max(white, min_white)

    # adjust intensity
    table = []
    for val in np.arange(0, 256):
        val_float = val / 255.0
        val_white = val_float * 255 / white
        val_gamma = val_white ** (255 / white) ** 0.75
        val_clipped = np.clip(val_gamma * 255, 0, 255)
        table.append(val_clipped)
    table = np.array(table, np.uint8)
    yuv[:,:,0] = cv2.LUT(yuv[:,:,0], table)
    output = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    if _debug:
        import matplotlib.pyplot as plt
        plt.plot(hist)
        plt.show()
        print(hist)

    return output


@profile
def extract_page(img):
    contour = get_contour_canny(img)
    page = crop_page(img, contour)
    page = correct_brightness(page)
    return page


class JpegCodec:
    def __init__(self):
        self.turbo = None
        try:
            import turbojpeg
        except ImportError:
            print("INFO: libjpeg-turbo is not installed. Install with pip install PyTurboJPEG")
        else:
            try:
                self.turbo = turbojpeg.TurboJPEG()
            except RuntimeError as exp:
                print("WARNING:", exp)

    def imread(self, path):
        if self.turbo:
            with open(path, "rb") as f:
                return self.turbo.decode(f.read())
        else:
            return cv2.imread(path)

    def imwrite(self, path, img, quality=70):
        if self.turbo:
            with open(path, "wb") as f:
                f.write(self.turbo.encode(img, quality=quality))
        else:
            cv2.imwrite(path, img, (cv2.IMWRITE_JPEG_QUALITY, quality))


@profile
def convert_to_pdf(input_files, output_files, pdf_path):
    codec = JpegCodec()
    for i, input_path, output_path in zip(range(len(input_files)), input_files, output_files):
        with time_ctx("page {}/{}".format(i + 1, len(input_files))):
            img = codec.imread(input_path)
            page = extract_page(img)
            codec.imwrite(output_path, page)

    with time_ctx("pdf"):
        with open(pdf_path, "wb") as f:
            f.write(img2pdf.convert(output_files))


def delete_path(path, folder=True, ask_overwrite=True):
    """Ask the user to delete the folder or file. Otherwise exit."""
    if path.exists():
        if (folder and not path.is_dir()) or (not folder and not path.is_file()):
            print("FATAL: output path '{}' already exists. Please delete it.".format(path))
            sys.exit(2)
        else:
            if ask_overwrite and input("Warning: output path '{}' exists. Delete? [yes/No] ".format(path)).lower() \
                    not in ["y", "yes"]:
                sys.exit(1)
            if folder:
                shutil.rmtree(str(path))
            else:
                path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder with *.jpg files.")
    parser.add_argument("-y", "--yes", help="overwrite files by defailt.", action="store_true")
    args = parser.parse_args()

    # calculate all paths
    folder = pathlib.Path(args.folder)
    inputs = list(sorted(folder.glob("*.[jJ][pP][gG]")))
    output_folder = folder.parent / ("crop_" + folder.name)
    outputs = [output_folder / ("crop_" + path.name) for path in inputs]
    pdfpath = folder.parent / (folder.name + ".pdf")

    # prepare file system
    ask_overwrite = not args.yes
    delete_path(pdfpath, folder=False, ask_overwrite=ask_overwrite)
    delete_path(output_folder, folder=True, ask_overwrite=ask_overwrite)
    output_folder.mkdir()

    # invoke
    def to_str(iterable):
        return list(map(str, iterable))
    convert_to_pdf(to_str(inputs), to_str(outputs), str(pdfpath))


if __name__ == "__main__":
    with time_ctx("total time:"):
        main()
