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
import glob
import time

import numpy as np
try:
    import cv2.cv2 as cv2
except ImportError:
    import cv2
import img2pdf

from papercrop.utils import show_images, time_ctx


try:
    profile
except NameError:
    def profile(f):
        return f

OUTPUT_FOLDER_PREFIX = "_crop_"
OUTPUT_JPEG_PREFIX = "crop_"


@profile
def resize(img, factor):
    return cv2.resize(img, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def morphEx(img, morph_type, ksize):
    bsize = ksize // 2
    bordered = cv2.copyMakeBorder(img, bsize, bsize, bsize, bsize, borderType=cv2.BORDER_CONSTANT, value=0)

    kernel = np.ones((ksize,)*2, np.uint8)
    filtered = cv2.morphologyEx(bordered, morph_type, kernel, borderValue=0)

    result = filtered[bsize:-bsize, bsize:-bsize]
    assert result.shape == img.shape, (result.shape, img.shape)
    return result


def increasing_close_open(img, max_ksize):
    assert max_ksize % 2 == 1
    ksize = 3
    while ksize <= max_ksize:
        img = morphEx(img, cv2.MORPH_CLOSE, ksize)
        img = morphEx(img, cv2.MORPH_OPEN, ksize)
        ksize += 2
    return img


def with_contour(img, contour):
    output = img.copy()
    cv2.drawContours(output, [np.array(contour).astype(int)], 0, (0, 255, 0), 10)
    return output


@profile
def get_contour_canny(img, ksize_smooth=7, ksize_close=30, _debug=False):
    scale = 500 / img.shape[0]
    resized = resize(img, scale)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurr = cv2.medianBlur(gray, ksize_smooth)
    opened = increasing_close_open(blurr, ksize_smooth)
    edged = cv2.Canny(opened, 20, 100)
    closed_edged = morphEx(edged, cv2.MORPH_CLOSE, ksize_close)

    contours = cv2.findContours(closed_edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    ####hulls = map(cv2.convexHull, contours)
    raw_contour = max(contours, key=cv2.contourArea)
    contour = np.round(raw_contour.reshape(-1, 2) / scale).astype(int)

    if _debug:
        show_images([opened, edged, closed_edged, with_contour(img, contour)], show=False)

    return contour


def to_mask(contour, img):
    res = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(res, [contour], 0, 255, -1)
    return res


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


def calc_hist(gray):
    hist = cv2.calcHist([gray], channels=[0], mask=None, histSize=[256], ranges=[0,255])[:,0]
    hist /= (gray.shape[0] * gray.shape[1])
    hist[:5] = 0
    return hist


@profile
def correct_brightness(img, threshold=0.05, min_white=150, max_black=50, _debug=True):
    blurred = cv2.medianBlur(img, 3)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    # adaptive histogram equalization
    lab[:,:,0] = cv2.createCLAHE(clipLimit=0.7, tileGridSize=(8, 8)).apply(lab[:,:,0])

    # calculate white and black point
    hist = calc_hist(lab[:,:,0])
    cutoff = np.max(hist) * threshold
    white = np.max(np.arange(256)[hist > cutoff]) + 10
    white = max(white, min_white)
    black = np.min(np.arange(256)[hist > cutoff]) / 3
    black = min(black, max_black)

    # adjust lightness
    val_black = np.arange(0, 256) - black
    val_white = val_black / (white - black) * 255
    table = np.clip(val_white, 0, 255).astype(np.uint8)
    lab[:,:,0] = cv2.LUT(lab[:,:,0], table)

    # increase saturation
    sat = 127 * 0.15  # percent
    val_sat = (np.arange(0, 256) - sat) / (255 - 2 * sat) * 255
    table = np.clip(val_sat, 0, 255).astype(np.uint8)
    lab[:,:,1] = cv2.LUT(lab[:,:,1], table)
    lab[:,:,2] = cv2.LUT(lab[:,:,2], table)

    output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if _debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(hist)
        plt.plot(calc_hist(cv2.cvtColor(output, cv2.COLOR_BGR2LAB)[:,:,0]))
        show_images([img, output], show=False)

    return output


@profile
def extract_page(img, _debug=False):
    contour = get_contour_canny(img, _debug=_debug)
    page = crop_page(img, contour)
    page = correct_brightness(page, _debug=_debug)
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

    def imwrite(self, path, img, quality=30):
        if self.turbo:
            with open(path, "wb") as f:
                f.write(self.turbo.encode(img, quality=quality))
        else:
            cv2.imwrite(path, img, (cv2.IMWRITE_JPEG_QUALITY, quality))


@profile
def convert_to_pdf(input_files, output_files, pdf_path, rotation_flag=None, _debug=False):
    codec = JpegCodec()
    for i, input_path, output_path in zip(range(len(input_files)), input_files, output_files):
        with time_ctx("page {}/{}".format(i + 1, len(input_files))):
            img = codec.imread(input_path)
            if rotation_flag is not None:
                img = cv2.rotate(img, rotation_flag)
            page = extract_page(img, _debug=_debug)
            codec.imwrite(output_path, page)

        if _debug:
            import matplotlib.pyplot as plt
            plt.show()

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
            if ask_overwrite and input("Warning: output path '{}' exists.\n"
                                       "Delete? [yes/No] ".format(path)).lower() not in ["y", "yes"]:
                sys.exit(1)
            print("INFO: deleted '{}'".format(path))
            if folder:
                shutil.rmtree(str(path))
                # wait until folder is deleted, especially on windows
                while True:
                    try:
                        if not path.is_dir():
                            break
                    except PermissionError:
                        pass
                    time.sleep(0.1)
            else:
                path.unlink()


def convert_folder(foldername, ask_overwrite=True, rotation_flag=None, _debug=False):
    # calculate all paths
    folder = pathlib.Path(foldername)
    inputs = list(sorted(folder.glob("*.[jJ][pP][gG]")))
    output_folder = folder.parent / (OUTPUT_FOLDER_PREFIX + folder.name)
    outputs = [output_folder / (OUTPUT_JPEG_PREFIX + path.name) for path in inputs]
    pdfpath = folder.parent / (folder.name + ".pdf")

    # prepare file system
    delete_path(output_folder, folder=True, ask_overwrite=ask_overwrite)
    delete_path(pdfpath, folder=False, ask_overwrite=ask_overwrite)
    output_folder.mkdir()

    # invoke
    def to_str(iterable):
        return list(map(str, iterable))

    convert_to_pdf(to_str(inputs), to_str(outputs), str(pdfpath), rotation_flag, _debug=_debug)


def main():
    with time_ctx("total time:"):
        parser = argparse.ArgumentParser()
        parser.add_argument("folder", help="folder with *.jpg files.", nargs="+")
        parser.add_argument("-y", "--yes", help="overwrite files by defailt.", action="store_true")
        parser.add_argument("-ccw", "--rotate-ccw", help="rotate images by 90 degrees counter-clockwise.", action="store_true")
        parser.add_argument("-cw", "--rotate-cw", help="rotate images by 90 degrees clockwise.", action="store_true")
        parser.add_argument("--rotate-180", help="rotate images by 180 degrees.", action="store_true")
        parser.add_argument("--debug", help="show debug plots.", action="store_true")
        args = parser.parse_args()

        if args.rotate_ccw + args.rotate_cw + args.rotate_180 > 1:
            print("You can only enable eith --rotate-ccw, --rotate-cw or --rotate-180, not both.")
            sys.exit(1)

        if args.rotate_ccw:
            rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif args.rotate_cw:
            rotation = cv2.ROTATE_90_CLOCKWISE
        elif args.rotate_180:
            rotation = cv2.ROTATE_180
        else:
            rotation = None
        
        # expand * patterns in paths
        for folder in args.folder:
            # enable glob * patterns also on windows
            for foldername in glob.glob(folder):
                if not os.path.isdir(foldername) or foldername.startswith(OUTPUT_FOLDER_PREFIX):
                    print("Skipping '{}'".format(foldername))
                    print()
                else:
                    print("Processing '{}':".format(foldername))
                    convert_folder(foldername, ask_overwrite=not args.yes,
                                   rotation_flag=rotation, _debug=args.debug)
                    print()


if __name__ == "__main__":
    main()
