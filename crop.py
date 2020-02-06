
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
    kernprof -l crop.py
    python -m line_profiler crop.py.lprof
"""

import os
import math

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
    width = rect[1][0]
    height = rect[1][1]

    # The lowest point of the rectangle will always be the first element.
    # All other points will follow in a clockwise-direction.
    # We make sure that the first element is always the lower left corner.
    corners = cv2.boxPoints(rect)
    if corners[0][0] > corners[1][0]:
        corners = np.roll(corners, -1, axis=0)
        height, width = width, height

    dst = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype=np.float32)
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

    def imwrite(self, path, img):
        if self.turbo:
            with open(path, "wb") as f:
                f.write(self.turbo.encode(img, quality=70))
        else:
            cv2.imwrite(path, img, (cv2.IMWRITE_JPEG_QUALITY, 70))


@profile
def main():
    codec = JpegCodec()
    names_jpg = []
    i = 0
    for name in sorted(os.listdir(".")):
        if name.startswith("test_"):
            i += 1

            with time_ctx("read image"):
                test = codec.imread(name)

            with time_ctx("extract page"):
                page = extract_page(test)

            with time_ctx("store jpg"):
                outname = 'output_{}.jpg'.format(i)
                codec.imwrite(outname, page)
                names_jpg.append(outname)

    with open("output_jpg.pdf", "wb") as f:
        f.write(img2pdf.convert(names_jpg))


if __name__ == "__main__":
    with time_ctx("total time:"):
        main()
