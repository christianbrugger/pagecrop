
"""
pip install img2pdf
"""

import os

import numpy as np
import matplotlib.pyplot as plt
try:
    import cv2.cv2 as cv2
except ImportError:
    import cv2
import img2pdf

from utils import show_images, time_ctx


def resize(img, factor):
    return cv2.resize(img, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def convex_area(contour):
    hull = cv2.convexHull(contour)
    return cv2.contourArea(hull)


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
    res = np.zeros(img.shape[:2])
    cv2.drawContours(res, [contour], 0, 255, -1)
    return res


def with_contour(img, contour):
    output = img.copy()
    cv2.drawContours(output, [np.array(contour).astype(int)], 0, (0, 255, 0), 10)
    return output


def get_cropped_min_area_rect(contour, img):
    rect = cv2.minAreaRect(contour)
    # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D(center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    out = cv2.getRectSubPix(dst, size, center)
    return out


def crop_page(img, contour):
    page = img.copy()
    mask = to_mask(contour, img)
    page[mask == 0] = 0, 0, 0

    crop = get_cropped_min_area_rect(contour, page)
    return crop


def correct_brightness(img, threshold=0.05, min_white=150, _debug=False):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # white point
    hist = np.histogram(yuv[:,:,0], 255, (0, 255))[0] / (img.shape[0] * img.shape[1])
    hist[0] = 0
    cutoff = np.max(hist) * threshold
    white = np.max(np.arange(255)[hist > cutoff])
    white = max(white, min_white)

    # adjust intensity
    gray = yuv[:,:,0].astype(float) / 255.
    gray_white = gray * 255 / white
    gray_gamma = gray_white ** ((255 / white) ** 0.75)

    yuv[:,:,0] = np.clip(gray_gamma * 255, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    if _debug:
        plt.plot(hist)
        plt.show()
        print(hist)

    return output


def extract_page(img):
    contour = get_contour_canny(img)
    page = crop_page(img, contour)
    page = correct_brightness(page)
    return page


def main():
    names_jpg = []
    i = 0
    for name in sorted(os.listdir(".")):
        if name.startswith("test_"):
            i += 1

            with time_ctx("read image"):
                test = cv2.imread(name)

            with time_ctx("extract page"):
                page = extract_page(test)

            with time_ctx("store jpg"):
                outname = 'output_{}.jpg'.format(i)
                cv2.imwrite(outname, page, (cv2.IMWRITE_JPEG_QUALITY, 70))
                names_jpg.append(outname)

    with open("output_jpg.pdf", "wb") as f:
        f.write(img2pdf.convert(names_jpg))


if __name__ == "__main__":
    main()