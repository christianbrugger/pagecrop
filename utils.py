
import time

import numpy as np
try:
    import cv2.cv2 as cv2
except ImportError:
    import cv2


def show_images(images, horizontal_image=2, filename=None):
    """
    Show multiple images in a GUI window or save them to a file
    """
    import matplotlib.pyplot as plt
    plt.clf()
    columns = min(len(images), horizontal_image)
    rows = int(np.ceil(len(images) / float(horizontal_image)))

    for i, image in enumerate(images):
        plt.subplot(rows, columns, i + 1)
        plt.axis('off')
        if len(image.shape) == 3:
            img_plt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_plt = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        plt.imshow(img_plt, interpolation="bilinear")

    plt.gcf().subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename + ".png", dpi=300)


class time_ctx(object):
    """contect manager that measures time"""
    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type_, value_, traceback_):
        self.end = time.time()
        print("{}: {:.2f} s".format(self.description, self.end - self.start))


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)

