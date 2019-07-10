import cv2
import numpy as np


class ImageProcessor():

    def __init__(self, max_class=1000):
        self.head_data = []
        self.le_imgs   = []
        self.re_imgs   = []
        self.grid      = []
        self.max_class_len = max_class

    def crop_img(self, box, img):
        b = box
        crop = img[b[1]:b[3], b[0]:b[2]]
        return crop

    def resize_img(self, img, out_size):
        maxsize = np.max(img.shape[:2])
        r = out_size/float(maxsize)
        dim = (int(img.shape[1]*r), int(img.shape[0]*r))
        rsz = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        black = np.zeros((out_size, out_size, 3), np.uint8)
        black[0:rsz.shape[0], 0:rsz.shape[1]] = rsz
        return black

    def gamma_correction(self, img, gamma):
        lut = np.empty((1,256), np.uint8)
        for i in range(256):
            lut[0,i] = np.clip(pow(i/255.0, gamma) * 255.0, 0, 255)
        return cv2.LUT(img, lut)