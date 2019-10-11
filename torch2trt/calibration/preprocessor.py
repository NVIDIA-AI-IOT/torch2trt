from PIL import Image
import numpy as np
import cv2

class Transformer(object):
    def __init__(self, size, mean):
        self.mean = mean
        if not isinstance(size, tuple):
            self.size = (size, size)
        else:
            self.size = size
        self.transform = Compose([
                Resize(self.size),
                SubtractMeans(self.mean)
                ])

    def __call__(self, img):
        return self.transform(img)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        img = cv2.resize(image, self.size)
        return img

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype = np.float32)

    def __call__(self, image):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32)

