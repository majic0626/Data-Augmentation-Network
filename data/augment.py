# some augmentation
import torchvision.transforms.functional as trnf


class Rot:
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, x):
        x = trnf.rotate(x, angle=self.degree)
        return x


class Nothing:
    def __init__(self):
        return None

    def __call__(self, x):
        return x
