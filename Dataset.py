import numpy as np
from dataclasses import dataclass


@dataclass
class Letters:
    def __init__(self):
        self.images = []

        # Agoston Szabo
        self.images.append(np.matrix(([0,0,0,0,0], [0,0,1,0,0], [0,1,0,1,0], [0,1,1,1,0], [0,1,0,1,0])))
        self.images[0].target = 0 #A
        self.images.append(np.matrix(([1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0])))
        self.images[1].target = 1 #T
        self.images.append(np.matrix(([0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0])))
        self.images[2].target = 2 #O
        self.images.append(np.matrix(([1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1])))

        # Xiaoyang Sun
        self.images[3].target = 0  # X
        self.images.append(np.matrix(([1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1])))
        self.images[4].target = 1  # S
        self.images.append(np.matrix(([1, 0, 0, 0, 1], [1, 1, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 1, 1], [1, 0, 0, 0, 1])))
        self.images[5].target = 2  # N
        self.images.append(np.matrix(([1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 0])))

        # Liu Dian
        self.images[6].target = 0  # D
        self.images.append(np.matrix(([1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1])))
        self.images[7].target = 1  # L
        self.images.append(np.matrix(([1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0])))
        self.images[8].target = 2  # U
        self.images.append(np.matrix(([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])))
        self.images[9].target = 0  # All 1

def flatten(image):
    return image.reshape(1, -1)

def blur(image):
    blur1 = np.array([[0.9, 0.1, 0, 0, 0], [0.1, 0.8, 0.1, 0, 0], [0, 0.1, 0.8, 0.1, 0], [0, 0, 0.1, 0.8, 0.1],[0, 0, 0, 0.1, 0.9]])
    return np.matmul(image, blur1), np.matmul(blur1, np.matmul(blur1, image))

def darken(image, factor = 0.9):
    return factor * image

def letter(num, map=0):
    mapping = {0: {0: 'A', 1: 'T', 2: 'O'}, 1: {0: 'X', 1: 'S', 2: 'N'}, 2: {0: 'D', 1: 'L', 2: 'U'}}
    return (mapping[map])[num]
