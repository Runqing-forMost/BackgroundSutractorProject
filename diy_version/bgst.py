"""
bgst means background subtraction
this file will implement background subtraction by GMM
for simplicity, gray color space will be used
"""
import math
import random as rand
import numpy as np

__all__ = ["createBackgroundSubtractorGMM"]


def createBackgroundSubtractorGMM(history=200, lr=0.01, width=160, height=120):
    # create a GMM instance
    return GMM(default_history=history, lr=lr, width=width, height=height)


class GMM:

    def __init__(self, default_history=200, e=1e-15, lr=0.01, k=5, width=160, height=120):
        """

        :param default_history:
        :param e: judge whether the model convergence
        :param lr: learning rate
        :param k: number of single gaussian
        """
        self.e = e
        self.c = k
        self.history = default_history
        self.lr = lr
        self.vars_init = 10.
        self.vars = np.zeros((height, width, k), dtype=float)
        self.means = np.zeros((height, width, k), dtype=float)
        self.w = np.zeros((height, width, k), dtype=float)
        self.p2u_dis = np.zeros((height, width, k), dtype=float)
        self.height = height
        self.width = width
        self.initialize()

    def initialize(self):
        """-------------initialize the GMM----------------"""
        pixel_range = 255
        for i in range(self.height):
            for j in range(self.width):
                for k in range(self.c):
                    self.means[i, j, k] = rand.random() * pixel_range
                    self.w[i, j, k] = 1. / self.c
                    self.vars[i, j, k] = self.vars_init
        """-----------------------------------------------"""

    def apply(self, frame, frame_num):
        """
        :param frame_num: current frame number
        :param frame: a grey color image
        :return: a masked image with white foreground and black background
        """
        fg_bg = frame
        fg = np.zeros((self.height, self.width), dtype=np.uint8)

        """------------calculate the distance-------------"""
        for m in range(self.c):
            self.p2u_dis[:, :, m] = abs(fg_bg - self.means[:, :, m])
        """-----------------------------------------------"""
        """---------------train the model-----------------"""
        for i in range(self.height):
            for j in range(self.width):
                match = 0
                for k in range(self.c):
                    if abs(self.p2u_dis[i, j, k]) < 2.5 * self.vars[i, j, k]:
                        match = 1
                        if frame_num > self.history:  # no parameter updating
                            fg[i, j] = 0
                        else:
                            self.w[i, j, k] = (1 - self.lr) * self.w[i, j, k] + self.lr
                            p = self.lr / self.w[i, j, k]
                            self.means[i, j, k] = (1. - p) * self.means[i, j, k] + p * float(fg_bg[i, j])
                            self.vars[i, j, k] = math.sqrt((1. - p) * self.vars[i, j, k] ** 2 +
                                                           p * (fg_bg[i, j] - self.means[i, j, k]) ** 2)
                            fg[i, j] = 0
                    else:
                        if frame_num < self.history:
                            self.w[i, j, k] = (1 - self.lr) * self.w[i, j, k]

                if not match:
                    min_w = np.min(self.w[i, j, :])
                    min_w_idx = np.argwhere(self.w[i, j, :] == min_w)

                    self.means[i, j, int(min_w_idx[0])] = float(fg_bg[i, j])
                    self.vars[i, j, int(min_w_idx[0])] = self.vars_init
                    fg[i, j] = 255
        return fg

