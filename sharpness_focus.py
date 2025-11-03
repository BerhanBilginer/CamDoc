import cv2
import numpy as np
from matplotlib import pyplot as plt


class FocusCalculator:
    def __init__(self):
        pass

    def variance_laplacian(self, image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        score = lap.var()
        return score
    
    def tenengrand(self, image):
        pass

    def fft_hf_energy(self, image):
        pass

    def edge_density(self, image):
        pass

    def entropy(self, image):
        pass

    def dark_channel(self, image):
        pass

    def brisque_niqe(self, image):
        pass

    def visualize(self, image):
        pass

    def save(self, image):
        pass