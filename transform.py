import random
import numpy as np   
import torch
import scipy


class MinMaxInstance(object):
    def __call__(self, sample):
        maximum, minimum = sample.max(), sample.min()
        sample = (sample - minimum) / (maximum - minimum)
        
        return sample


class FlipBrain(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        return sample.flip(1) if self.prob > random.uniform(0, 1) else sample


class GaussianBlur(object):
   
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image):    
        if self.prob > random.uniform(0, 1):
            sigma = np.random.uniform(0.0,1.0,1)[0]
            image = scipy.ndimage.filters.gaussian_filter(image, sigma, truncate=8)
            return torch.from_numpy(image)
        else:  
            return image