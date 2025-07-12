import cv2
import imageio
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from collections import Counter
from joblib import cpu_count, delayed, Parallel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.displaytools import *
from src.ftextraction import *
from src.improcessing import *
from src.maskcreation import *


image_stack = cv2.imreadmulti('source_images/Adherent.tif', flags=cv2.IMREAD_GRAYSCALE)[1]

image_gray = scale_image(image_stack[-1])
# params = 'src/params_gray_mask.yml'
# params = 'src/params/foreground_mask.yml'
# params = 'src/params/histogram_normalize.yml'
# params = 'src/params/median_filter.yml' 
params = 'src/params/gaussian_dif_manual.yml'
mask_image, search_range = get_mask_image_with_refined_offset(image_gray, params, verbosity=2)


