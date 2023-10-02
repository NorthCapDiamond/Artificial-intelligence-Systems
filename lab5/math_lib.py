import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import math
from numpy.linalg import inv

cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100

def PCA(X, n_components=1):
	X_meaned = X - np.mean(X , axis = 0)
	cov_mat = np.cov(X_meaned , rowvar = False)
	eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)

	sorted_index = np.argsort(eigen_values)[::-1]
	sorted_eigenvalue = eigen_values[sorted_index]
	sorted_eigenvectors = eigen_vectors[:,sorted_index]
	eigenvector_subset = sorted_eigenvectors[:,0:n_components]
	X_reduced = np.dot(eigenvector_subset.transpose(),X_meaned.transpose()).transpose()

	return X_reduced
