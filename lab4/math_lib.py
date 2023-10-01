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



def gauss(matrix):
    for i in range(len(matrix)):
        if matrix[i][i] == 0:
            for j in range(len(matrix)):
                if matrix[j][i] != 0:
                    for k in range(len(matrix) + 1):
                        matrix[i][k] += matrix[j][k]
                    break

    if matrix[len(matrix) - 1][len(matrix) - 1] == 0:
        for i in range(len(matrix)):
            if matrix[i][len(matrix) - 1] != 0:
                for j in range(len(matrix) + 1):
                    matrix[len(matrix)][j] += matrix[i][j]
                break

    # forward stroke

    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            c = matrix[j][i] / matrix[i][i]
            for k in range(len(matrix), i - 1, -1):
                matrix[j][k] -= c * matrix[i][k]

    # creating diagonals of ones

    for i in range(len(matrix)):
        tmp = matrix[i][i]
        for j in range(len(matrix) + 1):
            matrix[i][j] /= tmp

    # reverse stroke

    all_x = [0] * len(matrix)
    all_x[len(matrix) - 1] = matrix[len(matrix) - 1][len(matrix)]

    for i in range(len(matrix) - 2, -1, -1):
        all_x[i] = matrix[i][len(matrix)]
        for j in range(i + 1, len(matrix)):
            all_x[i] -= matrix[i][j] * all_x[j]

    return all_x