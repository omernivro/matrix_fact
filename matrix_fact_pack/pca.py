# coding: utf-8

# Sources:
# /Users/omer/Documents/studies/applied_ml/matrix_facto.pdf
# /Users/omer/Documents/studies/applied_ml/andrew_ng_notes_lec_7_till_lec_15.pdf
# http://www.deeplearningbook.org/contents/linear_algebra.html#pf5

# **** consider kernel PCA as an alternative - source J.shaw-taylor

# This is an unsupervised learning problem
# find directions where data is widely spaced out.
# we pay attention that from the pythgeron theorem, both reconstruction
# error and direction with max variance are equivalent
# nice visualization:
# https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues


# PCA (principal component analysis) is good for:

# 1. Visualization in low dim - when you have high dimensional data,
# a common practice is to project it to lower dim (2D/3D) to search for
# structure in the data.

# 2. compress

# 3. learning - compress data and then make learning to run faster.

# 4. distance calculations  - In nearest neighbours classification,
# we need to compute the distance between datapoints.
# For high-dimensional data computing the squared Euclidean distance
# between vectors can be expensive, and also sensitive to noise.
# It is therefore often useful to project the data to a
# lower dimensional representation. For example, in making a classifier
# to distinguish between the digit 1 and the digit 7.

# NOT RECOMMENDED 5. anamoly detection - if it is far from your subspace
# tag as an anamoly - if you ever find future point lying far,you tag it

# Suppose that A is an m×n matrix. Then U is defined to be an
# m×m matrix, D to be an m×n matrix, and V to be an n×n matrix.

# if each data point is a column
# we want to take the mean and std of rows - i.e. the mean pixel i
# example if each data point is an image, we want to put all pixels number 1
# of all images on the same scale. example 2: every column is a car
# with rows representing its characteristics, so if one is door length
# in cm and other row is door length in meters, we want to standarize
# meter and cm rows.

# we then take the SVD of X this time we want the eigen vectors of X*X^T.
# which correspond to U

# if each data point is a row
# we want to take the mean and std of columns
# we then take the SVD of X this time we want the eigen vectors of X^T*X
# which correspond to V


# please insert data s.t each data point is a column
from __future__ import division
from plot import plot_largest_eigenvalues_to_detrmine_subspace_dim
from fake_data import gen_fake_matrix_of_ratings_full, get_shifted_mat_by_mean_n_std
import numpy as np
import scipy.linalg


def pca(matrix, k_top):
    N = matrix.shape[1]
    shift_matrix, mean = get_shifted_mat_by_mean_n_std(matrix)
    U, D, V = scipy.linalg.svd(shift_matrix)
    basis = U[:, :k_top]
    lower_dim_rep = np.matmul(np.transpose(basis), shift_matrix)
    reconstruction = np.matmul(basis, lower_dim_rep)
    reconstruction_err = sum(D**2) / (N-1)
    return(lower_dim_rep, basis, reconstruction, reconstruction_err, U, D, V)


def main():
    mat = gen_fake_matrix_of_ratings_full(500, 70)
    low_rep, b, reco, reco_err, U, D, V = pca(mat, 700)
    plot_largest_eigenvalues_to_detrmine_subspace_dim(D, 700)

if __name__ == "__main__":
    main()
