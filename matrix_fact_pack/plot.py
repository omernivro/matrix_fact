import matplotlib
# this should not be mentioned usually
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def plot_largest_eigenvalues_to_detrmine_subspace_dim(Diagonal_mat, largest_k):
    plt.title('largest k scaled eigenvalues')
    y_largest_k = (Diagonal_mat[:largest_k]**2) / (Diagonal_mat[0]**2)
    x_lar_k = range(largest_k)
    plt.plot(x_lar_k, y_largest_k)
    plt.show('hold')
