from __future__ import division
from sklearn.model_selection import train_test_split
import numpy as np
# from x_val import rand_idx_split
from fake_data import gen_fake_matrix_implicit_full, gen_fake_matrix_of_ratings_full

# true_matrix = gen_fake_matrix_of_ratings_full(30, 50)
# # print true_matrix.reshape(-1)
# # print true_matrix

# # a, b = rand_idx_split(15, 10, 5)
# # print a
# # print b


# a, b = train_test_split(true_matrix, test_size=0.2)

# print b.shape

a= np.array([3,5 ])
b = np.array([4, 7])

print 620/4