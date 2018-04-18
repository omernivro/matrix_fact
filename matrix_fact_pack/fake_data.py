from x_val import rand_idx_split
import numpy as np

def gen_fake_matrix_of_ratings_full(dim_ro, dim_col):
    mat = np.random.randint(1, 6, size=(dim_ro, dim_col))
    return(mat)


def gen_fake_matrix_of_ratings_with_missing(dim_ro, dim_col, percent_miss):
    original_mat = np.random.randint(1, 6, size=(dim_ro, dim_col))
    mat = original_mat.reshape(-1).astype('float')
    num_val_to_repllace = int(float(len(mat)) * percent_miss)
    random_cells = np.random.randint(
        0, len(mat), size=[int(num_val_to_repllace)])
    for i in random_cells:
        mat[i] = np.nan
    return(mat.reshape(-1, dim_col), original_mat)


def get_shifted_mat_by_mean_n_std(mat):
    mean_vec = np.sum(mat, axis=1) / (float(mat.shape[1]))
    shifted_mat = (mat - mean_vec.reshape(-1, 1))
    #stand_dev = np.sum(np.square(shifted_mat), axis = 1) / (float(mat.shape[1]) - 1.)
    #shifted_mat = shifted_mat / np.sqrt(stand_dev)
    return(shifted_mat, mean_vec)


def gen_fake_matrix_implicit_full(dim_ro, dim_col):
    mat = np.random.choice(np.arange(0, 2), p=[
                           0.6, 0.4], size=(dim_ro, dim_col))
    return(mat)


def gen_fake_matrix_implicit_confid(dim_ro, dim_col):
    '''turn every row representing confidence per user to diagonal mat'''
    true_matrix = gen_fake_matrix_implicit_full(dim_ro, dim_col)
    mat = np.random.random(dim_ro * dim_col).reshape(dim_ro, dim_col)
    true_matrix = true_matrix.astype('float32')
    confidence_matrix = (np.copy(mat) * np.copy(true_matrix)).astype('float32')

    return(true_matrix, confidence_matrix)


def main():
    tr, matrix = gen_fake_matrix_implicit_confid(10, 30)
    print tr[:, 13]
    print matrix[:, 13]


if __name__ == "__main__":
    main()
