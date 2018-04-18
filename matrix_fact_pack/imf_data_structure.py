import numpy as np
from fake_data import gen_fake_matrix_implicit_confid


def build_conf_diags(confidence_matrix):

    ro, col = confidence_matrix.shape
    c_u = {}  # user confidence
    for i in range(int(ro)):
        c_u[i] = np.diag(confidence_matrix[i, :])

    c_i = {}  # item confidence
    for i in range(int(col)):
        c_i[i] = np.diag(confidence_matrix[:, i])

    return(c_u, c_i)


def build_pref_vecs(matrix):

    ro, col = matrix.shape

    user_pref = {}
    for i in range(int(ro)):
        user_pref[i] = matrix[i, :]

    item_pref = {}
    for i in range(int(col)):
        item_pref[i] = matrix[:, i]

    return(user_pref, item_pref)


def zero_out_test_vals(matrix_1, matrix_2, te_idx):
    matrix_1.reshape(-1)[te_idx] = 0
    matrix_2.reshape(-1)[te_idx] = 0

    return (matrix_1.copy(), matrix_2.copy())

def main():
    true_matrix, confidence_matrix = gen_fake_matrix_implicit_confid(2, 12)
    a, b = zero_out_test_vals(np.copy(true_matrix), np.copy(confidence_matrix), [3, 7, 17])
    print a.reshape(-1)[3]
    print b.reshape(-1)[3]

if __name__ == "__main__":
    main()