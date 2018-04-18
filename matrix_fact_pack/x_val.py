import numpy as np


def split_miss_matrix_tr_n_val_idx(miss_matrix, tr_precent, val_precent):
    n_items = miss_matrix.shape[1]
    idx_col = int(n_items * tr_precent)
    tr_matrix = miss_matrix[:, :idx_col]
    val_matrix = miss_matrix[:, idx_col:]

    tr_idx = np.concatenate((np.array(np.where(~np.isnan(tr_matrix)))[0].reshape(-1, 1),
                             np.array(np.where(~np.isnan(tr_matrix)))[1].reshape(-1, 1)), axis=1)

    tr_num = tr_idx.shape[0]

    val_idx = np.concatenate((np.array(np.where(~np.isnan(val_matrix)))[0].reshape(-1, 1),
                              np.array(np.where(~np.isnan(val_matrix)))[1].reshape(-1, 1)), axis=1)

    val_num = val_idx.shape[0]

    return(tr_matrix, tr_idx, tr_num, val_matrix, val_idx, val_num)


def cross_val_miss_mat(k, iter_num, x):
    # k - number of folds
    # iter - iteration number
    # x - training dataset
    # RETURNS: new train and validation sets

    val_index = x.shape[1] * iter_num / k

    x_val = range((val_index - x.shape[1] / k), val_index, 1)

    x_tr = np.concatenate([np.array(range(
        0, (val_index - (x.shape[1] / k)))), np.array(range(val_index, x.shape[1], 1))])

    non_nan = np.concatenate((np.where(~np.isnan(x))[0].astype(int).reshape(-1, 1),
                              np.where(~np.isnan(x))[1].astype(int).reshape(-1, 1)), axis=1)

    tr_idx = non_nan[np.isin(non_nan[:, 1], x_tr), :]

    num_tr = float(tr_idx.shape[0])

    val_idx = non_nan[np.isin(non_nan[:, 1], x_val), :]

    num_val = float(val_idx.shape[0])

    return(tr_idx, num_tr, val_idx, num_val)


def cross_val_imf_mat(k, iter_num, tr_idx, sort=True, same=True, n_user=None):
    """ In general when conducting split for videos, we want to throw
    out different videos for different users. It is not realistic that
    all users didn't watch the same videos unless it's
    a timed programme schedule."""

    # k - number of folds
    # iter - iteration number
    # tr_idx - training indices
    # RETURNS: new train and validation indices

    if same == True:
        _tr_idx = tr_idx.reshape(n_user, -1)
        n_cols = int(len(tr_idx) / n_user)
        val_st_idx = n_cols * (iter_num - 1) / k
        val_end_idx = n_cols * iter_num / k
        val_idx = _tr_idx[:, val_st_idx:val_end_idx].reshape(-1)
        tr_idx = tr_idx[~np.isin(tr_idx, val_idx)]

    else:
        val_st_idx = len(tr_idx) * (iter_num - 1) / k
        val_end_idx = len(tr_idx) * iter_num / k

        val_idx = tr_idx[val_st_idx: val_end_idx]
        tr_idx = tr_idx[~np.isin(tr_idx, val_idx)]

        if (sort is True):
            tr_idx, val_idx = sorted(tr_idx), sorted(val_idx)

        assert (any(np.isin(val_idx, tr_idx)) is False)
    return(tr_idx, val_idx)


def rand_idx_split(num_obser, tr_precent, n_row=None, sort=True, same=True):
    # randomaly permutate numbers, then split train & test data accordingly.

    if same == False:
        rand_0_1 = np.random.rand(num_obser, 1)

        indices = np.random.permutation(rand_0_1.shape[0])

        tr_abs = int(num_obser * tr_precent)

        tr_idx, te_idx = indices[:tr_abs], indices[tr_abs:(num_obser)]

        if (sort is True):
            tr_idx = np.asarray(sorted(tr_idx))

        te_idx = np.asarray(sorted(te_idx))
    else:
        n_col = int(tr_precent * (num_obser / n_row))
        all_idx = np.array(range(num_obser)).reshape(n_row, -1)
        tr_idx = all_idx[:, :n_col].reshape(-1)
        te_idx = all_idx[:, n_col:].reshape(-1)

    return(tr_idx, te_idx)


def main():
    aa, cc = rand_idx_split(100, 0.8, sort=False)
    print cc


if __name__ == main:
    main()
