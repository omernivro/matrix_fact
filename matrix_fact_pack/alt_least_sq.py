# <h1> Alternating least squares IMF </h1>
# Source: http://yifanhu.net/PUB/cf.pdf
# see calculations:
# /Users/omer/Documents/programming/machine_learning/code_calculations
from __future__ import division
import numpy as np
from fake_data import gen_fake_matrix_implicit_full, gen_fake_matrix_implicit_confid
from x_val import cross_val_imf_mat, rand_idx_split
from imf_data_structure import build_conf_diags, build_pref_vecs, zero_out_test_vals


class base_model:
    _alpha = 40

    def __init__(self, true_matrix, conf_matrix=None, c_u=None, c_i=None, user_pref=None, item_pref=None, factors=40, _beta=0.05):
        self.true_matrix = true_matrix
        self.num_users, self.num_items = true_matrix.shape
        self.Y = np.random.normal(size=[self.num_items, factors], loc=0.0,
                                  scale=0.1)

        self.X = np.random.normal(size=[self.num_users, factors], loc=0.0,
                                  scale=0.1)
        self.c_u = c_u
        self.c_i = c_i
        self.user_pref = user_pref
        self.item_pref = item_pref
        self.conf_matrix = conf_matrix
        self._beta = _beta

    def prediction(self):
        self.predict = np.matmul(self.X, np.transpose(self.Y))
        return(self.predict)

    def loss(self, idx):
        self.confidence_gathered = self.conf_matrix.reshape(-1)[idx]

        self.conf = ((self._alpha * self.confidence_gathered) +
                     np.reshape(np.ones(shape=[np.shape(self.confidence_gathered)[0]]), [-1, 1]))

        self.regularizer_Y = np.linalg.norm(self.Y)
        self.regularizer_X = np.linalg.norm(self.X)

        self.sq_diff = np.square(
            self.true_matrix.reshape(-1)[idx] - (self.prediction().reshape(-1)[idx]))

        # self.mse = (np.sum(self.conf * self.sq_diff) / len(idx)) \
        #              + (self._beta * (self.regularizer_X + self.regularizer_Y))

        self.mse = (np.sum(self.sq_diff) / len(idx)) \
                     + (self._beta * (self.regularizer_X + self.regularizer_Y))

        return(self.mse)


class train_model(base_model):

    def __init__(self, true_matrix, conf_matrix=None, c_u=None, c_i=None, user_pref=None, item_pref=None, factors=40, _beta=0.05):
        base_model.__init__(self, true_matrix, conf_matrix, c_u, c_i,
                            user_pref, item_pref, factors, _beta)

    def user_updt(self):
        self.Y_t_Y = np.matmul(np.transpose(self.Y), self.Y)
        for i in range(self.num_users):
            self.confid_minus_eye = self.c_u[i] - \
                np.eye(np.shape(self.c_u[i])[0])
            self.Y_t_conf_Y = np.matmul(
                np.matmul(np.transpose(self.Y), self.confid_minus_eye), self.Y)
            self.first_term_u = np.linalg.inv(
                self.Y_t_Y + self.Y_t_conf_Y + (self._beta * np.eye(np.shape(self.Y_t_conf_Y)[0])))
            self.second_term_u = np.matmul(
                np.matmul(np.transpose(self.Y), self.c_u[i]), self.user_pref[i])
            self.X[i, :] = np.matmul(self.first_term_u, self.second_term_u)

        return(self.X)

    def item_updt(self):
        self.X_t_X = np.matmul(np.transpose(self.X), self.X)
        for j in range(self.num_items):
            self.confid_minus_eye = (self.c_i[j] -
                                     np.eye(np.shape(self.c_i[j])[0]))
            self.X_t_conf_X = np.matmul(
                np.matmul(np.transpose(self.X), self.confid_minus_eye), self.X)
            self.first_term = np.linalg.inv(
                self.X_t_X + self.X_t_conf_X + (self._beta * np.eye(np.shape(self.X_t_conf_X)[0])))
            self.second_term = np.matmul(
                np.matmul(np.transpose(self.X), self.c_i[j]), self.item_pref[j])
            self.Y[j, :] = np.matmul(self.first_term, self.second_term)

        return(self.Y)


class eval_model(base_model):

    def __init__(self, X, Y, true_matrix, conf_matrix=None, c_u=None, c_i=None, user_pref=None, item_pref=None, factors=40, _beta=0.05):

        base_model.__init__(self, true_matrix, conf_matrix, c_u, c_i,
                            user_pref, item_pref, factors, _beta)
        self.X = X
        self.Y = Y

    def rmse(self):
        pass

## insert two states if you're training before choosing regularization or after.
def main():
    k_fold = 5
    sweep = 10
    true_matrix, confidence_matrix = gen_fake_matrix_implicit_confid(
        50, 500)
    num_users, num_items = true_matrix.shape
    tr_idx, te_idx = rand_idx_split(num_users * num_items, 0.8, False)
    print len(tr_idx)
    tr_matrix, tr_conf_mat = zero_out_test_vals(
        true_matrix, confidence_matrix, te_idx)
    val_c_u, val_c_i = build_conf_diags(np.copy(tr_matrix))
    val_user_pref, val_item_pref = build_pref_vecs(np.copy(tr_conf_mat))

    fact = [10, 40, 70, 100]
    beta = [0.05, 1, 10, 30]
    best_lambda_fac = []
    for factors in fact:
        for bet in beta:
            build_model = train_model(true_matrix, factors=factors, _beta=bet)
            for i in range(sweep):
                mean_tr_err = []
                mean_val_err = []
                for cv_iter in range(1, k_fold + 1, 1):
                    n_tr_idx, val_idx = cross_val_imf_mat(
                        k_fold, cv_iter, tr_idx, True)
                    n_train_matrix, n_tr_conf = zero_out_test_vals(
                        np.copy(tr_matrix), np.copy(tr_conf_mat), val_idx)
                    build_model.c_u, build_model.c_i = build_conf_diags(np.copy(n_tr_conf))
                    build_model.user_pref, build_model.item_pref = build_pref_vecs(np.copy(n_train_matrix))
                    build_model.true_matrix = np.copy(n_train_matrix)
                    build_model.conf_matrix = np.copy(n_tr_conf)

                    X = train_model.user_updt(build_model)
                    Y = train_model.item_updt(build_model)

                    evaluate = eval_model(X, Y, true_matrix=np.copy(tr_matrix), conf_matrix=np.copy(tr_conf_mat), c_u=val_c_u, c_i=val_c_i,
                                          user_pref=val_user_pref, item_pref=val_item_pref, factors=factors, _beta=bet)

                    tr_err = build_model.loss(n_tr_idx)
                    val_err = evaluate.loss(val_idx)
                    mean_tr_err.append(tr_err)
                    mean_val_err.append(val_err)



                    # print('# factors:', factors, 'beta:', bet, 'cv no:', cv_iter, 'sweep no:', i, 'train mse:',
                    #       tr_err)
                    # print('# factors:', factors, 'beta:', bet, 'cv no:', cv_iter, 'sweep no:', i, 'test mse:',
                    #       val_err)
                best_lambda_fac.append([i, factors, bet, np.mean(mean_tr_err), np.mean(mean_val_err)])
    print(best_lambda_fac)


if __name__ == "__main__":
    main()
