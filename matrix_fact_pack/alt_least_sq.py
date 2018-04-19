# <h1> Alternating least squares IMF </h1>
# Source: http://yifanhu.net/PUB/cf.pdf
# see calculations:
# /Users/omer/Documents/programming/machine_learning/code_calculations
from __future__ import division
import numpy as np
from fake_data import gen_fake_matrix_implicit_full
from fake_data import gen_fake_matrix_implicit_confid
from x_val import cross_val_imf_mat, rand_idx_split
from imf_data_structure import build_conf_diags, build_pref_vecs
from imf_data_structure import zero_out_test_vals


class base_model:
    _alpha = 40

    def __init__(self, true_matrix, conf_matrix=None, c_u=None, c_i=None, user_pref=None, item_pref=None, factors=20, _beta=0.05):
        self.true_matrix = np.copy(true_matrix)
        self.num_users, self.num_items = np.copy(true_matrix).shape
        self.Y = np.random.normal(size=[self.num_items, factors], loc=0.0,
                                  scale=0.1)

        self.X = np.random.normal(size=[self.num_users, factors], loc=0.0,
                                  scale=0.1)
        self.c_u = np.copy(c_u)
        self.c_i = np.copy(c_i)
        self.user_pref = user_pref
        self.item_pref = item_pref
        self.conf_matrix = conf_matrix.copy()
        self._beta = _beta

    def prediction(self):
        self.predict = np.matmul(self.X.copy(), np.transpose(self.Y.copy()))
        return(self.predict)

    def loss(self, idx):
        predicted = np.copy(self.prediction().reshape(-1)[idx])

        self.confidence_gathered = self.conf_matrix.copy().reshape(-1)[idx]

        # print self.confidence_gathered.shape

        self.conf = ((self._alpha * np.copy(self.confidence_gathered).reshape(-1, 1)) +
                     np.reshape(np.ones(shape=[np.shape(np.copy(self.confidence_gathered))[0]]), [-1, 1]))

        ##### still need to take only the X, Y
        self.regularizer_Y = np.copy(np.linalg.norm(self.Y.copy()))
        self.regularizer_X = np.copy(np.linalg.norm(self.X.copy()))

        self.sq_diff = np.square(
            (np.copy(self.true_matrix).reshape(-1)[idx]) - (np.copy(predicted))).reshape(-1, 1)

        # print self.sq_diff

        # print sum(np.dot(self.conf, self.sq_diff))
        # print len(idx)
        # print self.conf
        # print self.conf.shape
        # print len(self.sq_diff)
        # print ('check:', (np.copy(self.conf) * np.copy(self.sq_diff))[1])
        self.mse = (np.sum((np.copy(self.conf) *
                                np.copy(self.sq_diff))) / len(idx))
        # \
            # + (self._beta * (self.regularizer_X + self.regularizer_Y))

        # self.mse = (np.sum(self.sq_diff) / len(idx)) \
        #     + (self._beta * (self.regularizer_X + self.regularizer_Y))

        return(np.copy(self.mse))


class train_model(base_model):

    def __init__(self, true_matrix, conf_matrix=None, c_u=None, c_i=None, user_pref=None, item_pref=None, factors=40, _beta=0.05):
        base_model.__init__(self, np.copy(true_matrix), np.copy(conf_matrix), np.copy(c_u), np.copy(c_i),
                            np.copy(user_pref), np.copy(item_pref), factors, _beta)

    def user_updt(self):
        self.Y_t_Y = np.matmul(np.transpose(self.Y.copy()), self.Y.copy())
        # print ('yty:', self.Y_t_Y)
        for i in range(self.num_users):
            self.confid_minus_eye = (self.c_u[i].copy() -
                                     np.eye(np.shape(self.c_u[i].copy())[0]))
            self.Y_t_conf_Y = np.matmul(
                np.matmul(np.transpose(self.Y.copy()), self.confid_minus_eye.copy()), self.Y.copy())
            self.first_term_u = np.linalg.inv(
                self.Y_t_Y.copy() + self.Y_t_conf_Y.copy() + (self._beta * np.eye(np.shape(self.Y_t_conf_Y.copy())[0])))
            self.second_term_u = np.matmul(
                np.matmul(np.transpose(self.Y.copy()), self.c_u[i].copy()), self.user_pref[i].copy())
            self.X[i, :] = np.matmul(
                self.first_term_u.copy(), self.second_term_u.copy())

        return(self.X.copy())

    def item_updt(self):
        self.X_t_X = np.matmul(np.transpose(self.X.copy()), self.X.copy())
        # print ('xtx:', self.X_t_X)
        for j in range(self.num_items):

            self.confidx_minus_eye = (self.c_i[j].copy() -
                                      np.eye(np.shape(self.c_i[j].copy())[0]))
            self.X_t_conf_X = np.matmul(
                np.matmul(np.transpose(self.X.copy()), self.confidx_minus_eye.copy()), self.X.copy())

            self.first_term = np.linalg.inv(
                self.X_t_X.copy() + self.X_t_conf_X.copy() + (self._beta * np.eye(np.shape(self.X_t_conf_X.copy())[0])))
            self.second_term = np.matmul(
                np.matmul(np.transpose(self.X.copy()), self.c_i[j].copy()), self.item_pref[j].copy())
            self.Y[j, :] = np.matmul(self.first_term.copy(
            ), self.second_term.copy().reshape(-1, 1)).reshape(-1)
            if j == 0:
                pass
                # print ('item pref', self.item_pref[0])
                # print ('confidence_diag', self.c_i[0])
                # print ('confidence_diag shape', self.c_i[0].shape)
                # print ('conf - eye', self.confidx_minus_eye)
                # print ('xt_conf_x', self.X_t_conf_X)
                # print ('first', self.first_term)
                # print ('first', self.first_term.shape)
                # print('second', self.second_term)
                # print ('second', self.second_term.reshape(-1, 1).shape)
                # print('y_j:', j, self.Y[j, :])

        return(self.Y.copy())


class eval_model(base_model):

    def __init__(self, X, Y, true_matrix, conf_matrix=None, c_u=None, c_i=None, user_pref=None, item_pref=None, factors=40, _beta=0.05):

        base_model.__init__(self, true_matrix, conf_matrix, c_u, c_i,
                            user_pref, item_pref, factors, _beta)
        self.X = X.copy()
        self.Y = Y.copy()

    def percentile_rank(self, idx):

        predicted = np.copy(self.prediction().reshape(-1))

        per_rank = 0

        for i in range(self.num_users):
            if i == 0:
                indices = idx[idx < (i * self.num_items) + self.num_items]
            else:
                indices = idx[(idx > ((i-1) * self.num_items) + self.num_items)
                              & (idx < (i * self.num_items) + self.num_items)]

            ordered_list_idx = (-predicted).argsort()[indices]

            percentiles = np.array(range(len(ordered_list_idx))).astype(
                'float32') * ((1 / (len(ordered_list_idx) - 1)))

            # print self.true_matrix.reshape(-1)[ordered_list_idx]
            # print self.true_matrix.reshape(-1)[ordered_list_idx] * percentiles

            # print (self.true_matrix.reshape(-1)[ordered_list_idx] * percentiles).shape
            per_rank += sum(self.true_matrix.reshape(-1)
                            [ordered_list_idx] * percentiles)

        # print indices
        # print ordered_list_idx
        # print per_rank
        # print percentiles
        # print self.true_matrix.reshape(-1)[idx]

        exp_per_rank = per_rank / sum(self.true_matrix.reshape(-1)[idx])

        return(exp_per_rank)

# insert two states if you're training before choosing regularization or after.


def main():
    row = 0
    k_fold = 5
    sweep = 10
    true_matrix, confidence_matrix = gen_fake_matrix_implicit_confid(50, 200)

    # print ('confidence_matrix original item 0', confidence_matrix[:, 0])
    # print ('true_matrix original item 0', true_matrix[:, 0])
    num_users, num_items = np.copy(true_matrix).shape
    should_restart = True
    while should_restart:
        should_restart = False
        for i in range(num_items):
            if(np.all(true_matrix[:, i] == 0)==True):
                should_restart = True
                print('should restart:', i)
                true_matrix, confidence_matrix = gen_fake_matrix_implicit_confid(
                    50, 200)
                break
    tr_idx, te_idx = rand_idx_split(
        num_users * num_items, tr_precent=0.8, n_row=num_users, sort=False, same=False)
    # print('test index:', te_idx)
    # print true_matrix.reshape(-1)[te_idx]

    tr_matrix, tr_conf_mat = zero_out_test_vals(
        np.copy(true_matrix), np.copy(confidence_matrix), te_idx)
    # print(tr_conf_mat[0, :])
    # print(np.copy(true_matrix).reshape(-1)[te_idx])
    # print(np.copy(tr_matrix).reshape(-1)[te_idx])

    val_c_u, val_c_i = build_conf_diags(np.copy(tr_conf_mat))
    val_user_pref, val_item_pref = build_pref_vecs(np.copy(tr_matrix))

    fact = [10, 40, 70]
    beta = [0.01, 0.05, 0.08, 0.5, 1, 3, 5, 10]
    best_lambda_fac = np.zeros((sweep * len(fact) * len(beta), 5))
    for factors in fact:
        for bet in beta:
            build_model = train_model(
                np.copy(true_matrix), factors=factors, _beta=bet)
            for i in range(sweep):
                mean_tr_err = []
                mean_val_err = []
                for cv_iter in range(1, k_fold + 1, 1):
                    n_tr_idx, val_idx = cross_val_imf_mat(
                        k_fold, cv_iter, tr_idx, sort=True, same=False, n_user=num_users)
                    # print ('val idx', val_idx)
                    n_train_matrix, n_tr_conf = zero_out_test_vals(
                        np.copy(tr_matrix), np.copy(tr_conf_mat), val_idx)
                    # print ('n_tr_conf 0', n_tr_conf[0, :])
                    # print ('val idx:', val_idx)
                    # print(np.copy(tr_matrix).reshape(-1)[val_idx])
                    # print(np.copy(n_train_matrix).reshape(-1)[val_idx])
                    # print('n_tr_conf 0 item', (np.copy(n_tr_conf[:, 0])))
                    for prob in range(num_items):
                        if(np.all(n_train_matrix[:, prob] == 0)==True):
                                # indices = np.array(range(num_items * num_users)).reshape(-1, num_items)
                                # print indices[:, i]
                                # print ('val idx:', val_idx)
                                print('problem is that we have an item which is\
                                 completely 0 for all users \
                                which means its confidence diag matrix \
                                would be all 0 causing Y to be 0 for some rows\
                                which in turn will produce 0 prediction to all\
                                users w.r.t these items. The problematic column is:', prob)
                                # print('true_matrix:', true_matrix[:, i])
                                # print('tr_matrix:', tr_matrix[:, i])
                                # print('n_train', n_train_matrix[:, i])

                    # print ('diag user 0',  np.diag(n_tr_conf[0, :]))
                    # print ('diag item 0',  np.diag(n_tr_conf[:, 0]))

                    build_model.c_u, build_model.c_i = build_conf_diags(
                        np.copy(n_tr_conf))
                    build_model.user_pref, build_model.item_pref = build_pref_vecs(
                        np.copy(n_train_matrix))
                    build_model.true_matrix = np.copy(n_train_matrix)
                    build_model.conf_matrix = np.copy(n_tr_conf)

                    X = np.copy(train_model.user_updt(build_model))
                    Y = np.copy(train_model.item_updt(build_model))

                    evaluate = eval_model(X.copy(), Y.copy(), true_matrix=np.copy(tr_matrix), conf_matrix=np.copy(tr_conf_mat), c_u=np.copy(val_c_u), c_i=np.copy(val_c_i),
                                          user_pref=np.copy(val_user_pref), item_pref=np.copy(val_item_pref), factors=factors, _beta=bet)

                    tr_err = build_model.loss(n_tr_idx)
                    val_err = evaluate.loss(val_idx)
                    mean_tr_err.append(tr_err.copy())
                    mean_val_err.append(val_err.copy())

                    print('# factors:', factors, 'beta:', bet, 'cv no:', cv_iter, 'sweep no:', i, 'train mse:',
                          tr_err)
                    print('# factors:', factors, 'beta:', bet, 'cv no:', cv_iter, 'sweep no:', i, 'val mse:',
                          val_err)
                a = np.array([i, factors, bet, np.mean(mean_tr_err),
                              np.mean(mean_val_err)]).reshape(1, 5)
                best_lambda_fac[row, :] = np.array(a)
                row = row + 1
    choose_params = best_lambda_fac[np.where(
        best_lambda_fac[:, 4] == min(best_lambda_fac[:, 4])), :][0][0][1:3]
    '''train with best params all the dataset'''
    print ('best params:', choose_params)
    full_model = train_model(true_matrix=np.copy(tr_matrix), factors=int(
        choose_params[0]), _beta=choose_params[1])

    full_model.c_u, full_model.c_i = build_conf_diags(np.copy(tr_conf_mat))
    full_model.user_pref, full_model.item_pref = build_pref_vecs(
        np.copy(tr_matrix))

    for sweep in range(sweep):
        X = np.copy(train_model.user_updt(full_model))
        Y = np.copy(train_model.item_updt(full_model))

    # print full_model.prediction().reshape(-1)[te_idx]

    test = eval_model(X.copy(), Y.copy(), true_matrix=np.copy(true_matrix), conf_matrix=np.copy(
        confidence_matrix), factors=int(choose_params[0]), _beta=choose_params[1])
    te_err = test.loss(te_idx)
    print ('test mse:', te_err)
    print ('expected percentile rank:', test.percentile_rank(te_idx))


if __name__ == "__main__":
    main()
