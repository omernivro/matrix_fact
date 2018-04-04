
# coding: utf-8


# Sources:
# /Users/omer/Documents/studies/applied_ml/matrix_facto.pdf
# /Users/omer/Documents/studies/applied_ml/andrew_ng_notes_lec_7_till_lec_15.pdf
# http://www.deeplearningbook.org/contents/linear_algebra.html#pf5

# import tensorflow as tf
import itertools
import numpy as np
import scipy.io
import csv
import os.path
import decimal
import time
import fnmatch
import fake_data

# <h2> PCA - with missing values tensorflow </h2>
# A function to get a series of numbers with a set jump between terms.

train_graph = tf.Graph()
eval_graph = tf.Graph()
infer_graph = tf.Graph()


# class


def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)


def init_plh():
    _idx = tf.placeholder("int32", [None, 2], name='utility_matrix_idx')
    _beta = tf.placeholder("float32", [], name='regularization')
    _num = tf.placeholder("float32", [])
    return(_idx, _beta, _num)


def init_vars():
    Y = {'low_m': tf.Variable(tf.random_normal([n_users, _rank], mean=0.0,
                                               stddev=0.1), name='low_p', validate_shape=False),
         'low_b': tf.Variable(tf.random_normal([n_users, 1], mean=0.0,
                                               stddev=0.1), name='low_p_b')}

    B = {'basis_m': tf.Variable(tf.random_normal([_rank, n_items], mean=0.0,
                                                 stddev=0.1), name='base', validate_shape=False),
         'basis_b': tf.Variable(tf.random_normal([1, n_items], mean=0.0,
                                                 stddev=0.1), name='base_b')}
    return(Y, B)


def main_network(Y, B, _beta, true_matrix, _num, _idx):

    app_x = tf.matmul((Y['low_m'] + Y['low_b']),
                      B['basis_m'] + B['basis_b'])

    _true = tf.reshape(tf.gather_nd(params=true_matrix,
                                    indices=_idx), [-1, 1])

    _pred = tf.reshape(tf.gather_nd(app_x, indices=_idx), [-1, 1])

    regularizer_Y = tf.nn.l2_loss(tf.abs(Y['low_m']))
    regularizer_B = tf.nn.l2_loss(tf.abs(B['basis_m']))

    diff = tf.subtract(_true, _pred)
    sq_diff = tf.square(diff)
    sum_sq_diff = tf.reduce_sum(sq_diff)

    mse = tf.divide(sum_sq_diff, _num) + _beta * \
        (regularizer_Y + regularizer_B)

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr_rat).minimize(mse)

    # accuracy
    rmse = tf.sqrt(tf.divide(sum_sq_diff, _num))

    return(app_x, mse, opt, rmse)


def graph_init():
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    return(sess)


def tb_summary_init(path, loss_param, accuracy_param):
    logs_path_1 = os.path.join('./' + path, 'run_%d' % run, 'plot_1')
    logs_path_2 = os.path.join('./' + path, 'run_%d' % run, 'plot_2')
    tf.summary.histogram("weights_y", Y['low_m'])
    tf.summary.histogram("weights_b", B['basis_m'])
    tf.summary.scalar("mse", loss_param)
    tf.summary.scalar("rmse",  accuracy_param)
    summary_op = tf.summary.merge_all()
    writer_1 = tf.summary.FileWriter(logs_path_1, graph=tf.get_default_graph())
    writer_2 = tf.summary.FileWriter(logs_path_2, graph=tf.get_default_graph())
    return(summary_op, writer_1, writer_2)


def writ_meth_to_csv(full_path):
    if(os.path.isfile(full_path + '_.csv')):
        writing_method = 'a'
    else:
        writing_method = 'wb'
    return (writing_method)


def init_network(full_path, path, loss_param, accuracy_param):
    summary_op, writer_1, writer_2 = tb_summary_init(
        path, loss_param, accuracy_param)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    sess = graph_init()
    writing_method = writ_meth_to_csv(full_path)
    return(saver, summary_op, writer_1, writer_2, writing_method, sess)


def train_network(full_path, path):
    with open(full_path + '_.csv', writing_method) as csvfile:
        wr = csv.writer(csvfile,  delimiter='\t',  lineterminator='\n')
        wr.writerow(['Run', 'Iteration', "mse", "accuracy", "time",
                     "beta", "factors", "lr_rat", "train/valid", 'restored'])

        # training parameters

        beta_power = np.asarray(list(drange(-4, 4.25, 0.25)))
        beta = (10 ** beta_power).astype('float32')
        n_iter = 100000
        k = 5

        for bet in beta:
            print('training for beta = %.4f' % bet)
            t0 = time.time()
            for iteration in range(int(n_iter)):
                for c_val_num in range(1, k + 1):
                    tr_idx, _num_t, val_idx, _num_v = cross_val_miss_mat(
                        k, c_val_num, m)

                    if ((iteration + 1) % 100 == 0):
                        _mse_val, _rmse_val, summary = sess.run([mse, rmse, summary_op], feed_dict={_idx: val_idx,
                                                                                                    _beta: bet,
                                                                                                    _num: _num_v,
                                                                                                    })
                        writer_2.add_summary(summary, iteration)

                        wr.writerow([run, iteration, _mse_val, _rmse_val, str(datetime.datetime.now()), beta, _rank,
                                     lr_rat, c_val_num, "valid", restored])

                    _mse_tr, _rmse_tr,  _, summary = sess.run([mse, rmse, opt, summary_op], feed_dict={
                                                              _idx: tr_idx, _beta: bet,  _num: _num_t})

                    if ((iteration + 1) % 100 == 0):
                        t1 = time.time()
                        print('cv:', c_val_num, 'Iteration', iteration, 'train mse:', _mse_tr,
                              'train rmse:', _rmse_tr)
                        print('cv:', c_val_num, 'Iteration', iteration, 'validation mse:', _mse_val,
                              'validation rmse:', _rmse_val)
                        print(iteration, 'iterations took:',
                              t1 - t0, 'seconds')

                        writer_1.add_summary(summary, iteration)

                        wr.writerow([run, iteration, _mse_tr, _rmse_tr, str(datetime.datetime.now()), beta,
                                     _rank, lr_rat, c_val_num, "train", restored])

                    if ((iteration + 1) % 100 == 0):
                        folder = os.path.join(
                            './' + path, 'run_%d' % run, 'iter_%d' % iteration, 'weights')
                        save_path = saver.save(sess,  folder + '_')


# def main():
    # rank = [50, 100, 150, 200, 250, 300, 350]
    # n_users = int(miss_matrix.shape[0])
    # n_items = int(miss_matrix.shape[1])
    # lr_rat = 0.0001
    # restored = False
    # miss_matrix, true_matrix = gen_fake_matrix_of_ratings_with_missing(
    #     500, 2000, 0.7)
    # true_matrix = true_matrix.astype('float32')
    # m = miss_matrix.astype('float32')
    # full_path = 'missing_explicit/mat_fact_w_missing_values'
    # path = 'missing_explicit/'

    # file_l = []
    # for file in os.listdir('./missing_explicit'):
    #     if fnmatch.fnmatch(file, 'run*'):
    #         file_l.append(int(file[-2:].strip('_')))
    # run = max(file_l) + 1
    # for _rank in rank:
    #     tf.reset_default_graph()
    #     _idx, _beta, _num = init_plh()
    #     Y, B = init_vars()
    #     app_x, mse, opt, rmse = main_network(
    #         Y, B, _beta, true_matrix, _num, _idx)
    #     saver, summary_op, writer_1, writer_2, writing_method, sess = init_network(
    #         full_path, path, loss_param=mse, accuracy_param=rmse)
    #     train_network(full_path, path)
    # sess.close();

if __name__ == "__main__":
    main()
