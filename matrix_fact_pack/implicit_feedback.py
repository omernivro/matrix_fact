

def main_network(Y, B, _beta, _alpha, true_matrix, confidence_matrix, _num, _idx):

    app_x = tf.matmul((Y['low_m'] + Y['low_b']),
                      B['basis_m'] + B['basis_b'])

    _true = tf.reshape(tf.gather_nd(params=true_matrix,
                                    indices=_idx), [-1, 1])

    _pred = tf.reshape(tf.gather_nd(app_x, indices=_idx), [-1, 1])

    confidence_gathered = tf.reshape(tf.gather_nd(
        confidence_matrix, indices=_idx), [-1, 1])

    num_u = tf.shape(true_matrix)[0]
    num_i = tf.shape(tf.reshape(_pred, [num_u, -1]))[1]

    conf = tf.add((_alpha * confidence_gathered),
                  tf.reshape(tf.ones(shape=[tf.size(confidence_gathered)]), [-1, 1]))

    regularizer_Y = tf.nn.l2_loss(tf.abs(Y['low_m']))
    regularizer_B = tf.nn.l2_loss(tf.abs(B['basis_m']))

    diff = tf.subtract(_true, _pred)
    sq_diff = tf.square(diff)
    confident_sq_diff = conf * sq_diff
    sum_sq_diff = tf.reduce_sum(confident_sq_diff)

    mse = tf.divide(sum_sq_diff, _num) + _beta * \
        (regularizer_Y + regularizer_B)

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr_rat).minimize(mse)

    top_k = tf.to_int32(tf.divide(num_i, tf.constant(4)))

    # only makes sense to use percentile_rank if data is sparse and users watch less than 50% of videos
    ordered_list_vals, ordered_list_idx = tf.nn.top_k(
        tf.reshape(_pred, [-1, num_i]), top_k)

    percentiles = (tf.to_float(tf.range(tf.size(ordered_list_idx)))
                   * tf.to_float(tf.divide(1, tf.size(ordered_list_idx))))

    idx_to_gather = tf.concat([tf.reshape(tf.transpose(tf.tile(input=[tf.range(num_u)],
                                                               multiples=[top_k, 1])), [-1, 1]), tf.reshape(ordered_list_idx, [-1, 1])], 1)

    r_ui = tf.gather_nd(params=tf.reshape(
        confidence_gathered, [-1, num_i]), indices=idx_to_gather)

    # accuracy
    ranking = tf.divide(tf.reduce_sum(r_ui * percentiles), tf.reduce_sum(r_ui))

    return(mse, opt, ranking)


def train_network(full_path, path):
    with open(full_path + '_.csv', writing_method) as csvfile:
        wr = csv.writer(csvfile,  delimiter='\t',  lineterminator='\n')
        wr.writerow(['Run', 'Iteration', "mse", "accuracy", "time", "beta",
                     "alpha", "factors", "lr_rat", "train/valid", 'restored'])

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
                        k, c_val_num, true_matrix)

                    if ((iteration + 1) % 1000 == 0):
                        _mse_val, _ranking_val, summary = sess.run([mse, ranking, summary_op], feed_dict={_idx: val_idx,
                                                                                                          _beta: bet,
                                                                                                          _num: _num_v
                                                                                                          })
                        writer_2.add_summary(summary, iteration)

                        wr.writerow([run, iteration, _mse_val, _ranking_val, str(datetime.datetime.now()), bet, _alpha, _rank,
                                     lr_rat, c_val_num, "valid", restored])

                    _mse_tr,  _, _ranking, summary = sess.run([mse, opt, ranking, summary_op], feed_dict={
                                                              _idx: tr_idx, _beta: bet,  _num: _num_t})

                    if ((iteration + 1) % 1000 == 0):
                        t1 = time.time()
                        print('cv:', c_val_num, 'Iteration', iteration, 'train mse:', _mse_tr,
                              'train accuracy:', _ranking)
                        print('cv:', c_val_num, 'Iteration', iteration, 'validation mse:', _mse_val,
                              'validation accuracy:', _ranking_val)
                        print(iteration, 'iterations took:',
                              t1 - t0, 'seconds')

                        writer_1.add_summary(summary, iteration)

                        wr.writerow([run, iteration, _mse_tr, _ranking, str(datetime.datetime.now()), bet, _alpha,
                                     _rank, lr_rat, c_val_num, "train", restored])

                    if ((iteration + 1) % 1000 == 0):
                        folder = os.path.join(
                            './' + path, 'run_%d' % run, 'iter_%d' % iteration, 'weights')
                        save_path = saver.save(sess,  folder + '_')


def main():

    # parameters
    alpha = [40, 30, 20, 10]

    rank = [50, 100, 150, 200, 250, 300, 350]
    n_users = int(true_matrix.shape[0])
    n_items = int(true_matrix.shape[1])
    lr_rat = 0.0001
    restored = False
    e = gen_fake_matrix_implicit_confid(20, 100) * 2

    true_matrix = r.astype('float32')

    confidence_matrix = (e * r).astype('float32')

    full_path = 'IMF/mat_fact_w_missing_values'
    path = 'IMF'

    file_l = []
    for file in os.listdir('./IMF'):
        if fnmatch.fnmatch(file, 'run*'):
            file_l.append(int(file[-2:].strip('_')))
    run = max(file_l) + 1
    for _rank in rank:
        for _alpha in alpha:
            tf.reset_default_graph()
            _idx, _beta, _num = init_plh()
            Y, B = init_vars()
            mse, opt, ranking = main_network(
                Y, B, _beta, _alpha, true_matrix, confidence_matrix, _num, _idx)
            saver, summary_op, writer_1, writer_2, writing_method, sess = init_network(
                full_path, path, mse, ranking)
            train_network(full_path, path)
    sess.close()


if __name__ == "__main__":
    main()
