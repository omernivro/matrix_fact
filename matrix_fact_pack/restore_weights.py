import tensorflow as tf
import os


# restore weights
restored = True
tf.reset_default_graph()
matrix, _idx, _beta, _num = init_plh()
Y, B = init_vars()
app_x, mse_tr, mse_val, opt, rmse_tr, rmse_val = main_network(
    Y, B, _beta, true_matrix, _num)
saver, summary_op, writer_opt, writing_method, sess = init_network()
save_MDir = './missing_explicit/run_35/iter_199/'
save_model = os.path.join(save_MDir, 'weights_')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess=sess, save_path=save_model)
