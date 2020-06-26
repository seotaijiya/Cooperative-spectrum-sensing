import tensorflow as tf
import numpy as np
import math
from sklearn import svm
#import matplotlib.pyplot as plt
from sklearn import metrics
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.set_random_seed(789)  # for reproducibility
#np.random.seed(789)


def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

def ch_gen(num_sens, num_prim, num_band, num_samples, p_t, max_band_span, mv_per_sample, opt=0):

    ## placing sensing entities and primary user randomly
    sen_loc = size_area*(np.random.rand(num_sens, 2)-0.5)
    pri_loc = size_area*(np.random.rand(num_prim, 2)-0.5)

    returned_list = []
    for i in range(num_samples):

        # make position change according to mv_per_sample
        if i % 10 == 0:
            pos_diff_sen_ang = 2 * math.pi * np.random.rand(num_sens, 1)
            pos_diff_pri_ang = 2 * math.pi * np.random.rand(num_prim, 1)

        pos_diff_sen = mv_per_sample*np.hstack((np.cos(pos_diff_sen_ang), np.sin(pos_diff_sen_ang)))
        pos_diff_pri = mv_per_sample*np.hstack((np.cos(pos_diff_pri_ang), np.sin(pos_diff_pri_ang)))
        pos_diff_pri = np.maximum(pos_diff_pri, 0.1)

        #updating position change
        sen_loc = sen_loc + pos_diff_sen
        pri_loc = pri_loc + pos_diff_pri


        # limiting the position diference
        sen_loc = np.minimum(sen_loc, size_area*np.ones([num_sens, 2]))
        sen_loc = np.maximum(sen_loc, -size_area*np.ones([num_sens, 2]))
        pri_loc = np.minimum(pri_loc, size_area*np.ones([num_prim, 2]))
        pri_loc = np.maximum(pri_loc, -size_area*np.ones([num_prim, 2]))

        pri_loc.reshape(num_prim, 1, 2)


        ## generate distance_vector
        ## dist_pr_su_vec is [pr_index][su_index][2]
        dist_pr_su_vec = pri_loc.reshape(num_prim, 1, 2) - sen_loc
        dist_pr_su_vec = np.maximum(dist_pr_su_vec, 0.1)
        dist_pr_su_vec = np.linalg.norm(dist_pr_su_vec, axis=2)

        dist_su_su_vec = sen_loc.reshape(num_sens, 1, 2) - sen_loc
        dist_su_su_vec = np.linalg.norm(dist_su_su_vec, axis=2)

        # find path loss and shadow fading
        pu_ch_gain_db = - pl_const - pl_alpha * np.log10(dist_pr_su_vec)
        pu_ch_gain = 10 ** (pu_ch_gain_db / 10)
        su_cor = np.exp(-dist_su_su_vec / d_ref)
        shadowing_dB = sh_sigma * np.random.multivariate_normal(np.zeros([num_sens]), su_cor, num_prim)
        shadowing = 10 ** (shadowing_dB / 10)

        if (opt == 1) | (i == 0):
            pu_power = np.zeros([len(su_cor), num_band])
            for j in range(num_prim):
                pri_power = np.zeros([num_band])
                if (np.random.rand() < pu_active_prob) | (j == 0):
                    pri_freq = np.random.randint(num_band)
                    pri_bw = np.random.randint(max_band_span)

                    pri_power[pri_freq:min(pri_freq + pri_bw, num_band + 1)] = p_t
                    pri_power[max(0, pri_freq - 1):pri_freq] = p_t / 100.
                    pri_power[
                    min(pri_freq + pri_bw, num_band + 1):min(pri_freq + pri_bw + 1, num_band + 1)] = p_t / 100.
                    pu_ch_gain_tot = pu_ch_gain[j] * shadowing[j]
                    pri_power = pri_power.reshape(num_band, 1)
                    pu_ch_gain_tot = pu_ch_gain_tot.reshape(len(su_cor), 1)
                    pu_power = pu_power +  pri_power.T*pu_ch_gain_tot
                    nan_val = np.isnan(pu_power)
                    nan_val = nan_val.astype("float")
                    if np.sum(nan_val) != 0:
                        print(dist_pr_su_vec[j])


        multi_fading = 0.5 * np.random.randn(num_sens, num_band) ** 2 + 0.5 * np.random.randn(num_sens, num_band) ** 2

        multi_fading = multi_fading ** 0.5
        final_ch = pu_power * multi_fading

        returned_list.append(final_ch)
    return returned_list


def model(X, w1, w2, w3, w4, w_o, b_4, b_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w1,
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,
                                  strides=[1, 1, 1, 1], padding='SAME'))

    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4) + b_4)
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o) + b_o
    return pyx


def conv_sensing(test, label, thr):
    test_val = np.sum(test, axis=1)
    predict = (test_val >= thr).astype(float)
    correct_mat = (predict == label[:, 0]).astype(float)
    correct = np.mean(correct_mat)
    return correct

tf.logging.set_verbosity(tf.logging.FATAL)



## set parameter

num_band = 16    #number of bands to be sensed
num_prim = 1    #number of primary users
pu_active_prob = 1.0

# size of bandwith - 10MHz
bw = 10*10**6

# tx power - 23dBm
p_t_dB = 23
p_t = 10*(p_t_dB/10)

# sensing threshold
sensing_thr = 10**(-107/10)


# size of area sensors are distributed = 1000
size_area = 200

# use pathloss constant   (wslee paper)
pl_const = 34.5
pl_alpha = 38

# shadow fading constant   (wslee paper)
d_ref = 50
sh_sigma = 7.9

# movement of entity   (wslee paper)
delta_time = 2    # units in seconds
speed = 3000/3600.            # units in km/hspeed = 3000/3600.            # units in km/h
mv_per_sample = delta_time*speed

max_band_span = 3

'''
In this simulation, we have changed the number of training sets

'''

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
train_set_num = 100
test_set_num = 1000
num_samples = train_set_num+test_set_num
batch_size = np.minimum(200, train_set_num)
opt_sensing_thr = 0      # 0: soft decision, 1: hard decision
N0W = bw*10**(-164.0/10)   # Noise: -174 dBm/Hz
num_sens = 16*2   #number of sensing entities
opt_pr_freq_hopping = 0   # 0: non hopping, 1: freq hopping
valid_set_size = 40
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# CNN development
# set parameters


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
learning_rate = 0.001



p_md_opt_fin = list()
p_md_svm_fin = list()
p_md_prop_fin = list()
p_md_prop_w_fin = list()

p_fa_opt_fin = list()
p_fa_svm_fin = list()
p_fa_prop_fin = list()
p_fa_prop_w_fin = list()


for j_3 in range(1):

    p_md_opt_tot = list()
    p_md_svm_tot = list()
    p_md_prop_tot = list()
    p_md_prop_w_tot = list()

    p_fa_opt_tot = list()
    p_fa_svm_tot = list()
    p_fa_prop_tot = list()
    p_fa_prop_w_tot = list()


    ## opt_sensing_thr = 1 => HD
    ## opt_sensing_thr = 0 => SD
    if j_3 == 1:  # HD, non hop
        opt_sensing_thr = 1
        opt_pr_freq_hopping = 0
    elif j_3 == 0:  # SD, non hop
        opt_sensing_thr = 0
        opt_pr_freq_hopping = 0



    for j_2 in range(1):


        num_samples = train_set_num + test_set_num
        batch_size = np.minimum(200, train_set_num)




        X = tf.placeholder("float", [None, num_sens, num_band, 1])
        Y = tf.placeholder("float", [None, 2])

        p_keep_conv = tf.placeholder("float")
        p_keep_hidden = tf.placeholder("float")

        w1 = tf.Variable(tf.random_normal((3, 3, 1, 8), stddev=0.01))
        w2 = tf.Variable(tf.random_normal((3, 3, 8, 8), stddev=0.01))
        w3 = tf.Variable(tf.random_normal((3, 3, 8, 8), stddev=0.01))
        # w4 = tf.get_variable("weight5", shape=[8*int(math.ceil(num_sens/8.))*int(math.ceil(num_band/8.)), 8], initializer=xavier_init(8*int(math.ceil(num_sens/8.))*int(math.ceil(num_band/8.)), 8))
        w4 = tf.Variable(
            tf.random_normal((8 * int(math.ceil(num_sens / 8.)) * int(math.ceil(num_band / 8.)), 8), stddev=0.01))
        b_4 = tf.Variable(tf.random_normal((1, 8), stddev=0.01))
        w_o = tf.Variable(tf.random_normal((8, 2), stddev=0.01))
        # w_o = tf.get_variable("weight6", shape=[8, 2], initializer=xavier_init(8, 2))
        b_o = tf.Variable(tf.random_normal((1, 2), stddev=0.01))
        py_x = model(X, w1, w2, w3, w4, w_o, b_4, b_o, p_keep_conv, p_keep_hidden)
        nn_for_roc = tf.nn.softmax(py_x)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=py_x))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        predict_op = tf.argmax(py_x, 1)
        correct_prediction = tf.equal(tf.argmax(Y, 1), predict_op)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


        p_md_opt = list()
        p_md_svm = list()
        p_md_prop = list()
        p_md_prop_w = list()

        p_fa_opt = list()
        p_fa_svm = list()
        p_fa_prop = list()
        p_fa_prop_w = list()

        granu_roc = 41
        roc_dnn_md = np.zeros((granu_roc, ))
        roc_dnn_fa = np.zeros((granu_roc, ))

        roc_koutn_md = np.zeros((granu_roc, ))
        roc_koutn_fa = np.zeros((granu_roc, ))

        roc_svm_md = np.zeros((granu_roc, ))
        roc_svm_fa = np.zeros((granu_roc, ))

        iter_num = 1000

        for j in range(iter_num):

            signal_with_pu = np.array(
                ch_gen(num_sens, num_prim, num_band, num_samples, p_t, max_band_span, mv_per_sample, opt=opt_pr_freq_hopping))

            noise_1 = np.random.randn(num_samples, num_sens, num_band)
            noise_1 = N0W / 2. * np.square(noise_1)
            signal_with_pu = signal_with_pu + noise_1
            signal_wo_pu = np.random.randn(num_samples, num_sens, num_band)
            signal_wo_pu = N0W / 2. * np.square(signal_wo_pu)

            _sample_ch = np.concatenate((signal_with_pu, signal_wo_pu))
            label_data_temp = np.concatenate((np.ones(num_samples), np.zeros(num_samples)))
            label_data = [list(label_data_temp), list(1 - label_data_temp)]
            label_data = np.transpose(label_data)



            if opt_sensing_thr == 1:    # use hard sensing results
                _sample_ch = _sample_ch > sensing_thr
                _sample_ch = _sample_ch.astype("float")
                sample_ch = (_sample_ch - np.mean(_sample_ch)) / np.sqrt(np.var(_sample_ch))
            else:
                _sample_ch = np.log10(_sample_ch)
                _sample_ch = (_sample_ch - np.mean(_sample_ch)) / np.sqrt(np.var(_sample_ch))
                sample_ch = _sample_ch



            # dividing data into two categories

            train_data = np.concatenate((sample_ch[:train_set_num], sample_ch[num_samples:num_samples+train_set_num]))
            test_data = np.concatenate((sample_ch[train_set_num:num_samples], sample_ch[num_samples+train_set_num:]))
            train_label = np.concatenate((label_data[:train_set_num], label_data[num_samples:num_samples + train_set_num]))
            test_label = np.concatenate((label_data[train_set_num:num_samples], label_data[num_samples + train_set_num:]))

            train_data = train_data.reshape(-1, num_sens, num_band, 1)
            test_data = test_data.reshape(-1, num_sens, num_band, 1)


            test_data_md = test_data[test_label[:,0]==1]
            test_data_fa = test_data[test_label[:,0]==0]
            test_label_md = test_label[test_label[:,0]==1]
            test_label_fa = test_label[test_label[:,0]==0]


            # organize data for Conventioanl scheme and SVM based scheme
            train_data_conv_md = _sample_ch[:train_set_num].reshape(train_set_num, -1)
            train_data_conv_fa = _sample_ch[num_samples:num_samples+train_set_num].reshape(train_set_num, -1)
            train_label_conv_md = np.ones([train_set_num, 1])
            train_label_conv_fa = np.zeros([train_set_num, 1])

            test_data_conv_md = _sample_ch[train_set_num:num_samples].reshape(num_samples-train_set_num, -1)
            test_data_conv_fa = _sample_ch[num_samples+train_set_num:].reshape(num_samples-train_set_num, -1)
            test_label_conv_md = np.ones([num_samples-train_set_num, 1])
            test_label_conv_fa = np.zeros([num_samples-train_set_num, 1])

            p_md_conv = list()
            p_fa_conv = list()


            for i in range(num_band*num_sens):
                p_md_t = 1-conv_sensing(train_data_conv_md, train_label_conv_md, i+1)
                p_fa_t = 1-conv_sensing(train_data_conv_fa, train_label_conv_fa, i+1)
                p_md_conv.append(p_md_t)
                p_fa_conv.append(p_fa_t)

            for i in range(granu_roc):
                tar_pf = 1-float(i/(granu_roc-1.0))
                idx = (np.abs(np.array(p_fa_conv) - tar_pf)).argmin()
                roc_koutn_md[i] = roc_koutn_md[i] + (1-p_md_conv[idx])/iter_num




            index_mat = np.abs(np.array(p_fa_conv) + np.array(p_md_conv))
            index = np.argmin(index_mat)

            p_md_opt.append(1 - conv_sensing(test_data_conv_md, test_label_conv_md, index+1))
            p_fa_opt.append(1 - conv_sensing(test_data_conv_fa, test_label_conv_fa, index+1))




            """
            Find the performance of SVM based scheme
            """

            clf_s = svm.LinearSVC()

            clf_s.fit(np.concatenate((train_data_conv_md, train_data_conv_fa), axis = 0), np.ravel(np.concatenate((train_label_conv_md, train_label_conv_fa), axis = 0)))

            svm_test_md = clf_s.predict(test_data_conv_md)
            svm_test_fa = clf_s.predict(test_data_conv_fa)


            fpr, tpr, thresholds = metrics.roc_curve(np.ravel(np.concatenate((test_label_conv_md, test_label_conv_fa), axis = 0)), clf_s.decision_function(np.concatenate((test_data_conv_md, test_data_conv_fa), axis = 0)))
            #print("fpr = ", fpr)
            #print("trp = ", tpr)

            fpr_temp = fpr[1:]
            tpr_temp = tpr[1:]
            for i in range(granu_roc):
                tar_pf = 1-float(i/(granu_roc-1.0))
                idx = (np.abs(np.array(fpr_temp) - tar_pf)).argmin()
                roc_svm_md[i] = roc_svm_md[i] + (tpr_temp[idx])/iter_num




            p_md_svm.append(np.mean(1-svm_test_md))
            p_fa_svm.append(np.mean(svm_test_fa))


            training_epoch = 400



            with tf.Session() as sess:

                md_val_valid_pre = 1.0
                fa_val_valid_pre = 1.0


                for k in range(1):
                    tf.initialize_all_variables().run()
                    perm_2 = np.random.permutation(num_sens)
                    train_data[:,:] = train_data[:,perm_2]
                    test_data[:,:] = test_data[:,perm_2]

                    avg_cost = 0.
                    for i in range(training_epoch):
                        perm_1 = np.random.permutation(train_data.shape[0])
                        train_data[:] = train_data[perm_1]
                        train_label[:] = train_label[perm_1]
                        training_batch = zip(range(0, len(train_data), batch_size), range(batch_size, len(train_data), batch_size))

                        for start, end in training_batch:
                            sess.run(train_op, feed_dict={X: train_data[start:end], Y: train_label[start:end],
                                                  p_keep_conv: 0.9, p_keep_hidden: 0.9})

                        accur_val = sess.run(accuracy, feed_dict={X: train_data, Y: train_label,
                                                                  p_keep_conv: 1.0, p_keep_hidden: 1.0})

                        cost_val = sess.run(cost, feed_dict={X: train_data, Y: train_label,
                                                                  p_keep_conv: 1.0, p_keep_hidden: 1.0})

                        if i % 10 == -1:
                            print("arcur_val_training = ", accur_val)
                            print("cost_val_training = ", cost_val)
                            print("")



                    result_val = sess.run(nn_for_roc, feed_dict={X: train_data, Y: train_label,
                                                       p_keep_conv: 1.0, p_keep_hidden: 1.0})
                    accur_val = sess.run(accuracy, feed_dict={X: train_data, Y: train_label,
                                                          p_keep_conv: 1.0, p_keep_hidden: 1.0})

                    md_val = 1 - sess.run(accuracy, feed_dict={X: test_data_md, Y: test_label_md,
                                                           p_keep_conv: 1.0, p_keep_hidden: 1.0})
                    fa_val = 1 - sess.run(accuracy, feed_dict={X: test_data_fa, Y: test_label_fa,
                                                           p_keep_conv: 1.0, p_keep_hidden: 1.0})


                    result_md_val = sess.run(nn_for_roc, feed_dict={X: test_data_md, Y: test_label_md,
                                                               p_keep_conv: 1.0, p_keep_hidden: 1.0})

                    result_fa_val = sess.run(nn_for_roc, feed_dict={X: test_data_fa, Y: test_label_fa,
                                                               p_keep_conv: 1.0, p_keep_hidden: 1.0})

                    sorted_fa = np.sort(result_fa_val[:, 0])
                    for k_roc in range(granu_roc):
                        roc_index = sorted_fa[min(int(test_set_num*k_roc/(granu_roc-1.0)), test_set_num-1)]
                        md_roc = np.sum(np.array(result_md_val[:,0] > roc_index).astype(float))/test_set_num         #Detection probl
                        fa_roc = np.sum(np.array(result_fa_val[:, 0] > roc_index).astype(float))/test_set_num    #False alarm
                        #print("md_roc = ", md_roc)
                        #print("fa_roc = ", fa_roc)
                        #print("result_fa_val = ", sorted_fa)
                        md_roc = max(md_roc, 1-k_roc/(granu_roc-1.0))
                        roc_dnn_md[k_roc] = roc_dnn_md[k_roc] + md_roc/iter_num
                        roc_dnn_fa[k_roc] = roc_dnn_fa[k_roc] + fa_roc/iter_num




                    print("")
                    print("")
                    print("j = ", j)
                    print("k = ", k)
                    #print("results_val (train) = ", result_val[:10])
                    #print("result_md_val (test) = ", result_md_val[:10])
                    #print("result_fa_val (test) = ", result_fa_val[:10])
                    print("roc_koutn_md = ", roc_koutn_md * iter_num / (j + 1))
                    print("roc_svm_md = ", roc_svm_md * iter_num / (j + 1))
                    print("roc_dnn_md = ", roc_dnn_md * iter_num / (j + 1))
                    print("")
                    print("")

                    #############################
                    ## Valid set is to find the best
                    ## permutation of the data
                    ############################
                    md_val_valid = 1 - sess.run(accuracy, feed_dict={X: test_data_md[:valid_set_size], Y: test_label_md[:valid_set_size],
                                                                   p_keep_conv: 1.0, p_keep_hidden: 1.0})
                    fa_val_valid = 1 - sess.run(accuracy, feed_dict={X: test_data_fa[:valid_set_size], Y: test_label_fa[:valid_set_size],
                                                               p_keep_conv: 1.0, p_keep_hidden: 1.0})


                    if (md_val_valid + fa_val_valid) < (md_val_valid_pre + fa_val_valid_pre):
                        accur_val_pre = accur_val
                        md_val_fin = md_val
                        fa_val_fin = fa_val
                        md_val_valid_pre = md_val_valid
                        fa_val_valid_pre = fa_val_valid
                    p_md_prop_w.append(md_val)
                    p_fa_prop_w.append(fa_val)

                p_md_prop.append(md_val_fin)
                p_fa_prop.append(fa_val_fin)


            if j % 25 == 0:
                print("%" * 50)
                print("iter_num =  ", j)
                print("Sensing SU numbers = ", num_sens)
                print("Detection type = ", ("Soft" if opt_sensing_thr == 0 else "Hard"))
                print("Number of training samples = ", train_set_num)
                print("Number of test samples = ", test_set_num)
                print("Frequency hopping = ", ("not hopped" if opt_pr_freq_hopping == 0 else "hopped"))

                print("="*50)
                print("conv pmd", np.mean(p_md_opt))
                print("conv pfa", np.mean(p_fa_opt))

                print("=" * 50)
                print("svm pmd", np.mean(p_md_svm))
                print("svm pfa", np.mean(p_fa_svm))

                print("=" * 50)
                print("proposed pmd (best)", np.mean(p_md_prop))
                print("proposed pfa (best)", np.mean(p_fa_prop))

                print("=" * 50)
                print("proposed pmd (worst)", np.mean(p_md_prop_w))
                print("proposed pfa (worst)", np.mean(p_fa_prop_w))
                print("%" * 50)

            sess.close()

        p_md_opt_tot.append(np.mean(p_md_opt)*100)
        p_md_svm_tot.append(np.mean(p_md_svm)*100)
        p_md_prop_tot.append(np.mean(p_md_prop)*100)
        p_md_prop_w_tot.append(np.mean(p_md_prop_w)*100)

        p_fa_opt_tot.append(np.mean(p_fa_opt)*100)
        p_fa_svm_tot.append(np.mean(p_fa_svm)*100)
        p_fa_prop_tot.append(np.mean(p_fa_prop)*100)
        p_fa_prop_w_tot.append(np.mean(p_fa_prop_w)*100)

        print("%" * 50)
        print("iter_num =  ", j)
        print("Sensing SU numbers = ", num_sens)
        print("Detection type = ", ("Soft" if opt_sensing_thr == 0 else "Hard"))
        print("Number of training samples = ", train_set_num)
        print("Number of test samples = ", test_set_num)
        print("Frequency hopping = ", ("not hopped" if opt_pr_freq_hopping == 0 else "hopped"))
        print("Noise power = ", N0W)

        print("=" * 50)
        print("conv pmd", np.array(p_md_opt_tot))
        print("conv pfa", np.array(p_fa_opt_tot))

        print("=" * 50)
        print("svm pmd", np.array(p_md_svm_tot))
        print("svm pfa", np.array(p_fa_svm_tot))

        print("=" * 50)
        print("proposed pmd (best)", np.array(p_md_prop_tot))
        print("proposed pfa (best)", np.array(p_fa_prop_tot))

        print("=" * 50)
        print("proposed pmd (worst)", np.array(p_md_prop_w_tot))
        print("proposed pfa (worst)", np.array(p_fa_prop_w_tot))
        print("%" * 50)

    p_md_opt_fin.append(p_md_opt_tot)
    p_md_svm_fin.append(p_md_svm_tot)
    p_md_prop_fin.append(p_md_prop_tot)
    p_md_prop_w_fin.append(p_md_prop_w_tot)

    p_fa_opt_fin.append(p_fa_opt_tot)
    p_fa_svm_fin.append(p_fa_svm_tot)
    p_fa_prop_fin.append(p_fa_prop_tot)
    p_fa_prop_w_fin.append(p_fa_prop_w_tot)


    print("%" * 50)
    print("iter_num =  ", j)
    print("Sensing SU numbers = ", num_sens)
    print("Number of training samples = ", train_set_num)
    print("Number of test samples = ", test_set_num)
    print("Noise power = ", N0W)

    print("=" * 50)
    print("HD")
    print("SD")

    print("=" * 50)
    print("conv accu", np.array(p_md_opt_fin) + np.array(p_fa_opt_fin))

    print("=" * 50)
    print("svm accu", np.array(p_md_svm_fin)+np.array(p_fa_svm_fin))

    print("=" * 50)
    print("proposed accu (best)", np.array(p_md_prop_fin)+np.array(p_fa_prop_fin))

    print("=" * 50)
    print("proposed accu (worst)", np.array(p_md_prop_w_fin)+np.array(p_fa_prop_w_fin))
    print("%" * 50)


