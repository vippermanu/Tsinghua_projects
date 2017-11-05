from __future__ import division
import tensorflow as tf
import numpy as np
import re
import random


def load_data(txt):
    with open(txt, 'r') as fid:
        read = fid.readlines()

    dat = np.zeros((len(read), 10))
    label = []
    i = 0
    for line in read:
        row = re.split('\t| ', line)
        while '' in row:
            row.remove('')
        dat[i, :] = row[0:10]
        label.append(row[10])
        i += 1

    for j in range(len(label)):
        label[j] = label[j].replace('\r\n', '')
        label[j] = label[j].replace('\n', '')

    l = np.zeros((len(label), 2))
    for j in range(len(label)):
        if label[j] == 'M':  # 1 for male, 0 for female
            l[j][:] = [1, 0]
        else:
            l[j][:] = [0, 1]
    # label = np.reshape(label, (len(label), 1))

    return dat, l


train_data, train_label = load_data('dataset3.txt')
test_data, test_label = load_data('dataset4.txt')

H = 10  # hidden layer size
train_num, feature_num = train_data.shape

iteration = 10
error = np.zeros((iteration, 1))
for it in range(iteration):
    f_index = [random.randint(0, 468) for _ in range(10)]
    m_index = [random.randint(469, 953) for _ in range(10)]

    f = train_data[f_index, 4:6]
    m = train_data[m_index, 4:6]
    f_l = train_label[f_index]
    m_l = train_label[m_index]

    tr_data = np.vstack((f, m))
    tr_label = np.vstack((f_l, m_l))

    x = tf.placeholder(tf.float32, shape=(None, 2))
    y = tf.placeholder(tf.float32, shape=(None, 2))

    init = tf.contrib.layers.xavier_initializer()
    h = tf.layers.dense(inputs=x, units=H, activation=tf.nn.sigmoid, kernel_initializer=init)
    y_pred = tf.layers.dense(inputs=h, units=2, kernel_initializer=init)

    # loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))

    loss = tf.losses.mean_squared_error(y_pred, y)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    updates = optimizer.minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        values = {x: tr_data, y: tr_label}

        for itr in range(1000):
            loss_val, _ = sess.run([loss, updates], feed_dict=values)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        a = sess.run(accuracy, feed_dict={x: test_data[:, 4:6], y: test_label})

    error[it] = 1 - a


min_err = np.min(error)
mean_err = np.mean(error)
print 'minimum error rate:'
print min_err
print 'mean error rate:'
print mean_err