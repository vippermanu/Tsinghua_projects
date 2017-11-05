from __future__ import division
import tensorflow as tf
import numpy as np
import re


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

H = 50  # hidden layer size
train_num, feature_num = train_data.shape
x = tf.placeholder(tf.float32, shape=(None, feature_num), name='input')
y = tf.placeholder(tf.float32, shape=(None, 2), name='label')

init = tf.contrib.layers.xavier_initializer()
h = tf.layers.dense(inputs=x, units=H, activation=tf.nn.sigmoid, kernel_initializer=init, name='hidden')
y_pred = tf.layers.dense(inputs=h, units=2, kernel_initializer=init, name='output')

# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))

loss = tf.losses.mean_squared_error(y_pred, y)
optimizer = tf.train.GradientDescentOptimizer(0.01)
updates = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {x: train_data, y: train_label}

    for it in range(1000):
        loss_val, _ = sess.run([loss, updates], feed_dict=values)

    # saver.save(sess, "model/mlp_all_10.ckpt")
    # print(sess.run(y_pred, feed_dict={x: test_data}))
    print(sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
