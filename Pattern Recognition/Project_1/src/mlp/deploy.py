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

# test_data = test_data[:, 4:6]     # if u want to use 2-d features, just remove the # at the beginning

ckpt = tf.train.get_checkpoint_state('model/model_all_10/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    accuracy = tf.get_default_graph().get_tensor_by_name('accuracy:0')

    x = tf.get_default_graph().get_tensor_by_name('input:0')
    y = tf.get_default_graph().get_tensor_by_name('label:0')
    ac = sess.run(accuracy, feed_dict={x: test_data, y: test_label})

print('error rate:')
print(1 - ac)


