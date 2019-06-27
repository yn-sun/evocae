import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from datetime import datetime
import numpy as np
import os
import get_data
import tensorflow as tf

'''
C[K:2-F:24-L2:0.0016]
C[K:3-F:57-L2:0.0001]
C[K:5-F:63-L2:0.0096]
C[K:5-F:35-L2:0.0071]
C[K:3-F:76-L2:0.0015]
P[K:2-S:2]

C[K:4-F:36-L2:0.0001]
P[K:2-S:2]

'''

batch_size = 128
total_epochs = 100

def model():

    x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3], name='Input')
    y = tf.placeholder(dtype=tf.float32, shape=[batch_size], name='True_Y')
    y = tf.cast(y, tf.int64)
    keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='dropout')
    is_training = tf.placeholder(tf.bool, shape=())

    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.crelu, normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training, 'decay': 0.95}):
        h = slim.conv2d(inputs=x, num_outputs=24, kernel_size=2, weights_regularizer=slim.l2_regularizer(0.0016))
        h = slim.conv2d(inputs=h, num_outputs=57, kernel_size=3, weights_regularizer=slim.l2_regularizer(0.0001))
        h = slim.conv2d(inputs=h, num_outputs=63, kernel_size=5, weights_regularizer=slim.l2_regularizer(0.0096))
        h = slim.conv2d(inputs=h, num_outputs=35, kernel_size=5, weights_regularizer=slim.l2_regularizer(0.0071))
        h = slim.conv2d(inputs=h, num_outputs=76, kernel_size=3, weights_regularizer=slim.l2_regularizer(0.0015))
        h = slim.max_pool2d(h, kernel_size=2, stride=2)
        h = slim.conv2d(inputs=h, num_outputs=36, kernel_size=4, weights_regularizer=slim.l2_regularizer(0.0001))
        h = slim.max_pool2d(h, kernel_size=2, stride=2)

        flatten = slim.flatten(h)
        full = slim.fully_connected(flatten, 512)
        drop_full = slim.dropout(full, keep_prob)
        with tf.name_scope('accuracy'):
            logits = slim.fully_connected(drop_full, 10, activation_fn=None)
            correct_prediction = tf.equal(tf.argmax(logits, 1), y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))+ tf.add_n(tf.losses.get_regularization_losses())
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer()
            step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
            train_op = slim.learning.create_train_op(loss, optimizer, global_step=step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)
                loss = control_flow_ops.with_dependencies([updates], loss)


        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                train_data, train_label = get_data.get_train_data(True)
                v1_data, v1_label = get_data.get_validate_data(True)
                train_data = np.concatenate((train_data, v1_data))
                train_label = np.concatenate((train_label, v1_label))
                print(train_data.shape, train_label.shape)
                total_count = train_data.shape[0]
                idx = np.arange(total_count)
                np.random.shuffle(idx)
#                 sample_count = 10000
#                 train_data = train_data[idx[0:sample_count]]
#                 train_label = train_label[idx[0:sample_count]]
                print(train_data.shape, train_label.shape)

                validate_data, validate_label = get_data.get_test_data(True)
                epochs = total_epochs
                for current_epoch in range(epochs):
                    train_loss_list = []
                    train_accu_list = []
                    total_length = train_data.shape[0]
                    idx = np.arange(total_length)
                    np.random.shuffle(idx)
                    train_data = train_data[idx]
                    train_label = train_label[idx]
                    total_steps = total_length // batch_size
                    for step in range(total_steps):
                        batch_train_data = train_data[step*batch_size:(step+1)*batch_size]
                        batch_train_label = train_label[step*batch_size:(step+1)*batch_size]
                        _, loss_v, accuracy_str = sess.run([train_op, loss, accuracy], {x:batch_train_data, y:batch_train_label, keep_prob:0.5, is_training:True})
                        train_loss_list.append(loss_v)
                        train_accu_list.append(accuracy_str)

                    #test
                    test_length = validate_data.shape[0]
                    test_steps = test_length // batch_size
                    test_loss_list = []
                    test_accu_list = []
                    for step in range(test_steps):
                        batch_test_data = validate_data[step*batch_size:(step+1)*batch_size]
                        batch_test_label = validate_label[step*batch_size:(step+1)*batch_size]
                        loss_v, accuracy_str = sess.run([loss, accuracy], {x:batch_test_data, y:batch_test_label, keep_prob:1.0, is_training:False})
                        test_loss_list.append(loss_v)
                        test_accu_list.append(accuracy_str)

                    print('{}, epoch:{}/{}, step:{}/{}, loss:{:.6f}, accu:{:.4f}, test loss:{:.6f}, accu:{:.4f}'.format(datetime.now(), current_epoch, total_epochs, total_steps*current_epoch+step, total_steps*epochs, np.mean(train_loss_list), np.mean(train_accu_list), np.mean(test_loss_list), np.mean(test_accu_list)))

if __name__ =='__main__':
    #cuda 3
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    tf.reset_default_graph()
    model()
