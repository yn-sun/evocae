import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from ae import CAE
import os
import get_data
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from libpasteurize.fixes.feature_base import Feature
from datetime import datetime
from utils import *

class FitnessAssignment():
    def __init__(self, pops, params):
        self.pops = pops
        self.params = params

    def evalue_all(self, gen_no):
        for i in range(self.pops.get_pop_size()):
            tf.reset_default_graph()
            cae = self.pops.indi[i]
            score = self.build_cae(cae)
            #score = np.random.random()
            cae.score = score
#             list_save_path = os.getcwd() + '/save_data/pop_{:03d}.txt'.format(gen_no)
#             save_append_individual(str(cae), list_save_path)
            print('{}, gen:{}, cae:{}/{}, mse:{:.6f}'.format(datetime.now(), gen_no, i, self.pops.get_pop_size(), score))

    def build_cae(self, cae):
        x = tf.placeholder(dtype=tf.float32, shape=[self.params['batch_size'], self.params['input_size'], self.params['input_size'], self.params['channel']], name='INPUT')
        out_channel_list = [self.params['channel']]
        output_list = [x]
        for i in range(len(cae.units)):
            unit = cae.units[i]
            if unit.type == 1:
                out_channel_list.append(unit.feature_size)
                h =slim.conv2d(inputs=output_list[-1], num_outputs=unit.feature_size, kernel_size=unit.kernel, scope='conv{}'.format(i), weights_regularizer=slim.regularizers.l2_regularizer(unit.l2))
                output_list.append(h)
            else:
                h = slim.max_pool2d(output_list[-1], unit.kernel, unit.stride, padding='SAME', scope='pool{}'.format(i))
                output_list.append(h)
        del out_channel_list[-1]
        #decoder
        out_channel_select_index = len(out_channel_list)-1
        for i in range(cae.get_length()-1, -1, -1):
            unit = cae.units[i]
            if unit.type == 1:
                h = slim.conv2d_transpose(inputs=output_list[-1], num_outputs=out_channel_list[out_channel_select_index], kernel_size=unit.kernel, scope='unconv{}'.format(i))
                output_list.append(h)
                out_channel_select_index -= 1
            else:
                h = self.max_unpool_2x2(output_list[-1], 'unpool{}'.format(i))
                output_list.append(h)

        reconstruct = output_list[-1]
        l1_l2_losses=tf.add_n(tf.losses.get_regularization_losses())
        pure_loss = tf.losses.mean_squared_error(x, reconstruct, scope='loss')
        loss = pure_loss + l1_l2_losses
        train_op = tf.train.AdamOptimizer().minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_data = self.params['train_data']
            epochs = self.params['epochs']
            batch_size = self.params['batch_size']
            total_length = train_data.shape[0]
            total_steps = total_length // batch_size
            for current_epoch in range(epochs):
                train_loss_list = []
                train_pure_loss_list = []
                idx = np.arange(total_length)
                np.random.shuffle(idx)
                for step in range(total_steps):
                    batch_index = idx[step:step+batch_size]
                    batch_train_data = train_data[batch_index]
                    _, loss_v, pure_loss_v = sess.run([train_op, loss, pure_loss], {x:batch_train_data})
                    train_loss_list.append(loss_v)
                    train_pure_loss_list.append(pure_loss_v)

                print('{}, epoch:{}/{}, step:{}/{}, loss:{:.6f}, pure_loss:{:.6f}'.format(datetime.now(), current_epoch, self.params['epochs'], total_steps*current_epoch+step, total_steps*epochs, np.mean(train_loss_list), np.mean(train_pure_loss_list)))
            #test
            validate_data = self.params['validate_data']
            test_length = validate_data.shape[0]
            test_steps = test_length // batch_size
            test_loss_list = []
            test_pure_loss_list = []
            for step in range(test_steps):
                batch_test_data = validate_data[step*batch_size:(step+1)*batch_size]
                loss_v, pure_loss_v = sess.run([loss, pure_loss], {x:batch_test_data})
                test_loss_list.append(loss_v)
                test_pure_loss_list.append(pure_loss_v)
            print('{}, validate loss:{:.6f}, pure loss:{:.6f}'.format(datetime.now(), np.mean(test_loss_list), np.mean(test_pure_loss_list)))

        return np.mean(test_pure_loss_list)

    def max_unpool_2x2(self, x, name):
        width = x.get_shape()[1].value
        height = x.get_shape()[2].value
        inference =  tf.image.resize_images(x, [width*2, height*2])
        return inference

if __name__ == '__main__':
    #cuda3
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    unlabeldata = get_data.get_unlabeled_data()
    train_data, train_label = get_data.get_train_data()
    test_data, test_label = get_data.get_test_data()

    params = {}
    params['unlabel_data'] = unlabeldata
    params['train_data'] = train_data
    params['train_label'] = train_label
    params['test_data'] = test_data
    params['test_label'] = test_label
    params['pop_size'] = 50
    params['num_class'] = 10
    params['cae_length'] = 5
    params['x_prob'] = 0.9
    params['x_eta'] = 20
    params['m_prob'] = 0.1
    params['m_eta'] = 20
    params['total_generation'] = 50

    params['batch_size'] = 128
    params['epochs'] = 5
    params['input_size'] = train_data.shape[2]
    params['channel'] = train_data.shape[3]

    cae = CAE(params['m_prob'], params['m_eta'])
    conv1 = cae.random_a_conv()
    conv2 = cae.random_a_conv()
    pool1 = cae.random_a_pool()
    cae.units.extend([conv1, conv2, pool1])

    pool1 = cae.random_a_pool()
    cae.units.extend([conv1, pool1])
    f= FitnessAssignment(None, params)
    f.build_cae(cae)





