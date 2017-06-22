import cifar10_input
#cifar10_input.maybe_download_and_extract()

import tensorflow as tf
import numpy as np
import time, os
from matplotlib import pyplot as plt
import utils

# Architecture
n_hidden_1 = 256
n_hidden_2 = 256

# Parameters
learning_rate = 0.01
training_epochs = 200
batch_size = 128
display_step = 1

from PIL import Image

def crop(image):

   width  = image.size[0]
   height = image.size[1]

   aspect = width / float(height)

   ideal_width = 200
   ideal_height = 200

   #ideal_aspect = ideal_width / float(ideal_height)
   ideal_aspect = 1

   if aspect > ideal_aspect:
      # Then crop the left and right edges:
      new_width = int(ideal_aspect * height)
      offset = (width - new_width) / 2

      if width - offset*2 == height:
         resize = (offset, 0, width - offset, height)
      else:
         resize = (offset, 0, width - offset - 1, height)
      
   else:
      # ... crop the top and bottom:
      new_height = int(width / ideal_aspect)
      offset = (height - new_height) / 2
      if width == height - offset*2:
         resize = (0, offset, width, height - offset)
      else:
         resize = (0, offset, width, height - offset -1)

   return image.crop(resize)

def convert_image(path):
    image = Image.open(path)
    image = crop(image)
    image.thumbnail((32,32),Image.ANTIALIAS)
    image = np.array(image).reshape(-1, 32, 32, 3)
    return image



def plot_conv_output(conv_img, plot_dir,name):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=False)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :,  l]
        # put it on the grid
        #ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        ax.imshow(img)
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')
    plt.close()


def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join('conv_weights', name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')



def inputs(eval_data=True):
  data_dir = os.path.join('data/cifar10_data', 'cifar-10-batches-bin')
  return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=batch_size)

def distorted_inputs():
  data_dir = os.path.join('data/cifar10_data', 'cifar-10-batches-bin')
  return cifar10_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=batch_size)

def conv_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)

    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var))

    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
        beta, gamma, 1e-3, True)
    return normed

def layer_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)

    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train,
        mean_var_with_update,
        lambda: (ema_mean, ema_var))

    reshaped_x = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(reshaped_x, mean, var,
        beta, gamma, 1e-3, True)
    return tf.reshape(normed, [-1, n_out])

def filter_summary(V, weight_shape):
    ix = weight_shape[0]
    iy = weight_shape[1]
    cx, cy = 8, 8
    V_T = tf.transpose(V, (3, 0, 1, 2))
    tf.summary.image("filters", V_T, max_outputs=64) 

def conv2d(input, weight_shape, bias_shape, phase_train, visualize=False):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)

    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    
    # add weights tensor to collection
    tf.add_to_collection('conv_weights', W)

    if visualize:
        filter_summary(W, weight_shape)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    logits = tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b)
    activations = tf.nn.relu(conv_batch_norm(logits, weight_shape[3], phase_train))
    
    # add output to collection
    tf.add_to_collection('conv_output', activations)
    return activations

def max_pool(input, k=2):
    activations = tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    tf.add_to_collection('maxpool_output', activations)
    return activations


def layer(input, weight_shape, bias_shape, phase_train):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    logits = tf.matmul(input, W) + b
    return tf.nn.relu(layer_batch_norm(logits, weight_shape[1], phase_train))

def inference(x, keep_prob, phase_train):

    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 3, 64], [64], phase_train, visualize=True)
        pool_1 = max_pool(conv_1)

    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [5, 5, 64, 64], [64], phase_train)
        pool_2 = max_pool(conv_2)

    with tf.variable_scope("fc_1"):

        dim = 1
        for d in pool_2.get_shape()[1:].as_list():
            dim *= d

        pool_2_flat = tf.reshape(pool_2, [-1, dim])
        fc_1 = layer(pool_2_flat, [dim, 384], [384], phase_train)
        
        # apply dropout
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    with tf.variable_scope("fc_2"):

        fc_2 = layer(fc_1_drop, [384, 192], [192], phase_train)
        
        # apply dropout
        fc_2_drop = tf.nn.dropout(fc_2, keep_prob)

    with tf.variable_scope("output"):
        output = layer(fc_2_drop, [192, 10], [10], phase_train)

    return output


def loss(output, y):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf.cast(y, tf.int64))    
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def evaluate(output, y):
    correct_prediction = tf.equal(tf.cast(tf.argmax(output, 1), dtype=tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("validation error", (1.0 - accuracy))
    return accuracy

if __name__ == '__main__':



    with tf.device("/gpu:0"):

        with tf.Graph().as_default():

            with tf.variable_scope("cifar_conv_bn_model"):

                x = tf.placeholder("float", [None, cifar10_input.IMAGE_SIZE, cifar10_input.IMAGE_SIZE, 3])
                y = tf.placeholder("int32", [None])
                
                keep_prob = tf.placeholder(tf.float32) # dropout probability
                phase_train = tf.placeholder(tf.bool) # training or testing

                distorted_images, distorted_labels = distorted_inputs()
                val_images, val_labels = inputs()

                output = inference(x, keep_prob, phase_train)

                cost = loss(output, y)

                global_step = tf.Variable(0, name='global_step', trainable=False)

                train_op = training(cost, global_step)

                eval_op = evaluate(output, y)

                summary_op = tf.summary.merge_all()

                saver = tf.train.Saver()

                sess = tf.Session()

                summary_writer = tf.summary.FileWriter("conv_cifar_bn_logs/",
                                                        graph_def=sess.graph_def)

                
                init_op = tf.global_variables_initializer()

                sess.run(init_op)

                tf.train.start_queue_runners(sess=sess)

                # Training cycle
                for epoch in range(training_epochs):

                    avg_cost = 0.
                    total_batch = int(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/batch_size)
                    # Loop over all batches
                    for i in range(total_batch):
                        # Fit training using batch data

                        train_x, train_y = sess.run([distorted_images, distorted_labels])

                        _, new_cost = sess.run([train_op, cost], feed_dict={x: train_x, y: train_y, keep_prob: 1, phase_train: True})
                        # Compute average loss
                        avg_cost += new_cost/total_batch
                        print "Epoch %d, minibatch %d of %d. Cost = %0.4f." %(epoch, i, total_batch, new_cost)
                    
                    # Display logs per epoch step
                    if epoch % display_step == 0:
                        print "Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost)

                        val_x, val_y = sess.run([val_images, val_labels])

                        accuracy = sess.run(eval_op, feed_dict={x: val_x, y: val_y, keep_prob: 1, phase_train: False})

                        print "Validation Error:", (1 - accuracy)

                        summary_str = sess.run(summary_op, feed_dict={x: train_x, y: train_y, keep_prob: 1, phase_train: False})
                        summary_writer.add_summary(summary_str, sess.run(global_step))

                        saver.save(sess, "conv_cifar_bn_logs/model-checkpoint", global_step=global_step)


                print "Optimization Finished!"

                val_x, val_y = sess.run([val_images, val_labels])
                accuracy = sess.run(eval_op, feed_dict={x: val_x, y: val_y, keep_prob: 1, phase_train: False})

                # get output of all convolutional layers
                # here we need to provide an input image
                dt = np.dtype([('label',np.uint8),('image',np.uint8,1024*3)])
                container_images = np.fromfile('/export/ddorroh/datasets/container/batches-bin/container_images.bin',dt)
                image = container_images['image'][0].reshape(3,32,32).transpose(1,2,0).astype(np.float32).reshape(-1,32, 32, 3)
                
                files = map(lambda i: os.path.join('/export/ddorroh/datasets/container/cropped/',i),os.listdir('/export/ddorroh/datasets/container/cropped/'))
                files.sort()
                for f in files[0:100]:
                    try:
                        image = convert_image(f)
                        conv_out = sess.run([tf.get_collection('conv_output')], feed_dict={x: image, phase_train: False})

                        for i, c in enumerate(conv_out[0]):
                            name = '{}'.format(os.path.basename(f).split('.')[0])
                            plot_dir = 'conv_output/conv{}'.format(i)
                            plot_conv_output(c, plot_dir, name)
                    except ValueError:
                        print 'skipping {}'.format(f)
                        
                #conv_img = conv_out[0][0]
                #plt.imshow(conv_img[0,:,:,2])


                print "Test Accuracy:", accuracy

