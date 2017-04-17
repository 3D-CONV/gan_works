import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mb_size = 32
Map_size = 64
X_dim = Map_size * Map_size
z_dim = 64
h_dim = 128
lr = 1e-3
d_steps = 3

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
file_name="test.tfrecords"


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(Map_size, Map_size), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def log(x):
    return tf.log(x + 1e-8)

def read_and_decode(filename_queue):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={'img_raw': tf.FixedLenFeature([], tf.string),})

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [X_dim])
    return img

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        [filenames],shuffle=True)
    example = read_and_decode(filename_queue)
    min_after_dequeue = 400
    capacity = min_after_dequeue + 3 * batch_size
    example_batch  = tf.train.shuffle_batch(
        [example], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch


X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

D_W1 = tf.Variable(xavier_init([X_dim + z_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
Q_W2 = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2 = tf.Variable(tf.zeros(shape=[z_dim]))

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [Q_W1, Q_W2, Q_b1, Q_b2, P_W1, P_W2, P_b1, P_b2]
theta_D = [D_W1, D_W2, D_b1, D_b2]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    h = tf.matmul(h, Q_W2) + Q_b2
    return h


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    h = tf.matmul(h, P_W2) + P_b2
    return tf.nn.sigmoid(h)


def D(X, z):
    inputs = tf.concat([X, z], axis=1)
    h = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    return tf.nn.sigmoid(tf.matmul(h, D_W2) + D_b2)


z_hat = Q(X)
X_hat = P(z)

D_enc = D(X, z_hat)
D_gen = D(X_hat, z)

D_loss = -tf.reduce_mean(log(D_enc) + log(1 - D_gen))
G_loss = -tf.reduce_mean(log(D_gen) + log(1 - D_enc))

D_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(G_loss, var_list=theta_G))

# Limit GPU usage
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

init_op = tf.global_variables_initializer()
sess.run(init_op)

example_batch = input_pipeline(file_name, mb_size, 1)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for it in range(1000000):
    # import raw data
    #X_mb, _ = mnist.train.next_batch(mb_size)
    X_mb = sess.run(example_batch)
    z_mb = sample_z(mb_size, z_dim)

    _, D_loss_curr = sess.run(
        [D_solver, D_loss], feed_dict={X: X_mb, z: z_mb}
    )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss], feed_dict={X: X_mb, z: z_mb}
    )

    if it % 1000 == 0:
        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))

        samples = sess.run(X_hat, feed_dict={z: sample_z(16, z_dim)})

        fig = plot(samples)
        #fig = plot(X_mb[0:16])
        plt.savefig('out/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

coord.request_stop()
coord.join(threads)
sess.close()
