import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected, batch_norm
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

N = 784

def show_image(x, name='image'):
    image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image(name, image, max_outputs=1)

def kl_divergence(x, mu=0.0, sigma=1.0):
    mu_x = tf.reduce_mean(x)
    variance = (x - mu_x)**2.0 / N
    sigma_x = variance**0.5
    kld = 0.5 * ((mu_x - mu)**2 + sigma_x**2 + sigma**2) * ((1.0 / sigma_x**2) + (1.0 / sigma**2))
    return kld

x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
mask = tf.placeholder(tf.float32, shape=[None, 784], name='mask')

masked = tf.maximum(tf.zeros_like(x), x - mask)

code = fully_connected(masked, 784, activation_fn=tf.nn.relu)

decoded = fully_connected(code, 784, activation_fn=None)

image_loss = tf.reduce_sum((x - decoded)**2)
kld_loss = tf.reduce_sum(kl_divergence(x))

tf.summary.scalar("image_loss", image_loss)
tf.summary.scalar("kld_loss", kld_loss)

# loss = image_loss + kld_loss
loss = kld_loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

show_image(x, 'input')
show_image(x, 'output')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter('./log', tf.get_default_graph())
    merged = tf.summary.merge_all()
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(5)
        loss_, summ, _ = sess.run([loss, merged, optimizer], feed_dict={
            x: batch_xs,
            mask: np.random.randint(0, 1, [5, 784])
        })
        print "%6i: %8.2f" % (i, loss_)
        writer.add_summary(summ, i)