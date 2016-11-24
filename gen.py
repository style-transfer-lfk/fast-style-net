import numpy as np
import tensorflow as tf
from PIL import Image
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", default='')
parser.add_option("-o", "--output", dest="output", default='')
parser.add_option("-g", "--graph", dest="graph", default='')
parser.add_option("-m", "--model", dest="model", default='seurat')

(options, args) = parser.parse_args()

def conv_layer(x, W, b, stride, mean, var, beta, gamma):
    h = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
    h = tf.nn.elu(h + b)
    h = tf.nn.batch_normalization(h, mean, var, beta, gamma, 2e-5)
    return h

def residual(x, W1, b1, W2, b2, mean1, mean2, var1, var2, beta1, beta2, gamma1, gamma2):
    h = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
    h = tf.nn.batch_normalization(h, mean1, var1, beta1, gamma1, 2e-5)
    h = tf.nn.relu(h)
    h = tf.nn.conv2d(h, W2, strides=[1, 1, 1, 1], padding='SAME') + b2
    h = tf.nn.batch_normalization(h, mean2, var2, beta2, gamma2, 2e-5)
    return x + h

def deconv_layer(x, W, b, stride, mean, var, beta, gamma):
    n, h, w, _ = x.get_shape().as_list()
    _, _, c, _ = W.get_shape().as_list()

    h = tf.nn.conv2d_transpose(x, W, [n, h * stride, w * stride, c], strides=[1, stride, stride, 1], padding='SAME')
    h = tf.nn.elu(h + b)
    h = tf.nn.batch_normalization(h, mean, var, beta, gamma, 2e-5)
    return h

if options.input != '':
    image = Image.open(options.input).convert('RGB')
    image = np.asarray(image, dtype=np.float32)
    image = image.reshape((1,) + image.shape)

model_name = options.model

graph = tf.Graph()
with graph.as_default():
    c1_W     = tf.constant(np.load('models/' + model_name + '/c1/W.npy').transpose(2, 3, 1, 0))
    c1_b     = tf.constant(np.load('models/' + model_name + '/c1/b.npy'))
    b1_mean  = tf.constant(np.load('models/' + model_name + '/b1/avg_mean.npy'))
    b1_var   = tf.constant(np.load('models/' + model_name + '/b1/avg_var.npy'))
    b1_beta  = tf.constant(np.load('models/' + model_name + '/b1/beta.npy'))
    b1_gamma = tf.constant(np.load('models/' + model_name + '/b1/gamma.npy'))

    c2_W     = tf.constant(np.load('models/' + model_name + '/c2/W.npy').transpose(2, 3, 1, 0))
    c2_b     = tf.constant(np.load('models/' + model_name + '/c2/b.npy'))
    b2_mean  = tf.constant(np.load('models/' + model_name + '/b2/avg_mean.npy'))
    b2_var   = tf.constant(np.load('models/' + model_name + '/b2/avg_var.npy'))
    b2_beta  = tf.constant(np.load('models/' + model_name + '/b2/beta.npy'))
    b2_gamma = tf.constant(np.load('models/' + model_name + '/b2/gamma.npy'))

    c3_W     = tf.constant(np.load('models/' + model_name + '/c3/W.npy').transpose(2, 3, 1, 0))
    c3_b     = tf.constant(np.load('models/' + model_name + '/c3/b.npy'))
    b3_mean  = tf.constant(np.load('models/' + model_name + '/b3/avg_mean.npy'))
    b3_var   = tf.constant(np.load('models/' + model_name + '/b3/avg_var.npy'))
    b3_beta  = tf.constant(np.load('models/' + model_name + '/b3/beta.npy'))
    b3_gamma = tf.constant(np.load('models/' + model_name + '/b3/gamma.npy'))

    r1_c1_W     = tf.constant(np.load('models/' + model_name + '/r1/c1/W.npy').transpose(2, 3, 1, 0))
    r1_c1_b     = tf.constant(np.load('models/' + model_name + '/r1/c1/b.npy'))
    r1_c2_W     = tf.constant(np.load('models/' + model_name + '/r1/c2/W.npy').transpose(2, 3, 1, 0))
    r1_c2_b     = tf.constant(np.load('models/' + model_name + '/r1/c2/b.npy'))
    r1_b1_mean  = tf.constant(np.load('models/' + model_name + '/r1/b1/avg_mean.npy'))
    r1_b1_var   = tf.constant(np.load('models/' + model_name + '/r1/b1/avg_var.npy'))
    r1_b1_beta  = tf.constant(np.load('models/' + model_name + '/r1/b1/beta.npy'))
    r1_b1_gamma = tf.constant(np.load('models/' + model_name + '/r1/b1/gamma.npy'))
    r1_b2_mean  = tf.constant(np.load('models/' + model_name + '/r1/b2/avg_mean.npy'))
    r1_b2_var   = tf.constant(np.load('models/' + model_name + '/r1/b2/avg_var.npy'))
    r1_b2_beta  = tf.constant(np.load('models/' + model_name + '/r1/b2/beta.npy'))
    r1_b2_gamma = tf.constant(np.load('models/' + model_name + '/r1/b2/gamma.npy'))

    r2_c1_W     = tf.constant(np.load('models/' + model_name + '/r2/c1/W.npy').transpose(2, 3, 1, 0))
    r2_c1_b     = tf.constant(np.load('models/' + model_name + '/r2/c1/b.npy'))
    r2_c2_W     = tf.constant(np.load('models/' + model_name + '/r2/c2/W.npy').transpose(2, 3, 1, 0))
    r2_c2_b     = tf.constant(np.load('models/' + model_name + '/r2/c2/b.npy'))
    r2_b1_mean  = tf.constant(np.load('models/' + model_name + '/r2/b1/avg_mean.npy'))
    r2_b1_var   = tf.constant(np.load('models/' + model_name + '/r2/b1/avg_var.npy'))
    r2_b1_beta  = tf.constant(np.load('models/' + model_name + '/r2/b1/beta.npy'))
    r2_b1_gamma = tf.constant(np.load('models/' + model_name + '/r2/b1/gamma.npy'))
    r2_b2_mean  = tf.constant(np.load('models/' + model_name + '/r2/b2/avg_mean.npy'))
    r2_b2_var   = tf.constant(np.load('models/' + model_name + '/r2/b2/avg_var.npy'))
    r2_b2_beta  = tf.constant(np.load('models/' + model_name + '/r2/b2/beta.npy'))
    r2_b2_gamma = tf.constant(np.load('models/' + model_name + '/r2/b2/gamma.npy'))

    r3_c1_W     = tf.constant(np.load('models/' + model_name + '/r3/c1/W.npy').transpose(2, 3, 1, 0))
    r3_c1_b     = tf.constant(np.load('models/' + model_name + '/r3/c1/b.npy'))
    r3_c2_W     = tf.constant(np.load('models/' + model_name + '/r3/c2/W.npy').transpose(2, 3, 1, 0))
    r3_c2_b     = tf.constant(np.load('models/' + model_name + '/r3/c2/b.npy'))
    r3_b1_mean  = tf.constant(np.load('models/' + model_name + '/r3/b1/avg_mean.npy'))
    r3_b1_var   = tf.constant(np.load('models/' + model_name + '/r3/b1/avg_var.npy'))
    r3_b1_beta  = tf.constant(np.load('models/' + model_name + '/r3/b1/beta.npy'))
    r3_b1_gamma = tf.constant(np.load('models/' + model_name + '/r3/b1/gamma.npy'))
    r3_b2_mean  = tf.constant(np.load('models/' + model_name + '/r3/b2/avg_mean.npy'))
    r3_b2_var   = tf.constant(np.load('models/' + model_name + '/r3/b2/avg_var.npy'))
    r3_b2_beta  = tf.constant(np.load('models/' + model_name + '/r3/b2/beta.npy'))
    r3_b2_gamma = tf.constant(np.load('models/' + model_name + '/r3/b2/gamma.npy'))

    r4_c1_W     = tf.constant(np.load('models/' + model_name + '/r4/c1/W.npy').transpose(2, 3, 1, 0))
    r4_c1_b     = tf.constant(np.load('models/' + model_name + '/r4/c1/b.npy'))
    r4_c2_W     = tf.constant(np.load('models/' + model_name + '/r4/c2/W.npy').transpose(2, 3, 1, 0))
    r4_c2_b     = tf.constant(np.load('models/' + model_name + '/r4/c2/b.npy'))
    r4_b1_mean  = tf.constant(np.load('models/' + model_name + '/r4/b1/avg_mean.npy'))
    r4_b1_var   = tf.constant(np.load('models/' + model_name + '/r4/b1/avg_var.npy'))
    r4_b1_beta  = tf.constant(np.load('models/' + model_name + '/r4/b1/beta.npy'))
    r4_b1_gamma = tf.constant(np.load('models/' + model_name + '/r4/b1/gamma.npy'))
    r4_b2_mean  = tf.constant(np.load('models/' + model_name + '/r4/b2/avg_mean.npy'))
    r4_b2_var   = tf.constant(np.load('models/' + model_name + '/r4/b2/avg_var.npy'))
    r4_b2_beta  = tf.constant(np.load('models/' + model_name + '/r4/b2/beta.npy'))
    r4_b2_gamma = tf.constant(np.load('models/' + model_name + '/r4/b2/gamma.npy'))

    r5_c1_W     = tf.constant(np.load('models/' + model_name + '/r5/c1/W.npy').transpose(2, 3, 1, 0))
    r5_c1_b     = tf.constant(np.load('models/' + model_name + '/r5/c1/b.npy'))
    r5_c2_W     = tf.constant(np.load('models/' + model_name + '/r5/c2/W.npy').transpose(2, 3, 1, 0))
    r5_c2_b     = tf.constant(np.load('models/' + model_name + '/r5/c2/b.npy'))
    r5_b1_mean  = tf.constant(np.load('models/' + model_name + '/r5/b1/avg_mean.npy'))
    r5_b1_var   = tf.constant(np.load('models/' + model_name + '/r5/b1/avg_var.npy'))
    r5_b1_beta  = tf.constant(np.load('models/' + model_name + '/r5/b1/beta.npy'))
    r5_b1_gamma = tf.constant(np.load('models/' + model_name + '/r5/b1/gamma.npy'))
    r5_b2_mean  = tf.constant(np.load('models/' + model_name + '/r5/b2/avg_mean.npy'))
    r5_b2_var   = tf.constant(np.load('models/' + model_name + '/r5/b2/avg_var.npy'))
    r5_b2_beta  = tf.constant(np.load('models/' + model_name + '/r5/b2/beta.npy'))
    r5_b2_gamma = tf.constant(np.load('models/' + model_name + '/r5/b2/gamma.npy'))

    d1_W     = tf.constant(np.load('models/' + model_name + '/d1/W.npy').transpose(2, 3, 1, 0))
    d1_b     = tf.constant(np.load('models/' + model_name + '/d1/b.npy'))
    b4_mean  = tf.constant(np.load('models/' + model_name + '/b4/avg_mean.npy'))
    b4_var   = tf.constant(np.load('models/' + model_name + '/b4/avg_var.npy'))
    b4_beta  = tf.constant(np.load('models/' + model_name + '/b4/beta.npy'))
    b4_gamma = tf.constant(np.load('models/' + model_name + '/b4/gamma.npy'))

    d2_W     = tf.constant(np.load('models/' + model_name + '/d2/W.npy').transpose(2, 3, 1, 0))
    d2_b     = tf.constant(np.load('models/' + model_name + '/d2/b.npy'))
    b5_mean  = tf.constant(np.load('models/' + model_name + '/b5/avg_mean.npy'))
    b5_var   = tf.constant(np.load('models/' + model_name + '/b5/avg_var.npy'))
    b5_beta  = tf.constant(np.load('models/' + model_name + '/b5/beta.npy'))
    b5_gamma = tf.constant(np.load('models/' + model_name + '/b5/gamma.npy'))

    d3_W = tf.constant(np.load('models/' + model_name + '/d3/W.npy').transpose(2, 3, 1, 0))
    d3_b = tf.constant(np.load('models/' + model_name + '/d3/b.npy'))

    HEIGHT = 192
    WIDTH = 256

    h0 = tf.placeholder('float', shape=[1, HEIGHT, WIDTH, 3], name='input')
    h1  = conv_layer(h0, c1_W, c1_b, 1, b1_mean, b1_var, b1_beta, b1_gamma)
    h2  = conv_layer(h1, c2_W, c2_b, 2, b2_mean, b2_var, b2_beta, b2_gamma)
    h3  = conv_layer(h2, c3_W, c3_b, 2, b3_mean, b3_var, b3_beta, b3_gamma)
    h4  = residual(h3, r1_c1_W, r1_c1_b, r1_c2_W, r1_c2_b, r1_b1_mean, r1_b2_mean, r1_b1_var, r1_b2_var, r1_b1_beta, r1_b2_beta, r1_b1_gamma, r1_b2_gamma)
    h5  = residual(h4, r2_c1_W, r2_c1_b, r2_c2_W, r2_c2_b, r2_b1_mean, r2_b2_mean, r2_b1_var, r2_b2_var, r2_b1_beta, r2_b2_beta, r2_b1_gamma, r2_b2_gamma)
    h6  = residual(h5, r3_c1_W, r3_c1_b, r3_c2_W, r3_c2_b, r3_b1_mean, r3_b2_mean, r3_b1_var, r3_b2_var, r3_b1_beta, r3_b2_beta, r3_b1_gamma, r3_b2_gamma)
    h7  = residual(h6, r4_c1_W, r4_c1_b, r4_c2_W, r4_c2_b, r4_b1_mean, r4_b2_mean, r4_b1_var, r4_b2_var, r4_b1_beta, r4_b2_beta, r4_b1_gamma, r4_b2_gamma)
    h8  = residual(h7, r5_c1_W, r5_c1_b, r5_c2_W, r5_c2_b, r5_b1_mean, r5_b2_mean, r5_b1_var, r5_b2_var, r5_b1_beta, r5_b2_beta, r5_b1_gamma, r5_b2_gamma)
    h9  = deconv_layer(h8, d1_W, d1_b, 2, b4_mean, b4_var, b4_beta, b4_gamma)
    h10 = deconv_layer(h9, d2_W, d2_b, 2, b5_mean, b5_var, b5_beta, b5_gamma)
    n, h, w, _ = h10.get_shape().as_list()
    _, _, c, _ = d3_W.get_shape().as_list()
    h11 = tf.nn.conv2d_transpose(h10, d3_W, [n, h, w, c], strides=[1, 1, 1, 1], padding='SAME')
    h12 = (tf.tanh(h11) + 1) * 127.5
    h13 = tf.cast(h12, tf.uint8, name='output')

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    if options.graph != '':
        graph_def = graph.as_graph_def()
        tf.train.write_graph(graph_def, './', options.graph, as_text=False)

    if options.input != '':
        result = sess.run(h13, feed_dict={ h0: image })
        result = result.reshape((result.shape[1:]))
        result = np.uint8(result)

        Image.fromarray(result).save(options.output)
