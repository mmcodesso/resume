+++
title = "Face Generation"
date = 2018-02-16T11:42:16-05:00
draft = false

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = []
categories = []

summary="In this project, we'll use generative adversarial networks to generate new images of faces."

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
# Use `caption` to display an image caption.
#   Markdown linking is allowed, e.g. `caption = "[Image credit](http://example.org)"`.
# Set `preview` to `false` to disable the thumbnail in listings.
[header]
image = "facegenerator/gan.jpg"
caption = ""
preview = true

+++



In this project, we'll use generative adversarial networks to generate new images of faces.
### Get the Data
We'll be using two datasets in this project:
- MNIST
- CelebA

Since the celebA dataset is complex and you're doing GANs in a project for the first time, we want test our neural network on MNIST before CelebA.  Running the GANs on MNIST will allow you to see how well your model trains sooner.

```python
data_dir = './data'

import helper

helper.download_extract('mnist', data_dir)
helper.download_extract('celeba', data_dir)
```

    Found mnist Data
    Found celeba Data


## Explore the Data
### MNIST
As we're aware, the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains images of handwritten digits. We can view the first number of examples by changing `show_n_images`.


```python
show_n_images = 25

%matplotlib inline
import os
from glob import glob
from matplotlib import pyplot

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')
```

![png](/img/facegenerator/output_3_1.png)


### CelebA
The [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations.  Since we're going to be generating faces, we won't need the annotations.  We can view the first number of examples by changing `show_n_images`.


```python
show_n_images = 25

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
pyplot.imshow(helper.images_square_grid(mnist_images, 'RGB'))
```

![png](/img/facegenerator/output_5_1.png)


## Preprocess the Data
Since the project's main focus is on building the GANs.  The values of the MNIST and CelebA dataset will be in the range of -0.5 to 0.5 of 28x28 dimensional images.  The CelebA images will be cropped to remove parts of the image that don't include a face, then resized down to 28x28.

The MNIST images are black and white images with a single [color channel](https://en.wikipedia.org/wiki/Channel_(digital_image%29) while the CelebA images have [3 color channels (RGB color channel)](https://en.wikipedia.org/wiki/Channel_(digital_image%29#RGB_Images).
## Build the Neural Network
We'll build the components necessary to build a GANs by implementing the following functions below:
- `model_inputs`
- `discriminator`
- `generator`
- `model_loss`
- `model_opt`
- `train`

### Check the Version of TensorFlow and Access to GPU
This will check to make sure we have the correct version of TensorFlow and access to a GPU

```python
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.1.0
    Default GPU Device: /gpu:0


### Input
Implement the `model_inputs` function to create TF Placeholders for the Neural Network. It should create the following placeholders:
- Real input images placeholder with rank 4 using `image_width`, `image_height`, and `image_channels`.
- Z input placeholder with rank 2 using `z_dim`.
- Learning rate placeholder with rank 0.

Return the placeholders in the following the tuple (tensor of real input images, tensor of z data)


```python
import problem_unittests as tests

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """

    inputs_real = tf.placeholder(tf.float32, (None, image_width , image_height , image_channels ), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    learning_rate = tf.placeholder(tf.float32, (None), name='learning_rate')

    return inputs_real, inputs_z, learning_rate

```

### Discriminator
Implement `discriminator` to create a discriminator neural network that discriminates on `images`.  This function should be able to reuse the variables in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "discriminator" to allow the variables to be reused.  The function should return a tuple of (tensor output of the discriminator, tensor logits of the discriminator).


```python
def discriminator(images, reuse=False):
    """
    Create the discriminator network
    :param images: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """

    alpha = 0.2

    with tf.variable_scope('discriminator', reuse=reuse):
        x1 = tf.layers.conv2d(images, 32, 5, strides=2, padding="same")
        x1 = tf.maximum(alpha*x1, x1)

        x2 = tf.layers.conv2d(x1, 64, 5, strides=2, padding="same")
        x2 = tf.layers.batch_normalization(x2, training=True)
        x2 = tf.maximum(alpha*x2, x2)

        x3 = tf.layers.conv2d(x2, 128, 5, strides=2, padding="same")
        x3 = tf.layers.batch_normalization(x3, training=True)
        x3 = tf.maximum(alpha*x3, x3)

        x4 = tf.layers.conv2d(x3, 256, 5, strides=2, padding="same")
        x4 = tf.layers.batch_normalization(x4, training=True)
        x4 = tf.maximum(alpha*x4, x4)

        x4 = tf.reshape(x3, (-1, 2*2*256))
        logits = tf.layers.dense(x4, 1)
        out = tf.sigmoid(logits)

    return out, logits

```

### Generator
Implement `generator` to generate an image using `z`. This function should be able to reuse the variables in the neural network.  Use [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "generator" to allow the variables to be reused. The function should return the generated 28 x 28 x `out_channel_dim` images.


```python
def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """

    alpha = 0.2

    with tf.variable_scope('generator', reuse= not is_train):
        h1 = tf.layers.dense(z, units=4*4*512)
        h1 = tf.reshape(h1, (-1, 4, 4, 512))
        h1 = tf.layers.batch_normalization(h1, training=is_train)
        h1 = tf.maximum( alpha * h1, h1)

        h2 = tf.layers.conv2d_transpose(h1, filters=128, kernel_size=4, strides=1, padding='valid')
        h2 = tf.layers.batch_normalization(h2, training=is_train)
        h2 = tf.maximum(alpha * h2, h2)

        h3 = tf.layers.conv2d_transpose(h2, filters=64, kernel_size=5, strides=2, padding='same')
        h3 = tf.layers.batch_normalization(h3, training=is_train)
        h3 = tf.maximum(alpha * h3, h3)

        h4= tf.layers.conv2d_transpose(h3, filters=32, kernel_size=5, strides=2, padding='same')
        h4 = tf.layers.batch_normalization(h4, training=is_train)
        h4 = tf.maximum(alpha * h4, h4)

        logits = tf.layers.conv2d_transpose(h4, filters=out_channel_dim, kernel_size=3, strides=1, padding='same')
        out = tf.tanh(logits)

    return out

```

### Loss
Implement `model_loss` to build the GANs for training and calculate the loss.  The function should return a tuple of (discriminator loss, generator loss).  Use the following functions you implemented:
- `discriminator(images, reuse=False)`
- `generator(z, out_channel_dim, is_train=True)`

```python
def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    # TODO: Implement Function
    gen_model = generator(input_z, out_channel_dim)
    disc_model_real, disc_logits_real = discriminator(input_real)
    disc_model_fake, disc_logits_fake = discriminator(gen_model, reuse=True)

    disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits_real,
                                                                           labels=tf.ones_like(disc_model_real)*(1-0.1)))
    disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits_fake,
                                                                           labels=tf.zeros_like(disc_model_fake)))
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits_fake,
                                                                     labels=tf.ones_like(disc_model_fake)))

    disc_loss = disc_loss_real + disc_loss_fake

    return disc_loss, gen_loss

```

### Optimization
Implement `model_opt` to create the optimization operations for the GANs. Use [`tf.trainable_variables`](https://www.tensorflow.org/api_docs/python/tf/trainable_variables) to get all the trainable variables.  Filter the variables with names that are in the discriminator and generator scope names.  The function should return a tuple of (discriminator training operation, generator training operation).

```python
def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

```

## Neural Network Training
### Show Output
Use this function to show the current output of the generator during training. It will help us determine how well the GANs is training.


```python
import numpy as np

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()
```

### Train
Implement `train` to build and train the GANs.  Use the following functions we implemented:
- `model_inputs(image_width, image_height, image_channels, z_dim)`
- `model_loss(input_real, input_z, out_channel_dim)`
- `model_opt(d_loss, g_loss, learning_rate, beta1)`

Use the `show_generator_output` to show `generator` output while you train. Running `show_generator_output` for every batch will drastically increase training time and increase the size of the notebook.  It's recommended to print the `generator` output every 100 batches.


```python
def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """

    print_every = 10
    show_every = 100
    step = 0

    samples, width, height, channels = data_shape

    input_real, input_z, lr = model_inputs(width, height, channels, z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, channels)
    d_train_opt, g_train_opt = model_opt(d_loss, g_loss, lr, beta1)

    saver = tf.train.Saver()

    steps = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                # TODO: Train Model


                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                batch_images = batch_images * 2.0

                # Run optimizers
                sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z, lr: learning_rate})
                sess.run(g_train_opt, feed_dict={input_real: batch_images, input_z: batch_z, lr: learning_rate})

                steps += 1
                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = d_loss.eval({input_real: batch_images, input_z: batch_z})
                    train_loss_g = g_loss.eval({input_real: batch_images, input_z: batch_z})

                    print("Epoch {}/{}...".format(epoch_i+1, epoch_count),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))

                if steps % show_every == 0:
                    show_generator_output(sess, 16, input_z, channels, data_image_mode)            
```

### MNIST
Test our GANs architecture on MNIST.  After 2 epochs, the GANs should be able to generate images that look like handwritten digits.  Make sure the loss of the generator is lower than the loss of the discriminator or close to 0.


```python
batch_size = 64
z_dim = 128
learning_rate = 0.0001
beta1 = 0.4

epochs = 2

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode)
```

    Epoch 1/2... Discriminator Loss: 2.0867... Generator Loss: 0.2324
    Epoch 1/2... Discriminator Loss: 1.5094... Generator Loss: 0.5193
    Epoch 1/2... Discriminator Loss: 1.0356... Generator Loss: 0.9100
    Epoch 1/2... Discriminator Loss: 0.9495... Generator Loss: 0.9795
    Epoch 1/2... Discriminator Loss: 1.1186... Generator Loss: 0.8746
    Epoch 1/2... Discriminator Loss: 1.6295... Generator Loss: 0.6214
    Epoch 1/2... Discriminator Loss: 1.8456... Generator Loss: 0.5919
    Epoch 1/2... Discriminator Loss: 1.4133... Generator Loss: 0.9339
    Epoch 1/2... Discriminator Loss: 1.6277... Generator Loss: 0.6748
    Epoch 1/2... Discriminator Loss: 1.2511... Generator Loss: 1.0289



![png](/img/facegenerator/output_23_1.png)


    Epoch 1/2... Discriminator Loss: 1.0522... Generator Loss: 1.2071
    Epoch 1/2... Discriminator Loss: 1.1059... Generator Loss: 1.0716
    Epoch 1/2... Discriminator Loss: 0.7196... Generator Loss: 1.5074
    Epoch 1/2... Discriminator Loss: 0.8322... Generator Loss: 1.3778
    Epoch 1/2... Discriminator Loss: 0.7403... Generator Loss: 1.5193
    Epoch 1/2... Discriminator Loss: 0.7883... Generator Loss: 1.4761
    Epoch 1/2... Discriminator Loss: 0.8295... Generator Loss: 1.3749
    Epoch 1/2... Discriminator Loss: 1.0960... Generator Loss: 1.0644
    Epoch 1/2... Discriminator Loss: 1.1658... Generator Loss: 1.0452
    Epoch 1/2... Discriminator Loss: 1.2194... Generator Loss: 0.9498



![png](/img/facegenerator/output_23_3.png)


    Epoch 1/2... Discriminator Loss: 1.2080... Generator Loss: 1.0334
    Epoch 1/2... Discriminator Loss: 1.1472... Generator Loss: 1.1309
    Epoch 1/2... Discriminator Loss: 1.1982... Generator Loss: 1.0109
    Epoch 1/2... Discriminator Loss: 1.2863... Generator Loss: 0.9914
    Epoch 1/2... Discriminator Loss: 1.1951... Generator Loss: 1.0701
    Epoch 1/2... Discriminator Loss: 1.3705... Generator Loss: 0.7629
    Epoch 1/2... Discriminator Loss: 1.2540... Generator Loss: 1.0033
    Epoch 1/2... Discriminator Loss: 1.2792... Generator Loss: 0.8744
    Epoch 1/2... Discriminator Loss: 1.2811... Generator Loss: 1.0062
    Epoch 1/2... Discriminator Loss: 1.4031... Generator Loss: 1.0596



![png](/img/facegenerator/output_23_5.png)


    Epoch 1/2... Discriminator Loss: 1.2331... Generator Loss: 0.9807
    Epoch 1/2... Discriminator Loss: 1.2764... Generator Loss: 0.6834
    Epoch 1/2... Discriminator Loss: 1.2258... Generator Loss: 0.8944
    Epoch 1/2... Discriminator Loss: 1.1946... Generator Loss: 0.7654
    Epoch 1/2... Discriminator Loss: 1.2413... Generator Loss: 1.1821
    Epoch 1/2... Discriminator Loss: 1.1961... Generator Loss: 1.0457
    Epoch 1/2... Discriminator Loss: 1.1864... Generator Loss: 1.1887
    Epoch 1/2... Discriminator Loss: 1.2148... Generator Loss: 0.8252
    Epoch 1/2... Discriminator Loss: 1.2167... Generator Loss: 1.0156
    Epoch 1/2... Discriminator Loss: 1.3174... Generator Loss: 0.5959



![png](/img/facegenerator/output_23_7.png)


    Epoch 1/2... Discriminator Loss: 1.1696... Generator Loss: 1.0757
    Epoch 1/2... Discriminator Loss: 1.3124... Generator Loss: 0.6242
    Epoch 1/2... Discriminator Loss: 1.2202... Generator Loss: 0.9363
    Epoch 1/2... Discriminator Loss: 1.2088... Generator Loss: 0.7008
    Epoch 1/2... Discriminator Loss: 1.1439... Generator Loss: 0.9151
    Epoch 1/2... Discriminator Loss: 1.3049... Generator Loss: 0.6306
    Epoch 1/2... Discriminator Loss: 1.1609... Generator Loss: 0.8051
    Epoch 1/2... Discriminator Loss: 1.1810... Generator Loss: 0.9045
    Epoch 1/2... Discriminator Loss: 1.1866... Generator Loss: 1.2970
    Epoch 1/2... Discriminator Loss: 1.1802... Generator Loss: 1.1859



![png](/img/facegenerator/output_23_9.png)


    Epoch 1/2... Discriminator Loss: 1.1831... Generator Loss: 0.7298
    Epoch 1/2... Discriminator Loss: 1.1400... Generator Loss: 1.0396
    Epoch 1/2... Discriminator Loss: 1.0721... Generator Loss: 1.1058
    Epoch 1/2... Discriminator Loss: 1.2129... Generator Loss: 0.6734
    Epoch 1/2... Discriminator Loss: 1.1377... Generator Loss: 0.8990
    Epoch 1/2... Discriminator Loss: 1.3203... Generator Loss: 0.5271
    Epoch 1/2... Discriminator Loss: 1.1493... Generator Loss: 1.1018
    Epoch 1/2... Discriminator Loss: 1.0622... Generator Loss: 1.1183
    Epoch 1/2... Discriminator Loss: 1.1419... Generator Loss: 0.8327
    Epoch 1/2... Discriminator Loss: 1.0768... Generator Loss: 0.9751



![png](/img/facegenerator/output_23_11.png)


    Epoch 1/2... Discriminator Loss: 1.1085... Generator Loss: 0.7544
    Epoch 1/2... Discriminator Loss: 1.1810... Generator Loss: 1.4955
    Epoch 1/2... Discriminator Loss: 1.0576... Generator Loss: 0.9837
    Epoch 1/2... Discriminator Loss: 1.0756... Generator Loss: 1.1582
    Epoch 1/2... Discriminator Loss: 1.0922... Generator Loss: 1.0645
    Epoch 1/2... Discriminator Loss: 1.1661... Generator Loss: 0.7769
    Epoch 1/2... Discriminator Loss: 1.2407... Generator Loss: 0.5921
    Epoch 1/2... Discriminator Loss: 1.0693... Generator Loss: 1.1128
    Epoch 1/2... Discriminator Loss: 1.1366... Generator Loss: 1.1153
    Epoch 1/2... Discriminator Loss: 1.2939... Generator Loss: 1.6455



![png](/img/facegenerator/output_23_13.png)


    Epoch 1/2... Discriminator Loss: 1.1344... Generator Loss: 0.8429
    Epoch 1/2... Discriminator Loss: 0.9981... Generator Loss: 1.1148
    Epoch 1/2... Discriminator Loss: 1.1439... Generator Loss: 0.7655
    Epoch 1/2... Discriminator Loss: 1.1887... Generator Loss: 0.7730
    Epoch 1/2... Discriminator Loss: 1.1581... Generator Loss: 0.7955
    Epoch 1/2... Discriminator Loss: 1.1732... Generator Loss: 1.3147
    Epoch 1/2... Discriminator Loss: 1.1468... Generator Loss: 1.0593
    Epoch 1/2... Discriminator Loss: 1.2321... Generator Loss: 0.6411
    Epoch 1/2... Discriminator Loss: 1.2414... Generator Loss: 0.6540
    Epoch 1/2... Discriminator Loss: 1.1789... Generator Loss: 1.0781



![png](/img/facegenerator/output_23_15.png)


    Epoch 1/2... Discriminator Loss: 1.1860... Generator Loss: 1.4194
    Epoch 1/2... Discriminator Loss: 1.1472... Generator Loss: 0.8351
    Epoch 1/2... Discriminator Loss: 1.1374... Generator Loss: 0.9677
    Epoch 1/2... Discriminator Loss: 1.1041... Generator Loss: 0.8191
    Epoch 1/2... Discriminator Loss: 1.1570... Generator Loss: 0.7028
    Epoch 1/2... Discriminator Loss: 1.2045... Generator Loss: 0.7549
    Epoch 1/2... Discriminator Loss: 1.2028... Generator Loss: 0.7856
    Epoch 1/2... Discriminator Loss: 1.2868... Generator Loss: 1.4184
    Epoch 1/2... Discriminator Loss: 1.1845... Generator Loss: 0.8564
    Epoch 1/2... Discriminator Loss: 1.3979... Generator Loss: 0.5206



![png](/img/facegenerator/output_23_17.png)


    Epoch 1/2... Discriminator Loss: 1.2247... Generator Loss: 1.0396
    Epoch 1/2... Discriminator Loss: 1.2053... Generator Loss: 1.1057
    Epoch 1/2... Discriminator Loss: 1.2552... Generator Loss: 0.6563
    Epoch 2/2... Discriminator Loss: 1.2044... Generator Loss: 1.0764
    Epoch 2/2... Discriminator Loss: 1.2413... Generator Loss: 1.0572
    Epoch 2/2... Discriminator Loss: 1.1445... Generator Loss: 1.0412
    Epoch 2/2... Discriminator Loss: 1.1758... Generator Loss: 0.8125
    Epoch 2/2... Discriminator Loss: 1.2081... Generator Loss: 0.6821
    Epoch 2/2... Discriminator Loss: 1.2141... Generator Loss: 0.8098
    Epoch 2/2... Discriminator Loss: 1.3289... Generator Loss: 0.6118



![png](/img/facegenerator/output_23_19.png)


    Epoch 2/2... Discriminator Loss: 1.3209... Generator Loss: 1.3094
    Epoch 2/2... Discriminator Loss: 1.2298... Generator Loss: 0.7401
    Epoch 2/2... Discriminator Loss: 1.2435... Generator Loss: 0.9291
    Epoch 2/2... Discriminator Loss: 1.1978... Generator Loss: 1.0000
    Epoch 2/2... Discriminator Loss: 1.1949... Generator Loss: 0.9605
    Epoch 2/2... Discriminator Loss: 1.2492... Generator Loss: 0.8094
    Epoch 2/2... Discriminator Loss: 1.2571... Generator Loss: 0.8651
    Epoch 2/2... Discriminator Loss: 1.2837... Generator Loss: 1.0014
    Epoch 2/2... Discriminator Loss: 1.2181... Generator Loss: 0.8379
    Epoch 2/2... Discriminator Loss: 1.3743... Generator Loss: 0.5693



![png](/img/facegenerator/output_23_21.png)


    Epoch 2/2... Discriminator Loss: 1.2555... Generator Loss: 0.7658
    Epoch 2/2... Discriminator Loss: 1.2038... Generator Loss: 0.8254
    Epoch 2/2... Discriminator Loss: 1.3464... Generator Loss: 0.9879
    Epoch 2/2... Discriminator Loss: 1.2209... Generator Loss: 0.7307
    Epoch 2/2... Discriminator Loss: 1.2993... Generator Loss: 0.6874
    Epoch 2/2... Discriminator Loss: 1.1857... Generator Loss: 0.8730
    Epoch 2/2... Discriminator Loss: 1.2955... Generator Loss: 0.9850
    Epoch 2/2... Discriminator Loss: 1.3440... Generator Loss: 1.0402
    Epoch 2/2... Discriminator Loss: 1.3281... Generator Loss: 0.6175
    Epoch 2/2... Discriminator Loss: 1.1911... Generator Loss: 0.8710



![png](/img/facegenerator/output_23_23.png)


    Epoch 2/2... Discriminator Loss: 1.3056... Generator Loss: 0.6775
    Epoch 2/2... Discriminator Loss: 1.2098... Generator Loss: 0.9500
    Epoch 2/2... Discriminator Loss: 1.3130... Generator Loss: 0.7307
    Epoch 2/2... Discriminator Loss: 1.2803... Generator Loss: 1.0129
    Epoch 2/2... Discriminator Loss: 1.3080... Generator Loss: 1.0621
    Epoch 2/2... Discriminator Loss: 1.2776... Generator Loss: 1.0465
    Epoch 2/2... Discriminator Loss: 1.2663... Generator Loss: 0.6722
    Epoch 2/2... Discriminator Loss: 1.2557... Generator Loss: 0.8396
    Epoch 2/2... Discriminator Loss: 1.3579... Generator Loss: 0.5681
    Epoch 2/2... Discriminator Loss: 1.2776... Generator Loss: 0.6637



![png](/img/facegenerator/output_23_25.png)


    Epoch 2/2... Discriminator Loss: 1.3308... Generator Loss: 0.7690
    Epoch 2/2... Discriminator Loss: 1.3306... Generator Loss: 0.5899
    Epoch 2/2... Discriminator Loss: 1.2653... Generator Loss: 0.8439
    Epoch 2/2... Discriminator Loss: 1.2762... Generator Loss: 1.0469
    Epoch 2/2... Discriminator Loss: 1.2430... Generator Loss: 0.7680
    Epoch 2/2... Discriminator Loss: 1.2945... Generator Loss: 0.6913
    Epoch 2/2... Discriminator Loss: 1.3199... Generator Loss: 1.1280
    Epoch 2/2... Discriminator Loss: 1.2628... Generator Loss: 0.6944
    Epoch 2/2... Discriminator Loss: 1.2870... Generator Loss: 0.7167
    Epoch 2/2... Discriminator Loss: 1.2928... Generator Loss: 1.1388



![png](/img/facegenerator/output_23_27.png)


    Epoch 2/2... Discriminator Loss: 1.1898... Generator Loss: 1.0141
    Epoch 2/2... Discriminator Loss: 1.2056... Generator Loss: 0.8590
    Epoch 2/2... Discriminator Loss: 1.2736... Generator Loss: 0.7476
    Epoch 2/2... Discriminator Loss: 1.3113... Generator Loss: 0.7062
    Epoch 2/2... Discriminator Loss: 1.2659... Generator Loss: 0.8806
    Epoch 2/2... Discriminator Loss: 1.2707... Generator Loss: 0.7873
    Epoch 2/2... Discriminator Loss: 1.2009... Generator Loss: 0.9228
    Epoch 2/2... Discriminator Loss: 1.2752... Generator Loss: 0.7757
    Epoch 2/2... Discriminator Loss: 1.3504... Generator Loss: 0.8774
    Epoch 2/2... Discriminator Loss: 1.2314... Generator Loss: 0.9217



![png](/img/facegenerator/output_23_29.png)


    Epoch 2/2... Discriminator Loss: 1.3306... Generator Loss: 0.8571
    Epoch 2/2... Discriminator Loss: 1.3156... Generator Loss: 0.6177
    Epoch 2/2... Discriminator Loss: 1.2483... Generator Loss: 0.8452
    Epoch 2/2... Discriminator Loss: 1.2862... Generator Loss: 0.6832
    Epoch 2/2... Discriminator Loss: 1.2899... Generator Loss: 0.6720
    Epoch 2/2... Discriminator Loss: 1.3249... Generator Loss: 0.6026
    Epoch 2/2... Discriminator Loss: 1.3809... Generator Loss: 0.5495
    Epoch 2/2... Discriminator Loss: 1.3165... Generator Loss: 0.6890
    Epoch 2/2... Discriminator Loss: 1.2753... Generator Loss: 1.0544
    Epoch 2/2... Discriminator Loss: 1.2969... Generator Loss: 0.7528



![png](/img/facegenerator/output_23_31.png)


    Epoch 2/2... Discriminator Loss: 1.2779... Generator Loss: 0.8854
    Epoch 2/2... Discriminator Loss: 1.3638... Generator Loss: 0.6127
    Epoch 2/2... Discriminator Loss: 1.2186... Generator Loss: 0.9226
    Epoch 2/2... Discriminator Loss: 1.2312... Generator Loss: 1.1079
    Epoch 2/2... Discriminator Loss: 1.2646... Generator Loss: 0.7210
    Epoch 2/2... Discriminator Loss: 1.2953... Generator Loss: 1.2433
    Epoch 2/2... Discriminator Loss: 1.1991... Generator Loss: 0.9160
    Epoch 2/2... Discriminator Loss: 1.2403... Generator Loss: 0.7974
    Epoch 2/2... Discriminator Loss: 1.2443... Generator Loss: 0.7666
    Epoch 2/2... Discriminator Loss: 1.1691... Generator Loss: 0.9141



![png](/img/facegenerator/output_23_33.png)


    Epoch 2/2... Discriminator Loss: 1.2296... Generator Loss: 0.8136
    Epoch 2/2... Discriminator Loss: 1.2693... Generator Loss: 0.6582
    Epoch 2/2... Discriminator Loss: 1.2373... Generator Loss: 0.8782
    Epoch 2/2... Discriminator Loss: 1.2765... Generator Loss: 1.0020
    Epoch 2/2... Discriminator Loss: 1.3004... Generator Loss: 0.6388
    Epoch 2/2... Discriminator Loss: 1.2955... Generator Loss: 1.1928
    Epoch 2/2... Discriminator Loss: 1.2272... Generator Loss: 0.7862
    Epoch 2/2... Discriminator Loss: 1.2759... Generator Loss: 1.0464
    Epoch 2/2... Discriminator Loss: 1.2225... Generator Loss: 0.8203
    Epoch 2/2... Discriminator Loss: 1.2413... Generator Loss: 1.0729



![png](/img/facegenerator/output_23_35.png)


    Epoch 2/2... Discriminator Loss: 1.2402... Generator Loss: 0.8544
    Epoch 2/2... Discriminator Loss: 1.2417... Generator Loss: 0.7286
    Epoch 2/2... Discriminator Loss: 1.2518... Generator Loss: 1.0633
    Epoch 2/2... Discriminator Loss: 1.2325... Generator Loss: 0.8128
    Epoch 2/2... Discriminator Loss: 1.2957... Generator Loss: 1.2616
    Epoch 2/2... Discriminator Loss: 1.2729... Generator Loss: 0.6427
    Epoch 2/2... Discriminator Loss: 1.1950... Generator Loss: 0.8738


### CelebA
Run our GANs on CelebA.  It will take around 20 minutes on the average GPU to run one epoch.  We can run the whole epoch or stop when it starts to generate realistic faces.


```python
batch_size = 64
z_dim = 100
learning_rate = 0.0003
beta1 = 0.3

epochs = 1

celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)
```

    Epoch 1/1... Discriminator Loss: 2.3149... Generator Loss: 0.2816
    Epoch 1/1... Discriminator Loss: 1.2416... Generator Loss: 0.7742
    Epoch 1/1... Discriminator Loss: 0.8748... Generator Loss: 1.3591
    Epoch 1/1... Discriminator Loss: 1.4324... Generator Loss: 2.4702
    Epoch 1/1... Discriminator Loss: 1.1738... Generator Loss: 0.7172
    Epoch 1/1... Discriminator Loss: 1.0136... Generator Loss: 2.1460
    Epoch 1/1... Discriminator Loss: 1.1662... Generator Loss: 1.0521
    Epoch 1/1... Discriminator Loss: 1.1394... Generator Loss: 0.9716
    Epoch 1/1... Discriminator Loss: 1.1296... Generator Loss: 0.8982
    Epoch 1/1... Discriminator Loss: 1.2313... Generator Loss: 1.0092



![png](/img/facegenerator/output_25_1.png)


    Epoch 1/1... Discriminator Loss: 1.1925... Generator Loss: 1.5360
    Epoch 1/1... Discriminator Loss: 1.3641... Generator Loss: 0.9303
    Epoch 1/1... Discriminator Loss: 1.2386... Generator Loss: 0.8219
    Epoch 1/1... Discriminator Loss: 1.3989... Generator Loss: 0.7562
    Epoch 1/1... Discriminator Loss: 1.3945... Generator Loss: 0.7217
    Epoch 1/1... Discriminator Loss: 1.3891... Generator Loss: 0.6059
    Epoch 1/1... Discriminator Loss: 1.4276... Generator Loss: 0.7372
    Epoch 1/1... Discriminator Loss: 1.3764... Generator Loss: 0.6309
    Epoch 1/1... Discriminator Loss: 1.3884... Generator Loss: 1.0268
    Epoch 1/1... Discriminator Loss: 1.2752... Generator Loss: 0.7784



![png](/img/facegenerator/output_25_3.png)


    Epoch 1/1... Discriminator Loss: 1.1962... Generator Loss: 1.5991
    Epoch 1/1... Discriminator Loss: 1.2140... Generator Loss: 2.2475
    Epoch 1/1... Discriminator Loss: 1.2710... Generator Loss: 0.6464
    Epoch 1/1... Discriminator Loss: 1.5610... Generator Loss: 1.7737
    Epoch 1/1... Discriminator Loss: 1.3079... Generator Loss: 0.6572
    Epoch 1/1... Discriminator Loss: 1.7443... Generator Loss: 0.3808
    Epoch 1/1... Discriminator Loss: 1.0157... Generator Loss: 1.8892
    Epoch 1/1... Discriminator Loss: 1.3779... Generator Loss: 2.2072
    Epoch 1/1... Discriminator Loss: 1.3951... Generator Loss: 1.3022
    Epoch 1/1... Discriminator Loss: 1.0371... Generator Loss: 1.2426



![png](/img/facegenerator/output_25_5.png)


    Epoch 1/1... Discriminator Loss: 1.4993... Generator Loss: 0.4867
    Epoch 1/1... Discriminator Loss: 0.9911... Generator Loss: 1.3262
    Epoch 1/1... Discriminator Loss: 1.3073... Generator Loss: 0.8244
    Epoch 1/1... Discriminator Loss: 1.1086... Generator Loss: 1.3685
    Epoch 1/1... Discriminator Loss: 1.6336... Generator Loss: 0.4269
    Epoch 1/1... Discriminator Loss: 1.3890... Generator Loss: 0.7221
    Epoch 1/1... Discriminator Loss: 1.2874... Generator Loss: 2.3135
    Epoch 1/1... Discriminator Loss: 1.2192... Generator Loss: 0.9024
    Epoch 1/1... Discriminator Loss: 1.6617... Generator Loss: 0.4317
    Epoch 1/1... Discriminator Loss: 1.5095... Generator Loss: 0.5994



![png](/img/facegenerator/output_25_7.png)


    Epoch 1/1... Discriminator Loss: 1.3813... Generator Loss: 1.0992
    Epoch 1/1... Discriminator Loss: 1.3092... Generator Loss: 0.6902
    Epoch 1/1... Discriminator Loss: 1.1585... Generator Loss: 0.9146
    Epoch 1/1... Discriminator Loss: 1.3528... Generator Loss: 0.8420
    Epoch 1/1... Discriminator Loss: 1.3202... Generator Loss: 0.7359
    Epoch 1/1... Discriminator Loss: 1.3768... Generator Loss: 0.7122
    Epoch 1/1... Discriminator Loss: 1.4131... Generator Loss: 0.6578
    Epoch 1/1... Discriminator Loss: 1.3587... Generator Loss: 0.8800
    Epoch 1/1... Discriminator Loss: 1.1875... Generator Loss: 0.9812
    Epoch 1/1... Discriminator Loss: 1.3291... Generator Loss: 0.7910



![png](/img/facegenerator/output_25_9.png)


    Epoch 1/1... Discriminator Loss: 1.3816... Generator Loss: 0.8396
    Epoch 1/1... Discriminator Loss: 1.2179... Generator Loss: 1.8638
    Epoch 1/1... Discriminator Loss: 1.3999... Generator Loss: 0.7043
    Epoch 1/1... Discriminator Loss: 1.4113... Generator Loss: 0.7141
    Epoch 1/1... Discriminator Loss: 1.3655... Generator Loss: 0.6968
    Epoch 1/1... Discriminator Loss: 1.5068... Generator Loss: 1.1586
    Epoch 1/1... Discriminator Loss: 1.4112... Generator Loss: 0.8485
    Epoch 1/1... Discriminator Loss: 1.2713... Generator Loss: 0.8209
    Epoch 1/1... Discriminator Loss: 1.4052... Generator Loss: 0.6998
    Epoch 1/1... Discriminator Loss: 1.4269... Generator Loss: 1.2699



![png](/img/facegenerator/output_25_11.png)


    Epoch 1/1... Discriminator Loss: 1.4125... Generator Loss: 0.6185
    Epoch 1/1... Discriminator Loss: 1.3750... Generator Loss: 0.8218
    Epoch 1/1... Discriminator Loss: 1.3075... Generator Loss: 1.1242
    Epoch 1/1... Discriminator Loss: 1.4407... Generator Loss: 0.6538
    Epoch 1/1... Discriminator Loss: 1.3310... Generator Loss: 0.8178
    Epoch 1/1... Discriminator Loss: 1.2202... Generator Loss: 0.9430
    Epoch 1/1... Discriminator Loss: 1.2181... Generator Loss: 0.7861
    Epoch 1/1... Discriminator Loss: 1.3697... Generator Loss: 0.7326
    Epoch 1/1... Discriminator Loss: 1.2229... Generator Loss: 0.9424
    Epoch 1/1... Discriminator Loss: 1.1598... Generator Loss: 1.0535



![png](/img/facegenerator/output_25_13.png)


    Epoch 1/1... Discriminator Loss: 1.2089... Generator Loss: 1.0640
    Epoch 1/1... Discriminator Loss: 1.1913... Generator Loss: 0.8731
    Epoch 1/1... Discriminator Loss: 1.3132... Generator Loss: 0.7522
    Epoch 1/1... Discriminator Loss: 1.5001... Generator Loss: 0.5009
    Epoch 1/1... Discriminator Loss: 1.2221... Generator Loss: 1.2435
    Epoch 1/1... Discriminator Loss: 1.4104... Generator Loss: 0.6448
    Epoch 1/1... Discriminator Loss: 1.4843... Generator Loss: 0.5003
    Epoch 1/1... Discriminator Loss: 1.3965... Generator Loss: 0.7937
    Epoch 1/1... Discriminator Loss: 1.2389... Generator Loss: 0.8937
    Epoch 1/1... Discriminator Loss: 1.1637... Generator Loss: 1.0135



![png](/img/facegenerator/output_25_15.png)


    Epoch 1/1... Discriminator Loss: 1.5302... Generator Loss: 0.6909
    Epoch 1/1... Discriminator Loss: 1.3750... Generator Loss: 1.0790
    Epoch 1/1... Discriminator Loss: 1.4153... Generator Loss: 0.6359
    Epoch 1/1... Discriminator Loss: 1.2187... Generator Loss: 0.9002
    Epoch 1/1... Discriminator Loss: 1.3082... Generator Loss: 0.8110
    Epoch 1/1... Discriminator Loss: 1.5445... Generator Loss: 0.4746
    Epoch 1/1... Discriminator Loss: 1.3988... Generator Loss: 0.8564
    Epoch 1/1... Discriminator Loss: 1.4058... Generator Loss: 0.9279
    Epoch 1/1... Discriminator Loss: 1.1958... Generator Loss: 0.8365
    Epoch 1/1... Discriminator Loss: 1.0999... Generator Loss: 0.9948



![png](/img/facegenerator/output_25_17.png)


    Epoch 1/1... Discriminator Loss: 1.2857... Generator Loss: 1.0030
    Epoch 1/1... Discriminator Loss: 1.4372... Generator Loss: 0.6809
    Epoch 1/1... Discriminator Loss: 1.3551... Generator Loss: 0.9058
    Epoch 1/1... Discriminator Loss: 1.3624... Generator Loss: 0.7823
    Epoch 1/1... Discriminator Loss: 1.3621... Generator Loss: 0.7567
    Epoch 1/1... Discriminator Loss: 1.3296... Generator Loss: 0.8264
    Epoch 1/1... Discriminator Loss: 1.3879... Generator Loss: 0.7329
    Epoch 1/1... Discriminator Loss: 1.4125... Generator Loss: 0.8207
    Epoch 1/1... Discriminator Loss: 1.3492... Generator Loss: 0.7753
    Epoch 1/1... Discriminator Loss: 1.3706... Generator Loss: 0.7321



![png](/img/facegenerator/output_25_19.png)


    Epoch 1/1... Discriminator Loss: 0.9864... Generator Loss: 1.3506
    Epoch 1/1... Discriminator Loss: 1.3990... Generator Loss: 0.7291
    Epoch 1/1... Discriminator Loss: 1.3076... Generator Loss: 0.9040
    Epoch 1/1... Discriminator Loss: 1.3990... Generator Loss: 0.6924
    Epoch 1/1... Discriminator Loss: 1.3367... Generator Loss: 0.7349
    Epoch 1/1... Discriminator Loss: 1.3990... Generator Loss: 0.6318
    Epoch 1/1... Discriminator Loss: 1.2399... Generator Loss: 0.8559
    Epoch 1/1... Discriminator Loss: 1.4255... Generator Loss: 0.6898
    Epoch 1/1... Discriminator Loss: 1.4534... Generator Loss: 1.0882
    Epoch 1/1... Discriminator Loss: 1.0179... Generator Loss: 1.3672



![png](/img/facegenerator/output_25_21.png)


    Epoch 1/1... Discriminator Loss: 1.4213... Generator Loss: 0.6941
    Epoch 1/1... Discriminator Loss: 1.1932... Generator Loss: 1.0422
    Epoch 1/1... Discriminator Loss: 1.2666... Generator Loss: 0.9393
    Epoch 1/1... Discriminator Loss: 1.2826... Generator Loss: 0.9103
    Epoch 1/1... Discriminator Loss: 1.1003... Generator Loss: 1.0246
    Epoch 1/1... Discriminator Loss: 1.3026... Generator Loss: 0.8595
    Epoch 1/1... Discriminator Loss: 1.5127... Generator Loss: 0.5016
    Epoch 1/1... Discriminator Loss: 1.3469... Generator Loss: 0.8147
    Epoch 1/1... Discriminator Loss: 1.3403... Generator Loss: 0.9104
    Epoch 1/1... Discriminator Loss: 1.4491... Generator Loss: 0.6292



![png](/img/facegenerator/output_25_23.png)


    Epoch 1/1... Discriminator Loss: 1.3977... Generator Loss: 0.7109
    Epoch 1/1... Discriminator Loss: 1.3887... Generator Loss: 0.7925
    Epoch 1/1... Discriminator Loss: 1.2540... Generator Loss: 0.8163
    Epoch 1/1... Discriminator Loss: 1.3751... Generator Loss: 0.7604
    Epoch 1/1... Discriminator Loss: 1.4031... Generator Loss: 0.7976
    Epoch 1/1... Discriminator Loss: 1.3734... Generator Loss: 1.0501
    Epoch 1/1... Discriminator Loss: 1.3806... Generator Loss: 0.8318
    Epoch 1/1... Discriminator Loss: 1.3547... Generator Loss: 0.9260
    Epoch 1/1... Discriminator Loss: 1.1292... Generator Loss: 1.0131
    Epoch 1/1... Discriminator Loss: 1.3749... Generator Loss: 0.8000



![png](/img/facegenerator/output_25_25.png)


    Epoch 1/1... Discriminator Loss: 1.3617... Generator Loss: 0.7539
    Epoch 1/1... Discriminator Loss: 1.1532... Generator Loss: 1.1888
    Epoch 1/1... Discriminator Loss: 1.4558... Generator Loss: 0.6205
    Epoch 1/1... Discriminator Loss: 1.6047... Generator Loss: 0.7907
    Epoch 1/1... Discriminator Loss: 1.3141... Generator Loss: 0.8944
    Epoch 1/1... Discriminator Loss: 1.3552... Generator Loss: 0.8042
    Epoch 1/1... Discriminator Loss: 1.3849... Generator Loss: 0.7067
    Epoch 1/1... Discriminator Loss: 1.2351... Generator Loss: 0.9189
    Epoch 1/1... Discriminator Loss: 1.3557... Generator Loss: 0.6968
    Epoch 1/1... Discriminator Loss: 1.3785... Generator Loss: 0.7279



![png](/img/facegenerator/output_25_27.png)


    Epoch 1/1... Discriminator Loss: 1.3492... Generator Loss: 0.8392
    Epoch 1/1... Discriminator Loss: 1.2809... Generator Loss: 0.8491
    Epoch 1/1... Discriminator Loss: 1.3917... Generator Loss: 0.7274
    Epoch 1/1... Discriminator Loss: 1.5575... Generator Loss: 0.6136
    Epoch 1/1... Discriminator Loss: 1.2944... Generator Loss: 0.6731
    Epoch 1/1... Discriminator Loss: 1.2865... Generator Loss: 0.7802
    Epoch 1/1... Discriminator Loss: 1.4422... Generator Loss: 0.6142
    Epoch 1/1... Discriminator Loss: 0.9473... Generator Loss: 1.3395
    Epoch 1/1... Discriminator Loss: 1.2625... Generator Loss: 0.9060
    Epoch 1/1... Discriminator Loss: 1.4201... Generator Loss: 0.6008



![png](/img/facegenerator/output_25_29.png)


    Epoch 1/1... Discriminator Loss: 1.4472... Generator Loss: 0.6684
    Epoch 1/1... Discriminator Loss: 1.2699... Generator Loss: 0.8177
    Epoch 1/1... Discriminator Loss: 1.1282... Generator Loss: 1.0019
    Epoch 1/1... Discriminator Loss: 1.2274... Generator Loss: 0.9027
    Epoch 1/1... Discriminator Loss: 1.3671... Generator Loss: 0.8393
    Epoch 1/1... Discriminator Loss: 1.3623... Generator Loss: 0.7299
    Epoch 1/1... Discriminator Loss: 1.1020... Generator Loss: 1.0048
    Epoch 1/1... Discriminator Loss: 1.3621... Generator Loss: 0.7084
    Epoch 1/1... Discriminator Loss: 1.3290... Generator Loss: 0.8587
    Epoch 1/1... Discriminator Loss: 1.2966... Generator Loss: 0.7183



![png](/img/facegenerator/output_25_31.png)


    Epoch 1/1... Discriminator Loss: 1.4234... Generator Loss: 0.7774
    Epoch 1/1... Discriminator Loss: 1.4385... Generator Loss: 0.7451
    Epoch 1/1... Discriminator Loss: 1.3204... Generator Loss: 0.9087
    Epoch 1/1... Discriminator Loss: 1.3274... Generator Loss: 0.8213
    Epoch 1/1... Discriminator Loss: 1.1239... Generator Loss: 1.1774
    Epoch 1/1... Discriminator Loss: 1.4184... Generator Loss: 0.6825
    Epoch 1/1... Discriminator Loss: 1.4437... Generator Loss: 0.6565
    Epoch 1/1... Discriminator Loss: 1.2447... Generator Loss: 0.9294
    Epoch 1/1... Discriminator Loss: 1.3252... Generator Loss: 0.8418
    Epoch 1/1... Discriminator Loss: 1.2688... Generator Loss: 0.9125



![png](/img/facegenerator/output_25_33.png)


    Epoch 1/1... Discriminator Loss: 1.3636... Generator Loss: 0.7188
    Epoch 1/1... Discriminator Loss: 1.3419... Generator Loss: 0.8352
    Epoch 1/1... Discriminator Loss: 1.3294... Generator Loss: 0.9335
    Epoch 1/1... Discriminator Loss: 1.1950... Generator Loss: 0.9622
    Epoch 1/1... Discriminator Loss: 1.3473... Generator Loss: 0.7522
    Epoch 1/1... Discriminator Loss: 1.3788... Generator Loss: 0.6587
    Epoch 1/1... Discriminator Loss: 1.1851... Generator Loss: 0.8257
    Epoch 1/1... Discriminator Loss: 1.3360... Generator Loss: 0.7985
    Epoch 1/1... Discriminator Loss: 1.3686... Generator Loss: 0.7144
    Epoch 1/1... Discriminator Loss: 1.3770... Generator Loss: 0.8312



![png](/img/facegenerator/output_25_35.png)


    Epoch 1/1... Discriminator Loss: 1.2954... Generator Loss: 0.8060
    Epoch 1/1... Discriminator Loss: 1.3128... Generator Loss: 0.8790
    Epoch 1/1... Discriminator Loss: 1.3055... Generator Loss: 0.8695
    Epoch 1/1... Discriminator Loss: 1.1582... Generator Loss: 0.9782
    Epoch 1/1... Discriminator Loss: 1.3420... Generator Loss: 0.9030
    Epoch 1/1... Discriminator Loss: 1.1851... Generator Loss: 0.9982
    Epoch 1/1... Discriminator Loss: 1.3686... Generator Loss: 0.7583
    Epoch 1/1... Discriminator Loss: 1.3648... Generator Loss: 0.7411
    Epoch 1/1... Discriminator Loss: 1.2680... Generator Loss: 0.9324
    Epoch 1/1... Discriminator Loss: 1.3726... Generator Loss: 0.7018



![png](/img/facegenerator/output_25_37.png)


    Epoch 1/1... Discriminator Loss: 1.3153... Generator Loss: 0.7678
    Epoch 1/1... Discriminator Loss: 1.3920... Generator Loss: 0.7677
    Epoch 1/1... Discriminator Loss: 1.3415... Generator Loss: 0.8016
    Epoch 1/1... Discriminator Loss: 1.2949... Generator Loss: 0.7652
    Epoch 1/1... Discriminator Loss: 1.2062... Generator Loss: 0.9255
    Epoch 1/1... Discriminator Loss: 1.4185... Generator Loss: 0.7812
    Epoch 1/1... Discriminator Loss: 1.4017... Generator Loss: 0.9312
    Epoch 1/1... Discriminator Loss: 1.3149... Generator Loss: 1.1798
    Epoch 1/1... Discriminator Loss: 1.4739... Generator Loss: 0.6627
    Epoch 1/1... Discriminator Loss: 1.3684... Generator Loss: 0.7918



![png](/img/facegenerator/output_25_39.png)


    Epoch 1/1... Discriminator Loss: 1.4715... Generator Loss: 0.7140
    Epoch 1/1... Discriminator Loss: 1.2843... Generator Loss: 0.7990
    Epoch 1/1... Discriminator Loss: 1.2161... Generator Loss: 0.8644
    Epoch 1/1... Discriminator Loss: 1.2104... Generator Loss: 0.8849
    Epoch 1/1... Discriminator Loss: 1.5012... Generator Loss: 0.6990
    Epoch 1/1... Discriminator Loss: 1.3124... Generator Loss: 0.8760
    Epoch 1/1... Discriminator Loss: 1.2560... Generator Loss: 0.8102
    Epoch 1/1... Discriminator Loss: 1.0677... Generator Loss: 1.0095
    Epoch 1/1... Discriminator Loss: 1.3556... Generator Loss: 0.6629
    Epoch 1/1... Discriminator Loss: 1.4408... Generator Loss: 0.5373



![png](/img/facegenerator/output_25_41.png)


    Epoch 1/1... Discriminator Loss: 1.3078... Generator Loss: 0.7650
    Epoch 1/1... Discriminator Loss: 1.0900... Generator Loss: 1.0246
    Epoch 1/1... Discriminator Loss: 1.4812... Generator Loss: 1.0133
    Epoch 1/1... Discriminator Loss: 1.3468... Generator Loss: 0.6327
    Epoch 1/1... Discriminator Loss: 1.3533... Generator Loss: 0.9688
    Epoch 1/1... Discriminator Loss: 1.3225... Generator Loss: 0.7116
    Epoch 1/1... Discriminator Loss: 1.5624... Generator Loss: 0.4262
    Epoch 1/1... Discriminator Loss: 1.4535... Generator Loss: 0.7086
    Epoch 1/1... Discriminator Loss: 1.3774... Generator Loss: 0.7772
    Epoch 1/1... Discriminator Loss: 1.3708... Generator Loss: 0.7432



![png](/img/facegenerator/output_25_43.png)


    Epoch 1/1... Discriminator Loss: 1.3522... Generator Loss: 0.7890
    Epoch 1/1... Discriminator Loss: 1.2590... Generator Loss: 0.8186
    Epoch 1/1... Discriminator Loss: 1.4569... Generator Loss: 0.6861
    Epoch 1/1... Discriminator Loss: 1.3049... Generator Loss: 0.7738
    Epoch 1/1... Discriminator Loss: 1.4734... Generator Loss: 0.5271
    Epoch 1/1... Discriminator Loss: 1.1704... Generator Loss: 1.0307
    Epoch 1/1... Discriminator Loss: 1.2443... Generator Loss: 0.8466
    Epoch 1/1... Discriminator Loss: 1.3109... Generator Loss: 0.8697
    Epoch 1/1... Discriminator Loss: 1.2564... Generator Loss: 0.8187
    Epoch 1/1... Discriminator Loss: 1.2972... Generator Loss: 0.8461



![png](/img/facegenerator/output_25_45.png)


    Epoch 1/1... Discriminator Loss: 1.1784... Generator Loss: 0.9402
    Epoch 1/1... Discriminator Loss: 1.4024... Generator Loss: 0.7195
    Epoch 1/1... Discriminator Loss: 1.3197... Generator Loss: 0.6825
    Epoch 1/1... Discriminator Loss: 1.4874... Generator Loss: 0.4925
    Epoch 1/1... Discriminator Loss: 1.4779... Generator Loss: 0.9170
    Epoch 1/1... Discriminator Loss: 1.7313... Generator Loss: 0.3356
    Epoch 1/1... Discriminator Loss: 1.2387... Generator Loss: 0.9218
    Epoch 1/1... Discriminator Loss: 1.1896... Generator Loss: 0.8645
    Epoch 1/1... Discriminator Loss: 1.3893... Generator Loss: 0.8077
    Epoch 1/1... Discriminator Loss: 1.2284... Generator Loss: 0.8447



![png](/img/facegenerator/output_25_47.png)


    Epoch 1/1... Discriminator Loss: 1.3455... Generator Loss: 0.7420
    Epoch 1/1... Discriminator Loss: 1.4410... Generator Loss: 0.7239
    Epoch 1/1... Discriminator Loss: 1.3469... Generator Loss: 0.7676
    Epoch 1/1... Discriminator Loss: 1.3781... Generator Loss: 0.6891
    Epoch 1/1... Discriminator Loss: 1.2879... Generator Loss: 0.8008
    Epoch 1/1... Discriminator Loss: 1.3583... Generator Loss: 0.7410
    Epoch 1/1... Discriminator Loss: 1.3640... Generator Loss: 0.8243
    Epoch 1/1... Discriminator Loss: 1.1539... Generator Loss: 0.9609
    Epoch 1/1... Discriminator Loss: 1.2718... Generator Loss: 0.8644
    Epoch 1/1... Discriminator Loss: 1.2479... Generator Loss: 0.8401



![png](/img/facegenerator/output_25_49.png)


    Epoch 1/1... Discriminator Loss: 1.4485... Generator Loss: 0.5275
    Epoch 1/1... Discriminator Loss: 1.4431... Generator Loss: 0.7631
    Epoch 1/1... Discriminator Loss: 1.2541... Generator Loss: 0.9157
    Epoch 1/1... Discriminator Loss: 1.2299... Generator Loss: 0.8804
    Epoch 1/1... Discriminator Loss: 1.4235... Generator Loss: 0.5314
    Epoch 1/1... Discriminator Loss: 1.4212... Generator Loss: 0.7266
    Epoch 1/1... Discriminator Loss: 1.2835... Generator Loss: 0.7007
    Epoch 1/1... Discriminator Loss: 1.3568... Generator Loss: 0.7339
    Epoch 1/1... Discriminator Loss: 1.3212... Generator Loss: 0.7385
    Epoch 1/1... Discriminator Loss: 1.1365... Generator Loss: 1.0084



![png](/img/facegenerator/output_25_51.png)


    Epoch 1/1... Discriminator Loss: 1.4486... Generator Loss: 1.0020
    Epoch 1/1... Discriminator Loss: 1.3057... Generator Loss: 0.7669
    Epoch 1/1... Discriminator Loss: 1.5529... Generator Loss: 0.4767
    Epoch 1/1... Discriminator Loss: 1.4000... Generator Loss: 0.6269
    Epoch 1/1... Discriminator Loss: 1.3090... Generator Loss: 0.8894
    Epoch 1/1... Discriminator Loss: 1.2235... Generator Loss: 0.9030
    Epoch 1/1... Discriminator Loss: 1.2490... Generator Loss: 0.8701
    Epoch 1/1... Discriminator Loss: 1.3437... Generator Loss: 0.5722
    Epoch 1/1... Discriminator Loss: 1.2430... Generator Loss: 0.9822
    Epoch 1/1... Discriminator Loss: 1.3213... Generator Loss: 0.6595



![png](/img/facegenerator/output_25_53.png)


    Epoch 1/1... Discriminator Loss: 1.1092... Generator Loss: 0.9101
    Epoch 1/1... Discriminator Loss: 1.2494... Generator Loss: 0.9528
    Epoch 1/1... Discriminator Loss: 1.2662... Generator Loss: 0.8350
    Epoch 1/1... Discriminator Loss: 1.3351... Generator Loss: 0.7881
    Epoch 1/1... Discriminator Loss: 1.3444... Generator Loss: 0.7316
    Epoch 1/1... Discriminator Loss: 1.3791... Generator Loss: 0.5329
    Epoch 1/1... Discriminator Loss: 1.3428... Generator Loss: 0.7200
    Epoch 1/1... Discriminator Loss: 1.2052... Generator Loss: 0.9178
    Epoch 1/1... Discriminator Loss: 1.3815... Generator Loss: 0.7848
    Epoch 1/1... Discriminator Loss: 1.3434... Generator Loss: 0.8321



![png](/img/facegenerator/output_25_55.png)


    Epoch 1/1... Discriminator Loss: 1.3740... Generator Loss: 0.6343
    Epoch 1/1... Discriminator Loss: 1.1297... Generator Loss: 1.0645
    Epoch 1/1... Discriminator Loss: 1.2523... Generator Loss: 0.8868
    Epoch 1/1... Discriminator Loss: 1.2764... Generator Loss: 0.7321
    Epoch 1/1... Discriminator Loss: 1.3993... Generator Loss: 0.5500
    Epoch 1/1... Discriminator Loss: 1.3122... Generator Loss: 0.8831
    Epoch 1/1... Discriminator Loss: 1.3974... Generator Loss: 0.8300
    Epoch 1/1... Discriminator Loss: 1.4547... Generator Loss: 0.7671
    Epoch 1/1... Discriminator Loss: 1.3008... Generator Loss: 0.8082
    Epoch 1/1... Discriminator Loss: 1.3335... Generator Loss: 0.6352



![png](/img/facegenerator/output_25_57.png)


    Epoch 1/1... Discriminator Loss: 1.3042... Generator Loss: 0.7183
    Epoch 1/1... Discriminator Loss: 1.3713... Generator Loss: 0.5554
    Epoch 1/1... Discriminator Loss: 1.4189... Generator Loss: 0.8150
    Epoch 1/1... Discriminator Loss: 1.3934... Generator Loss: 0.6503
    Epoch 1/1... Discriminator Loss: 1.3552... Generator Loss: 0.8267
    Epoch 1/1... Discriminator Loss: 1.4232... Generator Loss: 0.4965
    Epoch 1/1... Discriminator Loss: 1.2116... Generator Loss: 1.0363
    Epoch 1/1... Discriminator Loss: 1.3030... Generator Loss: 0.7540
    Epoch 1/1... Discriminator Loss: 1.2195... Generator Loss: 0.9634
    Epoch 1/1... Discriminator Loss: 1.2943... Generator Loss: 0.7414



![png](/img/facegenerator/output_25_59.png)


    Epoch 1/1... Discriminator Loss: 1.2517... Generator Loss: 0.7875
    Epoch 1/1... Discriminator Loss: 1.3333... Generator Loss: 0.9016
    Epoch 1/1... Discriminator Loss: 1.1542... Generator Loss: 0.9181
    Epoch 1/1... Discriminator Loss: 1.3371... Generator Loss: 1.0411
    Epoch 1/1... Discriminator Loss: 1.2682... Generator Loss: 0.8428
    Epoch 1/1... Discriminator Loss: 1.3626... Generator Loss: 0.6750
    Epoch 1/1... Discriminator Loss: 1.1461... Generator Loss: 0.9491
    Epoch 1/1... Discriminator Loss: 1.0990... Generator Loss: 0.7883
    Epoch 1/1... Discriminator Loss: 1.4364... Generator Loss: 0.7612
    Epoch 1/1... Discriminator Loss: 1.4027... Generator Loss: 0.8417



![png](/img/facegenerator/output_25_61.png)
