# PROBLEM! - GAN COLLAPSE

import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras import models, optimizers, layers, preprocessing
from tensorflow.python.keras.models import Sequential
import tensorflow.python.keras.layers
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.models import Model

from matplotlib import pyplot as plt
import numpy as np
import pickle

# Loading the preprocessed data from pickle
X_test = pickle.load(open("X_test.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))
X_train = pickle.load(open("X_train.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))

# Displaying infromation about data
# print(X_train.shape) => (349, 64, 64, 3)
# print(type(X_train)) => <class 'numpy.ndarray'>

# Visualizing the data
plt.figure()
plt.imshow(X_train[0])
plt.show()

# Latent dimension of the random noise
LATENT_DIM = 10
# Weight initializer for G per DCGAN paper
WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Number of channels, 1 for gray scale and 3 for color images
CHANNELS = 3


# Generator
def build_generator():
    model = Sequential(name='generator')

    model.add(layers.Dense(8 * 8 * 512, input_dim=LATENT_DIM))
    model.add(layers.ReLU())

    model.add(layers.Reshape((8, 8, 512)))

    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=WEIGHT_INIT))
    model.add((layers.ReLU()))

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=WEIGHT_INIT))
    model.add((layers.ReLU()))

    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=WEIGHT_INIT))
    model.add((layers.ReLU()))

    model.add(layers.Conv2D(CHANNELS, (4, 4), padding="same", activation="tanh"))

    return model


# Building the generator model
generator = build_generator()
generator.summary()


# Discriminator
def build_discriminator(height, width, depth, alpha=0.2):
    model = Sequential(name='discriminator')
    input_shape = (height, width, depth)

    model.add(layers.Conv2D(64, (4, 4), padding="same", strides=(2, 2),
        input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=alpha))

    model.add(layers.Conv2D(128, (4, 4), padding="same", strides=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=alpha))

    model.add(layers.Conv2D(128, (4, 4), padding="same", strides=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=alpha))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1, activation="sigmoid"))

    return model


# Building the discriminator model
discriminator = build_discriminator(64, 64, 3)
discriminator.summary()


# Defining DCGAN class
class DCGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):

        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train the discriminator with real(1) and fake(0) images
        with tf.GradientTape() as tape:
            pred_real = self.discriminator(real_images, training=True)
            real_labels = tf.ones((batch_size, 1))
            real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))
            d_loss_real = self.loss_fn(real_labels, pred_real)

            fake_images = self.generator(noise)
            pred_fake = self.discriminator(fake_images, training=True)
            fake_labels = tf.zeros((batch_size, 1))
            d_loss_fake = self.loss_fn(fake_labels, pred_fake)

            # total discriminator loss
            d_loss = (d_loss_real + d_loss_fake)/2
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # Train the generator
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            pred_fake = self.discriminator(fake_images, training=True)
            g_loss = self.loss_fn(misleading_labels, pred_fake)
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=100):
        self.num_img = num_img
        self.latent_dim = latent_dim

        self.seed = tf.random.normal([16, latent_dim])

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator(self.seed)
        generated_images = (generated_images * 127.5) + 127.5
        generated_images.numpy()

        fig = plt.figure(figsize=(4, 4))
        for i in range(self.num_img):
            plt.subplot(4, 4, i+1)
            img = tensorflow.keras.preprocessing.image.array_to_img(generated_images[i])
            plt.imshow(img)
            plt.axis('off')
        plt.savefig('epoch_{:03d}.png'.format(epoch))
        plt.show()

    def on_train_end(self, logs=None):
        self.model.generator.save('generator.h5')


dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)

D_LR = 0.0001 # UPDATED: discriminator learning rate
G_LR = 0.0003 # UPDATED: generator learning rate

dcgan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=D_LR, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=G_LR, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

NUM_EPOCHS = 10
dcgan.fit(X_train, epochs=NUM_EPOCHS, callbacks=[GANMonitor(num_img=16, latent_dim=LATENT_DIM)])
