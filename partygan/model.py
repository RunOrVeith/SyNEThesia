import tensorflow as tf
import numpy as np
from partygan.ops import *
import math
import random
import numpy as np
import abc
from typing import List
import time
import os
import scipy.misc
import math
import random
import numpy as np
import abc
from typing import List
import time
import os
import scipy.misc


class NeuralEqualizer(object):

    def __init__(self,  latent_dim=32, color_dim=1):

        x_dim = 256
        self.gan = DCGAN(x_dim, z_dim=latent_dim, c_dim=color_dim, df_dim=64, gf_dim=64, gfc_dim=1024, dfc_dim=1024)

class GANModel(object, metaclass=abc.ABCMeta):

    def __init__(self, z_dim, img_shape):
        self.c_dim = img_shape[-1]
        self.z_dim = z_dim
        self.image_shape = img_shape

        self.image_input = tf.placeholder(tf.float32, shape=[None] + self.image_shape, name="image_input")
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name="z")
        self.is_training = tf.placeholder(tf.bool, name="is_training", shape=[])

        self.d_loss = None
        self.g_loss = None
        self.d_loss_real = None
        self.d_loss_fake = None
        self.d_vars = None
        self.g_vars = None
        self.d_summary = None
        self.g_summary = None
        self.build_model()

    def build_model(self):
        G = self.generator(self.z)

        # Real image discriminator
        D, D_logits = self.discriminator(self.image_input)

        # Fake image discrimiantor
        D_, D_logits_ = self.discriminator(G)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits,
                                                                             labels=tf.ones_like(D)),
                                     name="D_loss_real")
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_,
                                                                             labels=tf.zeros_like(D_)),
                                     name="D_loss_fake")

        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_,
                                                                        labels=tf.ones_like(D_)),
                                name="G_loss")
        d_loss = d_loss_real + d_loss_fake

        #g_loss = tf.Print(g_loss, [g_loss], " generator full loss")
        #d_loss = tf.Print(d_loss, [d_loss], " discriminator full loss")

        trainables = tf.trainable_variables()
        d_vars = [var for var in trainables if "discriminator" in var.name.lower()]
        g_vars = [var for var in trainables if "generator" in var.name.lower()]

        # Add some summaries for overview
        z_summary = tf.summary.histogram("z", self.z)

        D_summary = tf.summary.histogram("D", D)
        D__summary = tf.summary.histogram("D_", D_)
        G_summary = tf.summary.image("G", G)

        d_loss_real_summary = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_summary = tf.summary.scalar("d_loss_fake", d_loss_fake)

        d_loss_summary = tf.summary.scalar("d_loss", d_loss)
        g_loss_summary = tf.summary.scalar("g_loss", g_loss)

        g_summary = tf.summary.merge([z_summary, D__summary, G_summary, d_loss_fake_summary, g_loss_summary])
        d_summary = tf.summary.merge([z_summary, D_summary, d_loss_real_summary, d_loss_summary])

        self.d_loss = d_loss
        self.d_loss_real = d_loss_real
        self.d_loss_fake = d_loss_fake
        self.g_loss = g_loss
        self.d_vars = d_vars
        self.g_vars = g_vars
        self.d_summary = d_summary
        self.g_summary = g_summary
        self.G = G

    @abc.abstractmethod
    def discriminator(self, image):
        pass

    @abc.abstractmethod
    def generator(self, z):
        pass


class DCGAN(GANModel):

    def __init__(self, image_size, z_dim, c_dim, df_dim, gf_dim, gfc_dim, dfc_dim):
        self.image_size = image_size

        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        scope = "dcgan"
        image_shape = [image_size, image_size, c_dim]
        self.dbns = None
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.gbns = [BatchNorm(scope=f"gbn{i}") for i in range(int(math.log(self.image_size) / math.log(2)))]

            super(DCGAN, self).__init__(z_dim=z_dim, img_shape=image_shape)

    def discriminator(self, image, num_dense=3):

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            if self.dbns is None:
                self.dbns = [BatchNorm(scope=f"dbn{i}") for i in range(num_dense)]

            lay = lrelu(conv2d(image, self.df_dim, scope="d_h0_conv"))
            for i in range(num_dense):
                lay = lrelu(self.dbns[i](x=conv2d(lay, self.df_dim*(2 ** (i+1)), scope=f"d_h{i+1}_conv"),
                                         train=self.is_training))

            lay = linear(tf.reshape(lay, [-1, 8192]), 1, scope="d_h4_lin")  # TODO size as general as possible
            #lay = tf.Print(lay, [lay], " discriminator lay")
            return tf.nn.sigmoid(lay), lay

    def generator(self, z):
        with tf.variable_scope("generator"):

            _z = linear(z, self.gf_dim * 8 * 4 * 4, 'g_h0_lin', with_w=False)

            deconv = tf.nn.relu(self.gbns[0](x=tf.reshape(_z, [-1, 4, 4, self.gf_dim * 8]), train=self.is_training))

            i = 1
            depth_mul = 8
            size = 8

            while depth_mul > 1:
                name = f"g_h{i}"
                deconv = conv2d_transpose(x=deconv, output_dim=[tf.shape(z)[0], size, size, self.gf_dim*depth_mul],
                                          scope=name)
                deconv = tf.nn.relu(self.gbns[i](deconv, self.is_training))

                i += 1
                depth_mul //= 2
                size *= 2

            name = f"g_h{i}"
            deconv = conv2d_transpose(deconv, [tf.shape(z)[0], size, size, self.c_dim], scope=name)
            #deconv = tf.Print(deconv, [deconv], " generator deconv")
            return tf.nn.tanh(deconv)


class GANTrainer(object):

    def __init__(self, train_files: List[str], gan: GANModel,
                 model_name: str, sample_size: int=32,
                 log_dir: str="./logs", chkpt_dir: str="./checkpoints"):
        self.train_files = train_files

        self.gan = gan
        self.sample_size = sample_size

        self.model_name = model_name
        self.log_dir = log_dir
        self.checkpoint_dir = chkpt_dir

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def z_initializer(self, batch_size):
        return np.random.uniform(-1, 1, [batch_size, self.gan.z_dim]).astype(np.float32)

    def train(self, sess, batch_size, learning_rate, beta1, epochs=float("inf"), use_exp_decay=True,
              sample_every_n_steps=100, save_every_n_steps=500):
        assert len(self.train_files) > 0

        global_step_d = tf.get_variable(name="global_step_d", trainable=False, shape=[],
                                        initializer=tf.constant_initializer(0),
                                        dtype=tf.int64)
        global_step_g = tf.get_variable(name="global_step_g", trainable=False, shape=[],
                                        initializer=tf.constant_initializer(0),
                                        dtype=tf.int64)
        epoch_tensor = tf.get_variable(name="epoch", trainable=False, shape=[], initializer=tf.constant_initializer(0),
                                       dtype=tf.int64)
        if use_exp_decay:
            learning_rate_d = tf.train.exponential_decay(learning_rate, global_step_d, 1000, 0.96, staircase=False)
            learning_rate_g = tf.train.exponential_decay(learning_rate, global_step_g, 1000, 0.96, staircase=False)
        else:
            learning_rate_d, learning_rate_g = learning_rate, learning_rate

        d_optim = tf.train.AdamOptimizer(learning_rate_d, beta1=beta1).minimize(self.gan.d_loss,
                                                                                var_list=self.gan.d_vars,
                                                                                global_step=global_step_d)
        g_optim = tf.train.AdamOptimizer(learning_rate_g, beta1=beta1).minimize(self.gan.g_loss,
                                                                                var_list=self.gan.g_vars,
                                                                                global_step=global_step_g)
        saver = tf.train.Saver(max_to_keep=5)

        init = tf.global_variables_initializer()

        sample_z = self.z_initializer(min(len(self.train_files), self.sample_size))
        sample_files = self.train_files[:self.sample_size]
        sample_images = np.array([self.load_img(pth=sample_file) for sample_file in sample_files]).astype(np.float32)

        summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Loading existing model")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Initializing new model")
            sess.run(init)

        start_time = time.time()
        while sess.run(epoch_tensor) < epochs:
            random.shuffle(self.train_files)
            batch_idxs = len(self.train_files) // batch_size

            for idx in range(batch_idxs):

                batch_files = self.train_files[idx * batch_size: (idx+1) * batch_size]
                batch = np.array([self.load_img(pth=img_file) for img_file in batch_files]).astype(np.float32)
                batch_z = self.z_initializer(batch_size)

                # Train Generator
                for i in range(1):
                    _, summary = sess.run([g_optim, self.gan.g_summary],
                                          feed_dict={self.gan.is_training: True,
                                                     self.gan.z: batch_z})
                    summary_writer.add_summary(summary, tf.train.global_step(sess, global_step_g))

                # Train Discriminator
                _, summary = sess.run([d_optim, self.gan.d_summary],
                                      feed_dict={self.gan.image_input: batch,
                                                 self.gan.z: batch_z,
                                                 self.gan.is_training: True})
                summary_writer.add_summary(summary, tf.train.global_step(sess, global_step_d))

            error_D_fake = self.gan.d_loss_fake.eval({self.gan.z: batch_z, self.gan.is_training: False})
            error_D_real = self.gan.d_loss_real.eval({self.gan.image_input: batch, self.gan.is_training: False})
            error_G = self.gan.g_loss.eval({self.gan.z: batch_z, self.gan.is_training: False})

            epoch = sess.run(epoch_tensor)
            global_step_g += 1
            global_step_d += 1
            epoch_tensor += 1

            print(f"Epoch {epoch}: [{idx}/{batch_idxs - 1}] time: {time.time() - start_time}, " +
                  f"d_loss: {error_D_real + error_D_fake}, g_loss: {error_G}")

            if epoch % (sample_every_n_steps + 1) == sample_every_n_steps:
                # Create some samples
                samples, = sess.run([self.gan.G],
                                    feed_dict={self.gan.z: sample_z,
                                    self.gan.image_input: sample_images,
                                    self.gan.is_training: False})
                self.save_images(images=samples, pths=sample_files)

            if epoch % (save_every_n_steps + 1) == save_every_n_steps:
                # Save the model
                saver.save(sess, os.path.join(self.checkpoint_dir, self.model_name), epoch)

    def load_img(self, pth):
        img = scipy.misc.imresize(scipy.misc.imread(pth, mode='RGB'),
                                  size=(self.gan.image_size, self.gan.image_size),
                                  interp="bicubic", mode="RGB").astype(np.float) / 255
        return img

    def save_images(self, images, pths):
        assert len(images) == len(pths)
        store_folder = os.path.join(self.log_dir, "samples")
        os.makedirs(store_folder, exist_ok=True)
        for i, img in enumerate(images):
            img_name = os.path.split(pths[i])[-1]
            save_location = os.path.join(store_folder, img_name)
            scipy.misc.imsave(save_location, (255*img).astype(np.uint8))



class SampledSong(object):

    def __init__(self, song_path, samples_per_second=44100, sound_window_samples=25):
        audio_binary = tf.read_file(song_path)
        waveform = ffmpeg.decode_audio(audio_binary,
                                       file_format='mp3',
                                       samples_per_second=samples_per_second,
                                       channel_count=1)
        sample_duration = samples_per_second // sound_window_samples



        waveform = tf.squeeze(waveform)
        song_duration = tf.shape(waveform)[0] / samples_per_second
        window_batches = tf.cast(song_duration, tf.int32) * sound_window_samples
        waveform_padding = window_batches - tf.floormod(tf.shape(waveform)[0], window_batches)
        padding = tf.zeros([waveform_padding], dtype=tf.float32)
        waveform = tf.concat([waveform, padding], axis=0)
        waveform = tf.reshape(waveform, (window_batches, -1))
        spectrum = tf.abs(tf.fft(tf.complex(waveform, tf.zeros_like(waveform)), "fft"), "spectrum")
        self.features = tf.concat(values=[waveform, spectrum], axis=1)
