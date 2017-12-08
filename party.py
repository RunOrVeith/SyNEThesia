import tensorflow as tf
from tensorflow.contrib import ffmpeg, signal
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from partygan.model import NeuralEqualizer
from partygan.ops import rfft


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

def show_image(image_data, name, save=False):
    plt.subplot(1, 1, 1)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    c_dim = 3
    if c_dim > 1:
      plt.imshow(image_data, interpolation='nearest')
    else:
      plt.imshow(image_data.reshape(y_dim, x_dim), cmap='Greys', interpolation='nearest')
    plt.axis('off')
    if not save:
        plt.show()
    else:
        plt.savefig(f'/home/veith/Projects/PartyGAN/data/generated/420/{name}.png', bbox_inches='tight')

equalizer = NeuralEqualizer(batch_size=1, latent_dim=3532, color_dim=3, scale=1.0, net_size=128)
song = SampledSong(song_path = "/home/veith/Projects/PartyGAN/data/Bearded Skull - 420 [Hip Hop Instrumental]/audio/soundtrack.mp3")

with tf.Session() as sess:
    equalizer.init(sess)
    spectrum = sess.run(song.features)
    for time, latent_sound in enumerate(spectrum):
        print(latent_sound)
        img_batch = equalizer.generate(sess, x_dim=256, y_dim=256, scale=25.0, latent=np.expand_dims(latent_sound, 0))
        for i, img in enumerate(img_batch):
            show_image(img, name=time + i, save=True)
