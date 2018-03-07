import tensorflow as tf
from tensorflow.contrib import ffmpeg, signal
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from partygan.model import NeuralEqualizer, GANTrainer
import os

data_dir = "/home/veith/Projects/PartyGAN/data/Bearded Skull - 420 [Hip Hop Instrumental]/frames"
train_files = [os.path.join(data_dir, name) for name in os.listdir(data_dir)]


equalizer = NeuralEqualizer(latent_dim=1764, color_dim=3)
trainer = GANTrainer(train_files=train_files,
                      gan=equalizer.gan, sample_size=10, model_name="bearded_skull")
with tf.Session() as sess:
    trainer.train(sess, 8, learning_rate=0.0002, beta1=0.5, sample_every_n_steps=1, save_every_n_steps=50)
