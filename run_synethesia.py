#! /usr/bin/env python
import sys

from synethesia import Synethesia

# TODO tf.Flags or argparse


if __name__ == "__main__":

    train = bool(sys.argv[1])
    batch_size = 32
    img_size = (256, 128)
    model_name = "loss_tests"
    learning_rate = 0.0001
    target_dir = "/tmp"

    song_files=["/home/veith/Projects/PartyGAN/data/Bearded Skull - 420 [Hip Hop Instrumental]/audio/soundtrack.mp3",
                "/home/veith/Projects/PartyGAN/data/Gorillaz - Feel Good Inc. (Official Video)/audio/soundtrack.mp3",
                "/home/veith/Projects/PartyGAN/data/Gramatik   Just Jammin/audio/soundtrack.mp3"
                ]

    synethesia = Synethesia(song_files=song_files, batch_size=batch_size, img_size=img_size)
    if train:
        synethesia.train(model_name=model_name, learning_rate=learning_rate)
    else:
        synethesia.infer_and_store(model_name=model_name, target_dir=target_dir)
