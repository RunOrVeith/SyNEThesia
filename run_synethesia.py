#! /usr/bin/env python3
import sys
import argparse
from pathlib import Path

from synethesia import Synethesia
from synethesia.network import AudioRecorder

def parse_args():

    parser = argparse.ArgumentParser(description="""SyNEThesia is a deep neural network that
                                                 visualizes music (and other sound).""")

    subparsers = parser.add_subparsers(dest="mode", help="""Choose a mode of operation.""")
    subparsers.required = True

    train_parser = subparsers.add_parser("train", help="Train the model.")
    store_parser = subparsers.add_parser("infer", help="Infer songs or other sounds and store the frames in a folder.")
    stream_parser = subparsers.add_parser("stream", help="""Infer input from your standard microphone live.
                                                            Opens a window in full screen mode.
                                                            Close it by pressing 'q', or minimize it with 'Esc'.""")
    info_parser = subparsers.add_parser("info", help="Display additional information, such as available models.")

    for _parser in [train_parser, store_parser, stream_parser]:
        _parser.add_argument("model_name", type=str,
                            help="""Name of the model to be trained or used for inference.
                                    If it exists in the local checkpoints folder, the model will be loaded,
                                    otherwise it will be newly created.""")

        _parser.add_argument("-b", "--batch-size", default=32, type=int, dest="batch_size",
                             help="Batch size. Default is %(default)s. Ignored for streaming inference.")

        _parser.add_argument("-r", "--n-rows", default=256, type=int, dest="rows",
                             help="""Image rows (height). Should be a power of two.
                                     Defaults to %(default)s.""")
        _parser.add_argument("-c", "--n-cols", default=128, type=int, dest="cols",
                             help="""Image columns (width). Should be a power of two.
                                     Defaults to %(default)s.""")

    train_parser.add_argument("data", type=str,
                              help="""Either a file containing paths to .mp3's, or a folder containing .mp3's,
                                      or a single .mp3""")
    train_parser.add_argument("-l", "--learning-rate", default=0.0001, type=float, dest="learning_rate",
                              help="""Learning rate for training. Will be
                                      exponentially decayed over time.
                                      Defaults to %(default)s.""")
    train_parser.add_argument("-ns", "--no-shuffle-input", dest="allow_shuffle", action="store_false",
                              help="""Learning rate for training. Will be
                                      exponentially decayed over time.
                                      Defaults to %(default)s.""")
    train_parser.add_argument("-lrs", "--lambda-reconstruct-sound", type=float,
                              default=1.0, help="""Weight factor of sound reconstruction in the total loss.
                                                   Defaults to %(default)s.""")
    train_parser.add_argument("-lri", "--lambda-reconstruct-image", type=float,
                              default=0.0, help="""Weight factor of input image reconstruction in the total loss.
                                                   Defaults to %(default)s. The input image is currently random.""")
    train_parser.add_argument("-lc", "--lambda-change", type=float,
                              default=0.0, help="""Weight factor of difference to previous infered image
                                                   in the total loss.
                                                   Defaults to %(default)s.""")
    train_parser.add_argument("-ln", "--lambda-noise", type=float,
                              default=0.0, help="""Weight factor of penalizing noise in the total loss.
                                                   Defaults to %(default)s.""")
    train_parser.add_argument("-lco", "--lambda-colorfulness", type=float,
                              default=0.0, help="""Weight factor colorfulness in the total loss.
                                                   Defaults to %(default)s.
                                                   (This loss works the least well).""")
    train_parser.add_argument("-rc", "--with-random-crop", action="store_true",
                              help="""Use random crop of the generated image to reconstruct image.
                                   Defaults to %(default)s.""")

    store_parser.add_argument("data", type=str,
                              help="""Either a file containing paths to .mp3's, or a folder containing .mp3's,
                                      or a single .mp3""")
    store_parser.add_argument("target_dir", type=str,
                              help="""Target directory for storing the resulting video.
                                      Warning: There may be many frames stored here intermediately,
                                      so make sure you have enough disc space.""")

    stream_parser.add_argument("-d", "--device-index", dest="device_index", type=int, default=None,
                               help="""Device index to be used as audio input.
                                       Default is the system's standard audio input.""")

    arguments = parser.parse_args()
    return arguments


def info():
    pth = Path.cwd()
    ckpt_pth = pth / "checkpoints"
    if ckpt_pth.is_dir():
        print(f"Found the following models in {str(ckpt_pth)}:")
        print(list(_pth.parts[-1] for _pth in ckpt_pth.iterdir()))
    else:
        print(f"Could not fine any pretrained models in {str(ckpt_pth)}.")

    rec = AudioRecorder(feature_extractor=lambda _:None)
    rec.print_input_device_info()

if __name__ == "__main__":

    arguments = parse_args()
    mode = arguments.mode

    if mode == "info":
        info()
    else:
        batch_size = arguments.batch_size
        img_size = (arguments.rows, arguments.cols)
        model_name = arguments.model_name
        song_files = arguments.data if hasattr(arguments, "data") else None
        synethesia = Synethesia(song_files=song_files, batch_size=batch_size, img_size=img_size)
        if mode.lower() == "train":
            learning_rate = arguments.learning_rate
            shuffle = arguments.allow_shuffle

            synethesia.train(model_name=model_name, learning_rate=learning_rate, shuffle_input=shuffle,
                             lambda_reconstruct_sound=arguments.lambda_reconstruct_sound,
                             lambda_reconstruct_image=arguments.lambda_reconstruct_image,
                             lambda_noise=arguments.lambda_noise, lambda_change=arguments.lambda_change,
                             lambda_colorfulness=arguments.lambda_colorfulness,
                             with_random_crop=arguments.with_random_crop)
        elif mode.lower() == "infer":
            song_files = arguments.data
            target_dir = arguments.target_dir
            synethesia.infer_and_store(model_name=model_name, target_dir=target_dir)
        elif mode.lower() == "stream":
            device_idx = arguments.device_index
            synethesia.infer_and_stream(model_name=model_name, input_device_index=device_idx)
        elif mode.lower() == "info":
            info(__path__)
