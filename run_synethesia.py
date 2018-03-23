#! /usr/bin/env python3
import sys
import argparse

from synethesia import Synethesia

def parse_args():

    parser = argparse.ArgumentParser(description="""SyNEThesia is a deep neural network that
                                                 visualizes music (and other sound).""")
    subparsers = parser.add_subparsers(dest="mode", help="""Choose a mode of operation.""")
    subparsers.required = True
    train_parser = subparsers.add_parser("train", help="Train the model.")
    store_parser = subparsers.add_parser("infer", help="Infer songs or other sounds and store the frames in a folder.")
    stream_parser = subparsers.add_parser("stream", help="Infer songs or other sounds and visualize the frames live.")

    parser.add_argument("model_name", type=str,
                        help="""Name of the model to be trained or used for inference.
                                If it exists in the local checkpoints folder, the model will be loaded,
                                otherwise it will be newly created.""")

    parser.add_argument("data", type=str,
                        help="""Either a file containing paths to .mp3's, or a folder containing .mp3's.""")
    parser.add_argument("-b", "--batch-size", default=1, type=int, dest="batch_size",
                        help="Batch size. Default is %(default)s. Ignored for streaming inference.")

    parser.add_argument("-r", "--n-rows", default=256, type=int, dest="rows",
                        help="""Image rows (height). Should be a power of two.
                                An error will be thrown if the loaded model was not trained on the same size.
                                Defaults to %(default)s.""")
    parser.add_argument("-c", "--n-cols", default=128, type=int, dest="cols",
                        help="""Image columns (width). Should be a power of two.
                                An error will be thrown if the loaded model was not trained on the same size.
                                Defaults to %(default)s.""")

    train_parser.add_argument("-l", "--learning-rate", default=0.0001, type=float, dest="learning_rate",
                              help="""Learning rate for training. Will be
                                      exponentially decayed over time.
                                      Defaults to %(defauls)s.""")

    store_parser.add_argument("target_dir", default="/tmp", type=str,
                              help="""Target directory for storing the resulting frames.
                                      Warning: There may be many. Defaults to %(default)s.""")

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":

    arguments = parse_args()
    mode = arguments.mode
    batch_size = arguments.batch_size
    img_size = (arguments.rows, arguments.cols)
    model_name = arguments.model_name
    song_files = model_name.data

    synethesia = Synethesia(song_files=song_files, batch_size=batch_size, img_size=img_size)
    if mode.lower() == "train":
        learning_rate = arguments.learning_rate
        synethesia.train(model_name=model_name, learning_rate=learning_rate)
    elif mode.lower() == "store":
        target_dir = arguments.target_dir
        synethesia.infer_and_store(model_name=model_name, target_dir=target_dir)
    elif mode.lower() == "stream":
        synethesia.infer_and_stream(model_name=model_name)
