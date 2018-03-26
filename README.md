# WIP: SyNEThesia

SyNEThesia is a deep-learning-based music and sound visualizer, and a play of words on [Synesthesia](https://en.wikipedia.org/wiki/Synesthesia), a neurological condition where one perceives a stimulus in multiple ways (for example seeing sound).
Its main goal is to produce nice visuals, and act as "learned artist".

The current version produces nice samples most of the time.

## Examples

Click on the images, they redirect to Youtube! There are multiple samples from the same song to illustrate the
different results achievable using different loss functions.
More samples are available in a [playlist](https://www.youtube.com/playlist?list=PLYtGPepvlE7fnOHeHP7bkjPdXNW2cjzJL).


**Grizmatik - My People**| **Gorillaz - Feel Good Inc.**
:---:|:---:
[![/watch?v=9O20t8XsyWM](https://img.youtube.com/vi/9O20t8XsyWM/0.jpg)](https://www.youtube.com/watch?v=9O20t8XsyWM) | [![/watch?v=VdJmd9KLjgE](https://img.youtube.com/vi/VdJmd9KLjgE/0.jpg)](https://www.youtube.com/watch?v=VdJmd9KLjgE)
**Bearded Skull - 420 [Snippet]**| **Bearded Skull - 420 [Snippet 2]**
[![/watch?v=paLXtZr4P6k](https://img.youtube.com/vi/paLXtZr4P6k/0.jpg)](https://www.youtube.com/watch?v=paLXtZr4P6k)|[![/watch?v=kMR0hgHkgB8](https://img.youtube.com/vi/kMR0hgHkgB8/0.jpg)](https://www.youtube.com/watch?v=kMR0hgHkgB8)


## Installation and Setup

This network requires python version >= 3.6. (If you don't already have it, I recommend [pyenv](https://github.com/pyenv/pyenv)). I'm assuming that your global python interpreter is python3.6 from here on.

1. Clone this repository to your computer.
   Let's say you've cloned it to `$SYNETHESIA`.

2. Create a new virtual environment and activate it:

       cd $SYNETHESIA
       python -m venv syn-venv
       source syn-venv/bin/activate

3. Install the requirements:

       pip install -r requirements.txt

4. You're good to go.

## Running SyNEThesia

In the toplevel of `$SYNETHESIA`, there are 2 files that are relevant:

1. `./youtube-to-song-frames.sh link`: Downloads the youtube video `link` to `$SYNETHESIA/data`, and extracts an mp3 audio file, as well as the image frames from its video (the images are currently not needed, but will be in the future). You can also supply your own mp3 files instead of using this script.
You may have to install two dependencies:

       sudo apt install ffmpeg youtube-dl

2. `run_synethesia.py`: This is the main file to start. There are 4 modes with which you can call the program:

    1. train: Trains the network
    2. infer: Infers a song and creates a music video from the resulting images
    3. stream: Infers the input of your microphone live. This opens a new window, quit it with "Esc" or kill it with "q".
    4. info: Displays available pretrained model names (just checks the `./checkpoints` folder) and sound input devices.

  You can get further information by running `run_synethesia.py -h` and `run_synethesia {mode} -h`.


## General Information

Training takes a while before the images start to look nice and the sound is clearly reproduced. I let it run overnight for about 7-8 hours on a GTX 1070.
Of course, untrained versions might also look cool, but you won't be able to "see the sound".

If you use the `infer` mode, be aware that **many** images will be saved to your disk (think 24 * song duration in seconds).
They will be deleted automatically, but if you have little disc space it may become a problem.

The `framework` submodule will probably be moved to a independent repository at some point in the future. SyNEThesia uses it to implement its functionality, but it itself is an abstract set of classes to simply tensorflow development.


## Architecture

Comming soon

### Losses

There are a bunch of loss functions that are implemented and that were tested.
They are all defined in the bottom part of `network/synethesia_model.py`.
There is currently no easy way to configure them, but you can experiment with them by setting their
lambdas (i.e. constant factor to scale their contribution to the total loss) to something other than 0
in the `_build_loss` function.

The following losses are there:

1. Sound reconstruction loss (`_add_sound_reconstruction_loss`):
   Mean squared error loss for the input sound feature and the sound reproduced from the generated image.

2. Image reconstruction loss (`_add_image_reconstruction_loss`):
   Mean squared error loss for the produced image and the input image
   (at the moment only random images are used at input).

3. Noise loss (`_add_noise_loss`):
   Custom loss that penalizes difference in adjacent pixels. Using this loss during training
   produces sharper edges and more distinct colors.
   Personally, I think images look better without it.

4. Colorfulness loss (`_add_colorfulness_loss`):
   Custom loss that computes a histogram over a batch and penalizes the maximum number of entries in a bin.
   Thought to enforce use multiple colors, it seems to work OK for small number of bins (like 3 or 4)

5. Color variance loss:
   I removed this loss because it did not work.
   It penalized a low variance in each channel of RGB.

### Feature extraction methods

There are many options to extract sound features. There are 2 implemented and a third
is planned. Change which one is used in the constructor of `Synethesia`.

1. Currently, sound features based on logarithmic filterbank energies are used.

2. There is also a feature extractor based on the fast Fourier transform directly available.

3. One feature extraction method that I'd like to try is directly using the wav of a small sample.

## ToDo and Ideas

(You're welcome to create a pull request!)

### Open

- [ ] Allow more file formats than mp3
- [ ] Implement a discriminatory net that assesses aesthetic appeal
- [ ] Implement concurrent audio play and inference (streaming mode with audio)
- [ ] Inference only goes through one song at the moment, not multiple
- [ ] Unroll in time to compare current with previous frame
- [ ] Predict the wav directly, this allows to listen to images
- [ ] Enforce more colors
- [ ] Decay crop size during training
- [ ] Speed up training
- [ ] Implement all ToDo's in the code
- [ ] Write Docstrings
- [ ] Write unit tests
- [ ] The youtube video extractor should assign path-friendly folder and file names
- [ ] port shell script for youtube download to python
- [ ] CLI flag for **everything**

### Fixed/Implemented
- [x] Investigate large file amount: Wrong window length in fbank feature
- [x] `Infer` mode should delete the images and create the music video itself
- [x] Hook a microphone to live inference
- [x] Enforce base image similarity
