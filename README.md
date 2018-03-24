# WIP: SyNEThesia

SyNEThesia is a deep-learning-based music and sound visualizer, and a play of words on [Synesthesia](https://en.wikipedia.org/wiki/Synesthesia), a neurological condition where one perceives a stimulus in multiple ways (for example seeing sound).

The current version is already looking nice, but there are many ideas still in progress.



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

2. `run_synethesia.py`: This is the main file to start. Read the bottom of this README to see a description of the CLI. You can also get this information by running `run_synethesia.py -h` and `run_synethesia {mode} -h`.


## General Information

Training takes a while before the images start to look nice and the sound is clearly reproduced. I let it run overnight for about 7-8 hours on a GTX 1070.
Of course, untrained versions might also look cool, but you won't be able to "see the sound".

If you use the `infer` mode, be aware that **many** images will be saved to your disk (think 24 * song duration in seconds).
They will be deleted automatically, but if you have little disc space it may become a problem.

The `framework` submodule will probably be moved to a independent repository at some point in the future. SyNEThesia uses it to implement its functionality, but it itself is an abstract set of classes to simply tensorflow development.


## Architecture

Comming soon

### Losses

Comming soon

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
- [ ] Hook a microphone to inference
- [ ] Enforce base image similarity
- [ ] `Infer` mode should delete the images and create the music video itself
- [ ] Write Docstrings
- [ ] Write unit tests
- [ ] The youtube video extractor should assign path-friendly folder and file names
- [ ] port shell scripts to python

### Fixed/Implemented
- [x] Investigate large file amount: Wrong window length in fbank feature


## CLI Description

First, you need to supply a mode:

        positional arguments:
        {train,infer,stream,info}
                            Choose a mode of operation.
        train               Train the model.
        infer               Infer songs or other sounds and store the frames in a
                            folder.
        stream              Infer songs or other sounds and visualize the frames
                            live. Opens a window in full screen mode. Close it by
                            pressing 'q', or minimize it with 'Esc'.
        info                Display additional information, such as available
                            models.

        optional arguments:
        -h, --help            show this help message and exit

  Each mode has additional parameters:

  - train

          positional arguments:
          model_name            Name of the model to be trained or used for inference.
                                If it exists in the local checkpoints folder, the
                                model will be loaded, otherwise it will be newly
                                created.
          data                  Either a file containing paths to .mp3's, or a folder
                                containing .mp3's, or a single .mp3

          optional arguments:
          -h, --help            show this help message and exit
          -b BATCH_SIZE, --batch-size BATCH_SIZE
                                Batch size. Default is 1. Ignored for streaming
                                inference.
          -r ROWS, --n-rows ROWS
                                Image rows (height). Should be a power of two. An
                                error will be thrown if the loaded model was not
                                trained on the same size. Defaults to 256.
          -c COLS, --n-cols COLS
                                Image columns (width). Should be a power of two. An
                                error will be thrown if the loaded model was not
                                trained on the same size. Defaults to 128.
          -l LEARNING_RATE, --learning-rate LEARNING_RATE
                                Learning rate for training. Will be exponentially
                                decayed over time. Defaults to 0.0001.

  - infer

            positional arguments:
            model_name            Name of the model to be trained or used for inference.
                                  If it exists in the local checkpoints folder, the
                                  model will be loaded, otherwise it will be newly
                                  created.
            data                  Either a file containing paths to .mp3's, or a folder
                                  containing .mp3's, or a single .mp3
            target_dir            Target directory for storing the resulting frames.
                                  Warning: There may be many frames stored here intermediately,
                                           so make sure you have enough disc space.

            optional arguments:
            -h, --help            show this help message and exit
            -b BATCH_SIZE, --batch-size BATCH_SIZE
                                  Batch size. Default is 1. Ignored for streaming
                                  inference.
            -r ROWS, --n-rows ROWS
                                  Image rows (height). Should be a power of two. An
                                  error will be thrown if the loaded model was not
                                  trained on the same size. Defaults to 256.
            -c COLS, --n-cols COLS
                                  Image columns (width). Should be a power of two. An
                                  error will be thrown if the loaded model was not
                                  trained on the same size. Defaults to 128.

  - stream         

              positional arguments:
              model_name            Name of the model to be trained or used for inference.
                                    If it exists in the local checkpoints folder, the
                                    model will be loaded, otherwise it will be newly
                                    created.
              data                  Either a file containing paths to .mp3's, or a folder
                                    containing .mp3's, or a single .mp3

              optional arguments:
              -h, --help            show this help message and exit
              -b BATCH_SIZE, --batch-size BATCH_SIZE
                                    Batch size. Default is 1. Ignored for streaming
                                    inference.
              -r ROWS, --n-rows ROWS
                                    Image rows (height). Should be a power of two. An
                                    error will be thrown if the loaded model was not
                                    trained on the same size. Defaults to 256.
              -c COLS, --n-cols COLS
                                    Image columns (width). Should be a power of two. An
                                    error will be thrown if the loaded model was not
                                    trained on the same size. Defaults to 128.
