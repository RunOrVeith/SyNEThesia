from pathlib import Path

import numpy as np
import pydub
import scipy.io.wavfile


class SampledSong(object):

    def __init__(self, song_path, feature_extraction_method, fps=24):
        song_path = Path(song_path)

        self.feature_extraction_method = feature_extraction_method
        self.name = song_path.stem

        self.fps = fps

        with SampledSong._convert_to_wav(song_path) as wavfile_name:
            self.samplerate, self.signal = scipy.io.wavfile.read(wavfile_name)

        self._features = None

    @property
    def features(self):
        if self._features is None:
            self._features = self.extract_features()

        return self._features

    @property
    def feature_dim(self):
        return self.features[0].shape[-1]

    @staticmethod
    def _validate_input(song_path):
        if not song_path.exists():
            raise ValueError(f"Cannot find file {song_path}")

        audio_format = song_path.suffix
        if not audio_format.lower() == ".mp3":
            raise NotImplementedError(f"Unsupported audio format {audio_format}")

    @staticmethod
    def _convert_to_wav(song_path):

        class IntermediateWavFile(object):

            def __init__(self):
                self.intermediate_wav_fname = song_path.parents[0] / "waveform.wav"

            def __enter__(self):
                # TODO support other formats (should be easy with pydub)
                mp3 = pydub.AudioSegment.from_mp3(str(song_path))
                mp3.set_channels(1)  # Make sure everything is mono
                mp3.export(str(self.intermediate_wav_fname), format="wav")
                assert self.intermediate_wav_fname.exists(), "Could not create Intermediate wavfile"
                return str(self.intermediate_wav_fname)

            def __exit__(self, *args, **kwargs):
                self.intermediate_wav_fname.unlink()
                assert not self.intermediate_wav_fname.exists(), "Intermediate wavfile was not deleted"

        SampledSong._validate_input(song_path=song_path)
        return IntermediateWavFile()

    @property
    def duration(self):
        return self.signal.shape[0] / self.samplerate

    def __iter__(self):
        yield from self.features

    def extract_features(self):
        features = self.feature_extraction_method(signal=self.signal, samplerate=self.samplerate, fps=self.fps)
        return features
