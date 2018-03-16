from pathlib import Path
import numpy as np
import pydub
import scipy.io.wavfile
import math


def next_power_of_two(number):
    return 2 ** math.ceil(math.log(number, 2))


class SampledSong(object):

    def __init__(self, song_path, feature_extraction_method=np.fft.fft):
        song_path = Path(song_path)

        self.feature_extraction_method = feature_extraction_method
        self.name = song_path.stem
        with SampledSong._convert_to_wav(song_path) as wavfile_name:
            self.rate, multi_channel_audio = scipy.io.wavfile.read(wavfile_name)
            self.audio = np.mean(multi_channel_audio, axis=-1)

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
        return self.audio.shape[0] / self.rate

    def __iter__(self):
        yield from self.extract_features()

    def split_into_chunks(self, chunks_per_second=24):
        window_length_ms = 1/chunks_per_second * 1000
        intervals = np.arange(window_length_ms, self.audio.shape[0], window_length_ms, dtype=np.int32)
        chunks = np.array_split(self.audio, intervals, axis=0)
        pad_to = next_power_of_two(np.max([chunk.shape[0] for chunk in chunks]))
        padded_chunks = np.stack(np.concatenate([chunk, np.zeros((pad_to - chunk.shape[0],))]) for chunk in chunks)
        return padded_chunks

    def extract_features(self):
        chunks = self.split_into_chunks()
        features = self.feature_extraction_method(chunks)
        return features


if __name__ == "__main__":
    song = SampledSong("/home/veith/Projects/PartyGAN/data/Bearded Skull - 420 [Hip Hop Instrumental]/audio/soundtrack.mp3")
    song.extract_features()
