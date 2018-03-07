from pathlib import Path
import numpy as np
import pydub
import scipy.io.wavfile
import math

def next_power_of_two(number):
    return 2 ** math.ceil(math.log(number, 2))

class SampledSong(object):

    def __init__(self, song_path):
        song_path = Path(song_path)
        if not song_path.exists():
            raise ValueError(f"Unknown file location {song_path}")

        audio_format = song_path.suffix
        assert audio_format.lower() == ".mp3", f"Unsupported audio format {audio_format}"
        # TODO support other formats (should be easy with pydub)
        mp3 = pydub.AudioSegment.from_mp3(str(song_path))

        intermediate_wav_fname = song_path.parents[0] / "waveform.wav"
        mp3.set_channels(1) # Make sure everything is mono
        mp3.export(str(intermediate_wav_fname), format="wav")
        assert intermediate_wav_fname.exists()
        self.name = song_path.stem
        self.rate, self.audio = scipy.io.wavfile.read(intermediate_wav_fname)
        self.audio = np.mean(self.audio, axis=-1)
        intermediate_wav_fname.unlink()

    def split_into_chunks(self, window_length_ms=1/24 * 1000):

        time = self.audio.shape[0] / self.rate
        print(f"{self.name} is {time} seconds long.")
        intervals = np.arange(window_length_ms, self.audio.shape[0], window_length_ms, dtype=np.int32)
        chunks = np.array_split(self.audio, intervals, axis=0)
        pad_to = next_power_of_two(np.max([chunk.shape[0] for chunk in chunks]))
        padded_chunks = np.stack(np.concatenate([chunk, np.zeros((pad_to - chunk.shape[0],))]) for chunk in chunks)
        return padded_chunks

    def chunked_fft(self):
        fft = np.fft.fft(self.split_into_chunks())
        return fft

if __name__ == "__main__":
    song = SampledSong("/home/veith/Projects/PartyGAN/data/Bearded Skull - 420 [Hip Hop Instrumental]/audio/soundtrack.mp3")
    song.split_into_chunks()
