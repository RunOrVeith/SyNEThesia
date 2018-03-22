from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from sampled_song import SampledSong
from data_loaders import BatchCreator


class StaticSongLoader(object):

    # TODO abstract into reusable class
    # TODO convert to use tf.data

    def __init__(self, song_files, batch_size, feature_extractor,
                 to_infinity=False, load_n_songs_at_once=5, allow_shuffle=True):
        # Assume all song files exist for now
        self.songs = [SampledSong(song_file, feature_extraction_method=feature_extractor) for song_file in song_files]
        assert len({song.feature_dim for song in self.songs}) == 1, "Features must have the same size"
        load_n_songs_at_once = min(len(song_files), load_n_songs_at_once)
        self.song_batcher = BatchCreator(iterable=self.songs, batch_size=load_n_songs_at_once,
                                         allow_shuffle=allow_shuffle)
        self.loaded_snippets = []
        self.batch_size = batch_size
        self.to_infinity = to_infinity

    def __len__(self):
        return len(self.songs)

    @property
    def feature_dim(self):
        return self.songs[0].feature_dim

    def _load_songs_async(self, songs_to_load):

        def _load_song(song):
            features = [feat for feat in song]
            return features

        with ThreadPoolExecutor() as executor:

            future_to_song = {executor.submit(_load_song, song): song for song in songs_to_load}
            for future in as_completed(future_to_song):
                try:
                    song_snippets = future.result()
                except Exception as exc:
                    print(f"{song.name} generated an exception: {exc}")
                else:
                    self.loaded_snippets.extend(song_snippets)

    def _maybe_load_songs(self, refill_level=100):
        if len(self.loaded_snippets) < refill_level:
            next_songs = next(self.song_batcher)
            self._load_songs_async(songs_to_load=next_songs)

    def __iter__(self):
        while True:
            try:
                self._maybe_load_songs(refill_level=1000)
            except StopIteration:
                #print("Epoch done.")
                if self.to_infinity:
                    self.song_batcher.reset()
                    continue
                else:
                    raise StopIteration()
            else:
                feature_batcher = BatchCreator(iterable=self.loaded_snippets, batch_size=self.batch_size,
                                               allow_shuffle=True)
                yield from feature_batcher



if __name__ == "__main__":
    from feature_creators import logfbank_features
    song_loader = StaticSongLoader(song_files=["/home/veith/Projects/PartyGAN/data/Bearded Skull - 420 [Hip Hop Instrumental]/audio/soundtrack.mp3"],
                                   feature_extractor=logfbank_features,
                                   batch_size=1, load_n_songs_at_once=1)
    for audio_chunk in song_loader:
        print(audio_chunk)
