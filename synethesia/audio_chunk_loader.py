from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

import numpy as np

from sampled_song import SampledSong


class BatchCreator(object):

    def __init__(self, iterable, batch_size):
        self._iterator_source = iterable
        self.iterator = None
        self.batch_size = batch_size

        self.reset()

    def __iter__(self):
        return self

    def reset_and_shuffle(self):
        random.shuffle(self._iterator_source)
        self.reset()

    def reset(self):
        self.iterator = iter(self._iterator_source)

    def __iter__(self):
        return self

    def __next__(self):
        instances = []
        try:
            for _ in range(self.batch_size):
                instances.append(next(self.iterator))
        except StopIteration:
            # Number of elements in iterator is not a multuple of batch_size,
            # just batch the remaining instances
            if len(instances) == 0:
                raise StopIteration()

        return np.stack(instances, axis=0)


class StaticSongLoader(object):

    # TODO abstract into reusable class

    def __init__(self, song_files, batch_size, to_infinity=False, load_n_songs_at_once=5):
        # Assume all song files exist for now
        self.songs = [SampledSong(song_file) for song_file in song_files]
        load_n_songs_at_once = min(len(song_files), load_n_songs_at_once)
        self.song_batcher = BatchCreator(iterable=self.songs, batch_size=load_n_songs_at_once)
        self.loaded_snippets = []
        self.batch_size = batch_size
        self.to_infinity = to_infinity

    def _load_songs_async(self, songs_to_load):

        def _load_song(song):
            return [snippet for snippet in song]

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
                if self.to_infinity:
                    self.song_batcher.reset_and_shuffle()
                    continue
                else:
                    raise StopIteration()
            else:
                feature_batcher = BatchCreator(iterable=self.loaded_snippets, batch_size=self.batch_size)
                yield from feature_batcher




if __name__ == "__main__":
    song_loader = StaticSongLoader(song_files=["/home/veith/Projects/PartyGAN/data/Bearded Skull - 420 [Hip Hop Instrumental]/audio/soundtrack.mp3"],
                                   batch_size=1, load_n_songs_at_once=1)
    for audio_chunk in song_loader:
        print(audio_chunk)
