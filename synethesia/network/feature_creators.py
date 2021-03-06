import numpy as np
import python_speech_features as psf


# TODO raw wav feature

def _next_power_of_two(number):
    return int(2 ** np.ceil(np.log2(np.maximum(1., number))))


def _split_into_chunks(signal, chunks_per_second=24):
    # TODO currently broken
    raise NotImplemented("Splitting to chunks is currently broken.")
    window_length_ms = 1/chunks_per_second * 1000
    intervals = np.arange(window_length_ms, signal.shape[0], window_length_ms, dtype=np.int32)
    chunks = np.array_split(signal, intervals, axis=0)
    pad_to = _next_power_of_two(np.max([chunk.shape[0] for chunk in chunks]))
    padded_chunks = np.stack(np.concatenate([chunk, np.zeros((pad_to - chunk.shape[0],))]) for chunk in chunks)
    return padded_chunks


def fft_features(signal, fps, **kwargs):
    wav_chunks = _split_into_chunks(signal=signal, chunks_per_second=fps)
    fft = np.fft.fft(wav_chunks)
    sound_dB = 10 * np.log10(np.maximum(1., np.square(np.real(fft)) + np.square(np.imag(fft))))
    np.random.shuffle(sound_dB)

    return sound_dB


def logfbank_features(signal, samplerate=44100, fps=24, num_filt=40, num_cepstra=40, nfft=8192, **kwargs):
    winstep = 2 / fps
    winlen = winstep * 2
    feat, energy = psf.fbank(signal=signal, samplerate=samplerate,
                             winlen=winlen, winstep=winstep, nfilt=num_filt,
                             nfft=nfft)
    feat = np.log(feat)
    feat = psf.dct(feat, type=2, axis=1, norm='ortho')[:, :num_cepstra]
    feat = psf.lifter(feat, L=22)
    feat = np.asarray(feat)

    energy = np.log(energy)
    energy = energy.reshape([energy.shape[0],1])

    if feat.shape[0] > 1:
        std = 0.5 * np.std(feat, axis=0)
        mat = (feat - np.mean(feat, axis=0)) / std
    else:
        mat = feat

    mat = np.concatenate((mat, energy), axis=1)

    duration = signal.shape[0] / samplerate
    expected_frames = fps * duration
    assert mat.shape[0] - expected_frames <= 1, "Producted feature number does not match framerate"
    return mat
