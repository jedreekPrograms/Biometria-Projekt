"""Ekstrakcja cech z pojedynczego nagrania."""

import io
from functools import lru_cache

import librosa
import numpy as np
import onnxruntime as ort
import parselmouth
import soundfile as sf
from scipy import fft, signal


SAMPLE_RATE = 16_000
MIN_SECONDS = 2.0
VAD_FRAME = 512
VAD_CONTEXT = 64
FRAME_LENGTH = 400
HOP_LENGTH = 160
N_FFT = 512
N_MFCC = 13
TIME = ["SpeechAV", "SpeechSD", "SpeechProp", "MiRate", "MaRate"]
PROSODY = ["F0_mean", "F0_std", "F0_range_p90", "RMS_mean", "RMS_std"]
ACOUSTIC = [
    f"MFCC_{i}_{stat}" for i in range(N_MFCC) for stat in ("mean", "std")
] + ["centroid_mean", "centroid_std", "flatness_mean", "flatness_std"]
VARIANTS = {
    "A": TIME,
    "B": PROSODY,
    "C": ACOUSTIC,
    "D": TIME + PROSODY,
    "E": TIME + PROSODY + ACOUSTIC,
}
QUALITY_COLUMNS = ["duration_s", "n_segments", "n_micro", "n_macro", "n_voiced_frames"]


class AudioRejected(ValueError):
    def __init__(self, reason, duration):
        super().__init__(reason)
        self.reason = reason
        self.duration = duration


def load_audio(source):
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    samples, rate = sf.read(source, dtype="float32", always_2d=True)
    duration = len(samples) / rate
    if duration < MIN_SECONDS:
        raise AudioRejected("too_short", duration)
    audio = samples.mean(axis=1)
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
        raise AudioRejected("invalid_or_silent", duration)
    if rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
    return audio, duration


class SileroVAD:
    def __init__(self, model_path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
    def segments(self, audio):
        residual = (-len(audio)) % VAD_FRAME
        padded = np.pad(audio, (VAD_CONTEXT, residual))
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, VAD_CONTEXT + VAD_FRAME
        )[::VAD_FRAME]
        state = np.zeros((1, 1, 128), dtype=np.float32)
        probabilities, _, _ = self.session.run(
            None, {"input": windows, "h": state, "c": state}
        )
        return speech_segments(probabilities, len(audio))


def speech_segments(probabilities, length):
    # Histereza Silero. Bez dodawania marginesu do segmentu, aby zachować długość pauzy.
    detected = []
    start = silence_start = None
    min_silence = int(SAMPLE_RATE * 0.1)
    min_speech = int(SAMPLE_RATE * 0.25)
    for index, probability in enumerate(probabilities):
        position = index * VAD_FRAME
        if probability >= 0.5:
            silence_start = None
            if start is None:
                start = position
        elif probability < 0.35 and start is not None:
            if silence_start is None:
                silence_start = position
            if position - silence_start >= min_silence:
                if silence_start - start >= min_speech:
                    detected.append((start, silence_start))
                start = silence_start = None
    if start is not None and length - start >= min_speech:
        detected.append((start, length))
    return detected


def temporal_features(segments, sample_count):
    if not segments:
        return {
            **dict.fromkeys(TIME, np.nan), "n_segments": 0,
            "n_micro": 0, "n_macro": 0,
        }
    starts, ends = np.asarray(segments).T
    speech_samples = ends - starts
    speech_seconds = speech_samples / SAMPLE_RATE
    gaps = starts[1:] - ends[:-1]
    n_micro = np.count_nonzero((gaps >= 1_600) & (gaps < 8_000))
    n_macro = np.count_nonzero(gaps >= 8_000)
    trimmed_minutes = (ends[-1] - starts[0]) / SAMPLE_RATE / 60
    return {
        "SpeechAV": speech_seconds.mean(),
        "SpeechSD": speech_seconds.std(),
        "SpeechProp": speech_samples.sum() / sample_count,
        "MiRate": n_micro / trimmed_minutes,
        "MaRate": n_macro / trimmed_minutes,
        "n_segments": len(segments),
        "n_micro": n_micro,
        "n_macro": n_macro,
    }


def prosodic_features(audio, frames):
    pitch = parselmouth.Sound(audio, sampling_frequency=SAMPLE_RATE).to_pitch_ac(
        time_step=0.01, pitch_floor=75, pitch_ceiling=500
    )
    frequency = pitch.selected_array["frequency"]
    voiced = frequency[frequency > 0]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    f0 = dict.fromkeys(PROSODY[:3], np.nan)
    if voiced.size:
        low, high = np.percentile(voiced, [5, 95])
        f0.update(F0_mean=voiced.mean(), F0_std=voiced.std(), F0_range_p90=high - low)
    return {
        **f0, "RMS_mean": rms.mean(), "RMS_std": rms.std(),
        "n_voiced_frames": voiced.size,
    }


@lru_cache(maxsize=1)
def spectral_filters():
    window = signal.get_window("hann", FRAME_LENGTH).astype(np.float32)
    mel = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=40)
    frequencies = fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE).astype(np.float32)
    return window, mel, frequencies


def acoustic_features(frames):
    window, mel, frequencies = spectral_filters()
    spectrum = np.abs(fft.rfft(frames * window, n=N_FFT, axis=1))
    power = spectrum ** 2
    mel_db = 10 * np.log10(np.maximum(power @ mel.T, 1e-10))
    mel_db = np.maximum(mel_db, mel_db.max() - 80)
    mfcc = fft.dct(mel_db, type=2, norm="ortho", axis=1)[:, :N_MFCC]
    centroid = (spectrum @ frequencies) / np.maximum(spectrum.sum(axis=1), 1e-10)
    bounded_power = np.maximum(power, 1e-10)
    flatness = np.exp(np.log(bounded_power).mean(axis=1)) / bounded_power.mean(axis=1)
    summary = np.column_stack((mfcc, centroid, flatness))
    pairs = np.column_stack((summary.mean(axis=0), summary.std(axis=0)))
    return dict(zip(ACOUSTIC, pairs.ravel(), strict=True))


def extract_features(source, vad):
    audio, duration = load_audio(source)
    frames = np.lib.stride_tricks.sliding_window_view(audio, FRAME_LENGTH)[::HOP_LENGTH]
    return {
        "duration_s": duration,
        **temporal_features(vad.segments(audio), len(audio)),
        **prosodic_features(audio, frames),
        **acoustic_features(frames),
    }
