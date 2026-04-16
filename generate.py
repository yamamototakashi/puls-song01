import numpy as np
from moviepy import AudioClip, VideoClip

DUR = 5.0
SR = 44100
FPS = 30
W, H = 1280, 720
PULSE_HZ = 3.0

rng = np.random.default_rng(42)

# --- Audio: noise + clicks + glitch pulses at ~3 Hz ---
n_samples = int(DUR * SR)
t = np.arange(n_samples) / SR
noise = rng.standard_normal(n_samples) * 0.15
pulse_env = (np.sin(2 * np.pi * PULSE_HZ * t) > 0.9).astype(np.float32)
click_times = rng.uniform(0, DUR, size=20)
clicks = np.zeros(n_samples, dtype=np.float32)
for ct in click_times:
    i = int(ct * SR)
    L = int(rng.integers(30, 300))
    if i + L < n_samples:
        clicks[i:i + L] += (rng.standard_normal(L) * 0.8 * np.exp(-np.linspace(0, 5, L))).astype(np.float32)
glitch = rng.standard_normal(n_samples) * pulse_env * 0.9
audio = (noise + clicks + glitch).astype(np.float32)
audio = np.clip(audio, -1.0, 1.0)

def make_audio_frame(tt):
    tt = np.atleast_1d(np.asarray(tt, dtype=np.float64))
    idx = np.clip((tt * SR).astype(int), 0, n_samples - 1)
    return audio[idx][:, None]  # mono as (N,1)

audio_clip = AudioClip(make_audio_frame, duration=DUR, fps=SR)

# --- Video: black bg with flickering white rectangles/lines driven by amp + noise ---
WIN = int(SR / FPS)

def amp_at(tt):
    i = int(tt * SR)
    s = audio[max(0, i - WIN // 2): i + WIN // 2 + 1]
    return float(np.sqrt(np.mean(s * s))) if len(s) else 0.0

def make_frame(tt):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    a = amp_at(tt)
    brightness = int(np.clip(80 + a * 900, 0, 255))
    n_shapes = int(np.clip(a * 60, 1, 30)) + int(rng.integers(0, 5))
    for _ in range(n_shapes):
        if rng.random() < 0.5:
            x = int(rng.integers(0, W))
            y = int(rng.integers(0, H))
            w = int(rng.integers(10, 200))
            h = int(rng.integers(4, 80))
            frame[y:min(y + h, H), x:min(x + w, W)] = brightness
        else:
            y = int(rng.integers(0, H))
            th = int(rng.integers(1, 6))
            frame[y:min(y + th, H), :] = brightness
    return frame

video_clip = VideoClip(make_frame, duration=DUR).with_fps(FPS).with_audio(audio_clip)
video_clip.write_videofile("output.mp4", codec="libx264", audio_codec="aac", fps=FPS)
