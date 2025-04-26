from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os
import shlex
import subprocess
import tempfile
import numpy as np
import soundfile as sf


def video_audio_to_csv(
    video_path: str,
    csv_path: str = None,
    *,
    sample_rate: int = 44_100,
    mono: bool = True,
    rename: bool = False,
    chunk: int = 60_000,
):
    """
    Extract every audio sample from an MP4/MOV into a CSV.

    Parameters
    ----------
    video_path : str
        Input video file.
    csv_path : str, optional
        Destination CSV file (defaults to <video stem>.csv).
    sample_rate : int, optional
        Resample rate for output.  None = keep original rate.
    mono : bool, optional
        If True, down-mix to mono (1 column).  False → write 'L','R' columns.
    rename : bool, optional
        If True and the filename ends with double extensions (e.g. '.mov.mp4'),
        rename it in-place to '.mp4' before processing.
    chunk : int, optional
        Samples per streaming block (keeps memory < ~1 MB).

    Notes
    -----
    * Requires ffmpeg / ffprobe on PATH plus `pip install soundfile numpy`.
    * Produces a header row: index [,L] [,R]
    """
    # auto-rename/ensure correct format
    stem, ext = os.path.splitext(video_path)
    if rename and ext.lower() == ".mp4" and stem.lower().endswith(".mov"):
        fixed = stem + ".mp4"
        os.replace(video_path, fixed)
        video_path = fixed
        print(f"[info] renamed to {video_path}")

    if not csv_path:
        csv_path = os.path.splitext(video_path)[0] + ".csv"

    # create temporary .wav file (lossless)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    ac_opt = 1 if mono else 2
    ar_opt = f"-ar {sample_rate}" if sample_rate else ""
    cmd = (
        f"ffmpeg -y -loglevel error -i {shlex.quote(video_path)} "
        f"-ac {ac_opt} {ar_opt} -f wav {shlex.quote(wav_path)}"
    )
    subprocess.run(cmd, shell=True, check=True)

    # WAV -> CSV
    with sf.SoundFile(wav_path) as f, open(csv_path, "w") as out:
        if mono:
            out.write("index,sample\n")
        else:
            out.write("index,L,R\n")

        idx = 0
        while True:
            data = f.read(frames=chunk, dtype="int16")
            if len(data) == 0:
                break

            if mono:
                rows = np.column_stack((np.arange(idx, idx + len(data)), data))
            else:
                rows = np.column_stack(
                    (np.arange(idx, idx + len(data)), data[:, 0], data[:, 1])
                )

            np.savetxt(out, rows, fmt="%d", delimiter=",")
            idx += len(data)

    os.remove(wav_path)
    print(f"[✓] wrote {idx} samples → {csv_path}")


def plot_volume_envelope(
    csv_path: str,
    sample_rate: int = 44_100,
    window_ms: int = 50,
    output_path: str = "volume_envelope.png",
):
    """
    Plot the volume envelope of audio data from a CSV file.
    """
    df = pd.read_csv(csv_path)

    if {"L", "R"}.issubset(df.columns):  # stereo   → combine L+R
        mono = (df["L"].abs() + df["R"].abs()) / 2.0
        label = "Stereo RMS"
    elif "sample" in df.columns:  # mono
        mono = df["sample"].abs()
        label = "Mono |sample|"
    else:
        raise ValueError("CSV does not have the expected columns.")

    win = int(sample_rate * window_ms / 1000)  # samples per window
    pad = (-len(mono)) % win  # pad so length % win == 0
    mono_padded = np.pad(mono, (0, pad))

    env_rms = np.sqrt(np.mean(mono_padded.reshape(-1, win) ** 2, axis=1))

    t = (np.arange(len(env_rms)) * win + win / 2) / sample_rate

    plt.figure(figsize=(10, 4))
    plt.plot(t, env_rms)
    plt.title(f"Volume envelope (RMS, window = {window_ms} ms)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude (RMS)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[✓] saved plot to {output_path}")


video_audio_to_csv(
    "sample.mp4",
    csv_path="result.csv",
    sample_rate=48_000,
    mono=True,
    rename=True,
)


plot_volume_envelope(
    "result.csv",
    44_100,
    50,
    "graph.png",
)
