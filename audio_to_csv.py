import os
import shlex
import subprocess
import tempfile
import numpy as np
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt


def audio_to_csv(
    video_path: str,
    csv_path: str | None = None,
    *,
    sample_rate: int | None = 44_100,
    mono: bool = True,
    loops: int = 1,  # ← number of perfect repeats (5 ⇒ play 6 times total)
    rename_double_ext: bool = False,
    chunk: int = 60_000,
):
    """
    Extract audio samples from *video_path* into *csv_path*.

    Parameters
    ----------
    video_path : str
        Input media file (any format ffmpeg understands).
    csv_path : str, optional
        Output CSV (defaults to <video stem>.csv).
    sample_rate : int | None, optional
        Resample rate; None = keep original.
    mono : bool, optional
        Down-mix to mono if True (column 'sample'); else write 'L','R'.
    loops : int, optional
        Duplicate the CSV *loops* times after the first pass. 1 = no looping.
    rename_double_ext : bool, optional
        If the file ends in '.mov.mp4', rename it to '.mp4' before processing.
    chunk : int, optional
        Frames per read; keeps memory usage small.
    """
    # --------------------------------------------------- housekeeping -------
    stem, ext = os.path.splitext(video_path)
    if rename_double_ext and ext.lower() == ".mp4" and stem.lower().endswith(".mov"):
        fixed = stem + ".mp4"
        os.replace(video_path, fixed)
        video_path = fixed
        print(f"[info] renamed to {video_path}")

    if csv_path is None:
        csv_path = stem + ".csv"

    # ------------------------------------------- transcode → temp WAV -------
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    ac_opt = 1 if mono else 2
    ar_opt = f"-ar {sample_rate}" if sample_rate else ""
    cmd = (
        f"ffmpeg -y -loglevel error -i {shlex.quote(video_path)} "
        f"-ac {ac_opt} {ar_opt} -f wav {shlex.quote(wav_path)}"
    )
    subprocess.run(cmd, shell=True, check=True)

    # --------------------------------------------- stream WAV → CSV ---------
    with sf.SoundFile(wav_path) as f, open(csv_path, "w") as out:
        header = "index,sample\n" if mono else "index,L,R\n"
        out.write(header)

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

    # --------------------------------------------- duplicate rows ----------

    total_plays = max(1, loops)  # loops==1  → just the original
    if total_plays > 1:
        base = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.int32)
        with open(csv_path, "a") as out:
            for i in range(1, total_plays):
                dup = base.copy()
                dup[:, 0] += i * base.shape[0]
                np.savetxt(out, dup, fmt="%d", delimiter=",")
        print(
            f"[✓] CSV now contains {total_plays} contiguous plays "
            f"({base.shape[0] * total_plays:,} samples total)"
        )


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


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    audio_to_csv(
        "sample.mp4",
        csv_path="result.csv",
        sample_rate=48_000,
        mono=True,
        loops=5,  # <-- play the file 5× in a row
        rename_double_ext=True,
    )
    plot_volume_envelope(
        csv_path="result.csv",  # CSV produced by audio_to_csv
        sample_rate=48_100,  # Hz – must match the rate you used (or want to view)
        window_ms=20,  # analysis window width in milliseconds
        output_path="graph.png",  # destination image
    )
