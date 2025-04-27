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
    print("Starting conversion...")
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
    return csv_path


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


def convert_to_decibels(csv_path: str) -> str:
    """
    Read *csv_path*, convert its sample column(s) to dBFS, and write
    <original_stem>_db.csv alongside the source file.

    Returns
    -------
    out_path : str
        Path of the file that was written.
    """
    df = pd.read_csv(csv_path)

    # --- choose / down-mix channel(s) --------------------------------------
    if {"L", "R"}.issubset(df.columns):
        samples = 0.5 * (df["L"] + df["R"])  # simple L+R → mono
    elif "sample" in df.columns:
        samples = df["sample"]
    else:
        raise ValueError("CSV lacks expected 'sample' or 'L,R' columns.")

    # --- integer → float in [-1, 1] ---------------------------------------
    s_float = samples.astype(np.float32) / 32767.0

    # --- instantaneous dBFS ----------------------------------------------
    eps = 1e-10  # avoid log(0)
    df["dBFS"] = 20 * np.log10(np.abs(s_float) + eps)

    # --- write new CSV ----------------------------------------------------
    stem, _ = os.path.splitext(csv_path)
    out_path = f"{stem}_db.csv"
    df[["index", "dBFS"]].to_csv(out_path, index=False)
    print(f"[✓] wrote dBFS data → {out_path}")
    return out_path


if __name__ == "__main__":
    mp3_path = "./data/fan/fan.mp3"

    csv_path = audio_to_csv(
        mp3_path,
        sample_rate=1,
        mono=True,
        loops=5,  # <-- play the file 5 times in a row
        rename_double_ext=True,
    )

    plot_volume_envelope(
        csv_path=csv_path,  # CSV produced by audio_to_csv
        sample_rate=1,  # Hz – must match the rate you used (or want to view)
        window_ms=1000,  # analysis window width in milliseconds
        output_path=csv_path.replace(".csv", ".png"),  # destination image (png)
    )

    # convert csv values to decibel
    db_src = convert_to_decibels(csv_path)

    # normalize decibel values between -1 and 1 for the ol' A I
    df = pd.read_csv(db_src)
    if "dBFS" not in df.columns:
        raise ValueError(f"'dBFS' column not found in {db_src}")

    mean_val = df["dBFS"].mean()

    df["dBFS_norm"] = (df["dBFS"] - mean_val) / mean_val  # (x-mean)/mean

    out_norm = db_src.replace("_db.csv", "_db_norm.csv")
    df[["index", "dBFS_norm"]].to_csv(out_norm, index=False)
    print(f"[✓] wrote normalised file → {out_norm}")
