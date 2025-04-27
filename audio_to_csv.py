import matplotlib.pyplot as plt
import pandas as pd
import os
import shlex
import json
import subprocess
import tempfile
import numpy as np
import soundfile as sf


def _probe_duration(path: str) -> float:
    """Return media duration in **seconds** using ffprobe."""
    probe_cmd = f"ffprobe -v quiet -print_format json -show_format {shlex.quote(path)}"
    out = subprocess.check_output(probe_cmd, shell=True, text=True)
    return float(json.loads(out)["format"]["duration"])


def audio_to_csv(
    video_path: str,
    csv_path: str | None = None,
    *,
    sample_rate: int | None = 44_100,
    mono: bool = True,
    extend_hours: float | None = None,
    rename_double_ext: bool = False,
    chunk: int = 60_000,
):
    """
    Extract audio samples from *video_path* into *csv_path*.

    Parameters
    ----------
    video_path : str
        Input media file.
    csv_path : str, optional
        Output CSV (defaults to <video stem>.csv).
    sample_rate : int | None, optional
        Resample rate.  None = keep original.
    mono : bool, optional
        Down-mix to mono if True (column name 'sample'); otherwise write 'L','R'.
    extend_hours : float | None, optional
        If given, loop the audio so the total duration is ≥ this many hours.
        Looping is done with ffmpeg’s sample-accurate *-stream_loop* flag,
        so there is no gap between repeats.
    rename_double_ext : bool, optional
        If the filename ends with '.mov.mp4', rename in-place to '.mp4'.
    chunk : int, optional
        Frames per streaming block (keeps memory usage ~1 MB).
    """
    # --- housekeeping -------------------------------------------------------
    stem, ext = os.path.splitext(video_path)
    if rename_double_ext and ext.lower() == ".mp4" and stem.lower().endswith(".mov"):
        fixed = stem + ".mp4"
        os.replace(video_path, fixed)
        video_path = fixed
        print(f"[info] renamed to {video_path}")

    if csv_path is None:
        csv_path = stem + ".csv"

    # --- figure out looping -------------------------------------------------
    loop_opt = ""
    limit_opt = ""
    if extend_hours is not None and extend_hours > 0:
        target_sec = extend_hours * 3600
        src_sec = _probe_duration(video_path)
        if target_sec > src_sec:
            loops = math.ceil(target_sec / src_sec) - 1
            loop_opt = f"-stream_loop {loops}"
            # Trim in case the last repeat overshoots a little
            limit_opt = f"-t {target_sec:.3f}"

    # --- transcode to a temp WAV -------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    ac_opt = 1 if mono else 2
    ar_opt = f"-ar {sample_rate}" if sample_rate else ""
    cmd = (
        f"ffmpeg -y -loglevel error {loop_opt} -i {shlex.quote(video_path)} "
        f"{limit_opt} -ac {ac_opt} {ar_opt} -f wav {shlex.quote(wav_path)}"
    )
    subprocess.run(cmd, shell=True, check=True)

    # --- stream WAV -> CSV --------------------------------------------------
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


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    audio_to_csv(
        "sample.mp3",
        csv_path="result.csv",
        sample_rate=48_000,
        mono=True,
        extend_hours=2.0,  # 2-hour seamless loop
        rename_double_ext=True,
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


audio_to_csv(
    "sample.mp3",
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
