# SNR / max-abs between two WAV files (float32 read, peak-normalized like CLI).
import sys
import numpy as np

def read_wav(path):
    import wave
    with wave.open(path, 'rb') as w:
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    else:
        raise ValueError(f"sampwidth {sw}")
    x = x.reshape(-1, ch)[:, 0]
    return x, sr

def snr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    err = a - b
    return 10 * np.log10(np.dot(a, a) / max(np.dot(err, err), 1e-30)), float(np.max(np.abs(err))), n

if __name__ == "__main__":
    ref, sr1 = read_wav(sys.argv[1])
    tst, sr2 = read_wav(sys.argv[2])
    assert sr1 == sr2
    # peak-normalize both like the CLI does on write
    ref = ref / max(np.max(np.abs(ref)), 1e-9)
    tst = tst / max(np.max(np.abs(tst)), 1e-9)
    s, m, n = snr(ref, tst)
    print(f"samples {n}  SNR {s:.2f} dB  maxabs {m:.2e}")
