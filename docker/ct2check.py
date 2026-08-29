from faster_whisper import WhisperModel

for compute in ("int8", "float32"):
    try:
        WhisperModel("/app/model/faster-whisper-small", device="cpu", compute_type=compute)
    except ValueError as exc:
        print(f"compute_type={compute} unsupported: {exc}")
        continue
    print(f"faster-whisper model loads standalone OK (CPU, {compute})")
    break
else:
    raise SystemExit("faster-whisper model failed to load on CPU with any compute type")