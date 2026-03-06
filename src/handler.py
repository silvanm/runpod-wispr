"""
RunPod serverless handler: transcribe MP3/MP4 from URL → CSV with speaker_id, utterance.
Uses WhisperX + pyannote diarization. For MP4, extracts audio with ffmpeg first.
"""
from __future__ import annotations

import csv
import gc
import io
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import runpod

# Lazy init of heavy models (set in init_models())
_whisper_model = None
_device = "cuda"
_compute_type = "float16"
_batch_size = 16
_hf_token = os.environ.get("HF_TOKEN", "")


def _is_video(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")


def _ensure_audio_file(downloaded_path: str) -> str:
    """If path is video, extract audio to 16kHz mono WAV; return path to use for WhisperX."""
    if not _is_video(downloaded_path):
        return downloaded_path
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                downloaded_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                wav_path,
            ],
            check=True,
            capture_output=True,
        )
        return wav_path
    except Exception:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        raise


def _download_to_temp(url: str) -> str:
    """Download URL to a temp file; return path. Caller must clean up."""
    suffix = Path(url.split("?")[0]).suffix or ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception:
        if os.path.exists(path):
            os.unlink(path)
        raise


def init_models():
    """Load Whisper + align + diarize models once at worker startup."""
    global _whisper_model
    if _whisper_model is not None:
        return
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    _whisper_model = whisperx.load_model(
        "large-v3",
        device,
        compute_type=compute_type,
    )
    # Align and diarize models are loaded per-job to avoid holding GPU memory when idle
    # We keep only the whisper model cached
    globals().update(_device=device, _compute_type=compute_type)


def _segments_to_csv(segments: list[dict]) -> str:
    """Aggregate segments by speaker into rows (speaker_id, utterance)."""
    if not segments:
        return "speaker_id,utterance\n"
    out = io.StringIO()
    w = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
    w.writerow(["speaker_id", "utterance"])
    current_speaker = None
    current_text: list[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker") or "SPEAKER_00"
        if speaker == current_speaker:
            current_text.append(text)
        else:
            if current_speaker is not None and current_text:
                w.writerow([current_speaker, " ".join(current_text)])
            current_speaker = speaker
            current_text = [text]
    if current_speaker is not None and current_text:
        w.writerow([current_speaker, " ".join(current_text)])
    return out.getvalue()


def handler(job: dict) -> dict:
    """RunPod job handler: input.audio_url → download → (ffmpeg if MP4) → transcribe → CSV."""
    import whisperx

    job_input = job.get("input") or {}
    audio_url = (job_input.get("audio_url") or "").strip()
    if not audio_url:
        return {"error": "Missing required input: audio_url"}
    language = job_input.get("language") or "en"
    model_size = job_input.get("model_size") or "base"

    downloaded_path = None
    audio_path = None  # path actually passed to WhisperX (may be temp WAV from ffmpeg)
    try:
        runpod.serverless.progress_update(job, "Downloading file")
        downloaded_path = _download_to_temp(audio_url)
        audio_path = _ensure_audio_file(downloaded_path)
        if audio_path != downloaded_path:
            # We created a temp WAV; clean up the original download
            try:
                os.unlink(downloaded_path)
            except OSError:
                pass
            downloaded_path = None

        runpod.serverless.progress_update(job, "Loading audio")
        audio = whisperx.load_audio(audio_path)

        init_models()
        runpod.serverless.progress_update(job, "Transcribing")
        result = _whisper_model.transcribe(audio, _batch_size)
        if not result or "segments" not in result:
            return {"error": "Transcription produced no segments"}

        # Align for word-level timestamps (free GPU memory from align after)
        runpod.serverless.progress_update(job, "Aligning")
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_a, metadata = whisperx.load_align_model(
            language_code=result.get("language", language),
            device=_device,
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            _device,
        )
        del model_a
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Diarize
        runpod.serverless.progress_update(job, "Diarizing")
        if not _hf_token:
            return {"error": "HF_TOKEN environment variable required for speaker diarization"}
        from whisperx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(
            token=_hf_token,
            device=_device,
        )
        diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=20)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        csv_str = _segments_to_csv(result.get("segments") or [])
        return {"csv": csv_str, "error": None}
    except Exception as e:
        return {"csv": None, "error": str(e)}
    finally:
        for p in (downloaded_path, audio_path):
            if p and p != downloaded_path and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
