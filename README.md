# RunPod serverless meeting transcription

Transcribe MP3 or MP4 meeting recordings via a RunPod serverless endpoint. Input: a URL to an audio/video file (e.g. from Google Cloud Storage). Output: CSV with `speaker_id` and `utterance` (speaker diarization via WhisperX + pyannote).

## Features

- **Input:** MP3 or MP4 URL (payload limit 10–20 MB, so use a URL, not raw upload).
- **Output:** CSV with columns `speaker_id`, `utterance` (one row per speaker turn).
- **MP4:** Worker extracts audio with ffmpeg before transcription.
- **Local scripts:** Upload files to GCS and optionally call RunPod from your machine (Typer CLI).

## Prerequisites

- **RunPod** account and API key; a serverless endpoint running this worker image (GPU).
- **Hugging Face** account: accept [pyannote speaker-diarization](https://huggingface.co/pyannote/speaker-diarization-3.0) terms and set `HF_TOKEN` on the worker.
- **Google Cloud:** A GCS bucket for inputs; credentials on your machine for the local upload/transcribe scripts (`gcloud auth application-default login` or service account).

## Setup (local)

```bash
uv venv
uv sync --group client
```

For local handler testing (GPU + WhisperX):

```bash
uv sync --group client --group worker
```

Tasks (optional, [Taskfile](https://taskfile.dev)):

```bash
task venv
task upload -- path/to/meeting.mp4
task transcribe -- meeting.mp4 -o transcript.csv
```

## Local scripts (run on your machine)

### Upload only → signed URL

Upload a file to GCS and print a signed URL (paste into RunPod console or use with transcribe):

```bash
export GCS_BUCKET=your-bucket-name
uv run scripts/upload_to_gcs.py path/to/meeting.mp4
# Or: uv run scripts/upload_to_gcs.py meeting.mp4 --bucket your-bucket --expiry 2
```

### Full pipeline: upload + RunPod + save CSV

```bash
export GCS_BUCKET=your-bucket
export RUNPOD_API_KEY=your-runpod-api-key
export ENDPOINT_ID=your-runpod-endpoint-id
uv run scripts/transcribe.py meeting.mp4 -o transcript.csv
```

## RunPod worker

- **Handler:** [src/handler.py](src/handler.py). Input: `{"audio_url": "https://..."}`. Optional: `language`, `model_size`.
- **Build and run:**

```bash
docker build -t runpod-wispr .
# Run locally (GPU): docker run --rm --gpus all -e HF_TOKEN=your-hf-token runpod-wispr
```

Deploy the image to a RunPod serverless endpoint, set `HF_TOKEN`, and use a GPU (e.g. T4 or A10G).

## CSV format

```csv
speaker_id,utterance
SPEAKER_00,"Hello everyone, let's start the meeting."
SPEAKER_01,"Thanks. I'll take the notes."
```

## Testing

- **Handler locally:** `python src/handler.py` (uses `test_input.json` if present). Requires GPU, ffmpeg, and `HF_TOKEN`.
- **RunPod:** In the endpoint’s Requests tab, send `{"input": {"audio_url": "<signed_url>"}}`.

---

*Last updated: 2025-03-06. Doc SHA: (not a git repo).*
