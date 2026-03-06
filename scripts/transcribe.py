"""
Full pipeline: upload file to GCS, call RunPod endpoint, wait for result, save CSV.
Run locally: uv run scripts/transcribe.py meeting.mp4 -o transcript.csv
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import typer
from google.cloud import storage

app = typer.Typer(help="Upload to GCS, transcribe via RunPod, save CSV.")


def _get_env(name: str, opt: str | None) -> str:
    if opt:
        return opt
    v = os.environ.get(name)
    if not v:
        typer.echo(f"Set {name} or pass --{name.lower()}.", err=True)
        raise typer.Exit(1)
    return v


def _upload_and_signed_url(
    file_path: Path,
    bucket_name: str,
    prefix: str,
    expiry_hours: int,
) -> str:
    client = storage.Client()
    bucket_obj = client.bucket(bucket_name)
    object_name = f"{prefix.rstrip('/')}/{file_path.name}"
    blob = bucket_obj.blob(object_name)
    content_type = "audio/mpeg" if file_path.suffix.lower() == ".mp3" else "video/mp4" if file_path.suffix.lower() == ".mp4" else "application/octet-stream"
    blob.upload_from_filename(str(file_path), content_type=content_type)
    try:
        return blob.generate_signed_url(expiration=timedelta(hours=expiry_hours), method="GET")
    except AttributeError:
        blob.make_public()
        return blob.public_url


@app.command()
def main(
    file_path: Path = typer.Argument(..., help="Local MP3 or MP4 file", path_type=Path),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output CSV path (default: <input>.csv)", path_type=Path),
    bucket: str | None = typer.Option(None, "--bucket", "-b", help="GCS bucket (or GCS_BUCKET)"),
    prefix: str = typer.Option("meetings/", "--prefix", "-p", help="Object prefix in bucket"),
    runpod_api_key: str | None = typer.Option(None, "--runpod-api-key", envvar="RUNPOD_API_KEY"),
    endpoint_id: str | None = typer.Option(None, "--endpoint-id", envvar="ENDPOINT_ID"),
    language: str | None = typer.Option(None, "--language", "-l", help="Language code (e.g. de, en). Auto-detected if omitted."),
    timeout_sec: int = typer.Option(600, "--timeout", "-t", help="Max seconds to wait for job"),
) -> None:
    """Upload FILE_PATH to GCS, call RunPod, write CSV to OUTPUT."""
    if not file_path.exists() or not file_path.is_file():
        typer.echo(f"File not found: {file_path}", err=True)
        raise typer.Exit(1)

    if output is None:
        output = file_path.with_suffix(".csv")

    bucket_name = _get_env("GCS_BUCKET", bucket)
    api_key = _get_env("RUNPOD_API_KEY", runpod_api_key)
    ep_id = _get_env("ENDPOINT_ID", endpoint_id)

    # Show file info
    file_size = file_path.stat().st_size
    if file_size < 1024 * 1024:
        size_str = f"{file_size / 1024:.1f} KB"
    else:
        size_str = f"{file_size / (1024 * 1024):.1f} MB"
    duration_str = ""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            dur = float(result.stdout.strip())
            mins, secs = divmod(int(dur), 60)
            hours, mins = divmod(mins, 60)
            if hours:
                duration_str = f", duration: {hours}h {mins}m {secs}s"
            elif mins:
                duration_str = f", duration: {mins}m {secs}s"
            else:
                duration_str = f", duration: {secs}s"
    except (FileNotFoundError, ValueError):
        pass

    typer.echo(f"Uploading {file_path.name} ({size_str}{duration_str}) to GCS ...", err=True)
    audio_url = _upload_and_signed_url(file_path, bucket_name, prefix, expiry_hours=2)
    typer.echo("Calling RunPod (waiting for result) ...", err=True)

    import runpod

    runpod.api_key = api_key
    endpoint = runpod.Endpoint(ep_id)
    payload = {"audio_url": audio_url}
    if language:
        payload["language"] = language
    run_request = endpoint.run(payload)
    t0 = time.monotonic()
    try:
        out = run_request.output(timeout=timeout_sec)
    except Exception as e:
        typer.echo(f"RunPod error: {e}", err=True)
        raise typer.Exit(1)
    elapsed = time.monotonic() - t0
    mins, secs = divmod(int(elapsed), 60)
    if mins:
        typer.echo(f"Transcription completed in {mins}m {secs}s", err=True)
    else:
        typer.echo(f"Transcription completed in {secs}s", err=True)

    csv_content = None
    if isinstance(out, dict):
        csv_content = out.get("csv")
        if csv_content is None and isinstance(out.get("output"), dict):
            csv_content = out["output"].get("csv")
    if not csv_content:
        typer.echo("Job completed but no CSV in output.", err=True)
        raise typer.Exit(1)
    output.write_text(csv_content, encoding="utf-8")
    typer.echo(f"Wrote {output}", err=True)


if __name__ == "__main__":
    app()
