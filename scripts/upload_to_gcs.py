"""
Upload a local MP3/MP4 to GCS and print a signed URL for use with RunPod.
Run locally: uv run scripts/upload_to_gcs.py path/to/meeting.mp4
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import typer
from google.cloud import storage

app = typer.Typer(help="Upload file to GCS and print signed URL for RunPod.")


def _get_bucket_name(bucket: str | None) -> str:
    if bucket:
        return bucket
    import os
    b = os.environ.get("GCS_BUCKET")
    if not b:
        typer.echo("Set GCS_BUCKET or pass --bucket.", err=True)
        raise typer.Exit(1)
    return b


@app.command()
def main(
    file_path: Path = typer.Argument(..., help="Local MP3 or MP4 file to upload", path_type=Path),
    bucket: str | None = typer.Option(None, "--bucket", "-b", help="GCS bucket name (or set GCS_BUCKET)"),
    prefix: str = typer.Option("meetings/", "--prefix", "-p", help="Object name prefix in bucket"),
    expiry_hours: int = typer.Option(1, "--expiry", "-e", help="Signed URL expiry in hours"),
) -> None:
    """Upload FILE_PATH to GCS and print a signed URL to stdout."""
    if not file_path.exists():
        typer.echo(f"File not found: {file_path}", err=True)
        raise typer.Exit(1)
    if not file_path.is_file():
        typer.echo(f"Not a file: {file_path}", err=True)
        raise typer.Exit(1)

    bucket_name = _get_bucket_name(bucket)
    object_name = f"{prefix.rstrip('/')}/{file_path.name}"

    client = storage.Client()
    bucket_obj = client.bucket(bucket_name)
    blob = bucket_obj.blob(object_name)

    typer.echo(f"Uploading {file_path} to gs://{bucket_name}/{object_name} ...", err=True)
    blob.upload_from_filename(str(file_path), content_type=_content_type(file_path.suffix))

    # Use public URL if bucket allows, otherwise try signed URL.
    # Signed URLs require service-account credentials; ADC user creds can't sign.
    try:
        url = blob.generate_signed_url(
            expiration=timedelta(hours=expiry_hours),
            method="GET",
        )
    except AttributeError:
        # Fallback: make object publicly readable and use public URL
        blob.make_public()
        url = blob.public_url
        typer.echo("(Using public URL — signed URL requires service account credentials)", err=True)
    print(url)


def _content_type(suffix: str) -> str:
    s = suffix.lower()
    if s == ".mp3":
        return "audio/mpeg"
    if s == ".mp4":
        return "video/mp4"
    if s == ".wav":
        return "audio/wav"
    return "application/octet-stream"


if __name__ == "__main__":
    app()
