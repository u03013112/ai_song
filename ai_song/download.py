"""Download songs from video platforms and extract audio."""

from __future__ import annotations

import logging
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("downloads")


class DownloadError(Exception):
    """Failed to download or extract audio."""


def download_audio(
    url: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    filename: str | None = None,
) -> Path:
    """Download audio from a URL and save as WAV.

    Args:
        url: Video/audio URL (supports Bilibili, YouTube, etc.).
        output_dir: Directory to save the downloaded file.
        filename: Optional output filename (without extension).
            Defaults to the video title.

    Returns:
        Path to the downloaded WAV file.

    Raises:
        DownloadError: If download or audio extraction fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = str(output_dir / (filename or "%(title)s")) + ".%(ext)s"

    ydl_opts: dict[str, object] = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise DownloadError(f"Failed to extract info from: {url}")

            title = info.get("title", "audio")
            wav_path = output_dir / f"{filename or title}.wav"

            if not wav_path.exists():
                raise DownloadError(f"Expected output not found: {wav_path}")

            logger.info("Downloaded: %s -> %s", url, wav_path)
            return wav_path

    except yt_dlp.utils.DownloadError as e:
        raise DownloadError(f"Download failed for {url}: {e}") from e


def main() -> None:
    """CLI entry point for downloading audio."""
    import argparse

    parser = argparse.ArgumentParser(description="Download audio from URL as WAV")
    parser.add_argument("url", help="Video/audio URL (Bilibili, YouTube, etc.)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: downloads/)",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Output filename without extension (default: video title)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    path = download_audio(args.url, args.output_dir, args.filename)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
