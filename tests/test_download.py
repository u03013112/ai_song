"""Tests for the download module."""

from pathlib import Path

from ai_song.download import DEFAULT_OUTPUT_DIR, DownloadError


def test_download_error_is_exception() -> None:
    assert issubclass(DownloadError, Exception)


def test_default_output_dir() -> None:
    assert Path("downloads") == DEFAULT_OUTPUT_DIR
