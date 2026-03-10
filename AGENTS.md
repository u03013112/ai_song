# AGENTS.md — ai_song

> AI singing voice conversion pipeline. macOS (Apple Silicon), Python.

## Project Overview

Audio processing pipeline that automates: song download -> vocal/instrument separation -> voice conversion -> mixing.
Target platform: macOS with Apple Silicon (MPS acceleration). Primary language: Python.

## Repository Structure

```
ai_song/
  README.md          # Project plan and analysis (Chinese)
  .gitignore         # Python standard gitignore
  AGENTS.md          # This file
```

This is a greenfield project. Structure will evolve — update this file as modules are added.

## Build & Run Commands

### Environment Setup

```bash
# Create virtual environment (use Python 3.10+ for audio ML compatibility)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (once requirements.txt or pyproject.toml exists)
pip install -r requirements.txt
# or
pip install -e .
```

### Running

```bash
# Run main pipeline (update as entry point is created)
python -m ai_song

# Run a specific step
python -m ai_song.separate --input song.wav
python -m ai_song.convert --input vocals.wav --model model_name
python -m ai_song.mix --vocals converted.wav --instrumental instrumental.wav
```

### Testing

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_separate.py

# Run a single test function
pytest tests/test_separate.py::test_vocal_separation

# Run tests matching a keyword
pytest -k "separation"

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=ai_song --cov-report=term-missing
```

### Linting & Formatting

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Lint and auto-fix
ruff check --fix .

# Type checking
mypy ai_song/
```

## Code Style Guidelines

### Python Version & Standards

- Python 3.10+ (needed for modern type syntax and ML library compatibility)
- Follow PEP 8 with the exceptions enforced by ruff
- Maximum line length: 88 characters (ruff/black default)

### Imports

```python
# Order: stdlib -> third-party -> local (ruff enforces this via isort rules)
import os
from pathlib import Path

import numpy as np
import torch

from ai_song.utils import load_audio
```

- Use absolute imports for cross-module references
- Use `from __future__ import annotations` when needed for forward refs
- Prefer specific imports (`from module import func`) over wildcard imports

### Type Annotations

```python
# Use modern Python 3.10+ syntax
def process_audio(input_path: Path, sample_rate: int = 44100) -> np.ndarray:
    ...

# For complex types
from collections.abc import Sequence

def batch_process(files: Sequence[Path]) -> list[np.ndarray]:
    ...
```

- Annotate all public function signatures (params and return types)
- Use `Path` (not `str`) for filesystem paths
- Use `np.ndarray` for audio data types
- Never use `Any` to bypass type checking — find the correct type

### Naming Conventions

| Element         | Convention        | Example                        |
|-----------------|-------------------|--------------------------------|
| Modules         | snake_case        | `vocal_separator.py`           |
| Classes         | PascalCase        | `VocalSeparator`               |
| Functions       | snake_case        | `separate_vocals()`            |
| Constants       | UPPER_SNAKE_CASE  | `DEFAULT_SAMPLE_RATE = 44100`  |
| Private members | _leading_under    | `_internal_buffer`             |
| CLI args        | kebab-case        | `--sample-rate`                |

### Error Handling

```python
# Define project-specific exceptions
class AiSongError(Exception):
    """Base exception for ai_song."""

class AudioLoadError(AiSongError):
    """Failed to load audio file."""

class ModelNotFoundError(AiSongError):
    """Voice conversion model not found."""

# Always catch specific exceptions, never bare except
try:
    audio = load_audio(path)
except FileNotFoundError:
    raise AudioLoadError(f"Audio file not found: {path}")

# Never silently swallow errors
# BAD: except Exception: pass
# GOOD: except SpecificError as e: logger.error(...)
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Use structured logging levels appropriately
logger.debug("Processing frame %d/%d", current, total)
logger.info("Separation complete: %s", output_path)
logger.warning("Sample rate mismatch: expected %d, got %d", expected, actual)
logger.error("Failed to load model: %s", model_name)
```

- Use `logging` module, not `print()`, for operational output
- `print()` is acceptable only for CLI user-facing output
- Use lazy formatting (`%s`) not f-strings in logger calls

### File & Path Handling

```python
from pathlib import Path

# Always use pathlib for file operations
input_path = Path("data") / "songs" / "input.wav"

# Validate paths exist before processing
if not input_path.exists():
    raise AudioLoadError(f"File not found: {input_path}")
```

### Audio-Specific Conventions

- Default sample rate: 44100 Hz (CD quality)
- Audio format: WAV for intermediate files, avoid lossy compression mid-pipeline
- Use `numpy` arrays as the internal audio representation
- Normalize audio data to float32 range [-1.0, 1.0]
- Always preserve original sample rate metadata through the pipeline

### Configuration

- Use dataclasses or pydantic models for configuration
- Support both CLI args (via argparse or click) and config files (YAML)
- Sensitive values (API keys) go in `.env` — never commit them

```python
from dataclasses import dataclass

@dataclass
class SeparationConfig:
    model_name: str = "htdemucs"
    sample_rate: int = 44100
    output_dir: Path = Path("output")
```

### Project Conventions

- Each pipeline step is an independent module that can run standalone via CLI
- Modules should be composable — output of one step feeds into the next
- Use `click` or `argparse` for CLI interfaces
- Audio files are large — never commit them to git (already in .gitignore)
- Write docstrings for all public functions and classes (Google style)

```python
def separate_vocals(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Separate vocals from instrumental track.

    Args:
        input_path: Path to the input audio file (WAV format).
        output_dir: Directory to write separated tracks.

    Returns:
        Tuple of (vocals_path, instrumental_path).

    Raises:
        AudioLoadError: If the input file cannot be loaded.
    """
```

### Testing Guidelines

- Use `pytest` as the test framework
- Place tests in `tests/` directory, mirroring source structure
- Use fixtures for common audio test data
- Mock external API calls and model inference in unit tests
- Integration tests can use small audio samples (< 5s duration)
- Name test files `test_<module>.py`, test functions `test_<behavior>`

## Communication

- 与用户交流、思考过程尽可能使用中文
- 专有名词、技术缩写保持原文（如 MPS、RVC、demucs、WAV、CLI 等）
- README and inline comments may be in Chinese
- Code identifiers (variables, functions, classes) must be in English
- Git commit messages in English
- Docstrings in English

## Key Dependencies (Planned)

| Purpose              | Library                    |
|----------------------|----------------------------|
| Audio I/O            | librosa, soundfile         |
| Vocal separation     | demucs (Meta/Facebook)     |
| Voice conversion     | RVC, so-vits-svc           |
| Audio processing     | numpy, scipy               |
| ML framework         | torch (with MPS backend)   |
| CLI                  | click or argparse          |
| Testing              | pytest                     |
| Linting              | ruff                       |
| Type checking        | mypy                       |

## Git Workflow

- Branch from `main` for features: `feature/<name>`
- Keep commits atomic and descriptive
- Do not commit audio files, model weights, or .env files
- 当用户说"提交代码"时，默认执行 commit + push（即 CI & push）
