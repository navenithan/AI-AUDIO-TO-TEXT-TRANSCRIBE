# Audio Transcription with Speaker Diarization

Production-ready speech-to-text tool with speaker identification, content analysis, and batch processing — powered entirely by OpenAI.

## Quick Setup

### 1. Install Python 3.10+

```bash
python --version
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:
- **Windows**: `venv\Scripts\activate`
- **Linux/Mac**: `source venv/bin/activate`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file (or set the variable directly):
```bash
# .env file
OPENAI_API_KEY=your_key_here
```

Or set via terminal:
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="your_key_here"

# Linux/Mac
export OPENAI_API_KEY="your_key_here"
```

## Usage

### Single File
```bash
python transcribe.py meeting.mp4
python transcribe.py interview.mp3 --clean
python transcribe.py call.wav --no-analysis
```

### Batch Processing (entire folder)
```bash
python transcribe.py ./recordings/ --batch
python transcribe.py ./recordings/ --batch --clean
```

### Interactive Mode
```bash
python transcribe.py --interactive
```

### Cost Estimation (before processing)
```bash
python transcribe.py meeting.mp4 --estimate-cost
python transcribe.py ./recordings/ --estimate-cost
```

### All CLI Options
```
python transcribe.py <file_or_folder> [options]

Options:
  --model MODEL          Transcription model (default: gpt-4o-transcribe-diarize)
  --clean                Remove filler words (um, uh, like, you know, etc.)
  --no-analysis          Skip advanced summary and content analysis
  --batch                Process all supported files in a folder
  --interactive, -i      Launch interactive mode
  --extract-audio        Re-encode & compress audio (required for large/long files)
  --estimate-cost        Show estimated cost without processing
  --api-key KEY          OpenAI API key (overrides env variable)
```

## Features

### Core
- **Speaker Diarization** — Identifies different speakers automatically
- **15+ Formats** — WAV, MP3, M4A, FLAC, OGG, MP4, MPEG, WEBM, AVI, MOV, MKV, WMV, FLV
- **Video Support** — Auto-extracts audio from video files
- **Timestamps** — HH:MM:SS for each speaker segment

### Analysis (saved to `<filename>_analysis.txt`)
- **Advanced Summary** — Key points, action items, decisions, Q&A
- **Content Analysis** — Topics, keywords, named entities, sentiment
- **Confidence Scores** — Heuristic quality check per segment

### Processing
- **Batch Mode** — Process entire folders of files
- **Auto-Split** — Files over 25MB or 23 min are automatically split into chunks and merged
- **Smart Cleaning** — Remove filler words (um, uh, like, you know, etc.)
- **Cost Estimation** — See estimated API cost before committing
- **Cost Tracking** — Running total of actual API spend

### Interactive Mode
- REPL-style interface for processing files one-by-one
- Toggle cleaning, analysis, extraction on/off
- View running session cost
- Process files or entire folders

## Output Files

Each file produces:

| File | Content |
|------|---------|
| `<name>_transcript.txt` | Timestamped speaker transcript |
| `<name>_analysis.txt` | Summary, content analysis, confidence report, cost |

### Transcript Format
```
[00:00:01 - 00:00:08] Speaker 1:
Good morning everyone. Let's get started with the sprint review.

[00:00:09 - 00:00:15] Speaker 2:
Sure. First item is the payment module refactor.
```

### Analysis Report Includes
- Detailed Summary (key points, action items, decisions, questions)
- Content Analysis (topics, keywords, entities, sentiment)
- Confidence Report (per-segment quality scoring)
- Cost Breakdown (transcription + analysis costs)

## Supported Formats

**Audio (native OpenAI support):**
WAV, MP3, M4A, FLAC, OGG, MP4, MPEG, MPGA, WEBM

**Video (auto audio extraction):**
AVI, MOV, MKV, WMV, FLV

## Requirements

- Python 3.10+
- OpenAI API key (paid account)
- FFmpeg (bundled via `imageio-ffmpeg`, or install system-wide)

## Troubleshooting

**"Missing required package"** — Run `pip install -r requirements.txt`

**"API key not found"** — Ensure `.env` file exists or env variable is set

**"Model not available"** — The script auto-falls back to `whisper-1`. Check your OpenAI account model access.

**"File size exceeds 25MB"** — Use `--extract-audio` to re-encode and compress; the script will auto-split into chunks if still needed

**Audio extraction fails** — Ensure `imageio-ffmpeg` is installed (`pip install imageio-ffmpeg`), or install FFmpeg system-wide
