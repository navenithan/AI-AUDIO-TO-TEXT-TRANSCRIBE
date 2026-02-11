# Audio Transcription with Speaker Diarization

Production-ready speech-to-text tool with speaker identification using OpenAI's native diarization.
**Now supports video files** with automatic audio extraction!

## Quick Setup

### 1. Install Python 3.10+

Ensure Python 3.10 or newer is installed:
```bash
python --version
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:
- **Windows**: `venv\Scripts\activate`
- **Linux/Mac**: `source venv/bin/activate`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

1. Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)

2. Set environment variable:
   ```bash
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your_key_here"
   
   # Linux/Mac
   export OPENAI_API_KEY="your_key_here"
   ```

## Usage

```bash
# Transcribe audio file
python transcribe.py audio.mp3

# Transcribe video file (auto-extracts audio)
python transcribe.py meeting.mp4

# Force audio extraction for large video files
python transcribe.py large_video.mp4 --extract-audio
```

### Supported Formats

**Audio (native support):**
- WAV, MP3, M4A, FLAC, OGG
- MP4, MPEG, MPGA, WEBM

**Video (auto audio extraction):**
- AVI, MOV, MKV, WMV, FLV

### Command-Line Options

```bash
# Basic usage
python transcribe.py file.mp3

# Provide API key directly
python transcribe.py file.wav --api-key YOUR_OPENAI_KEY

# Extract audio from video first (helps with large files)
python transcribe.py video.mp4 --extract-audio
```

## Features

✅ **Single API Platform** - Uses only OpenAI (no HuggingFace needed)  
✅ **Video Support** - Automatically extracts audio from video files  
✅ **Speaker Diarization** - Identifies different speakers automatically  
✅ **Timestamp Precision** - HH:MM:SS format for each segment  
✅ **Multiple Formats** - Supports 15+ audio/video formats  
✅ **25MB Limit Check** - Warns if file exceeds OpenAI's limit  
✅ **Clean Output** - Console display + text file output  
✅ **Auto Cleanup** - Removes temporary extracted audio files

## How Video Extraction Works

For video files (AVI, MOV, MKV, etc.), the script:
1. Extracts audio track to temporary MP3 file using MoviePy
2. Sends audio to OpenAI for transcription with speaker diarization
3. Generates formatted transcript
4. Cleans up the temporary file automatically

**Note**: MP4/MPEG can be sent directly to OpenAI, but use `--extract-audio` if the file is large (>25MB) or if you encounter issues.

## Output

The script generates two outputs:

1. **Console**: Pretty-printed transcript
2. **File**: `<filename>_transcript.txt` in the same directory as the input file

### Output Format Example

```
[00:00:01 - 00:00:08] Speaker 1:
Good morning everyone. Let's get started with the sprint review.

[00:00:09 - 00:00:15] Speaker 2:
Sure. First item is the payment module refactor.
```

## Error Handling

The script handles:

## Requirements

- Python 3.10+
- OpenAI API key (paid account required for API access)

## Troubleshooting

**"Missing required package" error**:
```bash
pip install -r requirements.txt
```

**"API key not found" error**:
Ensure the OPENAI_API_KEY environment variable is set correctly.

**"API transcription failed" error**:
1. Verify your OpenAI API key is valid
2. Check your account has available credits
3. Ensure the audio file is less than 25 MB
4. Check internet connection
3. Check internet connection

**Audio conversion error**:
Install FFmpeg and ensure it's in your system PATH.
