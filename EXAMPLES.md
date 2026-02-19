# Usage Examples

## Basic Audio Transcription

```bash
# Transcribe an MP3 file
python transcribe.py meeting.mp3

# Transcribe a WAV recording
python transcribe.py interview.wav

# Transcribe with API key in command
python transcribe.py podcast.m4a --api-key sk-...
```

## Video File Transcription

```bash
# Transcribe MP4 video (sent directly to OpenAI)
python transcribe.py conference_call.mp4

# Transcribe MOV/AVI video (auto audio extraction)
python transcribe.py recording.mov
python transcribe.py webinar.avi

# Re-encode & compress for large MP4 (>25MB)
python transcribe.py large_meeting.mp4 --extract-audio
```

## Large / Long File Handling

```bash
# Large audio files (>25MB) — re-encodes and auto-splits into chunks
python transcribe.py long_interview.m4a --extract-audio

# Multi-hour recordings — auto-split to stay within API limits
python transcribe.py 3hr_meeting.m4a --extract-audio --clean

# Batch folder with large files
python transcribe.py ./recordings/ --batch --extract-audio
```

### Auto-Split Output:
```
  Re-encoding audio: 3hr_meeting.m4a
  Audio extracted to: 3hr_meeting.extracted.mp3 (87.3MB)
  File is 87.3MB, 5720s -- splitting into 5 chunks (~1144s each)
  Created 5 chunk(s)
  -- Chunk 1/5 --
  Transcribing: 3hr_meeting.extracted.chunk000.mp3
  Model: gpt-4o-transcribe-diarize
  File size: 17.5MB
  -- Chunk 2/5 --
  ...
```
Chunks are transcribed individually and merged with continuous timestamps.

## Expected Output

### Console Output:
```
Transcribing audio with speaker diarization: meeting.mp3

================================================================================
TRANSCRIPT
================================================================================

[00:00:01 - 00:00:08] Speaker 1:
Good morning everyone. Let's get started with the sprint review.

[00:00:09 - 00:00:15] Speaker 2:
Sure. First item is the payment module refactor.

[00:00:16 - 00:00:23] Speaker 1:
Great. Can you walk us through the changes?

================================================================================

Transcript saved to: meeting_transcript.txt
```

### File Output:
A text file named `meeting_transcript.txt` is created in the same directory with the formatted transcript.

## Video Examples

### Small Video Files (< 25MB)
```bash
# OpenAI accepts MP4 directly
python transcribe.py zoom_recording.mp4
```

### Large Video Files (> 25MB)
```bash
# Re-encode audio to compress; auto-splits if still over 25MB
python transcribe.py long_webinar.mp4 --extract-audio
```

### Non-MP4 Videos
```bash
# AVI, MOV, MKV - automatic audio extraction
python transcribe.py screen_recording.avi
python transcribe.py iphone_video.mov
python transcribe.py gameplay.mkv
```

## Error Handling

### File Size Error:
```
Error: File size (30.5MB) exceeds OpenAI's 25MB limit.
Use --extract-audio to compress, or split the file.
```
**Solution**: Use `--extract-audio` to re-encode and compress. The script will auto-split into chunks if needed.

### Missing API Key:
```
Error: OpenAI API key not found. Set OPENAI_API_KEY environment variable.
```
**Solution**: Set environment variable or use `--api-key` flag.

### Unsupported Format:
```
Error: Unsupported format: .wmv.
Supported formats: .wav, .mp3, .m4a, ...
```
**Solution**: Check supported formats or convert the file.

## Tips

1. **For meetings/conferences**: Use MP4/MP3 format for best compatibility
2. **For large files**: Always use `--extract-audio` — the script re-encodes to 128kbps mono MP3 and auto-splits into chunks if needed
3. **For best accuracy**: Ensure clear audio with minimal background noise
4. **For podcasts**: MP3 format works great and is smaller than WAV
5. **For video files**: The script extracts only the audio track, video content is not analyzed
6. **For multi-hour recordings**: `--extract-audio` is required — the API has a 25MB / 23-minute limit per request
