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

# Force extraction for large MP4 (>25MB)
python transcribe.py large_meeting.mp4 --extract-audio
```

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
# Extract audio first to avoid size limit errors
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
Please compress or split the audio file.
```
**Solution**: Use `--extract-audio` with compression or split the file.

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
2. **For large files**: Always use `--extract-audio` to compress before sending
3. **For best accuracy**: Ensure clear audio with minimal background noise
4. **For podcasts**: MP3 format works great and is smaller than WAV
5. **For video files**: The script extracts only the audio track, video content is not analyzed
