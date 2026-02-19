#!/usr/bin/env python3
"""
Speech-to-Text Transcription with Speaker Diarization
Supports: wav, mp3, m4a, flac, ogg, mp4, mpeg, mpga, webm, avi, mov, mkv
Uses OpenAI's gpt-4o-transcribe-diarize model

Features:
  - Single file or batch processing
  - Smart transcript cleaning (filler word removal)
  - Advanced AI summaries (key points, action items, Q&A)
  - Content analysis (topics, keywords, entities)
  - Cost estimation
  - Confidence scoring
  - Interactive mode
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing required package. Please run: pip install -r requirements.txt")
    print(f"Details: {e}")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Pricing (USD per minute) — update if OpenAI changes rates
# https://openai.com/api/pricing
# ──────────────────────────────────────────────────────────────────────────────
MODEL_COST_PER_MINUTE = {
    "gpt-4o-transcribe-diarize": 0.06,   # $0.06 / min (estimated)
    "gpt-4o-transcribe":         0.06,
    "gpt-4o-mini-transcribe":    0.03,
    "whisper-1":                 0.006,
}
GPT4O_MINI_COST_PER_1K_INPUT  = 0.00015   # for summaries / analysis
GPT4O_MINI_COST_PER_1K_OUTPUT = 0.0006

# Filler words to strip during cleaning
FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "erm", "ah", "ahh",
    "hmm", "hm", "mm", "mmm", "mhm", "uh-huh",
    "you know", "i mean", "like", "sort of", "kind of",
    "basically", "actually", "literally", "right",
}
# Build a regex that matches whole-word fillers (case-insensitive)
_filler_pattern = re.compile(
    r'\b(?:' + '|'.join(re.escape(f) for f in sorted(FILLER_WORDS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)


class CostTracker:
    """Track estimated API costs across operations."""

    def __init__(self):
        self.transcription_cost = 0.0
        self.analysis_cost = 0.0

    def add_transcription(self, duration_seconds: float, model: str):
        rate = MODEL_COST_PER_MINUTE.get(model, 0.06)
        self.transcription_cost += (duration_seconds / 60) * rate

    def add_chat_tokens(self, input_tokens: int, output_tokens: int):
        self.analysis_cost += (input_tokens / 1000) * GPT4O_MINI_COST_PER_1K_INPUT
        self.analysis_cost += (output_tokens / 1000) * GPT4O_MINI_COST_PER_1K_OUTPUT

    @property
    def total(self) -> float:
        return self.transcription_cost + self.analysis_cost

    def summary(self) -> str:
        return (
            f"Transcription: ${self.transcription_cost:.4f}  |  "
            f"Analysis: ${self.analysis_cost:.4f}  |  "
            f"Total: ${self.total:.4f}"
        )


class AudioTranscriber:
    """Handles audio transcription and speaker diarization using OpenAI API."""

    # Native OpenAI supported formats
    SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4', '.mpeg', '.mpga', '.webm'}

    # Video formats that require audio extraction
    SUPPORTED_VIDEO_FORMATS = {'.avi', '.mov', '.mkv', '.wmv', '.flv'}

    def __init__(self, api_key: str = None):
        """Initialize transcriber with API credentials."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.cost = CostTracker()

    # ── File helpers ──────────────────────────────────────────────────────────

    def validate_file(self, file_path: Path) -> None:
        """Validate audio/video file exists and has supported format."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in (self.SUPPORTED_AUDIO_FORMATS | self.SUPPORTED_VIDEO_FORMATS):
            raise ValueError(
                f"Unsupported format: {file_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_AUDIO_FORMATS | self.SUPPORTED_VIDEO_FORMATS)}"
            )

    def extract_audio_from_video(self, video_path: Path) -> Path:
        """Extract/re-encode audio to compressed MP3 using ffmpeg.

        Works for both video files (extracts audio track) and oversized
        audio files (re-encodes with compression to fit under 25 MB).
        """
        import subprocess

        ffmpeg_cmd = self._find_ffmpeg()
        if ffmpeg_cmd is None:
            raise ImportError(
                "ffmpeg is required for audio extraction/compression. "
                "Install it with: pip install imageio-ffmpeg  (or install ffmpeg on your system)"
            )

        print(f"  Re-encoding audio: {video_path.name}")

        try:
            audio_path = video_path.with_suffix('.extracted.mp3')
            cmd = [
                ffmpeg_cmd, '-y', '-i', str(video_path),
                '-vn',                   # discard video
                '-acodec', 'libmp3lame', # MP3 codec
                '-ab', '128k',           # 128 kbps – good quality for speech
                '-ar', '44100',          # 44.1 kHz sample rate
                '-ac', '1',              # mono (halves size, fine for speech)
                str(audio_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr[-500:] if result.stderr else 'unknown ffmpeg error')

            out_size_mb = audio_path.stat().st_size / (1024 * 1024)
            print(f"  Audio extracted to: {audio_path.name} ({out_size_mb:.1f}MB)")
            return audio_path
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg timed out while processing the file.")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio: {e}")

    @staticmethod
    def _find_ffmpeg() -> Optional[str]:
        """Return path to an ffmpeg binary, or None."""
        import shutil
        # 1. System ffmpeg
        path = shutil.which('ffmpeg')
        if path:
            return path
        # 2. imageio-ffmpeg bundled binary
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
        return None

    def _get_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds using ffprobe, ffmpeg, or mutagen."""
        import subprocess, shutil

        ffmpeg_cmd = self._find_ffmpeg()

        # 1. Try mutagen (works for m4a, mp3, ogg, flac, etc.)
        try:
            from mutagen import File as MutagenFile
            audio_info = MutagenFile(str(audio_path))
            if audio_info and audio_info.info and audio_info.info.length > 0:
                return audio_info.info.length
        except Exception:
            pass

        if not ffmpeg_cmd:
            return 0.0

        # 2. Try ffprobe (same directory as ffmpeg)
        ffprobe_cmd = shutil.which('ffprobe')
        if not ffprobe_cmd:
            ffprobe_candidate = str(Path(ffmpeg_cmd).with_name('ffprobe'))
            if Path(ffprobe_candidate).exists():
                ffprobe_cmd = ffprobe_candidate
            elif Path(ffprobe_candidate + '.exe').exists():
                ffprobe_cmd = ffprobe_candidate + '.exe'

        if ffprobe_cmd:
            try:
                result = subprocess.run(
                    [ffprobe_cmd, '-v', 'error', '-show_entries',
                     'format=duration', '-of', 'csv=p=0', str(audio_path)],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return float(result.stdout.strip())
            except Exception:
                pass

        # 3. Use ffmpeg -i to parse duration from stderr
        try:
            result = subprocess.run(
                [ffmpeg_cmd, '-i', str(audio_path), '-f', 'null', '-'],
                capture_output=True, text=True, timeout=60,
            )
            # ffmpeg prints "Duration: HH:MM:SS.xx" in stderr
            import re as _re
            m = _re.search(r'Duration:\s*(\d+):(\d+):(\d+)\.(\d+)', result.stderr)
            if m:
                h, mn, s, cs = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                return h * 3600 + mn * 60 + s + cs / 100.0
        except Exception:
            pass

        return 0.0

    def _split_audio(self, audio_path: Path, max_size_mb: float = 24.0,
                     max_duration_sec: float = 1300.0):
        """Split an audio file into chunks that each fit under max_size_mb
        and max_duration_sec.

        Returns (chunk_paths, chunk_duration_sec).  chunk_duration_sec is
        the target length of each chunk (0 when no split was needed).
        """
        import subprocess, math

        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        duration = self._get_duration(audio_path)

        # Check if splitting is needed at all
        if file_size_mb <= max_size_mb and (duration <= 0 or duration <= max_duration_sec):
            return [audio_path], 0

        ffmpeg_cmd = self._find_ffmpeg()
        if not ffmpeg_cmd:
            raise RuntimeError("ffmpeg is needed to split large files.")

        if duration <= 0:
            raise RuntimeError("Could not determine audio duration for splitting.")

        # Determine chunk duration based on BOTH constraints
        chunks_by_size = math.ceil(file_size_mb / max_size_mb) if file_size_mb > max_size_mb else 1
        chunks_by_duration = math.ceil(duration / max_duration_sec) if duration > max_duration_sec else 1
        num_chunks = max(chunks_by_size, chunks_by_duration)
        chunk_duration = math.ceil(duration / num_chunks)

        print(f"  File is {file_size_mb:.1f}MB, {duration:.0f}s "
              f"-- splitting into {num_chunks} chunks (~{chunk_duration}s each)")

        chunk_paths: List[Path] = []
        for i in range(num_chunks):
            start = i * chunk_duration
            if start >= duration:
                break  # no more audio left
            chunk_path = audio_path.with_name(f"{audio_path.stem}.chunk{i:03d}.mp3")
            cmd = [
                ffmpeg_cmd, '-y',
                '-ss', str(start),
                '-i', str(audio_path),
                '-t', str(chunk_duration),
                '-acodec', 'copy',   # fast copy, no re-encode
                str(chunk_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to split chunk {i}: {result.stderr[-300:]}")
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                chunk_paths.append(chunk_path)

        print(f"  Created {len(chunk_paths)} chunk(s)")
        return chunk_paths, chunk_duration

    # ── Cost estimation ───────────────────────────────────────────────────────

    def estimate_cost(self, file_path: Path, model: str) -> Dict:
        """Estimate cost before processing."""
        try:
            from mutagen import File as MutagenFile
            audio_info = MutagenFile(str(file_path))
            duration = audio_info.info.length if audio_info and audio_info.info else 0
        except Exception:
            # Rough estimate from file size (assume ~1MB per minute for compressed audio)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            duration = file_size_mb * 60  # rough guess

        rate = MODEL_COST_PER_MINUTE.get(model, 0.06)
        transcription_cost = (duration / 60) * rate
        # Estimate analysis cost (summary + content analysis ≈ 2K tokens in + 1K out)
        analysis_cost = (2 * GPT4O_MINI_COST_PER_1K_INPUT) + (1 * GPT4O_MINI_COST_PER_1K_OUTPUT)
        total = transcription_cost + analysis_cost

        return {
            'estimated_duration_sec': duration,
            'transcription_cost': transcription_cost,
            'analysis_cost': analysis_cost,
            'total_cost': total,
            'model': model,
        }

    # ── Transcription ─────────────────────────────────────────────────────────

    def transcribe_with_diarization(self, file_path: Path, model: str = "gpt-4o-transcribe-diarize") -> List[Dict]:
        """Transcribe audio with speaker diarization using OpenAI API."""
        print(f"  Transcribing: {file_path.name}")
        print(f"  Model: {model}")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 25:
            raise ValueError(
                f"File size ({file_size_mb:.1f}MB) exceeds OpenAI's 25MB limit. "
                "Use --extract-audio to compress, or split the file."
            )
        print(f"  File size: {file_size_mb:.1f}MB")

        try:
            with open(file_path, 'rb') as audio_file:
                if model == "gpt-4o-transcribe-diarize":
                    transcript = self.client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                        response_format="diarized_json",
                        chunking_strategy="auto",
                    )
                else:
                    transcript = self.client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                    )

            transcript_dict = transcript.model_dump()

            if 'segments' in transcript_dict:
                segments = transcript_dict['segments']
                for i, seg in enumerate(segments):
                    if 'speaker' not in seg:
                        seg['speaker'] = f"SPEAKER_{i % 2}"
            else:
                segments = [{
                    'speaker': 'Speaker 1',
                    'start': 0,
                    'end': 0,
                    'text': transcript_dict.get('text', ''),
                }]

            return segments

        except Exception as e:
            error_msg = str(e)
            if "model_not_found" in error_msg or "does not have access" in error_msg:
                print(f"\n  ⚠️  Model '{model}' is not available for your account.")
                print("  Trying fallback model 'whisper-1'...\n")
                if model != "whisper-1":
                    return self.transcribe_with_diarization(file_path, model="whisper-1")
            raise RuntimeError(f"OpenAI API transcription failed: {e}")

    # ── Cleaning ──────────────────────────────────────────────────────────────

    @staticmethod
    def clean_transcript(text: str) -> str:
        """Remove filler words and clean up whitespace."""
        cleaned = _filler_pattern.sub('', text)
        cleaned = re.sub(r' {2,}', ' ', cleaned)       # collapse double spaces
        cleaned = re.sub(r' ([,.\?!])', r'\1', cleaned) # fix space before punctuation
        return cleaned.strip()

    # ── Formatting ────────────────────────────────────────────────────────────

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def format_transcript(self, segments: List[Dict], clean: bool = False) -> str:
        """Format segments into readable transcript."""
        output = []
        speaker_map: Dict[str, str] = {}
        speaker_counter = 1

        for segment in segments:
            speaker_id = segment.get('speaker', 'UNKNOWN')
            if speaker_id not in speaker_map:
                speaker_map[speaker_id] = f"Speaker {speaker_counter}"
                speaker_counter += 1

            speaker_name = speaker_map[speaker_id]
            start = self.format_timestamp(segment.get('start', 0))
            end   = self.format_timestamp(segment.get('end', 0))
            text  = segment.get('text', '').strip()
            if clean:
                text = self.clean_transcript(text)

            output.append(f"[{start} - {end}] {speaker_name}:")
            output.append(f"{text}\n")

        return '\n'.join(output)

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_metadata(self, segments: List[Dict]) -> Dict:
        if not segments:
            return {'duration': 0, 'speaker_count': 0, 'speakers': [], 'full_text': ''}

        max_end  = max(seg.get('end', 0) for seg in segments)
        speakers = set(seg.get('speaker', 'UNKNOWN') for seg in segments)
        full_text = ' '.join(seg.get('text', '') for seg in segments)

        return {
            'duration': max_end,
            'speaker_count': len(speakers),
            'speakers': list(speakers),
            'full_text': full_text,
        }

    @staticmethod
    def format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"~{int(seconds)} seconds"
        elif seconds < 3600:
            m = int(seconds / 60)
            return f"~{m} minute{'s' if m != 1 else ''}"
        else:
            h = int(seconds / 3600)
            m = int((seconds % 3600) / 60)
            return f"~{h} hour{'s' if h != 1 else ''} {m} minute{'s' if m != 1 else ''}"

    # ── AI-powered analysis ───────────────────────────────────────────────────

    def _chat(self, system: str, user: str, max_tokens: int = 800) -> tuple[str, int, int]:
        """Helper: call gpt-4o-mini and return (text, input_tokens, output_tokens)."""
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        choice = resp.choices[0].message.content.strip()
        usage  = resp.usage
        return choice, usage.prompt_tokens, usage.completion_tokens

    def generate_summary(self, full_text: str) -> str:
        """One-line topic summary."""
        try:
            text, inp, out = self._chat(
                "You summarize transcripts in one concise sentence.",
                f"Summarize:\n\n{full_text[:2000]}",
                max_tokens=60,
            )
            self.cost.add_chat_tokens(inp, out)
            return text
        except Exception:
            return "Unable to generate summary"

    def generate_advanced_summary(self, full_text: str) -> str:
        """Detailed analysis: key points, action items, decisions, Q&A."""
        prompt = (
            "Analyze the following transcript and produce a structured report with these sections:\n"
            "## Key Points\n- bullet list of the main ideas discussed\n\n"
            "## Action Items\n- bullet list of tasks, owners if mentioned, deadlines if mentioned\n\n"
            "## Decisions Made\n- bullet list of any conclusions or agreements\n\n"
            "## Questions Raised\n- bullet list of open or answered questions\n\n"
            "## Overall Summary\n- 2-3 sentence high-level summary\n\n"
            "If a section has no content, write 'None identified.'\n\n"
            f"TRANSCRIPT:\n{full_text[:6000]}"
        )
        try:
            text, inp, out = self._chat(
                "You are a professional meeting analyst. Output clean Markdown.",
                prompt,
                max_tokens=1200,
            )
            self.cost.add_chat_tokens(inp, out)
            return text
        except Exception:
            return "Unable to generate advanced summary"

    def generate_content_analysis(self, full_text: str) -> str:
        """Topics, keywords, named entities, sentiment."""
        prompt = (
            "Analyze this transcript and return a structured report:\n\n"
            "## Topics Discussed\n- list each topic with a one-line description\n\n"
            "## Keywords\n- comma-separated list of important terms\n\n"
            "## Named Entities\n- People, Companies, Products, Locations mentioned\n\n"
            "## Sentiment Overview\n- overall tone and per-speaker sentiment if multiple speakers\n\n"
            "## Speaking Style\n- pace observations, formality level, notable patterns\n\n"
            f"TRANSCRIPT:\n{full_text[:6000]}"
        )
        try:
            text, inp, out = self._chat(
                "You are a content analyst. Output clean Markdown.",
                prompt,
                max_tokens=1000,
            )
            self.cost.add_chat_tokens(inp, out)
            return text
        except Exception:
            return "Unable to generate content analysis"

    def generate_confidence_report(self, segments: List[Dict]) -> str:
        """Heuristic confidence scoring based on segment characteristics."""
        lines = []
        total_score = 0
        count = 0

        for seg in segments:
            text = seg.get('text', '').strip()
            start = seg.get('start', 0)
            end   = seg.get('end', 0)
            duration = end - start

            # Heuristic scoring
            score = 100
            reasons = []

            # Very short segments with text → may be mis-split
            if duration < 1 and len(text.split()) > 5:
                score -= 15
                reasons.append("high word density for short segment")

            # Very long unbroken segment → may have missed speaker changes
            if duration > 60:
                score -= 10
                reasons.append("long unbroken segment")

            # Few words for long duration → possible silence / music
            word_count = len(text.split())
            if duration > 5 and word_count < 3:
                score -= 20
                reasons.append("sparse text for duration")

            # Contains repeated words → possible stutter / loop
            words = text.lower().split()
            if len(words) > 3 and len(set(words)) < len(words) * 0.5:
                score -= 15
                reasons.append("high word repetition")

            score = max(score, 0)
            total_score += score
            count += 1

            if score < 85:
                ts = self.format_timestamp(start)
                lines.append(f"  [{ts}] Score: {score}/100 — {', '.join(reasons)}")

        avg = total_score / count if count else 0
        header = f"Overall Confidence: {avg:.0f}/100  ({count} segments analyzed)\n"

        if lines:
            header += f"\nLow-confidence segments ({len(lines)}):\n"
            header += '\n'.join(lines)
        else:
            header += "\nAll segments scored 85+ — no concerns detected."

        return header

    # ── Save helpers ──────────────────────────────────────────────────────────

    def save_transcript(self, transcript: str, original_file: Path) -> Path:
        output_file = original_file.with_name(f"{original_file.stem}_transcript.txt")
        output_file.write_text(transcript, encoding='utf-8')
        return output_file

    def save_analysis(self, original_file: Path, advanced_summary: str,
                      content_analysis: str, confidence: str, cost_summary: str) -> Path:
        """Save advanced analysis to a separate _analysis.txt file."""
        output_file = original_file.with_name(f"{original_file.stem}_analysis.txt")
        sections = [
            "=" * 80,
            "ADVANCED ANALYSIS REPORT",
            f"Source: {original_file.name}",
            "=" * 80,
            "",
            "─── DETAILED SUMMARY ───",
            advanced_summary,
            "",
            "─── CONTENT ANALYSIS ───",
            content_analysis,
            "",
            "─── CONFIDENCE REPORT ───",
            confidence,
            "",
            "─── COST ESTIMATE ───",
            cost_summary,
        ]
        output_file.write_text('\n'.join(sections), encoding='utf-8')
        return output_file

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def process_audio(
        self,
        file_path: Path,
        extract_audio: bool = False,
        model: str = "gpt-4o-transcribe-diarize",
        clean: bool = False,
        analyze: bool = True,
    ) -> Dict:
        """Full processing pipeline. Returns a result dict."""
        self.validate_file(file_path)

        suffix = file_path.suffix.lower()
        extracted_audio_path = None

        if suffix in self.SUPPORTED_VIDEO_FORMATS or extract_audio:
            extracted_audio_path = self.extract_audio_from_video(file_path)
            processing_file = extracted_audio_path
        else:
            processing_file = file_path

        # Split into chunks if still over 25 MB or over duration limit
        chunk_paths: List[Path] = []
        chunk_dur = 0
        try:
            chunk_paths, chunk_dur = self._split_audio(processing_file)
        except Exception as e:
            raise RuntimeError(f"Failed to prepare audio: {e}")

        try:
            # Transcribe (possibly across multiple chunks)
            all_segments: List[Dict] = []

            for idx, chunk in enumerate(chunk_paths):
                if len(chunk_paths) > 1:
                    print(f"  -- Chunk {idx + 1}/{len(chunk_paths)} --")
                chunk_segments = self.transcribe_with_diarization(chunk, model=model)

                # Offset timestamps for chunks beyond the first
                if idx > 0 and chunk_dur > 0:
                    time_offset = idx * chunk_dur
                    for seg in chunk_segments:
                        seg['start'] = seg.get('start', 0) + time_offset
                        seg['end']   = seg.get('end', 0)   + time_offset

                all_segments.extend(chunk_segments)

            segments = all_segments
            metadata = self.get_metadata(segments)

            # Track transcription cost
            self.cost.add_transcription(metadata['duration'], model)

            # Format
            formatted = self.format_transcript(segments, clean=clean)
            transcript_file = self.save_transcript(formatted, file_path)

            # One-line topic
            print("  Generating topic summary...")
            topic = self.generate_summary(metadata['full_text'])

            # Advanced analysis
            advanced_summary = ""
            content_analysis = ""
            confidence = ""
            analysis_file = None

            if analyze:
                print("  Generating advanced summary...")
                advanced_summary = self.generate_advanced_summary(metadata['full_text'])

                print("  Generating content analysis...")
                content_analysis = self.generate_content_analysis(metadata['full_text'])

                print("  Calculating confidence scores...")
                confidence = self.generate_confidence_report(segments)

                analysis_file = self.save_analysis(
                    file_path, advanced_summary, content_analysis,
                    confidence, self.cost.summary(),
                )

            return {
                'file': file_path,
                'transcript': formatted,
                'transcript_file': transcript_file,
                'analysis_file': analysis_file,
                'segments': segments,
                'metadata': metadata,
                'topic': topic,
                'advanced_summary': advanced_summary,
                'content_analysis': content_analysis,
                'confidence': confidence,
            }

        finally:
            # Clean up temporary files
            for tmp in chunk_paths:
                if tmp != processing_file and tmp.exists():
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
            if extracted_audio_path and extracted_audio_path.exists():
                try:
                    extracted_audio_path.unlink()
                except Exception:
                    pass


# ══════════════════════════════════════════════════════════════════════════════
# Console output helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_result(transcriber: AudioTranscriber, result: Dict):
    """Pretty-print a single transcription result."""
    meta = result['metadata']

    print("\n" + "=" * 80)
    print("TRANSCRIPTION SUMMARY")
    print("=" * 80)
    print(f"  File:      {result['file'].name}")
    print(f"  Duration:  {transcriber.format_duration(meta['duration'])}")
    sc = meta['speaker_count']
    print(f"  Speakers:  {sc} speaker{'s' if sc != 1 else ''} detected")
    print(f"  Topic:     {result['topic']}")
    print(f"  Cost:      {transcriber.cost.summary()}")

    if result.get('confidence'):
        print(f"\n{'─' * 80}")
        print("CONFIDENCE REPORT")
        print(f"{'─' * 80}")
        print(result['confidence'])

    print(f"\n{'─' * 80}")
    print("FULL TRANSCRIPT")
    print(f"{'─' * 80}\n")
    print(result['transcript'])

    if result.get('advanced_summary'):
        print(f"{'─' * 80}")
        print("ADVANCED SUMMARY")
        print(f"{'─' * 80}")
        print(result['advanced_summary'])

    if result.get('content_analysis'):
        print(f"\n{'─' * 80}")
        print("CONTENT ANALYSIS")
        print(f"{'─' * 80}")
        print(result['content_analysis'])

    print("=" * 80)
    print(f"\nTranscript saved to:   {result['transcript_file']}")
    if result.get('analysis_file'):
        print(f"Analysis saved to:     {result['analysis_file']}")


def print_batch_summary(transcriber: AudioTranscriber, results: List[Dict]):
    """Print summary table for batch processing."""
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"  Files processed: {len(results)}")
    print(f"  Total cost:      {transcriber.cost.summary()}")
    print()

    for i, r in enumerate(results, 1):
        meta = r['metadata']
        print(f"  {i}. {r['file'].name}")
        print(f"     Duration: {transcriber.format_duration(meta['duration'])}  |  "
              f"Speakers: {meta['speaker_count']}  |  Topic: {r['topic'][:60]}")
        print(f"     → {r['transcript_file']}")
        if r.get('analysis_file'):
            print(f"     → {r['analysis_file']}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Interactive mode
# ══════════════════════════════════════════════════════════════════════════════

def interactive_mode(transcriber: AudioTranscriber, model: str):
    """Interactive REPL for processing files."""
    print("\n" + "=" * 80)
    print("  INTERACTIVE TRANSCRIPTION MODE")
    print("  Type a file/folder path, or a command below.")
    print("  Commands: help, cost, quit")
    print("=" * 80 + "\n")

    clean = False
    analyze = True
    extract = False

    while True:
        try:
            user_input = input("transcribe> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        elif cmd == 'help':
            print("""
  Commands:
    <file_path>         Transcribe a single file
    <folder_path>       Batch-transcribe all audio/video files in folder
    clean on/off        Toggle filler-word removal (current: {})
    analyze on/off      Toggle advanced analysis (current: {})
    extract on/off      Toggle force audio extraction (current: {})
    cost                Show total session cost so far
    help                Show this help
    quit                Exit interactive mode
""".format('ON' if clean else 'OFF', 'ON' if analyze else 'OFF', 'ON' if extract else 'OFF'))

        elif cmd == 'cost':
            print(f"  Session cost: {transcriber.cost.summary()}\n")

        elif cmd.startswith('clean '):
            val = cmd.split()[1]
            clean = val in ('on', 'true', '1', 'yes')
            print(f"  Filler-word cleaning: {'ON' if clean else 'OFF'}\n")

        elif cmd.startswith('analyze '):
            val = cmd.split()[1]
            analyze = val in ('on', 'true', '1', 'yes')
            print(f"  Advanced analysis: {'ON' if analyze else 'OFF'}\n")

        elif cmd.startswith('extract '):
            val = cmd.split()[1]
            extract = val in ('on', 'true', '1', 'yes')
            print(f"  Force audio extraction: {'ON' if extract else 'OFF'}\n")

        else:
            path = Path(user_input.strip('"').strip("'"))

            if path.is_dir():
                # Batch process
                results = batch_process(
                    transcriber, path, model=model,
                    clean=clean, analyze=analyze, extract_audio=extract,
                )
                if results:
                    print_batch_summary(transcriber, results)
                else:
                    print("  No supported files found in directory.\n")

            elif path.is_file():
                try:
                    result = transcriber.process_audio(
                        path, extract_audio=extract, model=model,
                        clean=clean, analyze=analyze,
                    )
                    print_result(transcriber, result)
                except Exception as e:
                    print(f"  Error: {e}\n")
            else:
                print(f"  Path not found: {path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Batch processing
# ══════════════════════════════════════════════════════════════════════════════

def batch_process(
    transcriber: AudioTranscriber,
    folder: Path,
    model: str,
    clean: bool = False,
    analyze: bool = True,
    extract_audio: bool = False,
) -> List[Dict]:
    """Process all supported audio/video files in a folder."""
    all_formats = transcriber.SUPPORTED_AUDIO_FORMATS | transcriber.SUPPORTED_VIDEO_FORMATS
    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in all_formats
    )

    if not files:
        return []

    print(f"\n  Found {len(files)} file(s) in {folder}\n")

    results = []
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
        print("-" * 60)
        try:
            result = transcriber.process_audio(
                file_path, extract_audio=extract_audio,
                model=model, clean=clean, analyze=analyze,
            )
            results.append(result)
            print(f"  ✓ Done — {result['topic'][:60]}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Transcribe audio/video with speaker diarization, analysis & more',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python transcribe.py meeting.mp4
  python transcribe.py meeting.mp4 --clean
  python transcribe.py ./recordings/ --batch
  python transcribe.py --interactive
  python transcribe.py meeting.mp4 --no-analysis
  python transcribe.py meeting.mp4 --estimate-cost
        """,
    )
    parser.add_argument(
        'file', nargs='?', type=str, default=None,
        help='Path to audio/video file, or folder (with --batch)',
    )
    parser.add_argument('--api-key', type=str, help='OpenAI API key')
    parser.add_argument(
        '--model', type=str, default='gpt-4o-transcribe-diarize',
        choices=['gpt-4o-transcribe-diarize', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe', 'whisper-1'],
        help='Transcription model (default: gpt-4o-transcribe-diarize)',
    )
    parser.add_argument('--extract-audio', action='store_true',
                        help='Force audio extraction from video')
    parser.add_argument('--clean', action='store_true',
                        help='Remove filler words (um, uh, like, you know, etc.)')
    parser.add_argument('--no-analysis', action='store_true',
                        help='Skip advanced summary and content analysis')
    parser.add_argument('--batch', action='store_true',
                        help='Process all supported files in a folder')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Launch interactive mode')
    parser.add_argument('--estimate-cost', action='store_true',
                        help='Show estimated cost without processing')

    args = parser.parse_args()

    try:
        transcriber = AudioTranscriber(api_key=args.api_key)

        # Interactive mode
        if args.interactive:
            interactive_mode(transcriber, model=args.model)
            return

        if not args.file:
            parser.print_help()
            sys.exit(1)

        file_path = Path(args.file)
        analyze = not args.no_analysis

        # Cost estimation only
        if args.estimate_cost:
            if file_path.is_dir():
                all_fmts = transcriber.SUPPORTED_AUDIO_FORMATS | transcriber.SUPPORTED_VIDEO_FORMATS
                files = [f for f in file_path.iterdir() if f.suffix.lower() in all_fmts]
            else:
                files = [file_path]

            total = 0
            print("\nCost Estimate:")
            print("-" * 50)
            for f in files:
                est = transcriber.estimate_cost(f, args.model)
                print(f"  {f.name}: ~${est['total_cost']:.4f} "
                      f"(~{est['estimated_duration_sec']:.0f}s, {args.model})")
                total += est['total_cost']
            print("-" * 50)
            print(f"  Estimated total: ~${total:.4f}\n")
            return

        # Batch mode
        if args.batch or file_path.is_dir():
            if not file_path.is_dir():
                print(f"Error: {file_path} is not a directory.", file=sys.stderr)
                sys.exit(1)

            results = batch_process(
                transcriber, file_path, model=args.model,
                clean=args.clean, analyze=analyze,
                extract_audio=args.extract_audio,
            )
            if results:
                print_batch_summary(transcriber, results)
            else:
                print("No supported files found in the directory.")
            return

        # Single file
        result = transcriber.process_audio(
            file_path, extract_audio=args.extract_audio,
            model=args.model, clean=args.clean, analyze=analyze,
        )
        print_result(transcriber, result)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
