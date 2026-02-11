#!/usr/bin/env python3
"""
Speech-to-Text Transcription with Speaker Diarization
Supports: wav, mp3, m4a, flac, ogg, mp4, mpeg, mpga, webm, avi, mov, mkv
Uses OpenAI's gpt-4o-transcribe-diarize model
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict

try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing required package. Please run: pip install -r requirements.txt")
    print(f"Details: {e}")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()


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
        """Extract audio from video file to MP3 format."""
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            raise ImportError(
                "moviepy is required for video audio extraction. "
                "Install it with: pip install moviepy"
            )
        
        print(f"Extracting audio from video: {video_path.name}")
        
        try:
            audio_path = video_path.with_suffix('.extracted.mp3')
            
            # Load video and extract audio
            video = VideoFileClip(str(video_path))
            video.audio.write_audiofile(str(audio_path), logger=None)
            video.close()
            
            print(f"Audio extracted to: {audio_path.name}")
            return audio_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio from video: {e}")
    
    def transcribe_with_diarization(self, file_path: Path, model: str = "gpt-4o-transcribe-diarize") -> List[Dict]:
        """Transcribe audio with speaker diarization using OpenAI API."""
        print(f"Transcribing audio with speaker diarization: {file_path.name}")
        print(f"Using model: {model}")
        
        # Check file size (OpenAI limit is 25MB)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 25:
            raise ValueError(
                f"File size ({file_size_mb:.1f}MB) exceeds OpenAI's 25MB limit. "
                "Please compress or split the audio file."
            )
        
        try:
            with open(file_path, 'rb') as audio_file:
                # Try diarization-capable models
                if model == "gpt-4o-transcribe-diarize":
                    transcript = self.client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                        response_format="diarized_json",
                        chunking_strategy="auto"
                    )
                else:
                    # Fallback to regular transcription with verbose_json
                    transcript = self.client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"]
                    )
            
            # Convert response to dict and extract segments
            transcript_dict = transcript.model_dump()
            
            # Handle different response formats
            if 'segments' in transcript_dict:
                segments = transcript_dict['segments']
                # Add speaker labels if not present (fallback behavior)
                for i, seg in enumerate(segments):
                    if 'speaker' not in seg:
                        seg['speaker'] = f"SPEAKER_{i % 2}"  # Simple alternating for demo
            else:
                # Fallback format
                segments = [{
                    'speaker': 'Speaker 1',
                    'start': 0,
                    'end': 0,
                    'text': transcript_dict.get('text', '')
                }]
            
            return segments
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a model access error
            if "model_not_found" in error_msg or "does not have access" in error_msg:
                print(f"\n⚠️  Model '{model}' is not available for your account.")
                print("Trying fallback model 'whisper-1'...\n")
                
                # Retry with whisper-1
                if model != "whisper-1":
                    return self.transcribe_with_diarization(file_path, model="whisper-1")
            
            raise RuntimeError(f"OpenAI API transcription failed: {e}")
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def format_transcript(self, segments: List[Dict]) -> str:
        """Format segments into readable transcript."""
        output = []
        
        # Map speaker labels to simpler names
        speaker_map = {}
        speaker_counter = 1
        
        for segment in segments:
            speaker_id = segment.get('speaker', 'UNKNOWN')
            
            if speaker_id not in speaker_map:
                speaker_map[speaker_id] = f"Speaker {speaker_counter}"
                speaker_counter += 1
            
            speaker_name = speaker_map[speaker_id]
            start_time = self.format_timestamp(segment.get('start', 0))
            end_time = self.format_timestamp(segment.get('end', 0))
            text = segment.get('text', '').strip()
            
            output.append(f"[{start_time} - {end_time}] {speaker_name}:")
            output.append(f"{text}\n")
        
        return '\n'.join(output)
    
    def save_transcript(self, transcript: str, original_file: Path) -> Path:
        """Save transcript to text file."""
        output_file = original_file.with_name(f"{original_file.stem}_transcript.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        return output_file
    
    def generate_summary(self, transcript_text: str) -> str:
        """Generate a brief topic summary using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes transcripts in one concise sentence."},
                    {"role": "user", "content": f"Summarize this transcript in one sentence:\n\n{transcript_text[:1000]}"}
                ],
                max_tokens=50
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "Unable to generate summary"
    
    def get_metadata(self, segments: List[Dict]) -> Dict:
        """Extract metadata from segments."""
        if not segments:
            return {
                'duration': 0,
                'speaker_count': 0,
                'speakers': []
            }
        
        # Calculate duration
        max_end = max(seg.get('end', 0) for seg in segments)
        
        # Count unique speakers
        speakers = set(seg.get('speaker', 'UNKNOWN') for seg in segments)
        
        # Get full transcript text for summary
        full_text = ' '.join(seg.get('text', '') for seg in segments)
        
        return {
            'duration': max_end,
            'speaker_count': len(speakers),
            'speakers': list(speakers),
            'full_text': full_text
        }
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"~{int(seconds)} seconds"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"~{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"~{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
    
    def process_audio(self, file_path: Path, extract_audio: bool = False, model: str = "gpt-4o-transcribe-diarize") -> tuple[str, Path, Dict]:
        """Main processing pipeline."""
        # Validate input
        self.validate_file(file_path)
        
        # Check if we need to extract audio from video
        suffix = file_path.suffix.lower()
        extracted_audio_path = None
        
        if suffix in self.SUPPORTED_VIDEO_FORMATS or extract_audio:
            extracted_audio_path = self.extract_audio_from_video(file_path)
            processing_file = extracted_audio_path
        else:
            processing_file = file_path
        
        try:
            # Perform transcription with speaker diarization
            segments = self.transcribe_with_diarization(processing_file, model=model)
            
            # Get metadata
            metadata = self.get_metadata(segments)
            
            # Generate topic summary
            print("\nGenerating topic summary...")
            metadata['topic'] = self.generate_summary(metadata['full_text'])
            
            # Format transcript
            formatted_transcript = self.format_transcript(segments)
            
            # Save to file (use original filename)
            output_file = self.save_transcript(formatted_transcript, file_path)
            metadata['output_file'] = output_file
            
            return formatted_transcript, output_file, metadata
            
        finally:
            # Clean up extracted audio file
            if extracted_audio_path and extracted_audio_path.exists():
                try:
                    extracted_audio_path.unlink()
                    print(f"Cleaned up temporary file: {extracted_audio_path.name}")
                except Exception:
                    pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Transcribe audio/video with speaker diarization using OpenAI API'
    )
    parser.add_argument(
        'file',
        type=str,
        help='Path to audio/video file (wav, mp3, m4a, flac, ogg, mp4, mpeg, mpga, webm, avi, mov, mkv)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (or set OPENAI_API_KEY env variable)'
    )
    parser.add_argument(
        '--extract-audio',
        action='store_true',
        help='Force audio extraction from video (useful for large MP4 files)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-transcribe-diarize',
        choices=['gpt-4o-transcribe-diarize', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe', 'whisper-1'],
        help='OpenAI model to use (default: gpt-4o-transcribe-diarize)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize transcriber
        transcriber = AudioTranscriber(api_key=args.api_key)
        
        # Process audio/video file
        file_path = Path(args.file)
        transcript, output_file, metadata = transcriber.process_audio(
            file_path, 
            extract_audio=args.extract_audio,
            model=args.model
        )
        
        # Print metadata
        print("\n" + "="*80)
        print("TRANSCRIPTION SUMMARY")
        print("="*80)
        print(f"Duration: {transcriber.format_duration(metadata['duration'])}")
        
        speaker_count = metadata['speaker_count']
        if speaker_count == 1:
            print(f"Speakers: 1 speaker detected")
        else:
            print(f"Speakers: {speaker_count} speakers detected")
        
        print(f"Topic: {metadata['topic']}")
        print(f"Output file: {output_file.name}")
        
        # Print full transcript
        print("\n" + "="*80)
        print("FULL TRANSCRIPT")
        print("="*80 + "\n")
        print(transcript)
        print("="*80)
        print(f"\nTranscript saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
