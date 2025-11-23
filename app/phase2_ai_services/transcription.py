import json
import logging
from pathlib import Path
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

def restore_punctuation(data):
    """
    EXACT COPY OF YOUR PUNCTUATION LOGIC
    Maps punctuation from the full text back onto the word-level timestamps.
    """
    full_text = data.get("text", "")
    words = data.get("words", [])
    
    if not full_text or not words:
        return data

    current_idx = 0
    
    for w in words:
        word_str = w['word'].strip()
        match_idx = full_text.find(word_str, current_idx)
        
        if match_idx == -1:
            continue
            
        end_idx = match_idx + len(word_str)
        
        if end_idx < len(full_text):
            next_char = full_text[end_idx]
            if next_char in [".", ",", "!", "?", ";", ":"]:
                w['word'] = word_str + next_char
                current_idx = end_idx + 1 
            else:
                current_idx = end_idx 
        else:
            current_idx = end_idx

    data['words'] = words
    return data

class TranscriptionService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate_timestamps(self, audio_path: Path, output_path: Path) -> Path:
        """
        Takes ANY audio file (Gemini, OpenAI, or Custom) and generates timestamps 
        using the shared, correct punctuation logic.
        """
        logger.info(f"Generating timestamps for: {audio_path.name}")
        
        # 1. Whisper Transcription
        with open(audio_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"]
            )
        
        data = transcription.model_dump()
        
        # 2. Apply Shared Punctuation Logic
        data = restore_punctuation(data)
        
        # 3. Save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Timestamps saved to {output_path}")
        return output_path