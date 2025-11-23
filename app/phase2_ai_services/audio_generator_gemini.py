import logging
import os
from pathlib import Path
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class GeminiAudioService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key, http_options={'api_version': 'v1alpha'})

    def generate(self, text: str, output_path: Path, job_id: str) -> Path:
        """Generates Audio ONLY."""
        logger.info(f"Job {job_id}: Generating Gemini Audio...")
        
        prompt = f"Read this text naturally and clearly: {text}"
        
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Puck"
                        )
                    )
                )
            )
        )

        if response.parts and response.parts[0].inline_data:
            import wave
            with wave.open(str(output_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(response.parts[0].inline_data.data)
            return output_path
        else:
            raise Exception("Gemini returned no audio data.")