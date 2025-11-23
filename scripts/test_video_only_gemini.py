import sys
import logging
from pathlib import Path
import time
from datetime import datetime

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from app.config import settings
from app.logging_config import setup_logging

# Import V2 modules
from app.phase2_ai_services.audio_generator_gemini import GeminiAudioService
from app.phase3_audio_processing.mastering import master_audio
from app.phase4_video_generation.renderer_v2 import render_video_v2

def main():
    start_time = time.time()
    job_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir = settings.JOBS_OUTPUT_PATH / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(job_id=job_id, log_level="INFO")
    logger = logging.getLogger(__name__)
    
    print(f"\n🚀 STARTING TEST: {job_id}") # Force console output
    
    try:
        # 1. Script
        script_path = script_dir / "stress_script.txt"
        print(str(script_path))
        if script_path.exists():
            text = script_path.read_text(encoding='utf-8')
        else:
            text = """Stress: Accused of killing longevity.
Many people seem older than they are. Research into the causes of premature aging has shown that stress has a lot to do with it, because the body wears down much faster during periods of crisis.
The American Institute of Stress investigated this degenerative process and concluded that most health problems are caused by stress.
Researchers at the Heidelberg University Hospital conducted a study in which they subjected a young doctor to a job interview, which they made even more stressful by forcing him to solve complex math problems for thirty minutes.
Afterward, they took a blood sample. What they discovered was that his antibodies had reacted to stress the same way they react to pathogens, activating the proteins that trigger an immune response.
The problem is that this response not only neutralizes harmful agents, it also damages healthy cells, leading them to age prematurely."""
            
        # 2. Audio (Gemini Puck)
        print("🔊 Generating Audio (Gemini)...")
        audio_service = GeminiAudioService()
        raw_audio, timestamps = audio_service.generate(text, job_dir, job_id)
        
        # 3. Master
        print("🎛️ Mastering Audio...")
        mastered_audio = job_dir / f"{job_id}_mastered.mp3"
        mastered_audio = master_audio(raw_audio, mastered_audio)
        
        # 4. Video (V2)
        print("🎬 Rendering Video (Founders Style)...")
        final_video = job_dir / f"{job_id}_final.mp4"
        render_video_v2(mastered_audio, timestamps, final_video)
        
        print(f"\n✅ DONE! Video: {final_video}")
        
    except Exception as e:
        logger.error("Test Failed", exc_info=True)
        print(f"\n❌ FAILED: {e}")

if __name__ == "__main__":
    main()