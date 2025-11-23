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
# from app.phase2_ai_services.audio_generator_gemini import GeminiAudioService
from app.phase2_ai_services.openai_client import OpenAIService
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
            # text = """"If they had uncles enough to fill ALL Cheapside," cried Bingley, "it would not make them one jot less agreeable." "But it must very materially lessen their chance of marrying men of any consideration in the world," replied Darcy. To this speech Bingley made no answer; but his sisters gave it their hearty assent, and indulged their mirth for some time at the expense of their dear friend's vulgar relations. With a renewal of tenderness, however, they returned to her room on leaving the dining-parlour, and sat with her till summoned to coffee. She was still very poorly, and Elizabeth would not quit her at all, till late in the evening, when she had the comfort of seeing her sleep, and when it seemed to her rather right than pleasant that she should go downstairs herself. On entering the drawing-room she found the whole party at loo, and was immediately invited to join them; but suspecting them to be playing high she declined it, and making her sister the excuse, said she would amuse herself for the short time she could stay below, with a book. Mr. Hurst looked at her with astonishment. "Do you prefer reading to cards?" said he; "that is rather singular." "Miss Eliza Bennet," said Miss Bingley, "despises cards."""
#             text = """Stress: Accused of killing longevity.
# Many people seem older than they are. Research into the causes of premature aging has shown that stress has a lot to do with it, because the body wears down much faster during periods of crisis.
# The American Institute of Stress investigated this degenerative process and concluded that most health problems are caused by stress.
# Researchers at the Heidelberg University Hospital conducted a study in which they subjected a young doctor to a job interview, which they made even more stressful by forcing him to solve complex math problems for thirty minutes.
# Afterward, they took a blood sample. What they discovered was that his antibodies had reacted to stress the same way they react to pathogens, activating the proteins that trigger an immune response.
# The problem is that this response not only neutralizes harmful agents, it also damages healthy cells, leading them to age prematurely.
# The University of California conducted a similar study, taking data and samples from thirty-nine women who had high levels of stress due to the illness of one of their children and comparing them to samples from women with healthy children and low levels of stress.
# They found that stress promotes cellular aging by weakening cell structures known as telomeres, which affect cellular regeneration and how our cells age.
# As the study revealed, the greater the stress, the greater the degenerative effect on cells."""
            text = """
            I found a USB drive taped under a park bench… with a note saying: ‘Please return this to me. I’m running out of time.’
            I plug it into my laptop. There’s only ONE file. Named… ‘DON’T WATCH.’ So obviously… I open it.
            The video shows my ROOM. My desk. My chair. Recorded last night. While I was sleeping.
            """
        # 1. AI Service (Text -> Audio + Timestamps)
        logger.info("Step 1: Calling OpenAIService...")
        openai_service = OpenAIService(voice="onyx")
        raw_audio_path, timestamps_path = openai_service.generate_audio_with_timestamps(
            text=text,
            output_dir=job_dir,
            job_id=job_id
        )
        logger.info(f"Raw audio at: {raw_audio_path}")
        logger.info(f"Timestamps at: {timestamps_path}")
        
        raw_audio = raw_audio_path
        timestamps = timestamps_path

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