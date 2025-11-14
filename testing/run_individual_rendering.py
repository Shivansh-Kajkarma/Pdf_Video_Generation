import sys
import os
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.phase4_video_generation.renderer import render_video
from app.config import settings  
from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def run_test():
    """
    Runs only the video rendering for a specific, existing job.
    """
    logger.info("--- Starting Individual Render Test ---")
    
   
    job_name = "ikigai_20251114_1841"
    
    job_dir = PROJECT_ROOT / "jobs" / job_name
    

    audio_path = job_dir / f"{job_name}_processed_audio.mp3"
    timestamps_path = job_dir / f"{job_name}_timestamps.json"
    

    output_path = job_dir / f"{job_name}_final_video_FIXED.mp4"

    # --- 4. Safety Check ---
    if not audio_path.exists():
        logger.error(f"FATAL: Audio file not found: {audio_path}")
        return
    if not timestamps_path.exists():
        logger.error(f"FATAL: Timestamps file not found: {timestamps_path}")
        return

    logger.info(f"Using Audio file:      {audio_path}")
    logger.info(f"Using Timestamps file: {timestamps_path}")
    logger.info(f"Saving New Video to: {output_path}")

    # --- 5. Call the Renderer ---
    try:
        render_video(
            audio_path=audio_path,
            timestamps_path=timestamps_path,
            output_path=output_path
        )
        logger.info(f"--- SUCCESS! New video saved to {output_path} ---")
        
    except Exception as e:
        logger.error(f"---!!! RENDER FAILED !!! {e}---", exc_info=True)

if __name__ == "__main__":
    run_test()