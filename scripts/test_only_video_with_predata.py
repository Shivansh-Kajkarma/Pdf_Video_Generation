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

# Import only video renderer
from app.phase4_video_generation.renderer_v2 import render_video_v2


def main():
    """
    Test video rendering ONLY using pre-existing audio and timestamps from a completed job.

    Usage:
        python scripts/test_only_video_with_predata.py [job_folder_path]

    If no path provided, uses the hardcoded default path below.
    """
    start_time = time.time()

    # Hardcoded job folder path (can be overridden via command line)
    # DEFAULT_JOB_PATH = "/home/rareboy/Internship/Kajkarma/Pdf_Video_Generation/jobs/test_20251120_225601"
    #DEFAULT_JOB_PATH = "/home/rareboy/Internship/Kajkarma/Pdf_Video_Generation/jobs/test_20251120_231007"
    DEFAULT_JOB_PATH = "/home/rareboy/Internship/Kajkarma/Pdf_Video_Generation/jobs/test_20251121_214414"
    

    # Check if user provided a job path as argument
    if len(sys.argv) > 1:
        job_folder = Path(sys.argv[1])
    else:
        job_folder = Path(DEFAULT_JOB_PATH)

    if not job_folder.exists():
        print(f"❌ ERROR: Job folder not found: {job_folder}")
        print(
            f"Please provide a valid job folder path or update DEFAULT_JOB_PATH in the script."
        )
        sys.exit(1)

    # Create new test output folder
    test_id = f"render_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_output_dir = settings.JOBS_OUTPUT_PATH / test_id
    test_output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(job_id=test_id, log_level="INFO")
    logger = logging.getLogger(__name__)

    print(f"\n🎬 STARTING RENDER TEST: {test_id}")
    print(f"📁 Loading data from: {job_folder}")

    try:
        # 1. Find mastered audio file
        mastered_audio = None
        for audio_file in job_folder.glob("*_mastered.mp3"):
            mastered_audio = audio_file
            break

        if not mastered_audio or not mastered_audio.exists():
            print(
                f"❌ ERROR: No mastered audio file (*_mastered.mp3) found in {job_folder}"
            )
            sys.exit(1)

        print(f"🔊 Found audio: {mastered_audio.name}")

        # 2. Find timestamps file
        timestamps_file = None
        for ts_file in job_folder.glob("*_timestamps.json"):
            timestamps_file = ts_file
            break

        if not timestamps_file or not timestamps_file.exists():
            print(
                f"❌ ERROR: No timestamps file (*_timestamps.json) found in {job_folder}"
            )
            sys.exit(1)

        print(f"⏱️  Found timestamps: {timestamps_file.name}")

        # 3. Render Video with V2 renderer
        print(f"\n🎬 Rendering Video (Founders Style)...")
        final_video = test_output_dir / f"{test_id}_final.mp4"

        render_video_v2(mastered_audio, timestamps_file, final_video)

        elapsed_time = time.time() - start_time
        print(f"\n✅ RENDER TEST COMPLETE!")
        print(f"📹 Video saved: {final_video}")
        print(f"⏱️  Time taken: {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error("Render Test Failed", exc_info=True)
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
