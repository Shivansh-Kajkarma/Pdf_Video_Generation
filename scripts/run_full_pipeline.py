import sys
import logging
from pathlib import Path
import time
from datetime import datetime
import argparse
import json

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from app.config import settings
from app.logging_config import setup_logging

from app.phase1_pdf_processing.service import PDFExtractorService
from app.phase1_pdf_processing.image_extractor import extract_images
from app.phase1_pdf_processing.text_cleaner import clean_text

from app.phase2_ai_services.openai_client import OpenAIService, detect_book_genre
from app.phase2_ai_services.book_summary import generate_book_summary
from app.phase3_audio_processing.mastering import master_audio
from app.phase4_video_generation.renderer import render_video


def main(pdf_file_path: Path):
    start_time = time.time()
    
    # --- A. Setup ---
    job_id = f"{pdf_file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    job_dir = settings.JOBS_OUTPUT_PATH / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(job_id=job_id, log_level="INFO")
    logger = logging.getLogger(__name__)
    
    logger.info(f"--- STARTING FULL PIPELINE FOR JOB: {job_id} ---")
    logger.info(f"Input PDF: {pdf_file_path}")
    logger.info(f"Job output will be in: {job_dir}")

    try:
        # ===== PHASE 1: PDF PROCESSING =====
        logger.info("--- PHASE 1: PDF Processing (with Adaptive Logic) ---")
        
        # 1.1: Run full text/table/index service
        extractor_service = PDFExtractorService(output_dir=settings.JOBS_OUTPUT_PATH)
        extraction_result = extractor_service.extract_from_pdf(
            pdf_path=str(pdf_file_path),
            job_id=job_id
        )
        book_type = extraction_result.get("book_type", "unknown")
        
        # Get the paths to the files service just created
        raw_text_path = Path(extraction_result["output_files"]["full_text"])
        
        # Handle case where no tables are found (tables_directory might be None)
        tables_dir_path = extraction_result["output_files"].get("tables_directory")
        if tables_dir_path:
            tables_dir = Path(tables_dir_path)
        else:
            # Create empty tables directory if none exists
            tables_dir = job_dir / "tables"
            tables_dir.mkdir(exist_ok=True)
            logger.info("No tables found, using empty tables directory")
        
        # 1.2: Run image extraction logic
        images_dir = extract_images(pdf_file_path, job_dir)
        
        logger.info(f"Book type detected: {book_type}")
        logger.info(f"Tables found: {extraction_result['summary']['tables_count']}")

        # Optionally filter text to a specific page range (current request: page 50 only)
        start_page = 50
        end_page = 50
        logger.warning(f"Filtering text to pages {start_page}-{end_page} for main video.")
        json_path = Path(extraction_result["output_files"]["json"])
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filtered_text = ""
        for page in data['text_extraction']['pages']:
            if start_page <= page['page_number'] <= end_page:
                filtered_text += page['text'] + "\n\n"

        if not filtered_text.strip():
            raise ValueError(f"No text found for pages {start_page}-{end_page}.")

        raw_text_path = job_dir / f"filtered_pages_{start_page}_to_{end_page}.txt"
        with open(raw_text_path, 'w', encoding='utf-8') as f:
            f.write(filtered_text)
        logger.info(f"Filtered text saved to {raw_text_path}")

        # ===== PHASE 1.5: TEXT CLEANING =====
        logger.info("--- PHASE 1.5: Text Cleaning ---")
        cleaned_script_path = clean_text(
            raw_text_path=raw_text_path,
            tables_dir=tables_dir,
            images_dir=images_dir,
            job_dir=job_dir
        )
        
        with open(cleaned_script_path, 'r', encoding='utf-8') as f:
            text_script = f.read()
            if not text_script.strip():
                raise ValueError("Cleaned script is empty. Cannot proceed.")
        
        logger.info(f"Using cleaned text for audio/video generation ({len(text_script)} characters)")
        
        # ===== PHASE 2: AI SERVICES =====
        logger.info("--- PHASE 2: AI Services ---")
        
        # Detect book genre from PDF filename
        logger.info("--- Detecting book genre ---")
        book_title = pdf_file_path.stem
        genre = detect_book_genre(book_title)
        logger.info(f"Detected genre: {genre}")

        job_metadata = {
            "job_id": job_id,
            "pdf_path": str(pdf_file_path),
            "book_title": book_title,
            "book_type": book_type,
            "genre": genre,
            "created_at": datetime.now().isoformat()
        }
        metadata_path = job_dir / "job_metadata.json"

        def _write_metadata():
            with open(metadata_path, 'w', encoding='utf-8') as meta_file:
                json.dump(job_metadata, meta_file, indent=2)

        _write_metadata()
        
        openai_service = OpenAIService(voice="onyx")
        raw_audio_path, timestamps_path = openai_service.generate_audio_with_timestamps(
            text=text_script,  # Using cleaned text (image references removed)
            output_dir=job_dir,
            job_id=job_id,
            genre=genre
        )

        # ===== PHASE 3: AUDIO PROCESSING =====
        logger.info("--- PHASE 3: Audio Mastering ---")
        processed_audio_path = job_dir / f"{job_id}_processed_audio.mp3"
        processed_audio_path = master_audio(
            raw_audio_path=raw_audio_path,
            processed_audio_path=processed_audio_path
        )

        # ===== PHASE 4: VIDEO GENERATION =====
        logger.info("--- PHASE 4: Video Rendering ---")
        final_video_path = job_dir / f"{job_id}_final_video.mp4"
        final_video_path = render_video(
            audio_path=processed_audio_path,
            timestamps_path=timestamps_path,
            output_path=final_video_path
        )
        
        end_time = time.time()
        logger.info(f"--- FULL PIPELINE SUCCESS (Total time: {end_time - start_time:.2f}s) ---")
        logger.info(f"Final Video: {final_video_path}")

        # ===== SUMMARY GENERATION =====
        summary_text = None
        summary_stats = None
        summary_path = job_dir / f"{job_id}_summary.txt"

        try:
            logger.info("--- SUMMARY GENERATION (Target ~1 hour narration) ---")
            summary_text, summary_stats = generate_book_summary(
                book_text=text_script,
                book_title=book_title,
                genre=genre,
                book_type=book_type,
                target_word_count=settings.SUMMARY_TARGET_WORDS,
                max_word_count=settings.SUMMARY_MAX_WORDS
            )
            summary_path.write_text(summary_text, encoding='utf-8')
            logger.info(
                f"Summary saved to {summary_path} (~{summary_stats['word_count']} words, est {summary_stats['estimated_minutes']} min)."
            )
            job_metadata["summary"] = {
                "path": str(summary_path),
                **summary_stats
            }
            _write_metadata()
        except Exception as summary_error:
            logger.error("Summary generation failed. Skipping summary video prompt.", exc_info=True)

        # ===== OPTIONAL SUMMARY VIDEO =====
        if summary_text:
            try:
                user_choice = input(
                    "Summary ready. Create ~1 hour summary video as well? (yes/no): "
                ).strip().lower()
            except EOFError:
                user_choice = "no"

            if user_choice in ("yes", "y"):
                logger.info("--- SUMMARY VIDEO PIPELINE ---")
                summary_job_dir = job_dir / "summary_video"
                summary_job_dir.mkdir(exist_ok=True)
                summary_job_id = f"{job_id}_summary"

                summary_raw_audio_path, summary_timestamps_path = openai_service.generate_audio_with_timestamps(
                    text=summary_text,
                    output_dir=summary_job_dir,
                    job_id=summary_job_id,
                    genre=genre
                )

                summary_processed_audio_path = summary_job_dir / f"{summary_job_id}_processed_audio.mp3"
                summary_processed_audio_path = master_audio(
                    raw_audio_path=summary_raw_audio_path,
                    processed_audio_path=summary_processed_audio_path
                )

                summary_final_video_path = summary_job_dir / f"{summary_job_id}_final_video.mp4"
                summary_final_video_path = render_video(
                    audio_path=summary_processed_audio_path,
                    timestamps_path=summary_timestamps_path,
                    output_path=summary_final_video_path
                )

                logger.info(f"Summary video created: {summary_final_video_path}")
                job_metadata["summary_video"] = str(summary_final_video_path)
                _write_metadata()
            else:
                logger.info("User skipped summary video generation.")
        
    except Exception as e:
        logger.error(f"--- FULL PIPELINE FAILED {e} ---", exc_info=True)
        end_time = time.time()
        logger.error(f"Failed after {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full PDF-to-Video pipeline.")
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file.")
    args = parser.parse_args()
    
    input_pdf = Path(args.pdf_path)
    if not input_pdf.exists():
        print(f"Error: PDF file not found at {input_pdf}")
        sys.exit(1)
        
    main(pdf_file_path=input_pdf)