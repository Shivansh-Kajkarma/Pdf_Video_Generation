import logging
import sys
from pathlib import Path

def setup_logging(job_id: str = None, log_level: str = "INFO"):
    """
    Configures the root logger.
    
    If job_id is provided, it will log to a file inside that job's dir.
    Otherwise, it just logs to the console.
    """
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # --- Formatter ---
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # --- Console Handler  ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File Handler (for the server) ---
    if job_id:
        from app.config import settings # Import here to avoid circular imports
        job_dir = settings.JOBS_OUTPUT_PATH / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        log_file_path = job_dir / f"{job_id}_run.log"
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging configured. Also logging to file: {log_file_path}")
    else:
        logger.info("Logging configured. Logging to console only.")