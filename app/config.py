import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# This is the root directory of *entire* project
PROJECT_ROOT = Path(__file__).parent.parent

# Explicitly load .env file using python-dotenv as a fallback
_env_file_path = PROJECT_ROOT / ".env"
if _env_file_path.exists():
    load_dotenv(_env_file_path, override=True) 

class Settings(BaseSettings):
    """
    Main application settings. Loads from .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        case_sensitive=False,
        env_file_encoding='utf-8',
        env_ignore_empty=True,
        extra='ignore'  # Ignore extra fields from .env file
    )
    
    # --- Project Paths ---
    ASSETS_PATH: Path = PROJECT_ROOT / "assets"
    FONTS_PATH: Path = ASSETS_PATH / "fonts"
    BACKGROUNDS_PATH: Path = ASSETS_PATH / "backgrounds"
    JOBS_OUTPUT_PATH: Path = PROJECT_ROOT / "jobs"
    
    # --- API Keys (Loaded from .env) ---
    # Try to get from environment first (loaded by dotenv), then from .env file
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', "sk-...")  # Default,
    SERPER_API_KEY: str = os.getenv('SERPER_API_KEY', "")  # Optional, for genre detection
    
    # --- Video & Text Settings (from your files) ---
    DEFAULT_FONT_REGULAR: str = str(FONTS_PATH / "PlayfairDisplay-Regular.ttf")
    DEFAULT_FONT_BOLD: str = str(FONTS_PATH / "PlayfairDisplay-Medium.ttf")
    DEFAULT_BACKGROUND: str = str(BACKGROUNDS_PATH / "1920x1080-white-solid-color-background.jpg") #for 1080 p quality
    # DEFAULT_BACKGROUND: str = str(BACKGROUNDS_PATH / "854x480-white-background.jpg")     #for 480 p quality
    
    VIDEO_FPS: int = 30
    VIDEO_WIDTH: int = 1920   #for 1080 p quality
    VIDEO_HEIGHT: int = 1080  #for 1080 p quality
    # VIDEO_WIDTH: int = 854    #for 480 p quality
    # VIDEO_HEIGHT: int = 480   #for 480 p quality
    VIDEO_CODEC: str = "libx264"
    
    # Text colors 
    TEXT_REGULAR_COLOR: tuple = (170, 170, 170, 255) # Grey
    TEXT_BOLD_COLOR: tuple = (0, 0, 0, 255) # Black

    # --- Summary Generation Settings ---
    SUMMARY_MODEL: str = "gpt-4o-mini"
    SUMMARY_TARGET_WORDS: int = 9000          # ~1 hour of narration at ~150 WPM
    SUMMARY_MAX_WORDS: int = 9500
    SUMMARY_WORDS_PER_MINUTE: int = 150
    SUMMARY_MAX_INPUT_CHARS: int = 400_000    # Roughly ~100k tokens of context
    SUMMARY_TEMPERATURE: float = 0.4

settings = Settings()

# Override with environment variable if it exists (dotenv loaded it)
# This ensures we use the value from dotenv even if pydantic-settings didn't pick it up
env_api_key = os.getenv('OPENAI_API_KEY')
if env_api_key and env_api_key != "sk-..." and len(env_api_key) > 10:
    settings.OPENAI_API_KEY = env_api_key

# Debug: Verify .env file loading
import logging
_logger = logging.getLogger(__name__)
if _env_file_path.exists():
    _logger.info(f".env file found at: {_env_file_path}")
    
    # Try to read the file directly to see what's in it
    try:
        with open(_env_file_path, 'r', encoding='utf-8') as f:
            env_content = f.read()
            # Find OPENAI_API_KEY line
            for line in env_content.split('\n'):
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#') and 'OPENAI_API_KEY' in stripped_line:
                    # Show first 30 chars of the line (masked)
                    masked_line = stripped_line[:30] + "..." if len(stripped_line) > 30 else stripped_line
                    _logger.info(f"Found in .env file: {masked_line}")
                    break
    except Exception as e:
        _logger.warning(f"Could not read .env file: {e}")
    
    # Check environment variable directly
    env_var_value = os.getenv('OPENAI_API_KEY')
    if env_var_value:
        masked_env = env_var_value[:7] + "..." + env_var_value[-4:] if len(env_var_value) > 11 else "***"
        _logger.info(f"OPENAI_API_KEY from os.getenv (length: {len(env_var_value)}): {masked_env}")
    
    # Check if API key was loaded (not the default)
    if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "sk-..." and len(settings.OPENAI_API_KEY) > 10:
        masked_setting = settings.OPENAI_API_KEY[:7] + "..." + settings.OPENAI_API_KEY[-4:] if len(settings.OPENAI_API_KEY) > 11 else "***"
        _logger.info(f"OPENAI_API_KEY loaded successfully (length: {len(settings.OPENAI_API_KEY)}): {masked_setting}")
    else:
        _logger.warning(f"OPENAI_API_KEY appears to be default value (length: {len(settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else 0}). Check .env file format.")
        _logger.warning(f"Expected format in .env: OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx (no quotes, no spaces around =)")
        if env_var_value and env_var_value != "sk-..." and len(env_var_value) > 10:
            _logger.info(f"Using environment variable value instead.")
            settings.OPENAI_API_KEY = env_var_value
else:
    _logger.warning(f".env file not found at: {_env_file_path}")