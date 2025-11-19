import logging
from typing import Tuple, Dict

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


def _truncate_book_text(book_text: str, max_chars: int) -> str:
    if len(book_text) <= max_chars:
        return book_text
    
    half = max_chars // 2
    logger.warning(
        "Book text exceeds %s characters. Truncating to keep context manageable.",
        max_chars
    )
    return (
        book_text[:half]
        + "\n\n[... content truncated for summary generation ...]\n\n"
        + book_text[-half:]
    )


def _enforce_word_limit(summary_text: str, max_words: int) -> Tuple[str, int]:
    words = summary_text.split()
    if len(words) <= max_words:
        return summary_text, len(words)
    
    logger.info(
        "Summary exceeds max word count (%s). Trimming to fit limit.",
        max_words
    )
    trimmed_text = " ".join(words[:max_words])
    return trimmed_text, max_words


def generate_book_summary(
    book_text: str,
    book_title: str,
    genre: str,
    book_type: str | None = None,
    target_word_count: int | None = None,
    max_word_count: int | None = None
) -> Tuple[str, Dict[str, float]]:
    """
    Generate a long-form book summary (~1 hour narration) using OpenAI.
    
    Returns:
        summary_text: Generated summary
        stats: Dictionary with word_count and estimated_minutes
    """
    if not book_text or not book_text.strip():
        raise ValueError("Cannot generate summary from empty book text.")
    
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    model = settings.SUMMARY_MODEL
    target_words = target_word_count or settings.SUMMARY_TARGET_WORDS
    max_words = max_word_count or settings.SUMMARY_MAX_WORDS
    wpm = settings.SUMMARY_WORDS_PER_MINUTE
    
    truncated_text = _truncate_book_text(book_text, settings.SUMMARY_MAX_INPUT_CHARS)
    
    system_msg = (
        "You are an expert narrator and editor who creates cinematic, engaging book summaries. "
        "Your summaries should feel like a compelling audio script designed for voice narration."
    )
    
    prompt = f"""
Book Title: {book_title}
Genre: {genre}
Book Type: {book_type or 'unknown'}

CRITICAL REQUIREMENT: You MUST generate a summary that is EXACTLY {target_words} words (±100 words). This is non-negotiable. The summary must be extensive, comprehensive, and detailed enough to fill one full hour of narration.

Goal:
- Write a comprehensive, extensive narrative summary that would take about one hour to narrate aloud.
- Target length: EXACTLY {target_words} words (minimum {target_words - 500} words, maximum {max_words} words).
- This must be a LONG, DETAILED summary covering all major plot points, themes, characters, and key moments.
- Structure it into logical sections/chapters with short headings, but keep prose flowing.
- Emphasize main ideas, plot arcs, key lessons, character development, and emotional beats.
- Use vivid, descriptive language suited for spoken audio. Avoid bullet points.
- Smoothly transition between sections so it feels like a single continuous story.
- Be thorough and comprehensive - do not skip important details or rush through sections.
- Expand on key scenes, character motivations, and thematic elements.

Source Material (truncated if necessary):
\"\"\"{truncated_text}\"\"\"

Now write the EXTENSIVE summary (plain text). The summary MUST be approximately {target_words} words. Start immediately with the summary content.
""".strip()
    
    logger.info(
        "Generating book summary with model %s (target ~%s words).",
        model,
        target_words
    )
    
    # Calculate required tokens: ~1.3-1.5 tokens per word for English text
    # For 9000 words, we need ~12,000-13,500 tokens, so use 15,000 to be safe
    required_tokens = int(target_words * 1.5) + 2000  # Add buffer for safety
    max_tokens = min(required_tokens, 16000)  # GPT-4o-mini supports up to 16k output tokens
    
    logger.info(f"Requesting summary with max_tokens={max_tokens} (target {target_words} words)")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=settings.SUMMARY_TEMPERATURE,
        max_tokens=max_tokens
    )
    
    summary_text = response.choices[0].message.content.strip()
    summary_text, word_count = _enforce_word_limit(summary_text, max_words)
    estimated_minutes = round(word_count / wpm, 2)
    
    stats = {
        "word_count": word_count,
        "estimated_minutes": estimated_minutes,
        "target_word_count": target_words
    }
    
    logger.info(
        "Summary generated (~%s words, est %s min).",
        word_count,
        estimated_minutes
    )
    return summary_text, stats

