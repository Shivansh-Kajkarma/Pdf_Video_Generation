import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _clean_punctuation_for_speech(text: str) -> str:
    """
    Clean punctuation that causes awkward speech (e.g., "slash" from "/").
    Preserves legitimate numeric expressions like decimals and ranges.
    
    Args:
        text: Text content
        
    Returns:
        Text with punctuation cleaned for natural speech
    """
    # Step 1: Remove URLs first (before processing slashes)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Step 2: Remove email addresses
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)
    
    # Step 3: Replace "/" in common word patterns (like "and/or" -> "and or")
    # But preserve it in dates and numeric contexts
    
    # First, handle specific common patterns
    text = re.sub(r'\band/or\b', 'and or', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhe/she\b', 'he or she', text, flags=re.IGNORECASE)
    text = re.sub(r'\bshe/he\b', 'she or he', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhim/her\b', 'him or her', text, flags=re.IGNORECASE)
    text = re.sub(r'\bher/him\b', 'her or him', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhis/her\b', 'his or her', text, flags=re.IGNORECASE)
    text = re.sub(r'\bher/his\b', 'her or his', text, flags=re.IGNORECASE)
    
    # Replace "/" between letters/words (but not in dates like 12/25/2024 or numeric like 52/3)
    # Pattern: letter/letter or word/word -> word or word
    # This preserves dates (which have numbers) and numeric fractions
    # Only match if both sides are purely alphabetic
    text = re.sub(r'\b([a-zA-Z]+)/([a-zA-Z]+)\b', r'\1 or \2', text)
    
    # Clean up other awkward punctuation
    # Remove standalone asterisks used for footnotes (but keep them in context if needed)
    text = re.sub(r'\s+\*\s+', ' ', text)  # Space asterisk space
    text = re.sub(r'\s+\*\*\s+', ' ', text)  # Space double asterisk space
    
    # Clean up multiple consecutive punctuation marks
    text = re.sub(r'[!?]{2,}', '!', text)  # Multiple ! or ? to single
    text = re.sub(r'\.{3,}', '...', text)  # More than 3 dots to ellipsis
    
    # Remove standalone special characters that might be read awkwardly
    # But preserve them in context (like in quotes or parentheses)
    # Remove standalone "#" unless it's part of a number
    text = re.sub(r'\s+#\s+', ' ', text)
    
    # Clean up extra spaces that might have been created
    text = re.sub(r'\s+', ' ', text)
    
    return text


def _remove_image_references(text: str) -> str:
    """
    Remove sentences that reference images, figures, diagrams, charts, etc.
    
    Args:
        text: Raw text content
        
    Returns:
        Text with image reference sentences removed
    """
    # Common image reference patterns (case-insensitive)
    image_patterns = [
        # Patterns like "following image", "above figure", "below diagram"
        r'\b(following|above|below|preceding|next|previous)\s+(image|figure|diagram|chart|photo|picture|illustration|graph)\b',
        # Patterns like "see image below", "see figure above"
        r'\bsee\s+(the\s+)?(image|figure|diagram|chart|photo|picture|illustration|graph)\s+(above|below|on\s+page|in\s+figure)\b',
        # Patterns like "as shown in the image", "as illustrated in the figure"
        r'\bas\s+(shown|illustrated|depicted|seen|displayed)\s+in\s+(the\s+)?(image|figure|diagram|chart|photo|picture|illustration|graph)\b',
        # Patterns like "refer to image", "refer to figure 1"
        r'\brefer\s+(to|back\s+to)\s+(the\s+)?(image|figure|diagram|chart|photo|picture|illustration|graph)\b',
        # Patterns like "in the image above", "in figure 2 below"
        r'\bin\s+(the\s+)?(image|figure|diagram|chart|photo|picture|illustration|graph)\s+(above|below|on\s+page|#?\d+)\b',
        # Standalone figure/image captions like "Figure 1:", "Image 2.3:", "Fig. 1"
        r'^\s*(Figure|Fig\.?|Image|Diagram|Chart|Photo|Picture|Illustration|Graph)\s*[:\-]?\s*\d+[\.\-\:]?\s*$',
        # Patterns like "this image shows", "the figure illustrates"
        r'\b(this|the)\s+(image|figure|diagram|chart|photo|picture|illustration|graph)\s+(shows|illustrates|depicts|displays|presents)\b',
    ]
    
    # Combine all patterns with case-insensitive flag
    combined_pattern = '|'.join(f'({pattern})' for pattern in image_patterns)
    image_reference_regex = re.compile(combined_pattern, re.IGNORECASE | re.MULTILINE)
    
    # Split text into sentences
    # Simple sentence splitting (period, exclamation, question mark followed by space or newline)
    sentences = re.split(r'([.!?]+\s+)', text)
    
    # Recombine sentences (split keeps delimiters)
    sentence_pairs = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence_pairs.append(sentences[i] + sentences[i + 1])
        else:
            sentence_pairs.append(sentences[i])
    
    # Filter out sentences containing image references
    cleaned_sentences = []
    removed_count = 0
    
    for sentence in sentence_pairs:
        # Check if sentence contains image reference pattern
        if image_reference_regex.search(sentence):
            removed_count += 1
            logger.debug(f"Removed image reference: {sentence.strip()[:100]}")
            continue
        cleaned_sentences.append(sentence)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} sentence(s) containing image references")
    
    # Join cleaned sentences
    cleaned_text = ''.join(cleaned_sentences)
    
    # Clean up extra whitespace (multiple newlines/spaces)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Max 2 consecutive newlines
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)  # Multiple spaces to single space
    
    return cleaned_text.strip()


def clean_text(
    raw_text_path: Path, 
    tables_dir: Path, 
    images_dir: Path, 
    job_dir: Path
) -> Path:
    """
    Cleans the raw text by:
    1. Removing image reference sentences (e.g., "see image below", "as shown in the figure")
    2. Cleaning punctuation for natural speech (e.g., "and/or" -> "and or", removing "slash")
    3. Preserving legitimate numeric expressions (decimals, ranges)
    4. Removing URLs and email addresses
    5. Table content removal (future implementation)
    6. Image caption removal (future implementation)
    
    Args:
        raw_text_path: Path to raw extracted text
        tables_dir: Directory containing extracted tables (for future use)
        images_dir: Directory containing extracted images (for future use)
        job_dir: Output directory for cleaned text
        
    Returns:
        Path to cleaned text file
    """
    logger.info("--- Starting text cleaning ---")
    
    cleaned_script_path = job_dir / "cleaned_script.txt"
    
    try:
        # Load raw text
        with open(raw_text_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        logger.info(f"Loaded raw text: {len(raw_text)} characters")
        
        # Step 1: Remove image references
        cleaned_text = _remove_image_references(raw_text)
        
        # Step 2: Clean punctuation for natural speech
        cleaned_text = _clean_punctuation_for_speech(cleaned_text)
        
        logger.info(f"Cleaned text: {len(cleaned_text)} characters (removed {len(raw_text) - len(cleaned_text)} chars)")
        
        # Save cleaned text
        with open(cleaned_script_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        logger.info(f"Text cleaning complete. Saved to: {cleaned_script_path}")
        return cleaned_script_path
        
    except Exception as e:
        logger.error(f"Failed to clean text: {e}", exc_info=True)
        raise