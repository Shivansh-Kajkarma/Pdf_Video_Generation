import logging
import re
import csv
from pathlib import Path
from typing import Set, List

logger = logging.getLogger(__name__)


def _clean_punctuation_for_speech(text: str) -> str:
    """
    Clean punctuation that causes awkward speech (e.g., "slash" from "/").
    Preserves legitimate numeric expressions like decimals and ranges.
    Removes commas from text for video generation.
    
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
    
    # Step 3: Remove all commas
    text = text.replace(',', '')
    
    # Step 4: Replace "/" in common word patterns (like "and/or" -> "and or")
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


def _extract_table_text(tables_dir: Path) -> Set[str]:
    """
    Extract all text content from table CSV files.
    
    Args:
        tables_dir: Directory containing table CSV files
        
    Returns:
        Set of unique text phrases from all tables (headers + cell values)
    """
    table_texts = set()
    
    if not tables_dir or not tables_dir.exists():
        return table_texts
    
    # Find all CSV files in tables directory
    csv_files = list(tables_dir.glob("*.csv"))
    
    if not csv_files:
        logger.debug(f"No CSV files found in {tables_dir}")
        return table_texts
    
    logger.info(f"Extracting text from {len(csv_files)} table CSV files...")
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    for cell in row:
                        if cell and cell.strip():
                            # Add the cell value
                            cell_text = cell.strip()
                            table_texts.add(cell_text)
                            
                            # Also add individual words for better matching
                            # (handles cases where table text appears as part of sentences)
                            words = re.findall(r'\b\w+\b', cell_text)
                            for word in words:
                                if len(word) > 2:  # Only add words longer than 2 chars
                                    table_texts.add(word)
        except Exception as e:
            logger.warning(f"Error reading table file {csv_file}: {e}")
            continue
    
    logger.info(f"Extracted {len(table_texts)} unique text phrases from tables")
    return table_texts


def _remove_table_content(text: str, tables_dir: Path) -> str:
    """
    Remove table content from text by matching table cell values.
    
    Args:
        text: Text content to clean
        tables_dir: Directory containing table CSV files
        
    Returns:
        Text with table content removed
    """
    if not tables_dir or not tables_dir.exists():
        return text
    
    table_texts = _extract_table_text(tables_dir)
    
    if not table_texts:
        return text
    
    cleaned_text = text
    removed_count = 0
    
    # Strategy: Remove table-like patterns more carefully
    # 1. Remove lines that look like table rows (multiple table values in one line)
    lines = cleaned_text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            filtered_lines.append(line)
            continue
        
        # Check if line contains multiple table values (likely a table row)
        # Only consider longer table values (5+ chars) to avoid false positives
        table_matches = []
        for table_val in table_texts:
            if len(table_val) >= 5 and table_val.lower() in line_stripped.lower():
                # Check if it's a word boundary match (not part of a larger word)
                pattern = r'\b' + re.escape(table_val) + r'\b'
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    table_matches.append(table_val)
        
        # If line contains 3+ table values, it's likely a table row
        if len(table_matches) >= 3:
            removed_count += 1
            logger.debug(f"Removed table row: {line_stripped[:80]}...")
            continue
        
        # Also check for tab-separated or pipe-separated values (common table formats)
        if '\t' in line or '|' in line:
            # Count non-empty cells
            cells = [c.strip() for c in re.split(r'[\t|]+', line) if c.strip()]
            if len(cells) >= 3:  # Likely a table row if 3+ columns
                # Check if cells match table values
                matching_cells = sum(1 for cell in cells if any(
                    len(val) >= 5 and val.lower() == cell.lower() for val in table_texts
                ))
                if matching_cells >= 2:  # 2+ matching cells = likely table row
                    removed_count += 1
                    logger.debug(f"Removed table row (tab/pipe separated): {line_stripped[:80]}...")
                    continue
        
        filtered_lines.append(line)
    
    cleaned_text = '\n'.join(filtered_lines)
    
    # 2. Remove standalone table cell values that appear as isolated phrases
    # Only remove if they're very likely to be table content (numeric, short, etc.)
    for table_text in sorted(table_texts, key=len, reverse=True):
        if len(table_text) < 4 or len(table_text) > 50:  # Skip very short or very long
            continue
        
        # Only remove if it's a standalone line or phrase
        # Pattern: line that is just the table value (possibly with whitespace)
        pattern = r'^\s*' + re.escape(table_text) + r'\s*$'
        if re.search(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE):
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
            removed_count += 1
    
    # Clean up extra whitespace created by removals
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)  # Multiple spaces/tabs to single space
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Multiple newlines to double
    
    if removed_count > 0:
        logger.info(f"Removed table content: {removed_count} table rows/phrases removed")
    
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
    2. Removing table content (headers and cell values from extracted tables)
    3. Cleaning punctuation for natural speech (e.g., "and/or" -> "and or", removing "slash")
    4. Preserving legitimate numeric expressions (decimals, ranges)
    5. Removing URLs and email addresses
    
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
        
        # Step 2: Remove table content
        cleaned_text = _remove_table_content(cleaned_text, tables_dir)
        
        # Step 3: Clean punctuation for natural speech
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