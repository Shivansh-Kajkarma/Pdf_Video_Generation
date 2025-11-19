import json
import logging
import os
import tempfile
import multiprocessing
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from functools import partial

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, CompositeVideoClip, VideoClip, ImageClip, ImageSequenceClip
import moviepy.video.fx as vfx
from pydantic import BaseModel
from tqdm import tqdm

from app.config import settings

logger = logging.getLogger(__name__)

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    confidence: Optional[float] = None
    probability: Optional[float] = None


class FrameGeneratorV11:
    """
    Generates TRANSPARENT frames with clean "karaoke-style" animated text.
    V11: Robust punctuation alignment, justified text alignment, proper margins and padding.
    """
    def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
        logger.info("Initializing FrameGeneratorV11 (Robust Punctuation)...")
        
        self.bg_width = bg_width
        self.bg_height = bg_height

        self.all_words, self.segments = self._load_data(timestamps_path)

        # --- 1. Load settings from config.py ---
        # Reduced font size to prevent text overflow
        self.font_size = max(36, int(self.bg_height / 8))  # Smaller font for better fit
        self.line_height = int(self.font_size * 1.2)  # Slightly more line spacing
        self.regular_font, self.bold_font = self._load_fonts(self.font_size)
        # Fixed 50px margins on all four sides (increased left margin for better padding)
        self.margin = 80
        # Use same margin for all sides, but ensure left has proper padding
        self.left_margin = 160  # Increased left margin for better visual padding
        self.right_margin = self.margin
        self.top_margin = self.margin
        self.bottom_margin = self.margin 
        # Calculate available text area (accounting for different left/right margins, text left-aligned)
        self.max_text_width = self.bg_width - self.left_margin - self.right_margin
        self.max_text_height = self.bg_height - (2 * self.margin)
        # self.min_words_per_slide = 8
        
        self.slides, self.slide_layouts, self.slide_start_times = self._build_grouped_slides()
        logger.info(f"FrameGeneratorV11 initialized: {len(self.segments)} segments grouped into {len(self.slides)} slides.")

    def _load_fonts(self, size: int) -> Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
        # --- 2. Load fonts from config.py paths ---
        try:
            regular_path = settings.DEFAULT_FONT_REGULAR
            bold_path = settings.DEFAULT_FONT_BOLD
            regular = ImageFont.truetype(regular_path, size)
            bold = ImageFont.truetype(bold_path, size)
            logger.info(f"Loaded custom font: {regular_path}")
            return regular, bold
        except Exception as e:
            logger.error(f"FATAL: Could not load custom font! {e}", exc_info=True)
            logger.warning("Falling back to default font.")
            return ImageFont.load_default(size=size), ImageFont.load_default(size=size)

   
    def _load_data(self, timestamps_path: Path) -> Tuple[List[WordTimestamp], List[Dict]]:
        logger.info(f"Loading data from: {timestamps_path}")
        with open(timestamps_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "words" not in data or "segments" not in data:
            raise ValueError("Invalid timestamp format: expected 'words' and 'segments' keys")
        if not data["segments"]:
            raise ValueError("Timestamp data is missing 'segments'. Please re-run the OpenAI service with timestamp_granularities=['word', 'segment']")
        words = [WordTimestamp(**w) for w in data["words"] if w.get('word', '').strip()]
        segments = data["segments"]
        logger.info(f"Loaded {len(words)} words and {len(segments)} segments.")
        return words, segments

    def _get_words_for_segment(self, segment_index: int) -> List[WordTimestamp]:
        segment = self.segments[segment_index]
        segment_start = segment["start"]
        segment_end = self.segments[segment_index + 1]["start"] if segment_index + 1 < len(self.segments) else float('inf')
        return [w for w in self.all_words if w.start >= segment_start and w.start < segment_end]

    def _build_grouped_slides(self) -> Tuple[List[List[List[WordTimestamp]]], Dict[int, Dict[int, Tuple[int, int]]], List[float]]:
        logger.info("Building grouped slides (Max Words strategy)...")
        slides, layouts, slide_start_times = [], {}, []
        dummy_img = Image.new("RGB", (self.bg_width, self.bg_height))
        draw = ImageDraw.Draw(dummy_img)
        space_bbox = draw.textbbox((0, 0), " ", font=self.bold_font); space_width = space_bbox[2] - space_bbox[0]

        MAX_WORDS_PER_SLIDE = 20 #allow up to 20 words per slide
        MAX_LINES_PER_SLIDE = 5 #maximum 5 lines per slide

        current_slide_words_ts, current_slide_start_time, current_slide_segments = [], -1, []

        def build_slide_layout(words_ts, segments_list, start_time):
            """Helper function to avoid repeating the build logic."""
            if not words_ts:
                return

            slide_index = len(slides)
            all_clean_words = [word for s in segments_list for word in s["text"].strip().split()]
            if len(all_clean_words) == len(words_ts):
                for j in range(len(all_clean_words)):
                    words_ts[j].word = all_clean_words[j]

            current_slide_lines, current_line = [], []
            for word in words_ts:
                word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font)
                word_width = word_bbox[2] - word_bbox[0]
                
                # Calculate current line width including the new word
                if current_line:
                    # Build test line with new word to get accurate width
                    test_line = current_line + [word]
                    line_text = " ".join(w.word for w in test_line)
                    line_bbox = draw.textbbox((0, 0), line_text, font=self.bold_font)
                    test_line_width = line_bbox[2] - line_bbox[0]
                    
                    # Use 90% of max width as threshold for safety (very conservative)
                    threshold = self.max_text_width * 0.90
                    if test_line_width > threshold:
                        # Current line is full, start new line
                        current_slide_lines.append(current_line)
                        current_line = [word]
                    else:
                        current_line.append(word)
                else:
                    # First word in line - check if single word fits
                    if word_width > self.max_text_width:
                        # Word is too long, truncate or handle (shouldn't happen often)
                        logger.warning(f"Word '{word.word}' is wider than max width ({word_width} > {self.max_text_width})")
                    current_line.append(word)
            if current_line: 
                current_slide_lines.append(current_line)

            # Check if slide has too many lines and reduce verbosity of warnings
            total_text_height = len(current_slide_lines) * self.line_height
            if len(current_slide_lines) > MAX_LINES_PER_SLIDE:
                # Only log as debug to reduce noise - slides will still render
                logger.debug(f"Slide {slide_index} has {len(current_slide_lines)} lines (max recommended: {MAX_LINES_PER_SLIDE})")
            elif total_text_height > self.bg_height * 0.95:
                logger.warning(f"Slide {slide_index} (starting {start_time}s) may be too tall! Has {len(current_slide_lines)} lines.")

            slides.append(current_slide_lines); layouts[slide_index] = {}

            # Text area boundaries (equal margins on all sides)
            text_area_start_x = self.left_margin
            text_area_end_x = self.bg_width - self.right_margin
            text_area_width = text_area_end_x - text_area_start_x
            
            # Center vertically: start_y = (total_height - text_block_height) / 2
            start_y = self.top_margin + (self.max_text_height - total_text_height) // 2
            current_y = start_y
            max_x = text_area_end_x
            
            for line_of_words in current_slide_lines:
                # Calculate justified text positions
                if len(line_of_words) == 0:
                    current_y += self.line_height
                    continue
                
                # Calculate total width of all words (without spaces)
                word_widths = []
                total_words_width = 0
                for word in line_of_words:
                    word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font)
                    word_width = word_bbox[2] - word_bbox[0]
                    word_widths.append(word_width)
                    total_words_width += word_width
                
                # Calculate available width for this line
                # Account for the last word's width to prevent overflow
                last_word_width = word_widths[-1] if word_widths else 0
                line_available_width = self.max_text_width - last_word_width
                
                # Calculate normal spacing (with space_width between words)
                num_spaces = len(line_of_words) - 1
                normal_total_width = total_words_width + (num_spaces * space_width)
                
                # For justified alignment: only justify if spacing won't be excessive
                # All text should be left-aligned (start at text_area_start_x)
                
                # If single word, left-align it
                if len(line_of_words) == 1:
                    # Single word: left-align
                    current_x = text_area_start_x
                    layouts[slide_index][id(line_of_words[0])] = (current_x, current_y)
                # If line is nearly full or normal spacing already fills most of the line, left-align
                elif total_words_width >= line_available_width * 0.95 or normal_total_width >= line_available_width * 0.90:
                    # Line is nearly full: left-align
                    current_x = text_area_start_x
                    for i, word in enumerate(line_of_words):
                        word_width = word_widths[i]
                        # Ensure word fits - adjust position if needed
                        if current_x + word_width > max_x:
                            # Try to fit by positioning at max_x - word_width
                            adjusted_x = max(text_area_start_x, max_x - word_width)
                            if adjusted_x + word_width <= max_x:
                                current_x = adjusted_x
                            else:
                                # Word is too wide, but still left-align it
                                current_x = text_area_start_x
                        layouts[slide_index][id(word)] = (current_x, current_y)
                        current_x += word_width + space_width
                else:
                    # Justified alignment: distribute extra space between words
                    # Calculate space needed excluding the last word (it will be positioned at the end)
                    words_width_excluding_last = total_words_width - last_word_width
                    if num_spaces > 0:
                        total_space_needed = line_available_width - words_width_excluding_last
                        space_between_words = total_space_needed / num_spaces
                        
                        # Limit maximum space between words to 2.5x normal space width
                        # This prevents huge gaps when there are few words
                        max_space = space_width * 2.5
                        if space_between_words > max_space:
                            # Space would be too large, fall back to left-aligned
                            current_x = text_area_start_x
                            for i, word in enumerate(line_of_words):
                                word_width = word_widths[i]
                                # Ensure word fits - adjust position if needed
                                if current_x + word_width > max_x:
                                    # Try to fit by positioning at max_x - word_width
                                    adjusted_x = max(text_area_start_x, max_x - word_width)
                                    if adjusted_x + word_width <= max_x:
                                        current_x = adjusted_x
                                    else:
                                        # Word is too wide, but still left-align it
                                        current_x = text_area_start_x
                                layouts[slide_index][id(word)] = (current_x, current_y)
                                current_x += word_width + space_width
                        else:
                            # Justified alignment with reasonable spacing
                            # Use full width for justified text (spans from left to right margin)
                            current_x = text_area_start_x
                            for i, word in enumerate(line_of_words):
                                word_width = word_widths[i]
                                
                                # Last word: position it so it ends at max_x (right-aligned for justified effect)
                                if i == len(line_of_words) - 1:
                                    current_x = max_x - word_width
                                    # Safety check: ensure word doesn't go before text area start
                                    if current_x < text_area_start_x:
                                        current_x = text_area_start_x
                                else:
                                    # Safety check: ensure word fits - adjust if needed
                                    if current_x + word_width > max_x:
                                        # Try to fit by positioning at max_x - word_width
                                        adjusted_x = max(text_area_start_x, max_x - word_width)
                                        if adjusted_x + word_width <= max_x:
                                            current_x = adjusted_x
                                        else:
                                            # Word is too wide, but still left-align it
                                            current_x = text_area_start_x
                                
                                layouts[slide_index][id(word)] = (current_x, current_y)
                                
                                # Add word width plus justified space (except after last word)
                                if i < len(line_of_words) - 1:
                                    current_x += word_width + space_between_words
                    else:
                        # No spaces (shouldn't happen with multiple words, but handle it)
                        space_between_words = space_width
                        # Left-align
                        current_x = text_area_start_x
                        for i, word in enumerate(line_of_words):
                            word_width = word_widths[i]
                            # Ensure word fits - adjust position if needed
                            if current_x + word_width > max_x:
                                # Try to fit by positioning at max_x - word_width
                                adjusted_x = max(text_area_start_x, max_x - word_width)
                                if adjusted_x + word_width <= max_x:
                                    current_x = adjusted_x
                                else:
                                    # Word is too wide, but still left-align it
                                    current_x = text_area_start_x
                            layouts[slide_index][id(word)] = (current_x, current_y)
                            current_x += word_width + space_width
                        
                current_y += self.line_height
            
            # Verify all words have been assigned positions
            words_in_layout = set(layouts[slide_index].keys())
            words_in_slide = set(id(word) for line in current_slide_lines for word in line)
            missing_words = words_in_slide - words_in_layout
            if missing_words:
                logger.warning(f"Slide {slide_index}: {len(missing_words)} words missing from layout. This should not happen.")
                # Try to add missing words at the end of the last line
                if current_slide_lines:
                    last_line = current_slide_lines[-1]
                    last_y = start_y + (len(current_slide_lines) - 1) * self.line_height
                    # Left-align missing words
                    current_x = text_area_start_x
                    for word in last_line:
                        if id(word) in missing_words:
                            word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font)
                            word_width = word_bbox[2] - word_bbox[0]
                            if current_x + word_width <= max_x:
                                layouts[slide_index][id(word)] = (current_x, last_y)
                                current_x += word_width + space_width
            
            slide_start_times.append(start_time)

        def _word_ends_sentence(word_text: str) -> bool:
            if not word_text:
                return False
            cleaned = word_text.strip().rstrip("”’\"')]} ")
            if not cleaned:
                return False
            return cleaned[-1] in {".", "!", "?"}

        last_word_text = None

        # main loop
        for i, segment in enumerate(self.segments):
            segment_words_ts = self._get_words_for_segment(i)
            if not segment_words_ts:
                continue

            for word_ts in segment_words_ts:
                word_text = word_ts.word.strip()
                # If previous word ended a sentence, start a new slide
                if last_word_text and _word_ends_sentence(last_word_text) and current_slide_words_ts:
                    build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)
                    current_slide_words_ts = []
                    current_slide_segments = []
                    current_slide_start_time = -1

                if not current_slide_words_ts:
                    current_slide_start_time = word_ts.start

                current_slide_words_ts.append(word_ts)

                if segment not in current_slide_segments:
                    current_slide_segments.append(segment)

                last_word_text = word_text

                if len(current_slide_words_ts) >= MAX_WORDS_PER_SLIDE:
                    build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)
                    current_slide_words_ts = []
                    current_slide_segments = []
                    current_slide_start_time = -1

        # --- Handle the VERY LAST slide after the loop finishes ---
        if current_slide_words_ts:
            build_slide_layout(current_slide_words_ts, current_slide_segments, current_slide_start_time)

        logger.info(f"Grouped slide building complete. Created {len(slides)} slides.")
        return slides, layouts, slide_start_times
    def make_frame_function(self, slide_index: int, slide_start_time: float):
        def generate_frame(t_local: float) -> np.ndarray:
            global_t = slide_start_time + t_local
            frame = Image.new("RGBA", (self.bg_width, self.bg_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(frame)
            slide_lines = self.slides[slide_index]
            layout = self.slide_layouts[slide_index]
            max_x = self.bg_width - self.right_margin
            
            for line in slide_lines:
                for word in line:
                    coords = layout.get(id(word))
                    if not coords: continue
                    
                    # Safety check: ensure word doesn't overflow screen
                    x, y = coords
                    word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font)
                    word_width = word_bbox[2] - word_bbox[0]
                    
                    if x + word_width > max_x:
                        # Adjust position to prevent overflow
                        x = max(self.left_margin, max_x - word_width)
                    
                    # --- 3. Load colors from config.py ---
                    if global_t >= word.start:
                        font = self.bold_font
                        color = settings.TEXT_BOLD_COLOR
                    else:
                        font = self.regular_font
                        color = settings.TEXT_REGULAR_COLOR
                    draw.text((x, y), word.word, font=font, fill=color)
            return np.array(frame)
        return generate_frame

    def generate_single_frame(
        self,
        frame_number: int,
        frame_timestamp: float,
        slide_index: int,
        slide_start_time: float
    ) -> Image.Image:
        """
        Generate a single frame at a specific timestamp.
        Used for batch processing.
        """
        global_t = frame_timestamp
        frame = Image.new("RGBA", (self.bg_width, self.bg_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(frame)
        slide_lines = self.slides[slide_index]
        layout = self.slide_layouts[slide_index]
        
        max_x = self.bg_width - self.right_margin
        
        for line in slide_lines:
            for word in line:
                coords = layout.get(id(word))
                if not coords:
                    continue
                
                # Safety check: ensure word doesn't overflow screen
                x, y = coords
                word_bbox = draw.textbbox((0, 0), word.word, font=self.bold_font)
                word_width = word_bbox[2] - word_bbox[0]
                
                # Adjust position if word would overflow - don't skip it
                if x + word_width > max_x:
                    # Try to fit by positioning at max_x - word_width
                    adjusted_x = max(self.left_margin, max_x - word_width)
                    if adjusted_x + word_width <= max_x:
                        x = adjusted_x
                    # If still too wide, render it anyway (will be clipped but visible)
                
                # State-based logic: check if word has started
                if global_t >= word.start:
                    font = self.bold_font
                    color = settings.TEXT_BOLD_COLOR
                else:
                    font = self.regular_font
                    color = settings.TEXT_REGULAR_COLOR
                
                draw.text((x, y), word.word, font=font, fill=color)
        
        return frame


# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def _calculate_frame_timestamps(duration: float, fps: int) -> List[Tuple[int, float]]:
    """
    Calculate all frame timestamps for the video.
    These are evenly spaced frame timestamps - word highlighting will use
    Whisper timestamps directly for accuracy.
    
    Returns:
        List of (frame_number, timestamp) tuples
    """
    total_frames = int(duration * fps)
    frame_timestamps = []
    frame_interval = 1.0 / fps
    
    for frame_num in range(total_frames):
        # Calculate precise timestamp for each frame
        timestamp = frame_num * frame_interval
        frame_timestamps.append((frame_num, timestamp))
    
    return frame_timestamps


def _map_frames_to_slides(
    frame_timestamps: List[Tuple[int, float]],
    slide_start_times: List[float],
    audio_duration: float
) -> List[Tuple[int, float, int, float]]:
    """
    Map each frame to its corresponding slide.
    
    Returns:
        List of (frame_number, timestamp, slide_index, slide_start_time) tuples
    """
    mapped_frames = []
    
    for frame_num, timestamp in frame_timestamps:
        # Find which slide this frame belongs to
        slide_index = 0
        slide_start = slide_start_times[0]
        
        for i, slide_start_time in enumerate(slide_start_times):
            if timestamp >= slide_start_time:
                slide_index = i
                slide_start = slide_start_time
            else:
                break
        
        # Handle last slide
        if slide_index == len(slide_start_times) - 1:
            # Check if we're still within the last slide
            if timestamp > audio_duration:
                continue
        
        mapped_frames.append((frame_num, timestamp, slide_index, slide_start))
    
    return mapped_frames


def _create_frame_batches(
    mapped_frames: List[Tuple[int, float, int, float]],
    batch_size: int
) -> List[List[Tuple[int, float, int, float]]]:
    """
    Split frames into batches for parallel processing.
    """
    batches = []
    for i in range(0, len(mapped_frames), batch_size):
        batch = mapped_frames[i:i + batch_size]
        batches.append(batch)
    return batches


def _generate_frame_batch_worker(
    batch_data: Tuple[
        List[Tuple[int, float, int, float]],  # Frame tasks
        Dict[str, Any],  # Frame generator data
        Path,  # Output directory
        int,  # Width
        int,  # Height
    ]
) -> List[str]:
    """
    Worker function to generate a batch of frames in parallel.
    Optimized for performance.
    
    Args:
        batch_data: Tuple containing:
            - List of (frame_number, timestamp, slide_index, slide_start_time)
            - Frame generator serialized data (slides, layouts, fonts, etc.)
            - Output directory for saving frames
            - Width and height
    
    Returns:
        List of generated frame file paths
    """
    frame_tasks, gen_data, output_dir, width, height = batch_data
    
    # Reconstruct frame generator data (cached per worker)
    slides = gen_data['slides']
    slide_layouts = gen_data['slide_layouts']
    font_size = gen_data['font_size']
    
    # Load fonts once per worker (cached)
    try:
        regular_font = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, font_size)
        bold_font = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, font_size)
    except Exception:
        regular_font = ImageFont.load_default(size=font_size)
        bold_font = ImageFont.load_default(size=font_size)
    
    # Pre-extract colors to avoid repeated lookups
    bold_color = settings.TEXT_BOLD_COLOR
    regular_color = settings.TEXT_REGULAR_COLOR
    
    generated_files = []
    
    # Process frames in batch
    for frame_num, timestamp, slide_index, slide_start in frame_tasks:
        # Generate frame
        frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(frame)
        slide_lines = slides[slide_index]
        layout = slide_layouts[slide_index]
        
        # Render all words in the slide
        # Use same margins as main renderer (80px left, 50px right)
        left_margin = 80  # Increased left margin for better visual padding
        right_margin = 50
        max_text_width = width - left_margin - right_margin
        max_x = width - right_margin
        
        for line_idx, line in enumerate(slide_lines):
            for word_idx, word in enumerate(line):
                # Look up coordinates using (line_index, word_index) as key
                unique_key = (line_idx, word_idx)
                coords = layout.get(unique_key)
                if not coords:
                    continue
                
                # Safety check: ensure word doesn't overflow screen
                x, y = coords
                word_bbox = draw.textbbox((0, 0), word['word'], font=bold_font)
                word_width = word_bbox[2] - word_bbox[0]
                
                # Adjust position if word would overflow - don't skip it
                if x + word_width > max_x:
                    # Try to fit by positioning at max_x - word_width
                    adjusted_x = max(left_margin, max_x - word_width)
                    if adjusted_x + word_width <= max_x:
                        x = adjusted_x
                    # If still too wide, render it anyway (will be clipped but visible)
                
                # State-based logic: use Whisper timestamp directly for accuracy
                # This ensures word highlighting matches exactly when words are spoken
                if timestamp >= word['start']:
                    draw.text((x, y), word['word'], font=bold_font, fill=bold_color)
                else:
                    draw.text((x, y), word['word'], font=regular_font, fill=regular_color)
        
        # Save frame with optimized PNG settings
        frame_filename = output_dir / f"frame_{frame_num:06d}.png"
        # Use optimize=False for faster saving (we'll delete these anyway)
        frame.save(frame_filename, "PNG", optimize=False, compress_level=1)
        generated_files.append(str(frame_filename))
    
    return generated_files


def _get_ffmpeg_path() -> str:
    """Get the path to ffmpeg executable."""
    import shutil
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    
    raise FileNotFoundError("FFmpeg not found. Please install ffmpeg.")


def _detect_hardware_codec() -> Tuple[str, List[str]]:
    """
    Detect available hardware acceleration codec.
    
    Returns:
        Tuple of (codec_name, additional_ffmpeg_params)
    """
    import subprocess
    
    ffmpeg_path = _get_ffmpeg_path()
    
    # Try NVIDIA NVENC
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "h264_nvenc" in result.stdout:
            logger.info("Detected NVIDIA GPU - using h264_nvenc")
            # Optimized NVENC settings for speed: p1 preset (fastest), VBR with CQ
            return "h264_nvenc", [
                "-preset", "p1",  # Fastest preset
                "-rc", "vbr",
                "-cq", "23",
                "-rc-lookahead", "32",
                "-b_ref_mode", "middle"
            ]
    except Exception:
        pass
    
    # Try Intel QuickSync
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "h264_qsv" in result.stdout:
            logger.info("Detected Intel QuickSync - using h264_qsv")
            return "h264_qsv", ["-preset", "veryfast", "-global_quality", "23"]
    except Exception:
        pass
    
    # Try AMD AMF
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "h264_amf" in result.stdout:
            logger.info("Detected AMD GPU - using h264_amf")
            return "h264_amf", ["-quality", "speed", "-rc", "vbr_peak", "-qmin", "18", "-qmax", "28"]
    except Exception:
        pass
    
    # Fallback to software encoding
    logger.info("No hardware acceleration detected - using libx264 (software)")
    return "libx264", ["-preset", "veryfast", "-crf", "23"]


def _encode_video_with_ffmpeg(
    video_clip: VideoClip,
    audio_path: Path,
    output_path: Path,
    fps: int
) -> Path:
    """
    Encode video directly with FFmpeg, bypassing MoviePy's slow encoding.
    
    Args:
        video_clip: The composited video clip from MoviePy
        audio_path: Path to audio file
        output_path: Output video path
        fps: Video FPS
    
    Returns:
        Path to encoded video
    """
    import subprocess
    
    ffmpeg_path = _get_ffmpeg_path()
    codec, codec_params = _detect_hardware_codec()
    
    # Use hardware encoding for temp file if available, otherwise use fastest software encoding
    temp_codec, temp_codec_params = _detect_hardware_codec()
    
    # For temp file, use faster/lower quality settings since it's just intermediate
    if temp_codec == "h264_nvenc":
        temp_params = ["-preset", "p1", "-rc", "vbr", "-cq", "28", "-b:v", "10M"]  # Faster, lower quality temp
    elif temp_codec == "h264_qsv":
        temp_params = ["-preset", "veryfast", "-global_quality", "28", "-b:v", "10M"]
    elif temp_codec == "h264_amf":
        temp_params = ["-quality", "speed", "-rc", "vbr_peak", "-qmin", "24", "-qmax", "32", "-b:v", "10M"]
    else:
        temp_params = ["-preset", "ultrafast", "-crf", "28", "-tune", "zerolatency"]  # Fastest software
    
    temp_video = output_path.parent / f"{output_path.stem}_temp.mp4"
    
    try:
        logger.info("Writing temporary video file (optimized for speed)...")
        # Use hardware encoding for temp file if available
        video_clip.write_videofile(
            str(temp_video),
            fps=fps,
            codec=temp_codec,
            audio=False,
            ffmpeg_params=temp_params,
            logger=None,
            threads=4  # Use multiple threads
        )
        
        # Build FFmpeg command for final encoding with hardware acceleration
        # Use higher quality settings for final output
        cmd = [
            ffmpeg_path,
            "-y",
            "-i", str(temp_video),
            "-i", str(audio_path),
            "-c:v", codec,
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest"
        ]
        cmd.extend(codec_params)
        cmd.append(str(output_path))
        
        logger.info(f"Encoding final video with {codec}...")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        logger.info("Video encoding complete")
        return output_path
        
    finally:
        # Cleanup temporary file
        if temp_video.exists():
            temp_video.unlink()


# This is the "callable" version 
def render_video(
    audio_path: Path,
    timestamps_path: Path,
    output_path: Path
) -> Path:
    """
    Renders the final karaoke-style video using batch processing for faster rendering.
    
    Args:
        audio_path: Path to the PROCESSED audio file.
        timestamps_path: Path to the timestamps.json file.
        output_path: Path to save the final .mp4 video.

    Returns:
        The path to the rendered video.
    """
    logger.info("--- Starting Video Rendering Pipeline (Batch Processing) ---")
    
    temp_frames_dir = None
    
    try:
        logger.info(f"Loading audio: {audio_path}")
        audio_clip = AudioFileClip(str(audio_path))
        audio_duration = audio_clip.duration
        
        # --- Load background/config from settings ---
        background_path = settings.DEFAULT_BACKGROUND
        fps = settings.VIDEO_FPS
        width = settings.VIDEO_WIDTH
        height = settings.VIDEO_HEIGHT
        
        logger.info(f"Loading background: {background_path}")
        bg_clip = ImageClip(background_path).with_duration(audio_duration)

        # Initialize frame generator
        frame_gen = FrameGeneratorV11(
            timestamps_path=timestamps_path,
            bg_width=width,
            bg_height=height
        )

        # ====================================================================
        # PHASE 1: Pre-calculation
        # ====================================================================
        logger.info("--- Phase 1: Pre-calculating frame timestamps ---")
        frame_timestamps = _calculate_frame_timestamps(audio_duration, fps)
        logger.info(f"Calculated {len(frame_timestamps)} frames for {audio_duration:.2f}s video at {fps}fps")
        
        mapped_frames = _map_frames_to_slides(
            frame_timestamps,
            frame_gen.slide_start_times,
            audio_duration
        )
        logger.info(f"Mapped {len(mapped_frames)} frames to {len(frame_gen.slide_start_times)} slides")
        
        # Calculate optimal batch size - larger batches reduce overhead
        cpu_count = os.cpu_count() or 4
        # Use larger batches to reduce multiprocessing overhead
        # Target: 200-500 frames per batch for better efficiency
        batch_size = max(200, len(mapped_frames) // (cpu_count * 2))
        batches = _create_frame_batches(mapped_frames, batch_size)
        logger.info(f"Created {len(batches)} batches (batch size: {batch_size}, workers: {cpu_count})")
        
        # Serialize frame generator data for multiprocessing
        # Convert WordTimestamp objects to dicts for serialization
        # Use (line_index, word_index) as unique key to prevent collisions
        serialized_slides = []
        serialized_layouts = {}
        
        for slide_idx, slide in enumerate(frame_gen.slides):
            serialized_slide = []
            serialized_layout = {}
            
            # Get the layout for this slide
            layout = frame_gen.slide_layouts.get(slide_idx, {})
            
            for line_idx, line in enumerate(slide):
                serialized_line = []
                for word_idx, word in enumerate(line):
                    # Create unique key: (line_index, word_index)
                    unique_key = (line_idx, word_idx)
                    
                    # Store word data
                    word_data = {
                        'word': word.word,
                        'start': word.start,
                        'end': word.end
                    }
                    serialized_line.append(word_data)
                    
                    # Store layout coordinates using unique key
                    word_id = id(word)
                    if word_id in layout:
                        serialized_layout[unique_key] = layout[word_id]
                
                serialized_slide.append(serialized_line)
            
            serialized_slides.append(serialized_slide)
            serialized_layouts[slide_idx] = serialized_layout
        
        gen_data = {
            'slides': serialized_slides,
            'slide_layouts': serialized_layouts,
            'slide_start_times': frame_gen.slide_start_times,
            'font_size': frame_gen.font_size,
            'line_height': frame_gen.line_height,
            'max_text_width': frame_gen.max_text_width
        }
        
        # ====================================================================
        # PHASE 2: Parallel Frame Generation
        # ====================================================================
        logger.info("--- Phase 2: Generating frames in parallel batches ---")
        
        # Create temporary directory for frames
        temp_frames_dir = Path(tempfile.mkdtemp(prefix="video_frames_"))
        logger.info(f"Temporary frames directory: {temp_frames_dir}")
        
        # Prepare batch data for workers
        batch_data_list = [
            (batch, gen_data, temp_frames_dir, width, height)
            for batch in batches
        ]
        
        # Generate frames in parallel
        all_frame_files = []
        num_workers = min(cpu_count, len(batches))
        
        logger.info(f"Starting parallel frame generation with {num_workers} workers...")
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use tqdm for progress tracking
            # Use imap_unordered for better performance, then sort
            results = list(tqdm(
                pool.imap_unordered(_generate_frame_batch_worker, batch_data_list),
                total=len(batches),
                desc="Generating frames",
                unit="batch"
            ))
            
            # Flatten results
            for batch_files in results:
                all_frame_files.extend(batch_files)
        
        # Sort frame files by frame number (important for video sequence)
        all_frame_files.sort(key=lambda x: int(Path(x).stem.split('_')[1]))
        logger.info(f"Generated {len(all_frame_files)} frames")
        
        # ====================================================================
        # PHASE 3: Video Assembly
        # ====================================================================
        logger.info("--- Phase 3: Assembling video from frames ---")
        
        # Create video clip from frame images
        frame_clip = ImageSequenceClip(all_frame_files, fps=fps)
        
        # Composite with background
        final_video = CompositeVideoClip([bg_clip, frame_clip])
        
        # ====================================================================
        # PHASE 4: Optimized Encoding (Bypass MoviePy's slow encoding)
        # ====================================================================
        logger.info("--- Phase 4: Encoding video ---")
        logger.info(f"Rendering {fps}fps video to: {output_path}")
        
        # Use direct FFmpeg encoding for much faster performance
        _encode_video_with_ffmpeg(
            video_clip=final_video,
            audio_path=audio_path,
            output_path=output_path,
            fps=fps
        )
        
        logger.info("--- Video Rendering Complete ---")
        return output_path

    except Exception as e:
        logger.error("Video rendering pipeline failed!", exc_info=True)
        raise
    
    finally:
        # ====================================================================
        # PHASE 5: Cleanup
        # ====================================================================
        if temp_frames_dir and temp_frames_dir.exists():
            logger.info(f"Cleaning up temporary frames directory: {temp_frames_dir}")
            try:
                shutil.rmtree(temp_frames_dir)
                logger.info("Temporary files cleaned up successfully")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")