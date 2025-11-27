"""
final Merged Video Renderer - Combining renderer_v2's MoviePy approach with scalability for 10+ hour videos.

Key Features:
- MoviePy ImageSequenceClip rendering (from renderer_v2)
- Semantic slide building with abbreviation handling (from renderer_v2)
- Chunked rendering for memory efficiency (scalability pattern)
- Batch processing with multiprocessing (scalability pattern)
- Progress tracking and resource cleanup (scalability pattern)
- Supports both horizontal and vertical (reels) videos
"""

import json
import logging
import multiprocessing
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import bisect
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, VideoClip
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
MAX_LINES_PER_SLIDE = 4
MARGIN_PERCENT_X = 0.10
MARGIN_PERCENT_Y = 0.10

FONT_HEIGHT_DIVISOR = 8
LINE_HEIGHT_RATIO = 1.4

MIN_WORD_DURATION = 0.2
MAX_SILENCE_GAP = 0.3
PAUSE_THRESHOLD = 0.8

# Scalability settings for long videos
CHUNK_SIZE_MINUTES = 10  # Process video in 10-minute chunks to manage memory
BATCH_SIZE_PER_CPU = 50  # Frames per batch per CPU core
MAX_FRAMES_IN_MEMORY = 500  # Maximum frames to keep in memory at once
MIN_FREE_DISK_SPACE_GB = 10  # Minimum free disk space required (GB)

# Common abbreviations
COMMON_ABBREVS = {
    "mr.",
    "mrs.",
    "dr.",
    "ms.",
    "jr.",
    "sr.",
    "st.",
    "ave.",
    "rd.",
    "blvd.",
    "etc.",
    "e.g.",
    "i.e.",
}


# --- DATA MODELS ---
class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    display_word: Optional[str] = None
    id: int = 0


def interpolate_color(start_color, end_color, progress):
    """Smoothly interpolate between two RGBA colors."""
    return tuple(
        int(start_color[i] + (end_color[i] - start_color[i]) * progress)
        for i in range(4)
    )


# --- SANITIZER ---
def sanitize_words(raw_words: List[Dict]) -> List[WordTimestamp]:
    """Clean and normalize word timestamps from raw data."""
    clean_words = []
    for i, w in enumerate(raw_words):
        word_str = w["word"].strip()
        if not word_str:
            continue
        if word_str in [".", ",", "!", "?", ";", ":", '"', "'"]:
            if clean_words:
                clean_words[-1].display_word += word_str
            continue

        obj = WordTimestamp(
            word=word_str,
            start=float(w["start"]),
            end=float(w["end"]),
            display_word=word_str,
            id=i,
        )
        clean_words.append(obj)

    if not clean_words:
        return []

    # Ensure minimum word duration
    for i in range(len(clean_words)):
        curr = clean_words[i]
        if (curr.end - curr.start) < MIN_WORD_DURATION:
            curr.end = curr.start + MIN_WORD_DURATION
        if i < len(clean_words) - 1:
            next_w = clean_words[i + 1]
            if curr.end > next_w.start:
                next_w.start = curr.end
                if (next_w.end - next_w.start) < MIN_WORD_DURATION:
                    next_w.end = next_w.start + MIN_WORD_DURATION

    # Bridge small silences
    for i in range(len(clean_words) - 1):
        curr = clean_words[i]
        next_w = clean_words[i + 1]

        gap = next_w.start - curr.end
        max_gap_for_bridge = MAX_SILENCE_GAP

        # Tolerant for abbreviations (e.g., "Mr." followed by name)
        if curr.display_word and curr.display_word.lower().endswith("."):
            if curr.display_word.lower()[:-1] + "." in COMMON_ABBREVS:
                max_gap_for_bridge = 1.0

        if 0 < gap < max_gap_for_bridge:
            curr.end = next_w.start

    return clean_words


# --- FRAME GENERATOR ---
class FrameGeneratorMerged:
    """
    Frame generator with semantic slide building and scalability features.
    Handles both horizontal and vertical video formats.
    Supports smart detection for reels/shorts (vertical videos).
    """

    def __init__(
        self,
        timestamps_path: Path,
        bg_width: int,
        bg_height: int,
        font_size: Optional[int] = None,
        is_reels: bool = False,
    ):
        self.bg_width = bg_width
        self.bg_height = bg_height
        self.is_reels = is_reels

        # --- SMART DETECTION FOR REELS/SHORTS (matching renderer.py) ---
        if is_reels:
            # SHORTS MODE: Vertical video detection
            is_vertical = self.bg_height > self.bg_width

            if is_vertical:
                logger.info("Detected vertical video (Shorts/Reels mode)")
                # SHORTS MODE:
                # 1. Big Margins (Top/Bottom) to avoid TikTok/Reels UI overlay
                self.margin_x = int(self.bg_width * 0.12)  # 12% horizontal margins
                self.margin_y = int(
                    self.bg_height * 0.30
                )  # 30% vertical margins for UI space
                # 2. Text size (Calculated based on WIDTH, not height) - smaller for reels
                self.font_size = (
                    font_size if font_size else int(self.bg_width / 9)
                )  # Based on width/9
                # 3. Constraint: Only 4 lines per slide max for reels
                self.max_lines = 4
                # 4. Left alignment for reels (matching renderer.py)
                self.text_align = "left"
                logger.info(
                    f"Reels config: margins=({self.margin_x}, {self.margin_y}), font={self.font_size}, lines={self.max_lines}"
                )
            else:
                # Horizontal reels (shouldn't happen, but fallback)
                logger.info("Reels mode but horizontal video - using standard settings")
                self.margin_x = int(self.bg_width * 0.10)
                self.margin_y = int(self.bg_height * 0.10)
                self.font_size = font_size if font_size else int(self.bg_height / 8)
                self.max_lines = 4
                self.text_align = "left"
        else:
            # LAPTOP/LANDSCAPE MODE (Main video generation)
            self.margin_x = int(self.bg_width * MARGIN_PERCENT_X)
            self.margin_y = int(self.bg_height * MARGIN_PERCENT_Y)
            # Use provided font_size or calculate default
            self.font_size = (
                font_size if font_size else int(self.bg_height / FONT_HEIGHT_DIVISOR)
            )
            self.max_lines = 4
            self.text_align = "left"
            logger.info(
                f"Landscape config: margins=({self.margin_x}, {self.margin_y}), font={self.font_size}, lines={self.max_lines}"
            )

        self.usable_height = self.bg_height - 2 * self.margin_y
        self.max_text_width = self.bg_width - (2 * self.margin_x)
        self.line_height = int(self.font_size * LINE_HEIGHT_RATIO)

        # Load fonts
        self.regular_font = self._get_font(self.font_size, is_bold=False)
        self.bold_font = self._get_font(self.font_size, is_bold=True)

        self.all_words = self._load_and_clean_data(timestamps_path)

        self.slides = []
        self.slide_layouts = []
        self.slide_timings = []  # Stores tuples of (start, end)

        self._build_semantic_slides()
        self.slide_start_times = [s[0] for s in self.slide_timings]

        logger.info(
            f"FrameGeneratorMerged initialized: {len(self.all_words)} words in {len(self.slides)} slides"
        )

    def _load_and_clean_data(self, path: Path) -> List[WordTimestamp]:
        """Load and sanitize timestamp data."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_words = data.get("words", [])
        return sanitize_words(raw_words)

    def _get_font(self, size, is_bold=True):
        """Load font with fallback to default."""
        font_path = (
            settings.DEFAULT_FONT_BOLD if is_bold else settings.DEFAULT_FONT_REGULAR
        )
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            return ImageFont.load_default(size)

    def _is_abbreviation(self, word: str) -> bool:
        """Check if word is a common abbreviation."""
        lower_word = word.lower()
        if lower_word.endswith("."):
            base = lower_word[:-1] + "."
            return base in COMMON_ABBREVS
        return False

    def split_into_sentences(
        self, words: List[WordTimestamp]
    ) -> List[List[WordTimestamp]]:
        """
        Split words into sentences based on punctuation and pauses.
        Special handling for abbreviations to prevent incorrect splits.
        """
        sentences = []
        current = []

        for i in range(len(words)):
            current.append(words[i])
            word = words[i].display_word

            # Punctuation-based split
            if word.endswith((".", "!", "?")) and not self._is_abbreviation(word):
                if i == len(words) - 1 or words[i + 1].display_word[0].isupper():
                    sentences.append(current)
                    current = []

            # Pause-based split (skip after abbreviations followed by capitalized words)
            if i < len(words) - 1:
                gap = words[i + 1].start - words[i].end
                if gap > PAUSE_THRESHOLD:
                    skip_pause_split = (
                        current
                        and current[-1].display_word.endswith(".")
                        and self._is_abbreviation(current[-1].display_word)
                        and words[i + 1].display_word[0].isupper()
                    )
                    if not skip_pause_split:
                        if current:
                            sentences.append(current)
                        current = []

        if current:
            sentences.append(current)

        return sentences

    def _layout_into_lines(
        self, words: List[WordTimestamp]
    ) -> List[List[WordTimestamp]]:
        """Layout words into lines based on available width."""
        font = self._get_font(self.font_size)
        dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        space_width = dummy.textlength(" ", font=font)

        lines = []
        current_line = []
        current_w = 0

        for word in words:
            wl = dummy.textlength(word.display_word, font=font)
            if current_line and current_w + space_width + wl > self.max_text_width:
                lines.append(current_line)
                current_line = [word]
                current_w = wl
            else:
                if current_line:
                    current_w += space_width
                current_line.append(word)
                current_w += wl

        if current_line:
            lines.append(current_line)

        # Merge short lines to avoid orphans (single words or very short lines)
        # Keep merging until no more short lines can be merged
        improved = True
        max_iterations = 10  # Prevent infinite loop
        iteration = 0

        while improved and len(lines) > 1 and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(len(lines) - 1, 0, -1):  # Iterate backwards
                current_line_words = lines[i]
                prev_line_words = lines[i - 1]

                # Merge if current line has ≤2 words
                if len(current_line_words) <= 2:
                    # Calculate combined width
                    current_w = sum(
                        dummy.textlength(w.display_word, font=font) + space_width
                        for w in current_line_words
                    )
                    if len(current_line_words) > 0:
                        current_w -= space_width  # Remove trailing space

                    prev_w = sum(
                        dummy.textlength(w.display_word, font=font) + space_width
                        for w in prev_line_words
                    )
                    if len(prev_line_words) > 0:
                        prev_w -= space_width  # Remove trailing space

                    combined_w = prev_w + space_width + current_w

                    # Merge if they fit together
                    if combined_w <= self.max_text_width:
                        lines[i - 1].extend(lines[i])
                        lines.pop(i)
                        improved = True
                        break  # Restart the loop

        return lines

    def _is_small_chunk(self, lines: List[List[WordTimestamp]]) -> bool:
        """Check if a chunk is small (2 words or less)."""
        return sum(len(line) for line in lines) <= 2

    def _split_slide_semantically(
        self, lines: List[List[WordTimestamp]]
    ) -> Tuple[List[List[WordTimestamp]], List[List[WordTimestamp]]]:
        """Split slide at semantic boundaries (commas, semicolons) for better flow."""
        words = [w for line in lines for w in line]
        if len(words) <= 3:
            return lines, []

        # Find punctuation candidates for splitting
        candidates = [
            i
            for i, w in enumerate(words)
            if w.display_word.endswith((",", ";", ":")) and i > len(words) * 0.2
        ]
        if candidates:
            best_i = min(candidates, key=lambda x: abs(x - len(words) / 2))
            split_idx = best_i + 1
        else:
            split_idx = len(words) // 2

        return self._layout_into_lines(words[:split_idx]), self._layout_into_lines(
            words[split_idx:]
        )

    def _build_semantic_slides(self):
        """Build slides with semantic awareness for better reading flow."""
        sentences = self.split_into_sentences(self.all_words)
        current_slide_lines = []
        MAX = self.max_lines

        for sent_words in sentences:
            sent_lines = self._layout_into_lines(sent_words)
            chunk_start = 0

            while chunk_start < len(sent_lines):
                chunk_lines = sent_lines[chunk_start : chunk_start + MAX]
                is_small = self._is_small_chunk(chunk_lines)

                # Start new slide if current is full and chunk is not small
                if chunk_start == 0 and len(current_slide_lines) >= 3 and not is_small:
                    if current_slide_lines:
                        self._commit_slide(current_slide_lines)
                    current_slide_lines = []

                potential = len(current_slide_lines) + len(chunk_lines)
                if potential > MAX:
                    if is_small and current_slide_lines:
                        # Split current slide semantically
                        first, second = self._split_slide_semantically(
                            current_slide_lines
                        )
                        self._commit_slide(first)
                        current_slide_lines = second

                        if len(current_slide_lines) + len(chunk_lines) <= MAX:
                            current_slide_lines.extend(chunk_lines)
                        else:
                            self._commit_slide(current_slide_lines)
                            current_slide_lines = chunk_lines
                    else:
                        self._commit_slide(current_slide_lines)
                        current_slide_lines = chunk_lines
                else:
                    current_slide_lines.extend(chunk_lines)

                chunk_start += MAX

        if current_slide_lines:
            self._commit_slide(current_slide_lines)

    def generate_single_frame_hybrid(
        self, t: float, bg_image: Image.Image
    ) -> Image.Image:
        """
        Generates a frame with:
        1. Slide Persistence (No white flash between slides)
        2. Smooth Color Interpolation (Fades)
        3. Bold/Regular Font switching
        """

        # 1. FIND ACTIVE SLIDE (PERSISTENT LOGIC)
        # Use bisect to find the slide that started most recently
        # bisect_right returns insertion point to right of t. Subtract 1 to get the index.
        idx = bisect.bisect_right(self.slide_start_times, t) - 1

        # Use this slide if valid. Even if t > slide.end, we keep showing it
        # until t >= next_slide.start (which will be caught by bisect in the next frame)
        slide_index = -1
        if 0 <= idx < len(self.slides):
            slide_index = idx

        # 2. Setup Frame
        frame = bg_image.copy().convert("RGBA")

        # If no slide has started yet (intro silence), return background
        if slide_index == -1:
            return frame

        draw = ImageDraw.Draw(frame)
        slide_lines = self.slides[slide_index]
        layout = self.slide_layouts[slide_index]

        # Colors & Timing Constants
        C_ACTIVE = (20, 20, 20, 255)  # Almost Black
        C_FUTURE = (210, 210, 210, 255)  # Light Gray

        # 3. Render Words with Animation
        for line in slide_lines:
            for word in line:
                coords = layout.get(word.id)
                if not coords:
                    continue
                x, y = coords

                start = word.start

                # Determine animation parameters based on word type
                is_abbrev = word.display_word.lower() in COMMON_ABBREVS
                fade_duration = 0.50 if is_abbrev else 0.30

                # --- ANIMATION LOGIC ---
                if t >= start:
                    # Active Word (Spoken)
                    color = C_ACTIVE
                    font = self.bold_font

                elif start - fade_duration <= t < start:
                    # Fading In (Smooth Transition)
                    progress = (t - (start - fade_duration)) / fade_duration
                    color = interpolate_color(C_FUTURE, C_ACTIVE, progress)
                    # Switch to bold halfway through fade for visual "pop"
                    font = self.bold_font if progress > 0.5 else self.regular_font

                else:
                    # Future Word (Waiting)
                    color = C_FUTURE
                    font = self.regular_font

                draw.text((x, y), word.display_word, font=font, fill=color)

        return frame

    def _commit_slide(self, lines: List[List[WordTimestamp]]):
        """Commit a slide with its layout and timing information."""
        if not lines:
            return

        # Remove trailing comma from last word
        last_word = lines[-1][-1]
        if last_word.display_word.endswith(","):
            last_word.display_word = last_word.display_word[:-1]

        # Calculate vertical centering
        total_h = len(lines) * self.line_height
        start_y = self.margin_y + (self.usable_height - total_h) // 2

        # Build layout for each word
        layout = {}
        font = self._get_font(self.font_size)
        dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        space_w = dummy.textlength(" ", font=font)

        y = start_y
        for line in lines:
            if self.text_align == "center":
                # Calculate line width for centering
                line_width = 0
                for w in line:
                    line_width += dummy.textlength(w.display_word, font=font)
                if len(line) > 1:
                    line_width += (len(line) - 1) * space_w
                x = (self.bg_width - line_width) // 2
            else:
                # Left align
                x = self.margin_x

            # Position words with space BETWEEN them, not after the last one
            for word_idx, word in enumerate(line):
                layout[word.id] = (int(x), int(y))
                x += dummy.textlength(word.display_word, font=font)
                # Only add space if not the last word in line
                if word_idx < len(line) - 1:
                    x += space_w
            y += self.line_height

        self.slides.append(lines)
        self.slide_layouts.append(layout)
        s_start = lines[0][0].start
        s_end = lines[-1][-1].end
        self.slide_timings.append((s_start, s_end))


# --- MAIN RENDER FUNCTION ---
# --- MAIN RENDER FUNCTION (FIXED) ---
def render_video(
    audio_path: Path,
    timestamps_path: Path,
    output_path: Path,
    background_path: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    font_size: Optional[int] = None,
) -> Path:
    """
    Renders the final karaoke-style video with hybrid rendering.

    Args:
        audio_path: Path to the PROCESSED audio file.
        timestamps_path: Path to the timestamps.json file.
        output_path: Path to save the final .mp4 video.
        background_path: Optional custom background image path (for reels/shorts).
        width: Optional custom video width (for reels/shorts).
        height: Optional custom video height (for reels/shorts).
        font_size: Optional custom font size (for reels/shorts - smaller text).

    Returns:
        The path to the rendered video.
    """
    logger.info("--- Starting Video Rendering (Hybrid Fixed) ---")

    try:
        # 1. Load Audio
        logger.info(f"Loading audio: {audio_path}")
        audio_clip = AudioFileClip(str(audio_path))
        audio_duration = audio_clip.duration
        fps = settings.VIDEO_FPS

        # 2. Setup Background & Dimensions (with reels detection)
        # --- Load or generate background (matching renderer.py logic) ---
        if background_path is None:
            # No background provided
            if width is not None and height is not None:
                # Dimensions explicitly provided - check if reels (9:16, typically 1080x1920)
                is_reels = (width == 1080 and height == 1920) or (height > width)

                if is_reels:
                    # Generate solid white background programmatically for reels
                    logger.info(
                        f"Generating solid white background for reels: {width}x{height}"
                    )
                    bg_pil = Image.new(
                        "RGB", (width, height), (255, 255, 235)
                    )  # Solid white
                    logger.info(
                        f"Created white background programmatically: {width}x{height}"
                    )
                else:
                    # Main video - use default background
                    background_path = Path(settings.DEFAULT_BACKGROUND)
                    logger.info(f"Using default background: {background_path}")
                    bg_pil = Image.open(str(background_path))

                    # Resize background if needed
                    if bg_pil.size != (width, height):
                        logger.info(
                            f"Resizing background from {bg_pil.size} to ({width}, {height})"
                        )
                        bg_pil = bg_pil.resize(
                            (width, height), Image.Resampling.LANCZOS
                        )

                    is_reels = height > width
            else:
                # No dimensions provided - use default background and get dimensions from it
                background_path = Path(settings.DEFAULT_BACKGROUND)
                logger.info(f"Using default background: {background_path}")
                bg_pil = Image.open(str(background_path))
                width, height = bg_pil.size
                logger.info(f"Background image dimensions: {width}x{height}")
                is_reels = height > width
        else:
            # Custom background provided - load from file
            background_path = Path(background_path)
            logger.info(f"Loading custom background: {background_path}")
            bg_pil = Image.open(str(background_path))
            actual_width, actual_height = bg_pil.size

            # Use provided dimensions or actual image dimensions
            if width is None:
                width = actual_width
            if height is None:
                height = actual_height

            logger.info(f"Background image dimensions: {actual_width}x{actual_height}")
            logger.info(f"Using video dimensions: {width}x{height}")

            # Resize background image if dimensions don't match
            if bg_pil.size != (width, height):
                logger.info(
                    f"Resizing background from {bg_pil.size} to ({width}, {height})"
                )
                bg_pil = bg_pil.resize((width, height), Image.Resampling.LANCZOS)

            # Detect reels based on final dimensions
            is_reels = height > width

        logger.info(f"Final video dimensions: {width}x{height}, is_reels={is_reels}")

        # 3. Initialize Generator with reels mode
        frame_gen = FrameGeneratorMerged(
            timestamps_path=timestamps_path,
            bg_width=width,
            bg_height=height,
            font_size=font_size,
            is_reels=is_reels,
        )

        # Background is already loaded in bg_pil from above logic
        # Convert to RGB for rendering (no alpha channel needed for background)
        bg_pil = bg_pil.convert("RGB")

        # 4. Make Frame Function (CRITICAL FIXES HERE)
        def make_frame(t):
            # Generate frame in RGBA
            frame_img = frame_gen.generate_single_frame_hybrid(t, bg_pil)

            # FIX 1: Convert RGBA -> RGB.
            # Removing the Alpha channel prevents the "barcode/glitch" effect.
            frame_rgb = frame_img.convert("RGB")

            return np.array(frame_rgb)

        # 5. Write Video (CRITICAL FIXES HERE)
        video = VideoClip(make_frame, duration=audio_duration)
        video = video.with_audio(audio_clip)

        logger.info(f"Rendering video to {output_path}...")

        video.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            # FIX 2: REMOVED fixed 'bitrate="8000k"'.
            # Replaced with 'ffmpeg_params' for optimized size/quality.
            preset="veryfast",  # Balanced speed/size (ultrafast is too big)
            ffmpeg_params=[
                "-crf",
                "23",
            ],  # CRF 23 is the standard for web video (small size, high quality)
            threads=multiprocessing.cpu_count(),
            logger="bar",
        )

        logger.info("--- Video Rendering Complete ---")
        return output_path

    except Exception:
        logger.error("Hybrid rendering failed!", exc_info=True)
        raise
