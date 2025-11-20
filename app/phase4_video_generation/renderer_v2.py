# import json
# import logging
# import os
# import tempfile
# import multiprocessing
# import shutil
# import sys
# from pathlib import Path
# from typing import List, Optional, Dict, Tuple, Any
# import re

# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# from moviepy import AudioFileClip, CompositeVideoClip, ImageSequenceClip, ImageClip
# from pydantic import BaseModel
# from tqdm import tqdm

# from app.config import settings

# logger = logging.getLogger(__name__)

# # --- DATA MODELS ---
# class WordTimestamp(BaseModel):
#     word: str
#     start: float
#     end: float
#     display_word: Optional[str] = None
#     is_sentence_end: bool = False  # New flag for logic
#     is_sentence_start: bool = False # New flag for logic

# # --- UTILS ---
# def interpolate_color(start_color, end_color, progress):
#     """Blends two colors based on progress (0.0 to 1.0)."""
#     return tuple(
#         int(start_color[i] + (end_color[i] - start_color[i]) * progress)
#         for i in range(4)
#     )

# # --- GENERATOR CLASS ---
# class FrameGeneratorFounders:
#     def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
#         self.bg_width = bg_width
#         self.bg_height = bg_height
#         self.all_words = self._load_data(timestamps_path)

#         # --- Founders Style Config ---
#         # 1080p logic:
#         self.font_size = int(self.bg_height / 14) # Slightly smaller for elegance
#         self.line_height = int(self.font_size * 1.4) 
        
#         # STRICT LEFT ALIGNMENT (Fixed Margin)
#         # The text will ALWAYS start here. No centering calculations.
#         self.margin_left = int(self.bg_width * 0.12) 
#         self.margin_right = int(self.bg_width * 0.12)
#         self.max_text_width = self.bg_width - self.margin_left - self.margin_right
        
#         self.regular_font, self.bold_font = self._load_fonts(self.font_size)
        
#         # Build "Smart" Slides
#         self.slides, self.slide_layouts, self.slide_timings = self._build_semantic_slides()

#     def _load_fonts(self, size):
#         try:
#             reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, size)
#             bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
#             return reg, bold
#         except:
#             return ImageFont.load_default(size), ImageFont.load_default(size)

#     def _load_data(self, path: Path) -> List[WordTimestamp]:
#         """
#         Loads data and RECONSTRUCTS sentence boundaries since Google TTS 
#         might strip punctuation from the 'word' field.
#         """
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
        
#         full_text = data.get("text", "")
        
#         # Flatten words
#         raw_words = data.get("words", [])
#         if not raw_words and data.get("segments"):
#             for seg in data["segments"]:
#                 raw_words.extend(seg.get("words", []))

#         # 1. Re-align with full text to find punctuation
#         # This is critical because 'mark_name' often lacks the '.' 
#         # We try to match the word stream to the text stream.
#         processed_words = []
        
#         # Simple split of raw text to check for punctuation
#         text_tokens = full_text.split()
        
#         for i, w in enumerate(raw_words):
#             clean_w = w.get('word', '').strip()
#             if not clean_w: continue
            
#             obj = WordTimestamp(**w)
#             obj.word = clean_w
#             obj.display_word = clean_w
            
#             # Heuristic: Check if the word looks like a sentence start/end
#             # (If we had perfect mapping from text_tokens we'd use that, 
#             # but for now we use the 'display_word' or simple capitalization logic)
            
#             # Check if this word ends with punctuation in the raw data
#             if clean_w[-1] in ['.', '!', '?']:
#                 obj.is_sentence_end = True
            
#             # Check if it is a sentence starter (Capitalized and previous was end)
#             if i > 0 and clean_w[0].isupper() and clean_w.lower() not in ["i", "i'm", "i'll"]:
#                 # If previous word ended sentence, or we assume it does if caps
#                 pass 

#             processed_words.append(obj)
            
#         # Second Pass: Force boundaries based on punctuation
#         for i in range(len(processed_words) - 1):
#             curr = processed_words[i]
#             nxt = processed_words[i+1]
            
#             # If current has punctuation, it's an end
#             if curr.word[-1] in ['.', '!', '?']:
#                 curr.is_sentence_end = True
#                 nxt.is_sentence_start = True
            
#             # Fix: If 'next' is "This", "The", "And" capitalized, it likely starts a sentence
#             # This handles the specific error you mentioned ("This" at end of line)
#             if nxt.word[0].isupper() and nxt.word.lower() not in ["i", "god", "monday", "sunday", "january"]:
#                  # Only split if previous word looks like an end OR time gap is large
#                  if curr.end - nxt.start < -0.3 or curr.is_sentence_end:
#                      nxt.is_sentence_start = True

#         return processed_words

#     def _build_semantic_slides(self):
#         slides = []
#         layouts = {}
#         timings = []

#         dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
#         space_width = dummy_draw.textlength(" ", font=self.bold_font)

#         HANGING_WORDS = {'to', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'for', 'and', 'but', 'or', 'with'}

#         def commit_slide(words_list):
#             if not words_list: return
            
#             lines = []
#             current_line = []
#             current_line_width = 0
            
#             for i, word in enumerate(words_list):
#                 w_len = dummy_draw.textlength(word.display_word, font=self.bold_font)
#                 is_overflow = (current_line_width + space_width + w_len > self.max_text_width)
                
#                 if current_line and is_overflow:
#                     # Hanging word check
#                     if current_line and current_line[-1].display_word.lower() in HANGING_WORDS:
#                         moved_word = current_line.pop()
#                         lines.append(current_line)
#                         current_line = [moved_word, word]
#                         current_line_width = dummy_draw.textlength(moved_word.display_word, font=self.bold_font) + space_width + w_len
#                     else:
#                         lines.append(current_line)
#                         current_line = [word]
#                         current_line_width = w_len
#                 else:
#                     if current_line: current_line_width += space_width
#                     current_line.append(word)
#                     current_line_width += w_len
            
#             if current_line: lines.append(current_line)

#             # Punctuation Polish
#             for line in lines:
#                 if not line: continue
#                 last_w = line[-1]
#                 # Strip trailing punctuation for the clean look, 
#                 # BUT keep it if it's a mid-sentence pause (comma) 
#                 # The user said: "if sentence is pausing by , don't display comma... but if mid then show it"
#                 # Actually, standard clean style: Remove periods at end of slide. Keep commas inside.
#                 if last_w.display_word[-1] == '.':
#                     last_w.display_word = last_w.display_word[:-1]

#             # Layout Calculation
#             slide_layout = {}
#             total_h = len(lines) * self.line_height
#             # Center Vertically
#             start_y = (self.bg_height - total_h) // 2
            
#             curr_y = start_y
#             for line in lines:
#                 # STRICT LEFT ALIGNMENT
#                 curr_x = self.margin_left 
#                 for word in line:
#                     slide_layout[id(word)] = (curr_x, curr_y)
#                     w_len = dummy_draw.textlength(word.display_word, font=self.bold_font)
#                     curr_x += w_len + space_width
#                 curr_y += self.line_height

#             slides.append(lines)
#             layouts[len(slides)-1] = slide_layout
#             timings.append((words_list[0].start, words_list[-1].end))

#         # --- THE FIX FOR "THIS" AT END ---
#         current_chunk = []
#         for i, w in enumerate(self.all_words):
#             # If this word explicitly starts a new sentence, commit previous chunk
#             if w.is_sentence_start and current_chunk:
#                 commit_slide(current_chunk)
#                 current_chunk = []
            
#             current_chunk.append(w)
            
#             # Check for Split Triggers
#             is_end_of_sentence = w.is_sentence_end
#             is_too_long = len(current_chunk) > 12
            
#             # Peek next word to prevent splitting "The [Next Word]"
#             next_word = self.all_words[i+1] if i < len(self.all_words) - 1 else None
            
#             if is_end_of_sentence:
#                 commit_slide(current_chunk)
#                 current_chunk = []
#             elif is_too_long and w.display_word.lower() not in HANGING_WORDS:
#                 # Don't split if next word is punctuation
#                 if next_word and next_word.word in ['.', ',', '!', '?']:
#                     continue
#                 commit_slide(current_chunk)
#                 current_chunk = []

#         if current_chunk: commit_slide(current_chunk)

#         return slides, layouts, timings

# # --- BATCH WORKER ---
# def _generate_frame_batch_worker_v2(batch_data):
#     frame_tasks, gen_data, output_dir, width, height = batch_data
    
#     try:
#         reg_font = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, gen_data['font_size'])
#         bold_font = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, gen_data['font_size'])
#     except:
#         reg_font = ImageFont.load_default(gen_data['font_size'])
#         bold_font = ImageFont.load_default(gen_data['font_size'])

#     # Visual Style: Clean Black Text
#     C_REG = (180, 180, 180, 255) 
#     C_BOLD = (0, 0, 0, 255)     
#     FADE_MS = 0.15               

#     generated = []
#     for frame_num, ts, slide_idx, _ in frame_tasks:
#         img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
#         draw = ImageDraw.Draw(img)
        
#         slide_lines = gen_data['slides'][slide_idx]
#         layout = gen_data['layouts'][slide_idx]
        
#         for line in slide_lines:
#             for w_data in line:
#                 coords = layout.get(str(w_data['id']))
#                 if not coords: continue
                
#                 start, end = w_data['start'], w_data['end']
                
#                 if ts < start:
#                     color = C_REG
#                     font = reg_font
#                 elif ts >= start + FADE_MS:
#                     color = C_BOLD
#                     font = bold_font
#                 else:
#                     prog = (ts - start) / FADE_MS
#                     color = interpolate_color(C_REG, C_BOLD, prog)
#                     font = bold_font if prog > 0.5 else reg_font

#                 draw.text(coords, w_data['display_word'], font=font, fill=color)
        
#         path = output_dir / f"f_{frame_num:06d}.png"
#         img.save(path, optimize=False, compress_level=0)
#         generated.append(str(path))
        
#     return generated

# # --- MAIN RENDER FUNCTION ---
# def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
#     """Main V2 Rendering Entry Point."""
#     logger.info("--- Starting V2 Render (Founders Style) ---")
#     temp_frames_dir = None
    
#     try:
#         # 1. Setup
#         audio_clip = AudioFileClip(str(audio_path))
#         duration = audio_clip.duration
#         fps = settings.VIDEO_FPS
#         width = settings.VIDEO_WIDTH
#         height = settings.VIDEO_HEIGHT
        
#         bg_clip = ImageClip(settings.DEFAULT_BACKGROUND).with_duration(duration)
        
#         gen = FrameGeneratorFounders(timestamps_path, width, height)
        
#         # 2. Map Frames
#         total_frames = int(duration * fps)
#         frame_tasks = []
        
#         for i in range(total_frames):
#             ts = i / fps
#             active_slide_idx = 0
#             for s_idx, (start, end) in enumerate(gen.slide_timings):
#                 if start <= ts <= end + 0.8: # Hold slide slightly longer
#                     active_slide_idx = s_idx
#                     break
#                 if ts > end:
#                     active_slide_idx = min(s_idx + 1, len(gen.slides)-1)
            
#             frame_tasks.append((i, ts, active_slide_idx, gen.slide_timings[active_slide_idx][0]))

#         # 3. Serialize
#         ser_slides = []
#         ser_layouts = {}
#         for idx, lines in enumerate(gen.slides):
#             s_lines = []
#             for line in lines:
#                 s_line = []
#                 for word in line:
#                     s_line.append({
#                         'id': id(word),
#                         'display_word': word.display_word,
#                         'start': word.start,
#                         'end': word.end
#                     })
#                 s_lines.append(s_line)
#             ser_slides.append(s_lines)
#             ser_layouts[idx] = {str(k): v for k, v in gen.slide_layouts[idx].items()}

#         gen_data = {'slides': ser_slides, 'layouts': ser_layouts, 'font_size': gen.font_size}

#         # 4. Render
#         temp_frames_dir = Path(tempfile.mkdtemp(prefix="v2_frames_"))
#         cpu_count = os.cpu_count() or 4
#         batch_size = max(50, len(frame_tasks) // cpu_count)
#         batches = [frame_tasks[i:i + batch_size] for i in range(0, len(frame_tasks), batch_size)]
        
#         worker_args = [(b, gen_data, temp_frames_dir, width, height) for b in batches]
        
#         logger.info(f"Rendering {len(frame_tasks)} frames...")
        
#         all_files = []
#         with multiprocessing.Pool(cpu_count) as pool:
#             # Using sys.stderr to force progress bar visibility
#             results = list(tqdm(pool.imap(_generate_frame_batch_worker_v2, worker_args), 
#                               total=len(batches), file=sys.stderr))
#             for r in results: all_files.extend(r)
            
#         all_files.sort(key=lambda x: int(Path(x).stem.split('_')[1]))
        
#         frame_clip = ImageSequenceClip(all_files, fps=fps)
#         final = CompositeVideoClip([bg_clip, frame_clip])
        
#         final.write_videofile(
#             str(output_path),
#             audio=str(audio_path),
#             fps=fps,
#             codec="libx264",
#             preset="veryfast",
#             threads=4,
#             logger=None 
#         )
        
#         return output_path

#     except Exception as e:
#         logger.error("Render Failed", exc_info=True)
#         raise
#     finally:
#         if temp_frames_dir and temp_frames_dir.exists():
#             shutil.rmtree(temp_frames_dir)



import json
import logging
import os
import tempfile
import multiprocessing
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, CompositeVideoClip, ImageSequenceClip, ImageClip
from pydantic import BaseModel
from tqdm import tqdm

from app.config import settings

logger = logging.getLogger(__name__)

# --- DATA MODELS ---
class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    display_word: Optional[str] = None

# --- UTILS ---
def interpolate_color(start_color, end_color, progress):
    """Blends two colors based on progress (0.0 to 1.0)."""
    return tuple(
        int(start_color[i] + (end_color[i] - start_color[i]) * progress)
        for i in range(4)
    )

# --- GENERATOR CLASS ---
class FrameGeneratorFounders:
    def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
        self.bg_width = bg_width
        self.bg_height = bg_height
        
        # Load data grouped by SENTENCES (Segments)
        self.segments_of_words = self._load_data_grouped(timestamps_path)

        # --- Founders Style Config ---
        # 1080p logic:
        self.font_size = int(self.bg_height / 15) # Good size for paragraph reading
        self.line_height = int(self.font_size * 1.4) 
        
        # STRICT LEFT ALIGNMENT (Fixed Margin)
        self.margin_left = int(self.bg_width * 0.10) # 10% margin
        self.margin_right = int(self.bg_width * 0.10)
        self.max_text_width = self.bg_width - self.margin_left - self.margin_right
        
        # Area safe for text (vertically)
        self.max_lines_per_slide = 8 # Force up to 8 lines per screen to "fill" it
        
        self.regular_font, self.bold_font = self._load_fonts(self.font_size)
        
        # Build "Smart" Slides
        self.slides, self.slide_layouts, self.slide_timings = self._build_paragraph_slides()

    def _load_fonts(self, size):
        try:
            reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, size)
            bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
            return reg, bold
        except:
            return ImageFont.load_default(size), ImageFont.load_default(size)

    def _load_data_grouped(self, path: Path) -> List[List[WordTimestamp]]:
        """
        Loads data but keeps it grouped by WHISPER SEGMENTS (Sentences).
        This is key to preventing the "This" word appearing at the end of a slide.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        grouped_segments = []
        
        # Use the 'segments' list from Whisper to group words
        raw_segments = data.get("segments", [])
        all_words_flat = data.get("words", [])
        
        # Logic: Map flat words to segments based on time
        # This ensures we have word-level precision but sentence-level grouping
        
        current_word_idx = 0
        
        for seg in raw_segments:
            seg_start = seg['start']
            seg_end = seg['end']
            segment_words = []
            
            # If segments have their own 'words' key (some Whisper versions)
            if 'words' in seg:
                for w in seg['words']:
                    clean_w = w['word'].strip()
                    obj = WordTimestamp(word=clean_w, start=w['start'], end=w['end'], display_word=clean_w)
                    segment_words.append(obj)
            else:
                # Fallback: grabbing words from the flat list that fit this time window
                # giving a small buffer for drift
                while current_word_idx < len(all_words_flat):
                    w = all_words_flat[current_word_idx]
                    # If word starts before segment ends, it belongs here (mostly)
                    if w['start'] < seg_end + 0.1: 
                        clean_w = w['word'].strip()
                        obj = WordTimestamp(word=clean_w, start=w['start'], end=w['end'], display_word=clean_w)
                        segment_words.append(obj)
                        current_word_idx += 1
                    else:
                        break
            
            # Add punctuation from the segment text to the last word if missing
            if segment_words:
                seg_text = seg['text'].strip()
                last_char = seg_text[-1] if seg_text else ""
                if last_char in ['.', '?', '!'] and segment_words[-1].display_word[-1] not in ['.', '?', '!']:
                    segment_words[-1].display_word += last_char

            if segment_words:
                grouped_segments.append(segment_words)

        return grouped_segments

    def _build_paragraph_slides(self):
        """
        Packs multiple segments (sentences) onto a slide until full.
        NEVER splits a sentence between slides.
        """
        slides = []
        layouts = {}
        timings = []

        dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        space_width = dummy_draw.textlength(" ", font=self.bold_font)

        current_slide_segments = []
        current_slide_line_count = 0
        
        def calculate_lines_for_segment(segment_words):
            """Simulates how many lines a sentence will take."""
            lines = 1
            current_width = 0
            for word in segment_words:
                w_len = dummy_draw.textlength(word.display_word, font=self.bold_font)
                if current_width + space_width + w_len > self.max_text_width:
                    lines += 1
                    current_width = w_len
                else:
                    current_width += space_width + w_len
            return lines

        def commit_slide(segments_list):
            if not segments_list: return
            
            # Flatten segments into one list of words for layout
            all_words_on_slide = [w for seg in segments_list for w in seg]
            
            lines = []
            current_line = []
            current_line_width = 0
            
            for word in all_words_on_slide:
                w_len = dummy_draw.textlength(word.display_word, font=self.bold_font)
                is_overflow = (current_line_width + space_width + w_len > self.max_text_width)
                
                if current_line and is_overflow:
                    lines.append(current_line)
                    current_line = [word]
                    current_line_width = w_len
                else:
                    if current_line: current_line_width += space_width
                    current_line.append(word)
                    current_line_width += w_len
            if current_line: lines.append(current_line)

            # --- LAYOUT CALCULATION ---
            slide_layout = {}
            total_h = len(lines) * self.line_height
            start_y = (self.bg_height - total_h) // 2 # Centered Block
            
            curr_y = start_y
            for line in lines:
                curr_x = self.margin_left
                for word in line:
                    slide_layout[id(word)] = (curr_x, curr_y)
                    w_len = dummy_draw.textlength(word.display_word, font=self.bold_font)
                    curr_x += w_len + space_width
                curr_y += self.line_height

            slides.append(lines)
            layouts[len(slides)-1] = slide_layout
            # Timing: Start of first word in first segment to end of last word in last segment
            s_start = segments_list[0][0].start
            s_end = segments_list[-1][-1].end
            timings.append((s_start, s_end))

        # --- MAIN LOOP: SEGMENT PACKING ---
        for segment in self.segments_of_words:
            # 1. How big is this sentence?
            lines_needed = calculate_lines_for_segment(segment)
            
            # 2. Does it fit on current slide?
            if current_slide_line_count + lines_needed > self.max_lines_per_slide:
                # Slide is full. Commit it.
                commit_slide(current_slide_segments)
                # Start new slide with this sentence
                current_slide_segments = [segment]
                current_slide_line_count = lines_needed
            else:
                # It fits! Add to pile.
                current_slide_segments.append(segment)
                current_slide_line_count += lines_needed
                
                # If we force a new line between sentences (optional, looks cleaner)
                # current_slide_line_count += 1 

        if current_slide_segments:
            commit_slide(current_slide_segments)

        return slides, layouts, timings

# --- BATCH WORKER ---
def _generate_frame_batch_worker_v2(batch_data):
    frame_tasks, gen_data, output_dir, width, height = batch_data
    
    try:
        reg_font = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, gen_data['font_size'])
        bold_font = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, gen_data['font_size'])
    except:
        reg_font = ImageFont.load_default(gen_data['font_size'])
        bold_font = ImageFont.load_default(gen_data['font_size'])

    C_REG = (180, 180, 180, 255) 
    C_BOLD = (0, 0, 0, 255)     
    FADE_MS = 0.15               

    generated = []
    for frame_num, ts, slide_idx, _ in frame_tasks:
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        slide_lines = gen_data['slides'][slide_idx]
        layout = gen_data['layouts'][slide_idx]
        
        for line in slide_lines:
            for w_data in line:
                coords = layout.get(str(w_data['id']))
                if not coords: continue
                
                start, end = w_data['start'], w_data['end']
                
                if ts < start:
                    color = C_REG
                    font = reg_font
                elif ts >= start + FADE_MS:
                    color = C_BOLD
                    font = bold_font
                else:
                    prog = (ts - start) / FADE_MS
                    color = interpolate_color(C_REG, C_BOLD, prog)
                    font = bold_font if prog > 0.5 else reg_font

                draw.text(coords, w_data['display_word'], font=font, fill=color)
        
        path = output_dir / f"f_{frame_num:06d}.png"
        img.save(path, optimize=False, compress_level=0)
        generated.append(str(path))
        
    return generated

# --- MAIN RENDER FUNCTION ---
def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
    """Main V2 Rendering Entry Point."""
    logger.info("--- Starting V2 Render (Founders Style - Paragraph Mode) ---")
    temp_frames_dir = None
    
    try:
        # 1. Setup
        audio_clip = AudioFileClip(str(audio_path))
        duration = audio_clip.duration
        fps = settings.VIDEO_FPS
        width = settings.VIDEO_WIDTH
        height = settings.VIDEO_HEIGHT
        
        bg_clip = ImageClip(settings.DEFAULT_BACKGROUND).with_duration(duration)
        
        gen = FrameGeneratorFounders(timestamps_path, width, height)
        
        # 2. Map Frames
        total_frames = int(duration * fps)
        frame_tasks = []
        
        for i in range(total_frames):
            ts = i / fps
            active_slide_idx = 0
            for s_idx, (start, end) in enumerate(gen.slide_timings):
                if start <= ts <= end + 0.5: 
                    active_slide_idx = s_idx
                    break
                if ts > end:
                    active_slide_idx = min(s_idx + 1, len(gen.slides)-1)
            
            frame_tasks.append((i, ts, active_slide_idx, gen.slide_timings[active_slide_idx][0]))

        # 3. Serialize
        ser_slides = []
        ser_layouts = {}
        for idx, lines in enumerate(gen.slides):
            s_lines = []
            for line in lines:
                s_line = []
                for word in line:
                    s_line.append({
                        'id': id(word),
                        'display_word': word.display_word,
                        'start': word.start,
                        'end': word.end
                    })
                s_lines.append(s_line)
            ser_slides.append(s_lines)
            ser_layouts[idx] = {str(k): v for k, v in gen.slide_layouts[idx].items()}

        gen_data = {'slides': ser_slides, 'layouts': ser_layouts, 'font_size': gen.font_size}

        # 4. Render
        temp_frames_dir = Path(tempfile.mkdtemp(prefix="v2_frames_"))
        cpu_count = os.cpu_count() or 4
        batch_size = max(50, len(frame_tasks) // cpu_count)
        batches = [frame_tasks[i:i + batch_size] for i in range(0, len(frame_tasks), batch_size)]
        
        worker_args = [(b, gen_data, temp_frames_dir, width, height) for b in batches]
        logger.info(f"Rendering {len(frame_tasks)} frames...")
        
        all_files = []
        with multiprocessing.Pool(cpu_count) as pool:
            results = list(tqdm(pool.imap(_generate_frame_batch_worker_v2, worker_args), 
                              total=len(batches), file=sys.stderr))
            for r in results: all_files.extend(r)
            
        all_files.sort(key=lambda x: int(Path(x).stem.split('_')[1]))
        
        frame_clip = ImageSequenceClip(all_files, fps=fps)
        final = CompositeVideoClip([bg_clip, frame_clip])
        
        final.write_videofile(
            str(output_path),
            audio=str(audio_path),
            fps=fps,
            codec="libx264",
            preset="veryfast",
            threads=4,
            logger=None 
        )
        
        return output_path

    except Exception as e:
        logger.error("Render Failed", exc_info=True)
        raise
    finally:
        if temp_frames_dir and temp_frames_dir.exists():
            shutil.rmtree(temp_frames_dir)