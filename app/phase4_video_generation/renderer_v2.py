# import json
# import logging
# import os
# import tempfile
# import multiprocessing
# import shutil
# import sys
# from pathlib import Path
# from typing import List, Optional, Dict, Tuple, Any

# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# from moviepy import AudioFileClip, CompositeVideoClip, ImageSequenceClip, ImageClip
# from pydantic import BaseModel
# from tqdm import tqdm

# from app.config import settings

# logger = logging.getLogger(__name__)

# # --- CONFIGURATION ---

# TEXT_ALIGN = "left"
# MAX_LINES_PER_SLIDE = 4
# MARGIN_PERCENT_X = 0.10
# MARGIN_PERCENT_Y = 0.10

# FONT_HEIGHT_DIVISOR = 8      # Smaller = bigger text. 8 works great for 4-line fill
# LINE_HEIGHT_RATIO = 1.4

# MIN_WORD_DURATION = 0.2
# MAX_SILENCE_GAP = 0.3

# PAUSE_THRESHOLD = 0.8        # Gap > 0.8s = sentence break

# # Common abbreviations that end with periods but are not sentence ends
# COMMON_ABBREVS = {'mr.', 'mrs.', 'dr.', 'ms.', 'jr.', 'sr.', 'st.', 'ave.', 'rd.', 'blvd.', 'etc.', 'e.g.', 'i.e.'}

# # --- DATA MODELS ---
# class WordTimestamp(BaseModel):
#     word: str
#     start: float
#     end: float
#     display_word: Optional[str] = None
#     id: int = 0 

# # --- SANITIZER ---
# def sanitize_words(raw_words: List[Dict]) -> List[WordTimestamp]:
#     clean_words = []
#     for i, w in enumerate(raw_words):
#         word_str = w['word'].strip()
#         if not word_str: 
#             continue
#         if word_str in ['.', ',', '!', '?', ';', ':'] and clean_words:
#             clean_words[-1].display_word += word_str
#             continue
#         obj = WordTimestamp(
#             word=word_str,
#             start=float(w['start']),
#             end=float(w['end']),
#             display_word=word_str,
#             id=i
#         )
#         clean_words.append(obj)

#     if not clean_words:
#         return []

#     for i in range(len(clean_words)):
#         curr = clean_words[i]
#         if (curr.end - curr.start) < MIN_WORD_DURATION:
#             curr.end = curr.start + MIN_WORD_DURATION
#         if i < len(clean_words) - 1:
#             next_w = clean_words[i+1]
#             if curr.end > next_w.start:
#                 next_w.start = curr.end
#                 if (next_w.end - next_w.start) < MIN_WORD_DURATION:
#                     next_w.end = next_w.start + MIN_WORD_DURATION

#     for i in range(len(clean_words) - 1):
#         curr = clean_words[i]
#         next_w = clean_words[i+1]
#         gap = next_w.start - curr.end
#         if 0 < gap < MAX_SILENCE_GAP:
#             curr.end = next_w.start

#     return clean_words

# def interpolate_color(start_color, end_color, progress):
#     return tuple(
#         int(start_color[i] + (end_color[i] - start_color[i]) * progress)
#         for i in range(4)
#     )

# # --- GENERATOR ---
# class FrameGeneratorBigFlow:
#     def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
#         self.bg_width = bg_width
#         self.bg_height = bg_height
        
#         self.margin_x = int(self.bg_width * MARGIN_PERCENT_X)
#         self.margin_y = int(self.bg_height * MARGIN_PERCENT_Y)
#         self.usable_height = self.bg_height - 2 * self.margin_y
#         self.max_text_width = self.bg_width - (2 * self.margin_x)
        
#         self.font_size = int(self.bg_height / FONT_HEIGHT_DIVISOR)
#         self.line_height = int(self.font_size * LINE_HEIGHT_RATIO)
        
#         self.all_words = self._load_and_clean_data(timestamps_path)
        
#         self.slides = []       
#         self.slide_layouts = [] 
#         self.slide_timings = [] 
        
#         self._build_semantic_slides()

#     def _load_and_clean_data(self, path: Path) -> List[WordTimestamp]:
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         raw_words = data.get("words", [])
#         return sanitize_words(raw_words)

#     def _get_font(self, size):
#         try:
#             return ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
#         except:
#             return ImageFont.load_default(size)

#     def _is_abbreviation(self, word: str) -> bool:
#         """
#         Check if the word is a common abbreviation that ends with a period but isn't a sentence end.
#         """
#         lower_word = word.lower()
#         # Check if it ends with '.' and the base form is in abbrevs
#         if lower_word.endswith('.'):
#             base = lower_word[:-1] + '.'  # Keep the period for matching
#             return base in COMMON_ABBREVS
#         return False

#     def split_into_sentences(self, words: List[WordTimestamp]) -> List[List[WordTimestamp]]:
#         sentences = []
#         current = []
#         for i in range(len(words)):
#             current.append(words[i])
#             word = words[i].display_word
#             # Only end sentence if it ends with . ! ? AND it's not an abbreviation
#             if word.endswith(('.', '!', '?')) and not self._is_abbreviation(word):
#                 # Confirm it's a sentence end if next word is capitalized or it's the last word
#                 if i == len(words) - 1 or words[i + 1].display_word[0].isupper():
#                     sentences.append(current)
#                     current = []
#             # Check for significant pause
#             if i < len(words) - 1:
#                 gap = words[i + 1].start - words[i].end
#                 if gap > PAUSE_THRESHOLD:
#                     if current:
#                         sentences.append(current)
#                     current = []
#         if current:
#             sentences.append(current)
#         return sentences

#     def _layout_into_lines(self, words: List[WordTimestamp]) -> List[List[WordTimestamp]]:
#         font = self._get_font(self.font_size)
#         dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
#         space_width = dummy.textlength(" ", font=font)
        
#         lines = []
#         current_line = []
#         current_w = 0
        
#         for word in words:
#             wl = dummy.textlength(word.display_word, font=font)
#             if current_line and current_w + space_width + wl > self.max_text_width:
#                 lines.append(current_line)
#                 current_line = [word]
#                 current_w = wl
#             else:
#                 if current_line: current_w += space_width
#                 current_line.append(word)
#                 current_w += wl
#         if current_line:
#             lines.append(current_line)
#         return lines

#     def _is_small_chunk(self, lines: List[List[WordTimestamp]]) -> bool:
#         return sum(len(l) for l in lines) <= 2

#     def _split_slide_semantically(self, lines: List[List[WordTimestamp]]) -> Tuple[List[List[WordTimestamp]], List[List[WordTimestamp]]]:
#         words = [w for line in lines for w in line]
#         if len(words) <= 3:
#             return lines, []
        
#         candidates = [i for i, w in enumerate(words) if w.display_word.endswith((',', ';', ':')) and i > len(words)*0.2]
#         if candidates:
#             best_i = min(candidates, key=lambda x: abs(x - len(words)/2))
#             split_idx = best_i + 1
#         else:
#             split_idx = len(words) // 2
        
#         return self._layout_into_lines(words[:split_idx]), self._layout_into_lines(words[split_idx:])

#     def _build_semantic_slides(self):
#         sentences = self.split_into_sentences(self.all_words)
#         current_slide_lines = []
#         MAX = MAX_LINES_PER_SLIDE
        
#         for sent_words in sentences:
#             sent_lines = self._layout_into_lines(sent_words)
#             chunk_start = 0
#             while chunk_start < len(sent_lines):
#                 chunk_lines = sent_lines[chunk_start : chunk_start + MAX]
#                 is_small = self._is_small_chunk(chunk_lines)
                
#                 if chunk_start == 0 and len(current_slide_lines) >= 3 and not is_small:
#                     if current_slide_lines:
#                         self._commit_slide(current_slide_lines)
#                     current_slide_lines = []
                
#                 potential = len(current_slide_lines) + len(chunk_lines)
#                 if potential > MAX:
#                     if is_small and current_slide_lines:
#                         first, second = self._split_slide_semantically(current_slide_lines)
#                         self._commit_slide(first)
#                         current_slide_lines = second
                        
#                         if len(current_slide_lines) + len(chunk_lines) <= MAX:
#                             current_slide_lines.extend(chunk_lines)
#                         else:
#                             self._commit_slide(current_slide_lines)
#                             current_slide_lines = chunk_lines
#                     else:
#                         self._commit_slide(current_slide_lines)
#                         current_slide_lines = chunk_lines
#                 else:
#                     current_slide_lines.extend(chunk_lines)
                
#                 chunk_start += MAX
        
#         if current_slide_lines:
#             self._commit_slide(current_slide_lines)

#     def _commit_slide(self, lines: List[List[WordTimestamp]]):
#         if not lines: return
        
#         # Remove trailing comma from last word (your original logic)
#         last_word = lines[-1][-1]
#         if last_word.display_word.endswith(','):
#             last_word.display_word = last_word.display_word[:-1]

#         total_h = len(lines) * self.line_height
#         start_y = self.margin_y + (self.usable_height - total_h) // 2
        
#         layout = {}
#         font = self._get_font(self.font_size)
#         dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
#         space_w = dummy.textlength(" ", font=font)
        
#         y = start_y
#         for line in lines:
#             x = self.margin_x
#             for word in line:
#                 layout[word.id] = (int(x), int(y))
#                 x += dummy.textlength(word.display_word, font=font) + space_w
#             y += self.line_height
            
#         self.slides.append(lines)
#         self.slide_layouts.append(layout)
#         s_start = lines[0][0].start
#         s_end = lines[-1][-1].end
#         self.slide_timings.append((s_start, s_end))

# # --- WORKER (FIXED UNPACK + PERSISTENT SLIDE + BLANK HANDLING) ---
# def _generate_frame_batch_worker_v5(batch_data):
#     frame_tasks, gen_data, output_dir, width, height = batch_data
    
#     C_ACTIVE = (20, 20, 20, 255)
#     C_FUTURE = (210, 210, 210, 255)
#     FADE_DURATION = 0.50

#     generated = []
#     for frame_num, ts, slide_idx in frame_tasks:   # ← FIXED: 3 values
#         img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
#         draw = ImageDraw.Draw(img)
        
#         if slide_idx == -1:
#             # Transparent frame (background only)
#             pass
#         else:
#             slide_lines = gen_data['slides'][slide_idx]
#             layout = gen_data['layouts'][slide_idx]
            
#             # Load fonts for this frame (different slides may theoretically have different sizes, safe)
#             try:
#                 font_bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, gen_data['font_size'])
#                 try:
#                     font_reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, gen_data['font_size'])
#                 except:
#                     font_reg = font_bold
#             except:
#                 font_bold = font_reg = ImageFont.load_default(gen_data['font_size'])

#             for line in slide_lines:
#                 for w_data in line:
#                     coords = layout.get(str(w_data['id']))
#                     if not coords: continue
#                     start = w_data['start']
                    
#                     if ts >= start:
#                         color = C_ACTIVE
#                         font = font_bold
#                     elif start - FADE_DURATION <= ts < start:
#                         progress = (ts - (start - FADE_DURATION)) / FADE_DURATION
#                         color = interpolate_color(C_FUTURE, C_ACTIVE, progress)
#                         font = font_bold if progress > 0.5 else font_reg
#                     else:
#                         color = C_FUTURE
#                         font = font_reg

#                     draw.text(coords, w_data['display_word'], font=font, fill=color)
        
#         path = output_dir / f"f_{frame_num:06d}.png"
#         img.save(path, optimize=False, compress_level=0)
#         generated.append(str(path))
        
#     return generated

# # --- MAIN RENDER FUNCTION (PERSISTENT SLIDE + CUMULATIVE INDEX) ---
# def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
#     logger.info("--- Starting V7 Render (Abbreviation Fix + Anti-Isolation + Persistent Text + Fixed Unpack) ---")
#     temp_frames_dir = None
#     try:
#         audio_clip = AudioFileClip(str(audio_path))
#         duration = audio_clip.duration
#         fps = settings.VIDEO_FPS
#         width = settings.VIDEO_WIDTH
#         height = settings.VIDEO_HEIGHT
        
#         gen = FrameGeneratorBigFlow(timestamps_path, width, height)
        
#         # === PERSISTENT SLIDE INDEX (text stays on screen during pauses & end) ===
#         total_frames = int(duration * fps + 1)
#         frame_tasks = []
#         active_slide_idx = -1
#         for i in range(total_frames):
#             ts = i / fps
#             while (active_slide_idx + 1 < len(gen.slide_timings) and 
#                    gen.slide_timings[active_slide_idx + 1][0] <= ts):
#                 active_slide_idx += 1
#             slide_idx = active_slide_idx if active_slide_idx >= 0 else -1
#             frame_tasks.append((i, ts, slide_idx))
        
#         # Serialize
#         ser_slides = []
#         ser_layouts = {}
#         for idx, lines in enumerate(gen.slides):
#             s_lines = []
#             for line in lines:
#                 s_line = []
#                 for word in line:
#                     s_line.append({
#                         'id': word.id,
#                         'display_word': word.display_word,
#                         'start': word.start,
#                         'end': word.end
#                     })
#                 s_lines.append(s_line)
#             ser_slides.append(s_lines)
#             ser_layouts[idx] = {str(k): v for k, v in gen.slide_layouts[idx].items()}

#         gen_data = {
#             'slides': ser_slides,
#             'layouts': ser_layouts,
#             'font_size': gen.font_size
#         }

#         temp_frames_dir = Path(tempfile.mkdtemp(prefix="v7_frames_"))
#         cpu = os.cpu_count() or 4
#         batch_size = max(50, len(frame_tasks) // cpu)
#         batches = [frame_tasks[i:i+batch_size] for i in range(0, len(frame_tasks), batch_size)]
#         worker_args = [(b, gen_data, temp_frames_dir, width, height) for b in batches]

#         logger.info(f"Rendering {len(frame_tasks)} frames across {len(batches)} batches...")
#         all_files = []
#         with multiprocessing.Pool(cpu) as pool:
#             for r in tqdm(pool.imap(_generate_frame_batch_worker_v5, worker_args), total=len(batches)):
#                 all_files.extend(r)

#         all_files.sort(key=lambda x: int(Path(x).stem.split('_')[1]))

#         text_clip = ImageSequenceClip(all_files, fps=fps)
#         final = CompositeVideoClip([ImageClip(settings.DEFAULT_BACKGROUND).with_duration(duration), text_clip])
#         final = final.with_audio(audio_clip)

#         final.write_videofile(
#             str(output_path),
#             fps=fps,
#             codec="libx264",
#             preset="veryfast",
#             threads=multiprocessing.cpu_count(),
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

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, CompositeVideoClip, ImageSequenceClip, ImageClip
from pydantic import BaseModel
from tqdm import tqdm

from app.config import settings

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---

# TEXT_ALIGN = "left"
MAX_LINES_PER_SLIDE = 4
MARGIN_PERCENT_X = 0.10
MARGIN_PERCENT_Y = 0.10

FONT_HEIGHT_DIVISOR = 8      # Smaller = bigger text. 8 works great for 4-line fill
LINE_HEIGHT_RATIO = 1.4

MIN_WORD_DURATION = 0.2
MAX_SILENCE_GAP = 0.3

PAUSE_THRESHOLD = 0.8        # Gap > 0.8s = sentence break

# Common abbreviations that end with periods but are not sentence ends
COMMON_ABBREVS = {'mr.', 'mrs.', 'dr.', 'ms.', 'jr.', 'sr.', 'st.', 'ave.', 'rd.', 'blvd.', 'etc.', 'e.g.', 'i.e.'}

# --- DATA MODELS ---
class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    display_word: Optional[str] = None
    id: int = 0 

# --- SANITIZER ---
def sanitize_words(raw_words: List[Dict]) -> List[WordTimestamp]:
    clean_words = []
    for i, w in enumerate(raw_words):
        word_str = w['word'].strip()
        if not word_str: 
            continue
        if word_str in ['.', ',', '!', '?', ';', ':'] and clean_words:
            clean_words[-1].display_word += word_str
            continue
        obj = WordTimestamp(
            word=word_str,
            start=float(w['start']),
            end=float(w['end']),
            display_word=word_str,
            id=i
        )
        clean_words.append(obj)

    if not clean_words:
        return []

    for i in range(len(clean_words)):
        curr = clean_words[i]
        if (curr.end - curr.start) < MIN_WORD_DURATION:
            curr.end = curr.start + MIN_WORD_DURATION
        if i < len(clean_words) - 1:
            next_w = clean_words[i+1]
            if curr.end > next_w.start:
                next_w.start = curr.end
                if (next_w.end - next_w.start) < MIN_WORD_DURATION:
                    next_w.end = next_w.start + MIN_WORD_DURATION

    for i in range(len(clean_words) - 1):
        curr = clean_words[i]
        next_w = clean_words[i+1]
        gap = next_w.start - curr.end
        if 0 < gap < MAX_SILENCE_GAP:
            curr.end = next_w.start

    return clean_words

def interpolate_color(start_color, end_color, progress):
    return tuple(
        int(start_color[i] + (end_color[i] - start_color[i]) * progress)
        for i in range(4)
    )

# --- GENERATOR ---
class FrameGeneratorBigFlow:
    def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
        self.bg_width = bg_width
        self.bg_height = bg_height
        
        is_vertical = self.bg_height > self.bg_width
        
        if is_vertical:
            self.margin_x = int(self.bg_width * 0.12)
            self.margin_y = int(self.bg_height * 0.30) # 30% down to be safe
            self.font_size = int(self.bg_width / 16)
            self.max_lines = 2 
            self.text_align = "center"
        else:
            self.margin_x = int(self.bg_width * MARGIN_PERCENT_X)
            self.margin_y = int(self.bg_height * MARGIN_PERCENT_Y)
            # 2. Standard Text
            self.font_size = int(self.bg_height / FONT_HEIGHT_DIVISOR)
            self.max_lines = 4
            self.text_align = "left"

        self.usable_height = self.bg_height - 2 * self.margin_y
        self.max_text_width = self.bg_width - (2 * self.margin_x)
        self.line_height = int(self.font_size * LINE_HEIGHT_RATIO)
        
        self.all_words = self._load_and_clean_data(timestamps_path)
        
        self.slides = []       
        self.slide_layouts = [] 
        self.slide_timings = [] 
        
        self._build_semantic_slides()

    def _load_and_clean_data(self, path: Path) -> List[WordTimestamp]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_words = data.get("words", [])
        return sanitize_words(raw_words)

    def _get_font(self, size):
        try:
            return ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
        except:
            return ImageFont.load_default(size)

    def _is_abbreviation(self, word: str) -> bool:
        """
        Check if the word is a common abbreviation that ends with a period but isn't a sentence end.
        """
        lower_word = word.lower()
        # Check if it ends with '.' and the base form is in abbrevs
        if lower_word.endswith('.'):
            base = lower_word[:-1] + '.'  # Keep the period for matching
            return base in COMMON_ABBREVS
        return False

    def split_into_sentences(self, words: List[WordTimestamp]) -> List[List[WordTimestamp]]:
        sentences = []
        current = []
        for i in range(len(words)):
            current.append(words[i])
            word = words[i].display_word
            # Only end sentence if it ends with . ! ? AND it's not an abbreviation
            if word.endswith(('.', '!', '?')) and not self._is_abbreviation(word):
                # Confirm it's a sentence end if next word is capitalized or it's the last word
                if i == len(words) - 1 or words[i + 1].display_word[0].isupper():
                    sentences.append(current)
                    current = []
            # Check for significant pause
            if i < len(words) - 1:
                gap = words[i + 1].start - words[i].end
                if gap > PAUSE_THRESHOLD:
                    if current:
                        sentences.append(current)
                    current = []
        if current:
            sentences.append(current)
        return sentences

    def _layout_into_lines(self, words: List[WordTimestamp]) -> List[List[WordTimestamp]]:
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
                if current_line: current_w += space_width
                current_line.append(word)
                current_w += wl
        if current_line:
            lines.append(current_line)
        return lines

    def _is_small_chunk(self, lines: List[List[WordTimestamp]]) -> bool:
        return sum(len(l) for l in lines) <= 2

    def _split_slide_semantically(self, lines: List[List[WordTimestamp]]) -> Tuple[List[List[WordTimestamp]], List[List[WordTimestamp]]]:
        words = [w for line in lines for w in line]
        if len(words) <= 3:
            return lines, []
        
        candidates = [i for i, w in enumerate(words) if w.display_word.endswith((',', ';', ':')) and i > len(words)*0.2]
        if candidates:
            best_i = min(candidates, key=lambda x: abs(x - len(words)/2))
            split_idx = best_i + 1
        else:
            split_idx = len(words) // 2
        
        return self._layout_into_lines(words[:split_idx]), self._layout_into_lines(words[split_idx:])

    def _build_semantic_slides(self):
        sentences = self.split_into_sentences(self.all_words)
        current_slide_lines = []
        MAX = self.max_lines
        
        for sent_words in sentences:
            sent_lines = self._layout_into_lines(sent_words)
            chunk_start = 0
            while chunk_start < len(sent_lines):
                chunk_lines = sent_lines[chunk_start : chunk_start + MAX]
                is_small = self._is_small_chunk(chunk_lines)
                
                if chunk_start == 0 and len(current_slide_lines) >= 3 and not is_small:
                    if current_slide_lines:
                        self._commit_slide(current_slide_lines)
                    current_slide_lines = []
                
                potential = len(current_slide_lines) + len(chunk_lines)
                if potential > MAX:
                    if is_small and current_slide_lines:
                        first, second = self._split_slide_semantically(current_slide_lines)
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

    def _commit_slide(self, lines: List[List[WordTimestamp]]):
        if not lines: return
        
        # Remove trailing comma from last word (your original logic)
        last_word = lines[-1][-1]
        if last_word.display_word.endswith(','):
            last_word.display_word = last_word.display_word[:-1]

        total_h = len(lines) * self.line_height
        start_y = self.margin_y + (self.usable_height - total_h) // 2
        
        layout = {}
        font = self._get_font(self.font_size)
        dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        space_w = dummy.textlength(" ", font=font)
        
        y = start_y
        for line in lines:
            if self.text_align == "center":
                # 1. Calculate the full width of this text line
                line_width = 0
                for w in line:
                    line_width += dummy.textlength(w.display_word, font=font)
                # Add spacing width
                if len(line) > 1:
                    line_width += (len(line) - 1) * space_w
                
                # 2. Center it: (Total Width - Line Width) / 2
                x = (self.bg_width - line_width) // 2
            else:
                # Left align (Standard)
                x = self.margin_x
            # --- ALIGNMENT LOGIC END ---
            
            for word in line:
                layout[word.id] = (int(x), int(y))
                x += dummy.textlength(word.display_word, font=font) + space_w
            y += self.line_height
            
        self.slides.append(lines)
        self.slide_layouts.append(layout)
        s_start = lines[0][0].start
        s_end = lines[-1][-1].end
        self.slide_timings.append((s_start, s_end))

# --- WORKER (FIXED UNPACK + PERSISTENT SLIDE + BLANK HANDLING) ---
def _generate_frame_batch_worker_v5(batch_data):
    frame_tasks, gen_data, output_dir, width, height = batch_data
    
    C_ACTIVE = (20, 20, 20, 255)
    C_FUTURE = (210, 210, 210, 255)
    FADE_DURATION = 0.50

    generated = []
    for frame_num, ts, slide_idx in frame_tasks:   # ← FIXED: 3 values
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        if slide_idx == -1:
            # Transparent frame (background only)
            pass
        else:
            slide_lines = gen_data['slides'][slide_idx]
            layout = gen_data['layouts'][slide_idx]
            
            # Load fonts for this frame (different slides may theoretically have different sizes, safe)
            try:
                font_bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, gen_data['font_size'])
                try:
                    font_reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, gen_data['font_size'])
                except:
                    font_reg = font_bold
            except:
                font_bold = font_reg = ImageFont.load_default(gen_data['font_size'])

            for line in slide_lines:
                for w_data in line:
                    coords = layout.get(str(w_data['id']))
                    if not coords: continue
                    start = w_data['start']
                    
                    if ts >= start:
                        color = C_ACTIVE
                        font = font_bold
                    elif start - FADE_DURATION <= ts < start:
                        progress = (ts - (start - FADE_DURATION)) / FADE_DURATION
                        color = interpolate_color(C_FUTURE, C_ACTIVE, progress)
                        font = font_bold if progress > 0.5 else font_reg
                    else:
                        color = C_FUTURE
                        font = font_reg

                    draw.text(coords, w_data['display_word'], font=font, fill=color)
        
        path = output_dir / f"f_{frame_num:06d}.png"
        img.save(path, optimize=False, compress_level=0)
        generated.append(str(path))
        
    return generated

# --- MAIN RENDER FUNCTION (PERSISTENT SLIDE + CUMULATIVE INDEX) ---
def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
    logger.info("--- Starting V7 Render (Abbreviation Fix + Anti-Isolation + Persistent Text + Fixed Unpack) ---")
    temp_frames_dir = None
    try:
        audio_clip = AudioFileClip(str(audio_path))
        duration = audio_clip.duration
        fps = settings.VIDEO_FPS
        bg_clip = ImageClip(settings.DEFAULT_BACKGROUND).with_duration(duration)
        width, height = bg_clip.size

        print(f"Video dimensions: {width}x{height}")
        
        gen = FrameGeneratorBigFlow(timestamps_path, width, height)
        
        # === PERSISTENT SLIDE INDEX (text stays on screen during pauses & end) ===
        total_frames = int(duration * fps + 1)
        frame_tasks = []
        active_slide_idx = -1
        for i in range(total_frames):
            ts = i / fps
            while (active_slide_idx + 1 < len(gen.slide_timings) and 
                   gen.slide_timings[active_slide_idx + 1][0] <= ts):
                active_slide_idx += 1
            slide_idx = active_slide_idx if active_slide_idx >= 0 else -1
            frame_tasks.append((i, ts, slide_idx))
        
        # Serialize
        ser_slides = []
        ser_layouts = {}
        for idx, lines in enumerate(gen.slides):
            s_lines = []
            for line in lines:
                s_line = []
                for word in line:
                    s_line.append({
                        'id': word.id,
                        'display_word': word.display_word,
                        'start': word.start,
                        'end': word.end
                    })
                s_lines.append(s_line)
            ser_slides.append(s_lines)
            ser_layouts[idx] = {str(k): v for k, v in gen.slide_layouts[idx].items()}

        gen_data = {
            'slides': ser_slides,
            'layouts': ser_layouts,
            'font_size': gen.font_size
        }

        temp_frames_dir = Path(tempfile.mkdtemp(prefix="v7_frames_"))
        cpu = os.cpu_count() or 4
        batch_size = max(50, len(frame_tasks) // cpu)
        batches = [frame_tasks[i:i+batch_size] for i in range(0, len(frame_tasks), batch_size)]
        worker_args = [(b, gen_data, temp_frames_dir, width, height) for b in batches]

        logger.info(f"Rendering {len(frame_tasks)} frames across {len(batches)} batches...")
        all_files = []
        with multiprocessing.Pool(cpu) as pool:
            for r in tqdm(pool.imap(_generate_frame_batch_worker_v5, worker_args), total=len(batches)):
                all_files.extend(r)

        all_files.sort(key=lambda x: int(Path(x).stem.split('_')[1]))

        text_clip = ImageSequenceClip(all_files, fps=fps)
        final = CompositeVideoClip([ImageClip(settings.DEFAULT_BACKGROUND).with_duration(duration), text_clip])
        final = final.with_audio(audio_clip)

        final.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            preset="veryfast",
            threads=multiprocessing.cpu_count(),
            logger=None
        )
        return output_path

    except Exception as e:
        logger.error("Render Failed", exc_info=True)
        raise
    finally:
        if temp_frames_dir and temp_frames_dir.exists():
            shutil.rmtree(temp_frames_dir)

