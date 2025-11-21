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

# # 1. Layout Controls
# TEXT_ALIGN = "left"
# MAX_LINES_PER_SLIDE = 5 
# MARGIN_PERCENT_X = 0.10    # Left/Right Margin (10% is safe)
# MARGIN_PERCENT_Y = 0.10    # Top/Bottom Margin

# # 2. Font Sizing
# # DECREASE this to make font BIGGER. INCREASE to make smaller.
# # 10 is standard "Huge". 8 is "Massive". 12 is "Large".
# FONT_HEIGHT_DIVISOR = 8   
# LINE_HEIGHT_RATIO = 1.4    

# # 3. Data Cleaning Settings
# MIN_WORD_DURATION = 0.2    # Seconds. Any word shorter than this gets inflated.
# MAX_SILENCE_GAP = 0.3      # Seconds. If gap < 0.3s, we bridge it (no flickering).

# # --- DATA MODELS ---
# class WordTimestamp(BaseModel):
#     word: str
#     start: float
#     end: float
#     display_word: Optional[str] = None
#     id: int = 0 

# # --- DATA SANITIZER (THE FIX) ---
# def sanitize_words(raw_words: List[Dict]) -> List[WordTimestamp]:
#     """
#     Fixes 0-duration words, overlaps, and jerky silences.
#     """
#     clean_words = []
    
#     # 1. Convert to Objects & Basic Cleanup
#     for i, w in enumerate(raw_words):
#         word_str = w['word'].strip()
#         # Skip empty strings
#         if not word_str: 
#             continue
            
#         # Fix Punctuation: Attach to previous word if it's just a symbol
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

#     # 2. Fix Durations & Overlaps
#     for i in range(len(clean_words)):
#         curr = clean_words[i]
        
#         # A. Force Minimum Duration (Fixes the "Flash" bug)
#         if (curr.end - curr.start) < MIN_WORD_DURATION:
#             curr.end = curr.start + MIN_WORD_DURATION
            
#         # B. Fix Overlaps with Next Word
#         if i < len(clean_words) - 1:
#             next_w = clean_words[i+1]
#             # If current ends AFTER next starts, push next start forward
#             if curr.end > next_w.start:
#                 next_w.start = curr.end
#                 # Ensure next word still has min duration after being pushed
#                 if (next_w.end - next_w.start) < MIN_WORD_DURATION:
#                     next_w.end = next_w.start + MIN_WORD_DURATION

#     # 3. Bridge Small Silences (Fixes the "Jerky" bug)
#     # If there is a tiny gap between words, extend the current word to touch the next.
#     for i in range(len(clean_words) - 1):
#         curr = clean_words[i]
#         next_w = clean_words[i+1]
        
#         gap = next_w.start - curr.end
#         if 0 < gap < MAX_SILENCE_GAP:
#             # Extend current word to fill the gap
#             curr.end = next_w.start

#     return clean_words

# def interpolate_color(start_color, end_color, progress):
#     """Blends two colors based on progress (0.0 to 1.0)."""
#     return tuple(
#         int(start_color[i] + (end_color[i] - start_color[i]) * progress)
#         for i in range(4)
#     )
# # --- GENERATOR CLASS ---
# class FrameGeneratorBigFlow:
#     def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
#         self.bg_width = bg_width
#         self.bg_height = bg_height
        
#         self.margin_x = int(self.bg_width * MARGIN_PERCENT_X)
#         self.max_text_width = self.bg_width - (2 * self.margin_x)
        
#         self.font_size = int(self.bg_height / FONT_HEIGHT_DIVISOR)
#         self.line_height = int(self.font_size * LINE_HEIGHT_RATIO)
        
#         self.font = self._load_font(self.font_size)
        
#         # LOAD AND SANITIZE
#         self.all_words = self._load_and_clean_data(timestamps_path)
        
#         self.slides = []       
#         self.slide_layouts = [] 
#         self.slide_timings = [] 
        
#         self._build_flow_slides()

#     def _load_font(self, size):
#         try:
#             return ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
#         except:
#             return ImageFont.load_default(size)

#     def _load_and_clean_data(self, path: Path) -> List[WordTimestamp]:
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         raw_words = data.get("words", [])
#         # APPLY THE SANITIZER
#         return sanitize_words(raw_words)

#     def _build_flow_slides(self):
#         dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
#         space_width = dummy_draw.textlength(" ", font=self.font)
        
#         current_slide_lines = []
#         current_line = []
#         current_line_width = 0
        
#         for word in self.all_words:
#             w_len = dummy_draw.textlength(word.display_word, font=self.font)
            
#             if current_line_width + space_width + w_len > self.max_text_width:
#                 current_slide_lines.append(current_line)
#                 current_line = [word]
#                 current_line_width = w_len
                
#                 if len(current_slide_lines) >= MAX_LINES_PER_SLIDE:
#                     self._commit_slide(current_slide_lines)
#                     current_slide_lines = [] 
#             else:
#                 if current_line: 
#                     current_line_width += space_width
#                 current_line.append(word)
#                 current_line_width += w_len

#         if current_line:
#             current_slide_lines.append(current_line)
#         if current_slide_lines:
#             self._commit_slide(current_slide_lines)

#     def _commit_slide(self, lines: List[List[WordTimestamp]]):
#         if not lines: return
        
#         if lines and lines[-1]:
#             last_word_obj = lines[-1][-1]
#             text = last_word_obj.display_word
            
#             # 2. Check if it ends with a comma
#             if text.endswith(","):
#                 # Remove the comma
#                 last_word_obj.display_word = text[:-1]


#         total_h = len(lines) * self.line_height
#         start_y = (self.bg_height - total_h) // 2
        
#         layout = {}
#         curr_y = start_y
        
#         for line in lines:
#             curr_x = self.margin_x
#             for word in line:
#                 layout[word.id] = (int(curr_x), int(curr_y))
#                 w_len = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(word.display_word, font=self.font)
#                 space_w = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(" ", font=self.font)
#                 curr_x += w_len + space_w
#             curr_y += self.line_height
            
#         self.slides.append(lines)
#         self.slide_layouts.append(layout)
        
#         s_start = lines[0][0].start
#         s_end = lines[-1][-1].end
#         self.slide_timings.append((s_start, s_end))

# # --- WORKER ---
# def _generate_frame_batch_worker_v5(batch_data):
#     frame_tasks, gen_data, output_dir, width, height = batch_data
    
#     # 1. Load Fonts
#     try:
#         font_bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, gen_data['font_size'])
#         # Try to load a regular version of the same size for the "inactive" state
#         # If you don't have a regular font file, use the bold one for both (just color changes)
#         try:
#             font_reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, gen_data['font_size'])
#         except:
#             font_reg = font_bold 
#     except:
#         font_bold = ImageFont.load_default(gen_data['font_size'])
#         font_reg = font_bold

#     # 2. Animation Settings
#     # COLOR_BG = (250, 248, 242, 255)    # Cream/Off-White
#     # COLOR_GHOST = (210, 210, 210, 255) # Light Gray (Future words)
#     # COLOR_ACTIVE = (20, 20, 20, 255)   # Soft Black (Active words)
#     C_ACTIVE = (20, 20, 20, 255)        # Black
#     C_FUTURE = (210, 210, 210, 255)    # Light Gray
#     FADE_DURATION = 0.30               # 0.25s transition window

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
                
#                 start = w_data['start']
                
#                 # --- THE ANIMATION LOGIC ---
                
#                 # Case 1: Fully Active (Time is past start)
#                 if ts >= start:
#                     color = C_ACTIVE
#                     font = font_bold
                
#                 # Case 2: Transitioning (Time is within FADE_DURATION before start)
#                 # e.g., Start is 2.0s. We are at 1.90s. Fade starts at 1.85s.
#                 elif start - FADE_DURATION <= ts < start:
#                     # Calculate progress (0.0 to 1.0)
#                     progress = (ts - (start - FADE_DURATION)) / FADE_DURATION
                    
#                     # Interpolate Color
#                     color = interpolate_color(C_FUTURE, C_ACTIVE, progress)
                    
#                     # Switch Font Weight (Switch to bold when 50% darker)
#                     font = font_bold if progress > 0.5 else font_reg
                    
#                 # Case 3: Fully Future
#                 else:
#                     color = C_FUTURE
#                     font = font_reg

#                 draw.text(coords, w_data['display_word'], font=font, fill=color)
        
#         path = output_dir / f"f_{frame_num:06d}.png"
#         img.save(path, optimize=False, compress_level=0)
#         generated.append(str(path))
        
#     return generated

#     # For 3d rendering, uncomment below and comment above
#     # frame_tasks, gen_data, output_dir, width, height = batch_data
    
#     # # 1. Load Fonts
#     # try:
#     #     font_bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, gen_data['font_size'])
#     #     try:
#     #         font_reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, gen_data['font_size'])
#     #     except:
#     #         font_reg = font_bold 
#     # except:
#     #     font_bold = ImageFont.load_default(gen_data['font_size'])
#     #     font_reg = font_bold

#     # # 2. Animation Settings
#     # C_ACTIVE = (0, 0, 0, 255)          # Pure Black
#     # C_SHADOW = (220, 220, 220, 255)    # Very Light Gray (The "Shadow" left behind)
#     # C_FUTURE = (180, 180, 180, 255)    # Darker Gray (Waiting to be read)
    
#     # FADE_DURATION = 0.20               # Slightly slower for the lift effect
#     # LIFT_PIXELS = 8                    # How many pixels the word moves UP

#     # generated = []
#     # for frame_num, ts, slide_idx, _ in frame_tasks:
#     #     img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
#     #     draw = ImageDraw.Draw(img)
        
#     #     slide_lines = gen_data['slides'][slide_idx]
#     #     layout = gen_data['layouts'][slide_idx]
        
#     #     for line in slide_lines:
#     #         for w_data in line:
#     #             coords = layout.get(str(w_data['id']))
#     #             if not coords: continue
                
#     #             start = w_data['start']
#     #             end = w_data['end']
                
#     #             x, y = coords

#     #             # --- 3D LIFT ANIMATION ---
                
#     #             # Case 1: FULLY ACTIVE (The "Pop")
#     #             if ts >= start and ts <= end:
#     #                 # Draw the "Shadow" (The word at its original position)
#     #                 draw.text((x, y), w_data['display_word'], font=font_bold, fill=C_SHADOW)
                    
#     #                 # Draw the "Active Word" (Moved UP by LIFT_PIXELS)
#     #                 draw.text((x, y - LIFT_PIXELS), w_data['display_word'], font=font_bold, fill=C_ACTIVE)

#     #             # Case 2: FADING IN (Transition)
#     #             elif start - FADE_DURATION <= ts < start:
#     #                 progress = (ts - (start - FADE_DURATION)) / FADE_DURATION
                    
#     #                 # Interpolate color
#     #                 color = interpolate_color(C_FUTURE, C_ACTIVE, progress)
#     #                 font = font_bold if progress > 0.5 else font_reg
                    
#     #                 # Slight Lift during transition? (Optional: linear interpolation of lift)
#     #                 current_lift = int(LIFT_PIXELS * progress)
                    
#     #                 draw.text((x, y - current_lift), w_data['display_word'], font=font, fill=color)
                    
#     #             # Case 3: PAST WORDS (Return to normal position?)
#     #             elif ts > end:
#     #                  # Client usually wants them to stay Black, or turn Gray.
#     #                  # Let's keep them Black but put them back on the ground (No lift)
#     #                  draw.text((x, y), w_data['display_word'], font=font_bold, fill=C_ACTIVE)

#     #             # Case 4: FUTURE WORDS
#     #             else:
#     #                 draw.text((x, y), w_data['display_word'], font=font_reg, fill=C_FUTURE)

#     #     path = output_dir / f"f_{frame_num:06d}.png"
#     #     img.save(path, optimize=False, compress_level=0)
#     #     generated.append(str(path))
        
#     # return generated

# # --- MAIN ---
# def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
#     logger.info("--- Starting V5 Render (Sanitized Data + Big Flow) ---")
#     temp_frames_dir = None
#     try:
#         audio_clip = AudioFileClip(str(audio_path))
#         duration = audio_clip.duration
#         fps = settings.VIDEO_FPS
#         width = settings.VIDEO_WIDTH
#         height = settings.VIDEO_HEIGHT
#         bg_clip = ImageClip(settings.DEFAULT_BACKGROUND).with_duration(duration)
        
#         gen = FrameGeneratorBigFlow(timestamps_path, width, height)
        
#         total_frames = int(duration * fps)
#         frame_tasks = []
#         current_slide_idx = 0
        
#         for i in range(total_frames):
#             ts = i / fps
#             if current_slide_idx < len(gen.slide_timings):
#                 slide_start, slide_end = gen.slide_timings[current_slide_idx]
#                 # Lookahead: If we are past end, go next
#                 if ts > slide_end and current_slide_idx < len(gen.slide_timings) - 1:
#                     current_slide_idx += 1
#             frame_tasks.append((i, ts, current_slide_idx, 0))

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

#         temp_frames_dir = Path(tempfile.mkdtemp(prefix="v5_frames_"))
#         cpu_count = os.cpu_count() or 4
#         batch_size = max(50, len(frame_tasks) // cpu_count)
#         batches = [frame_tasks[i:i + batch_size] for i in range(0, len(frame_tasks), batch_size)]
        
#         worker_args = [(b, gen_data, temp_frames_dir, width, height) for b in batches]
#         logger.info(f"Rendering {len(frame_tasks)} frames...")
        
#         all_files = []
#         with multiprocessing.Pool(cpu_count) as pool:
#             results = list(tqdm(pool.imap(_generate_frame_batch_worker_v5, worker_args), 
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

# #  182501

# # import json
# # import logging
# # import os
# # import tempfile
# # import multiprocessing
# # import shutil
# # import sys
# # from pathlib import Path
# # from typing import List, Optional, Dict, Tuple, Any
# # import re

# # import numpy as np
# # from PIL import Image, ImageDraw, ImageFont
# # from moviepy import AudioFileClip, CompositeVideoClip, ImageSequenceClip, ImageClip
# # from pydantic import BaseModel
# # from tqdm import tqdm

# # from app.config import settings

# # logger = logging.getLogger(__name__)

# # # --- DATA MODELS ---
# # class WordTimestamp(BaseModel):
# #     word: str
# #     start: float
# #     end: float
# #     display_word: Optional[str] = None

# # # --- UTILS ---
# # def interpolate_color(start_color, end_color, progress):
# #     return tuple(
# #         int(start_color[i] + (end_color[i] - start_color[i]) * progress)
# #         for i in range(4)
# #     )

# # # --- GENERATOR CLASS ---
# # class FrameGeneratorFounders:
# #     def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
# #         self.bg_width = bg_width
# #         self.bg_height = bg_height
        
# #         # Box Model (Safe Area)
# #         self.padding_x = int(self.bg_width * 0.12)
# #         self.padding_y = int(self.bg_height * 0.15)
# #         self.box_width = self.bg_width - (2 * self.padding_x)
# #         self.box_height = self.bg_height - (2 * self.padding_y)

# #         # Load Data & Group by Semantic Phrases
# #         self.phrases = self._load_and_group_phrases(timestamps_path)
        
# #         # Build Slides (Semantic Scaling)
# #         self.slides, self.slide_layouts, self.slide_timings, self.slide_fonts = self._build_semantic_scale_slides()

# #     def _load_fonts(self, size):
# #         try:
# #             reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, size)
# #             bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
# #             return reg, bold
# #         except:
# #             return ImageFont.load_default(size), ImageFont.load_default(size)

# #     def _load_and_group_phrases(self, path: Path) -> List[List[WordTimestamp]]:
# #         """
# #         Groups words into "Phrases" based on pauses and punctuation.
# #         This mimics the speaker's natural rhythm.
# #         """
# #         with open(path, "r", encoding="utf-8") as f:
# #             data = json.load(f)
        
# #         raw_words = []
# #         # Flatten
# #         if 'words' in data and data['words']:
# #             raw_words = data['words']
# #         elif 'segments' in data:
# #             for s in data['segments']:
# #                 raw_words.extend(s.get('words', []))
        
# #         if not raw_words: return []

# #         phrases = []
# #         current_phrase = []
        
# #         for i, w_dict in enumerate(raw_words):
# #             word = WordTimestamp(
# #                 word=w_dict.get('word', '').strip(),
# #                 start=w_dict.get('start', 0),
# #                 end=w_dict.get('end', 0),
# #                 display_word=w_dict.get('word', '').strip()
# #             )
            
# #             # Punctuation Logic
# #             txt = word.display_word
# #             is_end = txt[-1] in ['.', '!', '?', ',', ':'] if txt else False
            
# #             # Time Gap Logic (Pause detection)
# #             is_pause = False
# #             if i < len(raw_words) - 1:
# #                 next_start = raw_words[i+1].get('start', 0)
# #                 if next_start - word.end > 0.4: # 400ms pause = new phrase
# #                     is_pause = True
            
# #             current_phrase.append(word)
            
# #             # Break phrase if: End of sentence OR Significant Pause OR Phrase getting too long
# #             if is_end or is_pause or len(current_phrase) > 12:
# #                 phrases.append(current_phrase)
# #                 current_phrase = []
        
# #         if current_phrase: phrases.append(current_phrase)
        
# #         # Clean punctuation from display for the clean look
# #         for phrase in phrases:
# #             if not phrase: continue
# #             last = phrase[-1]
# #             if last.display_word and last.display_word[-1] in ['.', ',', '!', '?']:
# #                 last.display_word = last.display_word[:-1]

# #         return phrases

# #     def _calculate_layout_centered(self, words, font_size):
# #         """Calculates a CENTERED layout."""
# #         _, bold_font = self._load_fonts(font_size)
# #         dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
# #         space_w = dummy.textlength(" ", font=bold_font)
# #         lh = int(font_size * 1.2) # Tighter line height
        
# #         lines = []
# #         current_line = []
# #         current_w = 0
        
# #         for word in words:
# #             wl = dummy.textlength(word.display_word, font=bold_font)
# #             if current_line and (current_w + space_w + wl > self.box_width):
# #                 lines.append((current_line, current_w)) # Store width for centering
# #                 current_line = [word]
# #                 current_w = wl
# #             else:
# #                 if current_line: current_w += space_w
# #                 current_line.append(word)
# #                 current_w += wl
# #         if current_line: lines.append((current_line, current_w))
        
# #         total_h = len(lines) * lh
# #         return lines, total_h, lh

# #     def _build_semantic_scale_slides(self):
# #         slides = []
# #         layouts = {}
# #         timings = []
# #         fonts = {} 

# #         # Scaling Config (1080p)
# #         FONT_HUGE = int(self.bg_height / 10)   # Short phrases (3-5 words)
# #         FONT_LARGE = int(self.bg_height / 13)  # Medium phrases
# #         FONT_NORM = int(self.bg_height / 16)   # Long sentences
        
# #         for phrase in self.phrases:
# #             word_count = len(phrase)
            
# #             # 1. Determine Target Font based on Semantic Length
# #             if word_count <= 6:
# #                 target_font = FONT_HUGE
# #             elif word_count <= 12:
# #                 target_font = FONT_LARGE
# #             else:
# #                 target_font = FONT_NORM
            
# #             # 2. Validate Fit (Shrink if necessary)
# #             final_font = target_font
# #             lines_data, h, lh = self._calculate_layout_centered(phrase, final_font)
            
# #             # Shrink loop
# #             while h > self.box_height and final_font > FONT_NORM:
# #                 final_font -= 5
# #                 lines_data, h, lh = self._calculate_layout_centered(phrase, final_font)

# #             # 3. Build Coordinates (CENTER ALIGNED)
# #             _, bold_font = self._load_fonts(final_font)
# #             dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
# #             space = dummy.textlength(" ", font=bold_font)
            
# #             start_y = self.padding_y + (self.box_height - h) // 2
            
# #             slide_layout = {}
# #             curr_y = start_y
            
# #             plain_lines = []
            
# #             for line_words, line_width in lines_data:
# #                 # CENTER X CALCULATION
# #                 # Start X = Center of Screen - Half of Line Width
# #                 start_x = (self.bg_width - line_width) // 2
                
# #                 curr_x = start_x
# #                 for word in line_words:
# #                     slide_layout[id(word)] = (curr_x, curr_y)
# #                     wl = dummy.textlength(word.display_word, font=bold_font)
# #                     curr_x += wl + space
                
# #                 curr_y += lh
# #                 plain_lines.append(line_words)
            
# #             idx = len(slides)
# #             slides.append(plain_lines)
# #             layouts[idx] = slide_layout
# #             timings.append((phrase[0].start, phrase[-1].end))
# #             fonts[idx] = final_font

# #         return slides, layouts, timings, fonts

# # # --- WORKER ---
# # def _generate_frame_batch_worker_v2(batch_data):
# #     frame_tasks, gen_data, output_dir, width, height = batch_data
    
# #     font_cache = {}
# #     def get_fonts(size):
# #         if size not in font_cache:
# #             try:
# #                 r = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, size)
# #                 b = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
# #                 font_cache[size] = (r, b)
# #             except:
# #                 r = ImageFont.load_default(size)
# #                 b = ImageFont.load_default(size)
# #                 font_cache[size] = (r, b)
# #         return font_cache[size]

# #     # UPDATED COLORS TO MATCH REFERENCE
# #     C_REG = (210, 210, 210, 255) # Very Light Grey (Ghost text)
# #     C_BOLD = (20, 20, 20, 255)   # Soft Black
# #     FADE = 0.12 # Faster, snappier fade

# #     generated = []
# #     for frame_num, ts, slide_idx, _ in frame_tasks:
# #         img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
# #         draw = ImageDraw.Draw(img)
        
# #         slide_lines = gen_data['slides'][slide_idx]
# #         layout = gen_data['layouts'][slide_idx]
# #         font_size = gen_data['fonts'][slide_idx]
        
# #         reg_font, bold_font = get_fonts(font_size)
        
# #         for line in slide_lines:
# #             for w_data in line:
# #                 coords = layout.get(str(w_data['id']))
# #                 if not coords: continue
                
# #                 start, end = w_data['start'], w_data['end']
                
# #                 # Fade Logic
# #                 if ts < start:
# #                     col = C_REG
# #                     fnt = reg_font
# #                 elif ts >= start + FADE:
# #                     col = C_BOLD
# #                     fnt = bold_font
# #                 else:
# #                     p = (ts - start) / FADE
# #                     col = interpolate_color(C_REG, C_BOLD, p)
# #                     fnt = bold_font if p > 0.5 else reg_font
                
# #                 draw.text(coords, w_data['display_word'], font=fnt, fill=col)
                
# #         path = output_dir / f"f_{frame_num:06d}.png"
# #         img.save(path, optimize=False, compress_level=0)
# #         generated.append(str(path))
        
# #     return generated

# # # --- MAIN ---
# # def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
# #     logger.info("--- Starting V2 Render (Founders Semantic Center) ---")
# #     temp_frames_dir = None
# #     try:
# #         audio_clip = AudioFileClip(str(audio_path))
# #         duration = audio_clip.duration
# #         fps = settings.VIDEO_FPS
# #         width = settings.VIDEO_WIDTH
# #         height = settings.VIDEO_HEIGHT
# #         bg_clip = ImageClip(settings.DEFAULT_BACKGROUND).with_duration(duration)
        
# #         gen = FrameGeneratorFounders(timestamps_path, width, height)
        
# #         total_frames = int(duration * fps)
# #         frame_tasks = []
# #         for i in range(total_frames):
# #             ts = i / fps
# #             active = 0
# #             for idx, (s, e) in enumerate(gen.slide_timings):
# #                 if s <= ts <= e + 0.5: 
# #                     active = idx
# #                     break
# #                 if ts > e: active = min(idx + 1, len(gen.slides)-1)
# #             frame_tasks.append((i, ts, active, gen.slide_timings[active][0]))

# #         ser_slides = []
# #         ser_layouts = {}
# #         for idx, lines in enumerate(gen.slides):
# #             sl = []
# #             for line in lines:
# #                 wl = []
# #                 for w in line:
# #                     wl.append({'id': id(w), 'display_word': w.display_word, 'start': w.start, 'end': w.end})
# #                 sl.append(wl)
# #             ser_slides.append(sl)
# #             ser_layouts[idx] = {str(k): v for k, v in gen.slide_layouts[idx].items()}
        
# #         gen_data = {'slides': ser_slides, 'layouts': ser_layouts, 'fonts': gen.slide_fonts}

# #         temp_frames_dir = Path(tempfile.mkdtemp(prefix="sem_frames_"))
# #         cpu = os.cpu_count() or 4
# #         batch = max(50, len(frame_tasks) // cpu)
# #         batches = [frame_tasks[i:i+batch] for i in range(0, len(frame_tasks), batch)]
# #         args = [(b, gen_data, temp_frames_dir, width, height) for b in batches]
        
# #         logger.info(f"Rendering {len(frame_tasks)} frames...")
# #         all_f = []
# #         with multiprocessing.Pool(cpu) as pool:
# #             results = list(tqdm(pool.imap(_generate_frame_batch_worker_v2, args), total=len(batches), file=sys.stderr))
# #             for r in results: all_f.extend(r)
            
# #         all_f.sort(key=lambda x: int(Path(x).stem.split('_')[1]))
        
# #         clip = ImageSequenceClip(all_f, fps=fps)
# #         final = CompositeVideoClip([bg_clip, clip])
# #         final.write_videofile(str(output_path), audio=str(audio_path), fps=fps, codec="libx264", preset="veryfast", threads=4, logger=None)
# #         return output_path

# #     except Exception as e:
# #         logger.error("Render Failed", exc_info=True)
# #         raise
# #     finally:
# #         if temp_frames_dir and temp_frames_dir.exists(): shutil.rmtree(temp_frames_dir)



# # #182903
# # import json
# # import logging
# # import os
# # import tempfile
# # import multiprocessing
# # import shutil
# # import sys
# # from pathlib import Path
# # from typing import List, Optional, Dict, Tuple, Any

# # import numpy as np
# # from PIL import Image, ImageDraw, ImageFont
# # from moviepy import AudioFileClip, CompositeVideoClip, ImageSequenceClip, ImageClip
# # from pydantic import BaseModel
# # from tqdm import tqdm

# # from app.config import settings

# # logger = logging.getLogger(__name__)

# # # --- CONFIGURATION ---
# # # 1. Colors (Founders Style)
# # COLOR_BG = (250, 248, 242, 255)    # Cream/Off-White
# # COLOR_GHOST = (200, 200, 200, 255) # Very Light Gray (Inactive)
# # COLOR_ACTIVE = (20, 20, 20, 255)   # Sharp Black (Active)

# # # 2. Animation Config
# # TRANSITION_DURATION = 0.5  # Duration of the "Slide Up" entry
# # SLIDE_UP_DISTANCE = 40     # Pixels to slide up during entry
# # WORD_FADE_TIME = 0.30      # How fast a single word turns black

# # # --- DATA MODELS ---
# # class WordTimestamp(BaseModel):
# #     word: str
# #     start: float
# #     end: float
# #     display_word: Optional[str] = None

# # # --- UTILS ---
# # def interpolate_color(start_color, end_color, progress):
# #     return tuple(
# #         int(start_color[i] + (end_color[i] - start_color[i]) * progress)
# #         for i in range(4)
# #     )

# # # --- GENERATOR CLASS ---
# # class FrameGeneratorFounders:
# #     def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
# #         self.bg_width = bg_width
# #         self.bg_height = bg_height
        
# #         # Box Model (The "Safe Area" for text)
# #         # We give it nice wide margins so it looks "Premium"
# #         self.margin_x = int(self.bg_width * 0.15) 
# #         self.margin_y = int(self.bg_height * 0.15)
# #         self.box_width = self.bg_width - (2 * self.margin_x)
# #         self.box_height = self.bg_height - (2 * self.margin_y)

# #         # Load Data & Group by Semantic Phrases
# #         self.phrases = self._load_and_group_phrases(timestamps_path)
        
# #         # Build Slides (Dynamic Box Fitting)
# #         self.slides_data = self._build_dynamic_slides()

# #     def _load_fonts(self, size):
# #         try:
# #             # Load fonts (User should ensure these paths are correct in settings)
# #             reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, size)
# #             bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
# #             return reg, bold
# #         except:
# #             return ImageFont.load_default(size), ImageFont.load_default(size)

# #     def _load_and_group_phrases(self, path: Path) -> List[List[WordTimestamp]]:
# #         """Groups words into natural phrases based on pauses."""
# #         with open(path, "r", encoding="utf-8") as f:
# #             data = json.load(f)
        
# #         raw_words = []
# #         if 'words' in data and data['words']:
# #             raw_words = data['words']
# #         elif 'segments' in data:
# #             for s in data['segments']:
# #                 raw_words.extend(s.get('words', []))
        
# #         if not raw_words: return []

# #         phrases = []
# #         current_phrase = []
        
# #         for i, w_dict in enumerate(raw_words):
# #             word = WordTimestamp(
# #                 word=w_dict.get('word', '').strip(),
# #                 start=w_dict.get('start', 0),
# #                 end=w_dict.get('end', 0),
# #                 display_word=w_dict.get('word', '').strip()
# #             )
            
# #             # Punctuation/Pause Logic
# #             txt = word.display_word
# #             is_end = txt[-1] in ['.', '!', '?', ',', ':'] if txt else False
            
# #             is_pause = False
# #             if i < len(raw_words) - 1:
# #                 next_start = raw_words[i+1].get('start', 0)
# #                 if next_start - word.end > 0.4: 
# #                     is_pause = True
            
# #             current_phrase.append(word)
            
# #             # Semantic Break Conditions
# #             if is_end or is_pause or len(current_phrase) > 12:
# #                 phrases.append(current_phrase)
# #                 current_phrase = []
        
# #         if current_phrase: phrases.append(current_phrase)
        
# #         # Cleanup Punctuation for cleaner visual
# #         for phrase in phrases:
# #             if not phrase: continue
# #             last = phrase[-1]
# #             if last.display_word and last.display_word[-1] in ['.', ',', '!', '?']:
# #                 last.display_word = last.display_word[:-1]

# #         return phrases

# #     def _try_layout(self, words, font_size):
# #         """
# #         Attempts to fit words into lines with a specific font size.
# #         Returns: (lines, total_height, success_bool)
# #         """
# #         _, font = self._load_fonts(font_size) # Use bold for sizing
# #         dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
# #         space_w = dummy.textlength(" ", font=font)
# #         line_height = int(font_size * 1.3) # 1.3 is a good line-height for serif
        
# #         lines = []
# #         current_line = []
# #         current_w = 0
        
# #         for word in words:
# #             wl = dummy.textlength(word.display_word, font=font)
            
# #             # Check if single word is wider than the whole box (unlikely but possible)
# #             if wl > self.box_width:
# #                 return [], 0, False

# #             if current_line and (current_w + space_w + wl > self.box_width):
# #                 # Push line
# #                 lines.append((current_line, current_w))
# #                 current_line = [word]
# #                 current_w = wl
# #             else:
# #                 if current_line: current_w += space_w
# #                 current_line.append(word)
# #                 current_w += wl
                
# #         if current_line: lines.append((current_line, current_w))
        
# #         total_h = len(lines) * line_height
        
# #         # Success if total height fits in box
# #         return lines, total_h, (total_h <= self.box_height)

# #     def _build_dynamic_slides(self):
# #         """
# #         The Core Logic: For each phrase, start BIG and shrink until it fits.
# #         """
# #         slides_data = [] # List of dicts containing layout info

# #         # Font Constraints
# #         MAX_FONT = int(self.bg_height / 6)  # Huge (for 2-3 words)
# #         MIN_FONT = int(self.bg_height / 20) # Minimum legible size
        
# #         for phrase in self.phrases:
# #             if not phrase: continue
            
# #             # 1. Iterative Shrinking (The "Fit" Loop)
# #             current_size = MAX_FONT
# #             final_lines = []
# #             final_h = 0
            
# #             while current_size >= MIN_FONT:
# #                 lines, h, fits = self._try_layout(phrase, current_size)
# #                 if fits:
# #                     final_lines = lines
# #                     final_h = h
# #                     break
# #                 current_size -= 5 # Step down by 5px
            
# #             # If still doesn't fit at MIN_FONT, use MIN_FONT anyway (safety)
# #             if not final_lines:
# #                 final_lines, final_h, _ = self._try_layout(phrase, MIN_FONT)
# #                 current_size = MIN_FONT

# #             # 2. Build Coordinate Map (Vertical Center)
# #             _, font = self._load_fonts(current_size)
# #             dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
# #             space_w = dummy.textlength(" ", font=font)
# #             line_height = int(current_size * 1.3)
            
# #             # Vertical Center
# #             start_y = (self.bg_height - final_h) // 2
            
# #             layout_map = {}
# #             curr_y = start_y
            
# #             slide_structure = [] # For the worker to know line breaks
            
# #             for line_words, line_width in final_lines:
# #                 # Horizontal Center
# #                 start_x = (self.bg_width - line_width) // 2
# #                 curr_x = start_x
                
# #                 l_struct = []
# #                 for word in line_words:
# #                     layout_map[id(word)] = (int(curr_x), int(curr_y))
# #                     wl = dummy.textlength(word.display_word, font=font)
# #                     curr_x += wl + space_w
                    
# #                     l_struct.append({
# #                         'id': id(word),
# #                         'text': word.display_word,
# #                         'start': word.start,
# #                         'end': word.end
# #                     })
                
# #                 slide_structure.append(l_struct)
# #                 curr_y += line_height

# #             # 3. Store Slide Data
# #             slides_data.append({
# #                 'start_time': phrase[0].start,
# #                 'end_time': phrase[-1].end,
# #                 'font_size': current_size,
# #                 'lines': slide_structure,
# #                 'layout': layout_map
# #             })
            
# #         return slides_data

# # # --- WORKER ---
# # def _generate_frame_batch_worker_v3(batch_data):
# #     frame_tasks, gen_data, output_dir, width, height = batch_data
    
# #     # Font Cache
# #     font_cache = {}
# #     def get_fonts(size):
# #         if size not in font_cache:
# #             try:
# #                 # Assuming user has these fonts, otherwise fallbacks
# #                 r = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, size)
# #                 b = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
# #                 font_cache[size] = (r, b)
# #             except:
# #                 r = ImageFont.load_default(size)
# #                 b = ImageFont.load_default(size)
# #                 font_cache[size] = (r, b)
# #         return font_cache[size]

# #     generated = []
    
# #     for frame_num, ts, slide_idx, slide_start_ts in frame_tasks:
# #         # Create Base Image
# #         img = Image.new("RGBA", (width, height), COLOR_BG)
        
# #         # Handle Transition Logic (Global Slide Opacity/Position)
# #         time_into_slide = ts - slide_start_ts
        
# #         # 1. Calculate Global Slide Alpha & Offset (The "Entry" Animation)
# #         slide_alpha = 255
# #         y_offset = 0
        
# #         if time_into_slide < TRANSITION_DURATION:
# #             progress = time_into_slide / TRANSITION_DURATION
# #             # Ease Out Cubic
# #             progress = 1 - pow(1 - progress, 3) 
            
# #             slide_alpha = int(255 * progress)
# #             y_offset = int(SLIDE_UP_DISTANCE * (1 - progress)) # Move UP from offset to 0

# #         # If totally invisible, skip drawing text
# #         if slide_alpha < 5:
# #             path = output_dir / f"f_{frame_num:06d}.png"
# #             img.save(path, optimize=False, compress_level=0)
# #             generated.append(str(path))
# #             continue
            
# #         # Overlay for text with alpha
# #         txt_layer = Image.new("RGBA", (width, height), (0,0,0,0))
# #         draw = ImageDraw.Draw(txt_layer)
        
# #         slide = gen_data[slide_idx]
# #         layout = slide['layout']
# #         font_size = slide['font_size']
# #         reg_font, bold_font = get_fonts(font_size)
        
# #         # Draw Words
# #         for line in slide['lines']:
# #             for w in line:
# #                 base_x, base_y = layout.get(str(w['id']))
                
# #                 # Apply Entry Animation Offset
# #                 draw_y = base_y + y_offset
                
# #                 # Color Logic (Active vs Inactive)
# #                 # We mix the Slide Alpha with the Word Color Alpha
                
# #                 if ts < w['start']:
# #                     # Future Word (Ghost)
# #                     base_color = COLOR_GHOST
# #                     font = reg_font
# #                 elif ts > (w['start'] + WORD_FADE_TIME):
# #                     # Active Word (Bold)
# #                     base_color = COLOR_ACTIVE
# #                     font = bold_font
# #                 else:
# #                     # Transitioning Word
# #                     p = (ts - w['start']) / WORD_FADE_TIME
# #                     base_color = interpolate_color(COLOR_GHOST, COLOR_ACTIVE, p)
# #                     font = bold_font if p > 0.5 else reg_font

# #                 # Apply Slide Entry Fade to the color
# #                 final_color = (base_color[0], base_color[1], base_color[2], int(base_color[3] * (slide_alpha/255)))
                
# #                 draw.text((base_x, draw_y), w['text'], font=font, fill=final_color)
        
# #         # Composite Text onto BG
# #         img = Image.alpha_composite(img, txt_layer)
        
# #         path = output_dir / f"f_{frame_num:06d}.png"
# #         img.save(path, optimize=False, compress_level=0)
# #         generated.append(str(path))
        
# #     return generated

# # # --- MAIN ---
# # def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
# #     logger.info("--- Starting V3 Render (Founders Dynamic Box) ---")
# #     temp_frames_dir = None
# #     try:
# #         audio_clip = AudioFileClip(str(audio_path))
# #         duration = audio_clip.duration
# #         fps = settings.VIDEO_FPS
# #         width = settings.VIDEO_WIDTH
# #         height = settings.VIDEO_HEIGHT
        
# #         # Background is now handled per-frame in worker to allow clean alpha blending
# #         # But we need a dummy clip for MoviePy
# #         bg_clip = ImageClip(settings.DEFAULT_BACKGROUND).with_duration(duration) 
        
# #         gen = FrameGeneratorFounders(timestamps_path, width, height)
        
# #         # Serialize Data for Workers
# #         serialized_slides = []
# #         for s in gen.slides_data:
# #             # Convert keys to strings for JSON/Dict compat
# #             layout_str_keys = {str(k): v for k, v in s['layout'].items()}
# #             serialized_slides.append({
# #                 'start_time': s['start_time'],
# #                 'end_time': s['end_time'],
# #                 'font_size': s['font_size'],
# #                 'lines': s['lines'],
# #                 'layout': layout_str_keys
# #             })

# #         # Build Frame Tasks
# #         total_frames = int(duration * fps)
# #         frame_tasks = []
        
# #         for i in range(total_frames):
# #             ts = i / fps
            
# #             # Find active slide
# #             active_idx = 0
# #             slide_start = 0
            
# #             # Simple linear search (safe for short videos)
# #             for idx, slide in enumerate(serialized_slides):
# #                 if slide['start_time'] <= ts <= slide['end_time'] + 0.5:
# #                     active_idx = idx
# #                     slide_start = slide['start_time']
# #                     break
# #                 # Hold last slide if audio continues
# #                 if idx == len(serialized_slides) - 1 and ts > slide['end_time']:
# #                     active_idx = idx
# #                     slide_start = slide['start_time']

# #             frame_tasks.append((i, ts, active_idx, slide_start))

# #         temp_frames_dir = Path(tempfile.mkdtemp(prefix="founders_frames_"))
# #         cpu = os.cpu_count() or 4
# #         batch_size = max(50, len(frame_tasks) // cpu)
# #         batches = [frame_tasks[i:i+batch_size] for i in range(0, len(frame_tasks), batch_size)]
        
# #         worker_args = [(b, serialized_slides, temp_frames_dir, width, height) for b in batches]
        
# #         logger.info(f"Rendering {len(frame_tasks)} frames...")
# #         all_files = []
# #         with multiprocessing.Pool(cpu) as pool:
# #             results = list(tqdm(pool.imap(_generate_frame_batch_worker_v3, worker_args), 
# #                               total=len(batches), file=sys.stderr))
# #             for r in results: all_files.extend(r)
            
# #         all_files.sort(key=lambda x: int(Path(x).stem.split('_')[1]))
        
# #         # Composite
# #         # Since we drew the BG in the frame, we just sequence the images
# #         final_clip = ImageSequenceClip(all_files, fps=fps)
# #         final_clip = final_clip.with_audio(audio_clip)
        
# #         final_clip.write_videofile(
# #             str(output_path),
# #             fps=fps,
# #             codec="libx264",
# #             preset="veryfast",
# #             threads=4,
# #             logger=None 
# #         )
# #         return output_path

# #     except Exception as e:
# #         logger.error("Render Failed", exc_info=True)
# #         raise
# #     finally:
# #         if temp_frames_dir and temp_frames_dir.exists():
# #             shutil.rmtree(temp_frames_dir)


# # #184247
# # import json
# # import logging
# # import os
# # import tempfile
# # import multiprocessing
# # import shutil
# # import sys
# # from pathlib import Path
# # from typing import List, Optional, Dict, Tuple, Any

# # import numpy as np
# # from PIL import Image, ImageDraw, ImageFont
# # from moviepy import AudioFileClip, CompositeVideoClip, ImageSequenceClip, ImageClip
# # from pydantic import BaseModel
# # from tqdm import tqdm

# # from app.config import settings

# # logger = logging.getLogger(__name__)

# # # --- CONFIGURATION ---

# # # 1. Colors (Founders Style)
# # COLOR_BG = (250, 248, 242, 255)    # Cream/Off-White
# # COLOR_GHOST = (210, 210, 210, 255) # Light Gray (Future words)
# # COLOR_ACTIVE = (20, 20, 20, 255)   # Soft Black (Active words)

# # # 2. Layout
# # MARGIN_LEFT_PERCENT = 0.12  # 12% from left
# # MARGIN_RIGHT_PERCENT = 0.12 # 12% from right
# # MARGIN_TOP_PERCENT = 0.15   # 15% from top
# # MAX_LINES = 6               # Safety limit

# # # 3. Timing
# # WORD_FADE_TIME = 0.10       # Quick snap from gray to black

# # # --- DATA MODELS ---
# # class WordTimestamp(BaseModel):
# #     word: str
# #     start: float
# #     end: float
# #     display_word: Optional[str] = None

# # # --- UTILS ---
# # def interpolate_color(start_color, end_color, progress):
# #     return tuple(
# #         int(start_color[i] + (end_color[i] - start_color[i]) * progress)
# #         for i in range(4)
# #     )

# # # --- GENERATOR CLASS ---
# # class FrameGeneratorFounders:
# #     def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
# #         self.bg_width = bg_width
# #         self.bg_height = bg_height
        
# #         # Box Model
# #         self.margin_x = int(self.bg_width * MARGIN_LEFT_PERCENT)
# #         self.max_width = int(self.bg_width * (1 - MARGIN_LEFT_PERCENT - MARGIN_RIGHT_PERCENT))
# #         self.box_height = int(self.bg_height * (1 - (2 * MARGIN_TOP_PERCENT)))
# #         self.start_y = int(self.bg_height * MARGIN_TOP_PERCENT)

# #         self.phrases = self._load_and_group_phrases(timestamps_path)
# #         self.slides_data = self._build_dynamic_slides()

# #     def _load_fonts(self, size):
# #         try:
# #             reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, size)
# #             bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
# #             return reg, bold
# #         except:
# #             return ImageFont.load_default(size), ImageFont.load_default(size)

# #     def _load_and_group_phrases(self, path: Path) -> List[List[WordTimestamp]]:
# #         with open(path, "r", encoding="utf-8") as f:
# #             data = json.load(f)
        
# #         raw_words = []
# #         if 'words' in data and data['words']:
# #             raw_words = data['words']
# #         elif 'segments' in data:
# #             for s in data['segments']:
# #                 raw_words.extend(s.get('words', []))
        
# #         if not raw_words: return []

# #         phrases = []
# #         current_phrase = []
        
# #         # Smart Phrasing Logic
# #         for i, w_dict in enumerate(raw_words):
# #             word = WordTimestamp(
# #                 word=w_dict.get('word', '').strip(),
# #                 start=w_dict.get('start', 0),
# #                 end=w_dict.get('end', 0),
# #                 display_word=w_dict.get('word', '').strip()
# #             )
            
# #             txt = word.display_word
# #             # End of sentence punctuation
# #             is_end_sentence = txt[-1] in ['.', '!', '?'] if txt else False
            
# #             # Time Gap (Pause)
# #             is_pause = False
# #             if i < len(raw_words) - 1:
# #                 next_start = raw_words[i+1].get('start', 0)
# #                 if next_start - word.end > 0.5: # 0.5s silence triggers new slide
# #                     is_pause = True
            
# #             current_phrase.append(word)
            
# #             # Cut Phrase Condition:
# #             # 1. End of sentence
# #             # 2. Big pause
# #             # 3. Too many words (prevent clutter)
# #             if is_end_sentence or is_pause or len(current_phrase) >= 10:
# #                 phrases.append(current_phrase)
# #                 current_phrase = []
        
# #         if current_phrase: phrases.append(current_phrase)
        
# #         # Clean punctuation for display (Optional: Remove this loop if you WANT punctuation)
# #         # User usually prefers clean look, but keep punctuation if it's vital.
# #         # commenting out the stripper to keep punctuation as per 'mimic it as it is'
# #         # for phrase in phrases:
# #         #     if phrase and phrase[-1].display_word[-1] in [',']:
# #         #          phrase[-1].display_word = phrase[-1].display_word[:-1]

# #         return phrases

# #     def _try_layout(self, words, font_size):
# #         """
# #         Tries to fit words into the box using the given font size.
# #         """
# #         _, font = self._load_fonts(font_size) # Use bold for sizing calculation
# #         dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
# #         space_w = dummy.textlength(" ", font=font)
        
# #         # Line height: slightly tighter for that modern look (1.2 instead of 1.4)
# #         line_height = int(font_size * 1.25)
        
# #         lines = []
# #         current_line = []
# #         current_w = 0
        
# #         for word in words:
# #             wl = dummy.textlength(word.display_word, font=font)
            
# #             # Check width
# #             if current_line and (current_w + space_w + wl > self.max_width):
# #                 lines.append((current_line, current_w))
# #                 current_line = [word]
# #                 current_w = wl
# #             else:
# #                 if current_line: current_w += space_w
# #                 current_line.append(word)
# #                 current_w += wl
                
# #         if current_line: lines.append((current_line, current_w))
        
# #         total_h = len(lines) * line_height
# #         return lines, total_h, (total_h <= self.box_height)

# #     def _build_dynamic_slides(self):
# #         slides_data = []

# #         # We start MASSIVE. If it fits, great. If not, we shrink.
# #         # This ensures "Dynamic" sizing (Point 5).
# #         START_FONT = int(self.bg_height / 5)  # Very Big
# #         MIN_FONT = int(self.bg_height / 18)   # Smallest allowed
        
# #         for phrase in self.phrases:
# #             if not phrase: continue
            
# #             current_size = START_FONT
# #             final_lines = []
# #             final_h = 0
            
# #             # --- THE OPTIMIZED "BETTER WAY" LOOP ---
# #             while current_size >= MIN_FONT:
# #                 lines, h, fits = self._try_layout(phrase, current_size)
# #                 if fits:
# #                     final_lines = lines
# #                     final_h = h
# #                     break
# #                 # Semantic stepping: Decrease faster at high sizes, slower at low sizes
# #                 step = 10 if current_size > 100 else 5
# #                 current_size -= step
            
# #             # Fallback if it still doesn't fit
# #             if not final_lines:
# #                 final_lines, final_h, _ = self._try_layout(phrase, MIN_FONT)
# #                 current_size = MIN_FONT

# #             # Generate Coordinates (LEFT ALIGNED)
# #             _, font = self._load_fonts(current_size)
# #             dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
# #             space_w = dummy.textlength(" ", font=font)
# #             line_height = int(current_size * 1.25)
            
# #             # Vertical Center Logic: 
# #             # We calculate the top margin needed to center the text block vertically
# #             block_top = self.start_y + (self.box_height - final_h) // 2
            
# #             layout_map = {}
# #             curr_y = block_top
            
# #             slide_structure = []
            
# #             for line_words, _ in final_lines:
# #                 # LEFT ALIGN: Always start at margin_x
# #                 curr_x = self.margin_x 
                
# #                 l_struct = []
# #                 for word in line_words:
# #                     layout_map[id(word)] = (int(curr_x), int(curr_y))
# #                     wl = dummy.textlength(word.display_word, font=font)
# #                     curr_x += wl + space_w
                    
# #                     l_struct.append({
# #                         'id': id(word),
# #                         'text': word.display_word,
# #                         'start': word.start,
# #                         'end': word.end
# #                     })
                
# #                 slide_structure.append(l_struct)
# #                 curr_y += line_height

# #             slides_data.append({
# #                 'start_time': phrase[0].start,
# #                 'end_time': phrase[-1].end,
# #                 'font_size': current_size,
# #                 'lines': slide_structure,
# #                 'layout': layout_map
# #             })
            
# #         return slides_data

# # # --- WORKER ---
# # def _generate_frame_batch_worker_v4(batch_data):
# #     frame_tasks, gen_data, output_dir, width, height = batch_data
    
# #     font_cache = {}
# #     def get_fonts(size):
# #         if size not in font_cache:
# #             try:
# #                 r = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, size)
# #                 b = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
# #                 font_cache[size] = (r, b)
# #             except:
# #                 r = ImageFont.load_default(size)
# #                 b = ImageFont.load_default(size)
# #                 font_cache[size] = (r, b)
# #         return font_cache[size]

# #     generated = []
    
# #     for frame_num, ts, slide_idx in frame_tasks:
# #         # 1. Draw Background
# #         img = Image.new("RGB", (width, height), COLOR_BG)
        
# #         # 2. Handle "Silence" (No active slide)
# #         if slide_idx == -1:
# #             path = output_dir / f"f_{frame_num:06d}.png"
# #             img.save(path, optimize=False, compress_level=0)
# #             generated.append(str(path))
# #             continue
            
# #         draw = ImageDraw.Draw(img)
# #         slide = gen_data[slide_idx]
# #         layout = slide['layout']
# #         font_size = slide['font_size']
# #         reg_font, bold_font = get_fonts(font_size)
        
# #         # 3. Draw Words
# #         for line in slide['lines']:
# #             for w in line:
# #                 base_x, base_y = layout.get(str(w['id']))
                
# #                 # Color Logic (No animation, just color shift)
# #                 if ts < w['start']:
# #                     # Future = Gray
# #                     color = COLOR_GHOST
# #                     font = reg_font # or bold_font based on preference, reference uses Regular for inactive
# #                 elif ts > (w['start'] + WORD_FADE_TIME):
# #                     # Past/Active = Black
# #                     color = COLOR_ACTIVE
# #                     font = bold_font
# #                 else:
# #                     # Transition
# #                     p = (ts - w['start']) / WORD_FADE_TIME
# #                     color = interpolate_color(COLOR_GHOST, COLOR_ACTIVE, p)
# #                     font = bold_font if p > 0.5 else reg_font

# #                 draw.text((base_x, base_y), w['text'], font=font, fill=color)
        
# #         path = output_dir / f"f_{frame_num:06d}.png"
# #         img.save(path, optimize=False, compress_level=0)
# #         generated.append(str(path))
        
# #     return generated

# # # --- MAIN ---
# # def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
# #     logger.info("--- Starting V4 Render (Left Align, Dynamic Fit, No Glitch) ---")
# #     temp_frames_dir = None
# #     try:
# #         audio_clip = AudioFileClip(str(audio_path))
# #         duration = audio_clip.duration
# #         fps = settings.VIDEO_FPS
# #         width = settings.VIDEO_WIDTH
# #         height = settings.VIDEO_HEIGHT
        
# #         gen = FrameGeneratorFounders(timestamps_path, width, height)
        
# #         # Serialize Data
# #         serialized_slides = []
# #         for s in gen.slides_data:
# #             layout_str = {str(k): v for k, v in s['layout'].items()}
# #             serialized_slides.append({
# #                 'start_time': s['start_time'],
# #                 'end_time': s['end_time'],
# #                 'font_size': s['font_size'],
# #                 'lines': s['lines'],
# #                 'layout': layout_str
# #             })

# #         # Build Tasks
# #         total_frames = int(duration * fps)
# #         frame_tasks = []
        
# #         # GLITCH FIX: Explicitly find the slide, default to -1 if none found
# #         for i in range(total_frames):
# #             ts = i / fps
# #             active_idx = -1 
            
# #             for idx, slide in enumerate(serialized_slides):
# #                 # Buffer of 0.1s to keep text on screen slightly longer during pauses
# #                 if slide['start_time'] <= ts <= slide['end_time'] + 0.1:
# #                     active_idx = idx
# #                     break
            
# #             frame_tasks.append((i, ts, active_idx))

# #         temp_frames_dir = Path(tempfile.mkdtemp(prefix="final_frames_"))
# #         cpu = os.cpu_count() or 4
# #         batch_size = max(50, len(frame_tasks) // cpu)
# #         batches = [frame_tasks[i:i+batch_size] for i in range(0, len(frame_tasks), batch_size)]
        
# #         worker_args = [(b, serialized_slides, temp_frames_dir, width, height) for b in batches]
        
# #         logger.info(f"Rendering {len(frame_tasks)} frames...")
# #         all_files = []
# #         with multiprocessing.Pool(cpu) as pool:
# #             results = list(tqdm(pool.imap(_generate_frame_batch_worker_v4, worker_args), 
# #                               total=len(batches), file=sys.stderr))
# #             for r in results: all_files.extend(r)
            
# #         all_files.sort(key=lambda x: int(Path(x).stem.split('_')[1]))
        
# #         final_clip = ImageSequenceClip(all_files, fps=fps)
# #         final_clip = final_clip.with_audio(audio_clip)
        
# #         final_clip.write_videofile(
# #             str(output_path),
# #             fps=fps,
# #             codec="libx264",
# #             preset="veryfast",
# #             threads=4,
# #             logger=None 
# #         )
# #         return output_path

# #     except Exception as e:
# #         logger.error("Render Failed", exc_info=True)
# #         raise
# #     finally:
# #         if temp_frames_dir and temp_frames_dir.exists():
# #             shutil.rmtree(temp_frames_dir)


# #GROK
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

# # 1. Layout Controls
# TEXT_ALIGN = "left"
# MAX_LINES_PER_SLIDE = 4  # Changed to 4 as per requirements
# MARGIN_PERCENT_X = 0.10    # Left/Right Margin (10% is safe)
# MARGIN_PERCENT_Y = 0.10    # Top/Bottom Margin

# # 2. Font Sizing
# # DECREASE this to make font BIGGER. INCREASE to make smaller.
# # 10 is standard "Huge". 8 is "Massive". 12 is "Large".
# FONT_HEIGHT_DIVISOR = 8   
# LINE_HEIGHT_RATIO = 1.4    

# # 3. Data Cleaning Settings
# MIN_WORD_DURATION = 0.2    # Seconds. Any word shorter than this gets inflated.
# MAX_SILENCE_GAP = 0.3      # Seconds. If gap < 0.3s, we bridge it (no flickering).

# # 4. Semantic Grouping
# PAUSE_THRESHOLD = 0.8      # Gap > 0.8s triggers new sentence/slide consideration

# # --- DATA MODELS ---
# class WordTimestamp(BaseModel):
#     word: str
#     start: float
#     end: float
#     display_word: Optional[str] = None
#     id: int = 0 

# # --- DATA SANITIZER (THE FIX) ---
# def sanitize_words(raw_words: List[Dict]) -> List[WordTimestamp]:
#     """
#     Fixes 0-duration words, overlaps, and jerky silences.
#     """
#     clean_words = []
    
#     # 1. Convert to Objects & Basic Cleanup
#     for i, w in enumerate(raw_words):
#         word_str = w['word'].strip()
#         # Skip empty strings
#         if not word_str: 
#             continue
            
#         # Fix Punctuation: Attach to previous word if it's just a symbol
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

#     # 2. Fix Durations & Overlaps
#     for i in range(len(clean_words)):
#         curr = clean_words[i]
        
#         # A. Force Minimum Duration (Fixes the "Flash" bug)
#         if (curr.end - curr.start) < MIN_WORD_DURATION:
#             curr.end = curr.start + MIN_WORD_DURATION
            
#         # B. Fix Overlaps with Next Word
#         if i < len(clean_words) - 1:
#             next_w = clean_words[i+1]
#             # If current ends AFTER next starts, push next start forward
#             if curr.end > next_w.start:
#                 next_w.start = curr.end
#                 # Ensure next word still has min duration after being pushed
#                 if (next_w.end - next_w.start) < MIN_WORD_DURATION:
#                     next_w.end = next_w.start + MIN_WORD_DURATION

#     # 3. Bridge Small Silences (Fixes the "Jerky" bug)
#     # If there is a tiny gap between words, extend the current word to touch the next.
#     for i in range(len(clean_words) - 1):
#         curr = clean_words[i]
#         next_w = clean_words[i+1]
        
#         gap = next_w.start - curr.end
#         if 0 < gap < MAX_SILENCE_GAP:
#             # Extend current word to fill the gap
#             curr.end = next_w.start

#     return clean_words

# def interpolate_color(start_color, end_color, progress):
#     """Blends two colors based on progress (0.0 to 1.0)."""
#     return tuple(
#         int(start_color[i] + (end_color[i] - start_color[i]) * progress)
#         for i in range(4)
#     )

# # --- GENERATOR CLASS ---
# class FrameGeneratorBigFlow:
#     def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
#         self.bg_width = bg_width
#         self.bg_height = bg_height
        
#         self.margin_x = int(self.bg_width * MARGIN_PERCENT_X)
#         self.margin_y = int(self.bg_height * MARGIN_PERCENT_Y)  # Added Y margin
#         self.usable_height = self.bg_height - 2 * self.margin_y
#         self.max_text_width = self.bg_width - (2 * self.margin_x)
        
#         self.font_size = int(self.bg_height / FONT_HEIGHT_DIVISOR)
#         self.line_height = int(self.font_size * LINE_HEIGHT_RATIO)
        
#         self.font = self._load_font(self.font_size)
        
#         # LOAD AND SANITIZE
#         self.all_words = self._load_and_clean_data(timestamps_path)
        
#         self.slides = []       
#         self.slide_layouts = [] 
#         self.slide_timings = [] 
        
#         self._build_semantic_slides()  # Updated method name

#     def _load_font(self, size):
#         try:
#             return ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
#         except:
#             return ImageFont.load_default(size)

#     def _load_and_clean_data(self, path: Path) -> List[WordTimestamp]:
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         raw_words = data.get("words", [])
#         # APPLY THE SANITIZER
#         return sanitize_words(raw_words)

#     def split_into_sentences(self, words: List[WordTimestamp]) -> List[List[WordTimestamp]]:
#         """
#         Splits words into meaningful sentences based on punctuation and pauses.
#         """
#         sentences = []
#         current = []
#         for i in range(len(words)):
#             current.append(words[i])
#             word = words[i].display_word
#             # Check for sentence-ending punctuation
#             if word.endswith(('.', '!', '?')):
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
#         """
#         Wraps a list of words into lines based on max width.
#         """
#         dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
#         space_width = dummy_draw.textlength(" ", font=self.font)
        
#         current_line = []
#         current_line_width = 0
#         lines = []
        
#         for word in words:
#             w_len = dummy_draw.textlength(word.display_word, font=self.font)
            
#             if current_line_width + (space_width if current_line else 0) + w_len > self.max_text_width and current_line:
#                 lines.append(current_line)
#                 current_line = [word]
#                 current_line_width = w_len
#             else:
#                 if current_line:
#                     current_line_width += space_width
#                 current_line.append(word)
#                 current_line_width += w_len
        
#         if current_line:
#             lines.append(current_line)
        
#         return lines

#     def _build_semantic_slides(self):
#         """
#         Builds slides semantically: group by sentences, start new slide for new sentences if >=3 lines occupied,
#         split large sentences into chunks of max lines.
#         """
#         sentences = self.split_into_sentences(self.all_words)
#         current_slide_lines = []
#         MAX = MAX_LINES_PER_SLIDE
        
#         for sent_words in sentences:
#             sent_lines = self._layout_into_lines(sent_words)
#             chunk_start = 0
#             while chunk_start < len(sent_lines):
#                 chunk_end = min(chunk_start + MAX, len(sent_lines))
#                 chunk_lines = sent_lines[chunk_start:chunk_end]
                
#                 # For the first chunk of a new sentence, force new slide if current >=3 lines
#                 if chunk_start == 0 and len(current_slide_lines) >= 3:
#                     if current_slide_lines:
#                         self._commit_slide(current_slide_lines)
#                     current_slide_lines = []
                
#                 # Add the chunk to current slide if it fits
#                 potential_lines = current_slide_lines + chunk_lines
#                 if len(potential_lines) > MAX:
#                     if current_slide_lines:
#                         self._commit_slide(current_slide_lines)
#                     current_slide_lines = chunk_lines
#                 else:
#                     current_slide_lines = potential_lines
                
#                 chunk_start += MAX
        
#         if current_slide_lines:
#             self._commit_slide(current_slide_lines)

#     def _commit_slide(self, lines: List[List[WordTimestamp]]):
#         if not lines: return
        
#         # Optional: Remove trailing comma from last word of slide (as in original)
#         if lines and lines[-1]:
#             last_word_obj = lines[-1][-1]
#             text = last_word_obj.display_word
#             if text.endswith(","):
#                 last_word_obj.display_word = text[:-1]

#         # Compute total height and center vertically within usable area
#         total_h = len(lines) * self.line_height
#         start_y = self.margin_y + (self.usable_height - total_h) // 2
        
#         layout = {}
#         curr_y = start_y
        
#         dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
#         space_width = dummy_draw.textlength(" ", font=self.font)
        
#         for line in lines:
#             curr_x = self.margin_x
#             for word in line:
#                 layout[word.id] = (int(curr_x), int(curr_y))
#                 w_len = dummy_draw.textlength(word.display_word, font=self.font)
#                 curr_x += w_len + space_width
#             curr_y += self.line_height
            
#         self.slides.append(lines)
#         self.slide_layouts.append(layout)
        
#         s_start = lines[0][0].start
#         s_end = lines[-1][-1].end
#         self.slide_timings.append((s_start, s_end))

# # --- WORKER ---
# def _generate_frame_batch_worker_v5(batch_data):
#     frame_tasks, gen_data, output_dir, width, height = batch_data
    
#     # 1. Load Fonts (global font size for now; can extend to per-slide if dynamic needed)
#     try:
#         font_bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, gen_data['font_size'])
#         # Try to load a regular version of the same size for the "inactive" state
#         # If you don't have a regular font file, use the bold one for both (just color changes)
#         try:
#             font_reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, gen_data['font_size'])
#         except Exception as e:
#             print("inside exception", e)
#             font_reg = font_bold 
#     except Exception as e:
#         print("exception", e)
#         raise e
#         font_bold = ImageFont.load_default(gen_data['font_size'])
#         font_reg = font_bold

#     # 2. Animation Settings (Keep fade-in as client likes)
#     C_ACTIVE = (20, 20, 20, 255)        # Black
#     C_FUTURE = (210, 210, 210, 255)    # Light Gray
#     FADE_DURATION = 0.30               # 0.30s transition window

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
                
#                 start = w_data['start']
                
#                 # --- THE ANIMATION LOGIC (Fade-in preserved) ---
                
#                 # Case 1: Fully Active (Time is past start)
#                 if ts >= start:
#                     color = C_ACTIVE
#                     font = font_bold
                
#                 # Case 2: Transitioning (Time is within FADE_DURATION before start)
#                 elif start - FADE_DURATION <= ts < start:
#                     # Calculate progress (0.0 to 1.0)
#                     progress = (ts - (start - FADE_DURATION)) / FADE_DURATION
                    
#                     # Interpolate Color
#                     color = interpolate_color(C_FUTURE, C_ACTIVE, progress)
                    
#                     # Switch Font Weight (Switch to bold when 50% darker)
#                     font = font_bold if progress > 0.5 else font_reg
                    
#                 # Case 3: Fully Future
#                 else:
#                     color = C_FUTURE
#                     font = font_reg

#                 draw.text(coords, w_data['display_word'], font=font, fill=color)
        
#         path = output_dir / f"f_{frame_num:06d}.png"
#         img.save(path, optimize=False, compress_level=0)
#         generated.append(str(path))
        
#     return generated

# # --- MAIN ---
# def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
#     logger.info("--- Starting V5 Render (Semantic Grouping + 4-Line Max + Fade-In) ---")
#     temp_frames_dir = None
#     try:
#         audio_clip = AudioFileClip(str(audio_path))
#         duration = audio_clip.duration
#         fps = settings.VIDEO_FPS
#         width = settings.VIDEO_WIDTH
#         height = settings.VIDEO_HEIGHT
#         bg_clip = ImageClip(settings.DEFAULT_BACKGROUND).with_duration(duration)
        
#         gen = FrameGeneratorBigFlow(timestamps_path, width, height)
        
#         total_frames = int(duration * fps)
#         frame_tasks = []
#         current_slide_idx = 0
        
#         for i in range(total_frames):
#             ts = i / fps
#             if current_slide_idx < len(gen.slide_timings):
#                 slide_start, slide_end = gen.slide_timings[current_slide_idx]
#                 # Lookahead: If we are past end, go next
#                 if ts > slide_end and current_slide_idx < len(gen.slide_timings) - 1:
#                     current_slide_idx += 1
#             frame_tasks.append((i, ts, current_slide_idx, 0))

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

#         temp_frames_dir = Path(tempfile.mkdtemp(prefix="v5_frames_"))
#         cpu_count = os.cpu_count() or 4
#         batch_size = max(50, len(frame_tasks) // cpu_count)
#         batches = [frame_tasks[i:i + batch_size] for i in range(0, len(frame_tasks), batch_size)]
        
#         worker_args = [(b, gen_data, temp_frames_dir, width, height) for b in batches]
#         logger.info(f"Rendering {len(frame_tasks)} frames...")
        
#         all_files = []
#         with multiprocessing.Pool(cpu_count) as pool:
#             results = list(tqdm(pool.imap(_generate_frame_batch_worker_v5, worker_args), 
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


#Gemini animations 
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

# # 1. Layout Controls
# TEXT_ALIGN = "left"
# MAX_LINES_PER_SLIDE = 4  # Changed to 4 as per requirements
# MARGIN_PERCENT_X = 0.10    # Left/Right Margin (10% is safe)
# MARGIN_PERCENT_Y = 0.10    # Top/Bottom Margin

# # 2. Font Sizing
# # DECREASE this to make font BIGGER. INCREASE to make smaller.
# # 10 is standard "Huge". 8 is "Massive". 12 is "Large".
# FONT_HEIGHT_DIVISOR = 8   
# LINE_HEIGHT_RATIO = 1.4    

# # 3. Data Cleaning Settings
# MIN_WORD_DURATION = 0.2    # Seconds. Any word shorter than this gets inflated.
# MAX_SILENCE_GAP = 0.3      # Seconds. If gap < 0.3s, we bridge it (no flickering).

# # 4. Semantic Grouping
# PAUSE_THRESHOLD = 0.8      # Gap > 0.8s triggers new sentence/slide consideration

# # --- DATA MODELS ---
# class WordTimestamp(BaseModel):
#     word: str
#     start: float
#     end: float
#     display_word: Optional[str] = None
#     id: int = 0 

# # --- DATA SANITIZER (THE FIX) ---
# def sanitize_words(raw_words: List[Dict]) -> List[WordTimestamp]:
#     """
#     Fixes 0-duration words, overlaps, and jerky silences.
#     """
#     clean_words = []
    
#     # 1. Convert to Objects & Basic Cleanup
#     for i, w in enumerate(raw_words):
#         word_str = w['word'].strip()
#         # Skip empty strings
#         if not word_str: 
#             continue
            
#         # Fix Punctuation: Attach to previous word if it's just a symbol
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

#     # 2. Fix Durations & Overlaps
#     for i in range(len(clean_words)):
#         curr = clean_words[i]
        
#         # A. Force Minimum Duration (Fixes the "Flash" bug)
#         if (curr.end - curr.start) < MIN_WORD_DURATION:
#             curr.end = curr.start + MIN_WORD_DURATION
            
#         # B. Fix Overlaps with Next Word
#         if i < len(clean_words) - 1:
#             next_w = clean_words[i+1]
#             # If current ends AFTER next starts, push next start forward
#             if curr.end > next_w.start:
#                 next_w.start = curr.end
#                 # Ensure next word still has min duration after being pushed
#                 if (next_w.end - next_w.start) < MIN_WORD_DURATION:
#                     next_w.end = next_w.start + MIN_WORD_DURATION

#     # 3. Bridge Small Silences (Fixes the "Jerky" bug)
#     # If there is a tiny gap between words, extend the current word to touch the next.
#     for i in range(len(clean_words) - 1):
#         curr = clean_words[i]
#         next_w = clean_words[i+1]
        
#         gap = next_w.start - curr.end
#         if 0 < gap < MAX_SILENCE_GAP:
#             # Extend current word to fill the gap
#             curr.end = next_w.start

#     return clean_words

# def interpolate_color(start_color, end_color, progress):
#     """Blends two colors based on progress (0.0 to 1.0)."""
#     return tuple(
#         int(start_color[i] + (end_color[i] - start_color[i]) * progress)
#         for i in range(4)
#     )

# # --- GENERATOR CLASS ---
# class FrameGeneratorBigFlow:
#     def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
#         self.bg_width = bg_width
#         self.bg_height = bg_height
        
#         self.margin_x = int(self.bg_width * MARGIN_PERCENT_X)
#         self.margin_y = int(self.bg_height * MARGIN_PERCENT_Y)  # Added Y margin
#         self.usable_height = self.bg_height - 2 * self.margin_y
#         self.max_text_width = self.bg_width - (2 * self.margin_x)
        
#         self.font_size = int(self.bg_height / FONT_HEIGHT_DIVISOR)
#         self.line_height = int(self.font_size * LINE_HEIGHT_RATIO)
        
#         self.font = self._load_font(self.font_size)
        
#         # LOAD AND SANITIZE
#         self.all_words = self._load_and_clean_data(timestamps_path)
        
#         self.slides = []       
#         self.slide_layouts = [] 
#         self.slide_timings = [] 
        
#         self._build_semantic_slides()  # Updated method name

#     def _load_font(self, size):
#         try:
#             return ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
#         except:
#             return ImageFont.load_default(size)

#     def _load_and_clean_data(self, path: Path) -> List[WordTimestamp]:
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         raw_words = data.get("words", [])
#         # APPLY THE SANITIZER
#         return sanitize_words(raw_words)

#     def split_into_sentences(self, words: List[WordTimestamp]) -> List[List[WordTimestamp]]:
#         """
#         Splits words into meaningful sentences based on punctuation and pauses.
#         """
#         sentences = []
#         current = []
#         for i in range(len(words)):
#             current.append(words[i])
#             word = words[i].display_word
#             # Check for sentence-ending punctuation
#             if word.endswith(('.', '!', '?')):
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
#         """
#         Wraps a list of words into lines based on max width.
#         """
#         dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
#         space_width = dummy_draw.textlength(" ", font=self.font)
        
#         current_line = []
#         current_line_width = 0
#         lines = []
        
#         for word in words:
#             w_len = dummy_draw.textlength(word.display_word, font=self.font)
            
#             if current_line_width + (space_width if current_line else 0) + w_len > self.max_text_width and current_line:
#                 lines.append(current_line)
#                 current_line = [word]
#                 current_line_width = w_len
#             else:
#                 if current_line:
#                     current_line_width += space_width
#                 current_line.append(word)
#                 current_line_width += w_len
        
#         if current_line:
#             lines.append(current_line)
        
#         return lines

#     def _build_semantic_slides(self):
#         """
#         Builds slides semantically: group by sentences, start new slide for new sentences if >=3 lines occupied,
#         split large sentences into chunks of max lines.
#         """
#         sentences = self.split_into_sentences(self.all_words)
#         current_slide_lines = []
#         MAX = MAX_LINES_PER_SLIDE
        
#         for sent_words in sentences:
#             sent_lines = self._layout_into_lines(sent_words)
#             chunk_start = 0
#             while chunk_start < len(sent_lines):
#                 chunk_end = min(chunk_start + MAX, len(sent_lines))
#                 chunk_lines = sent_lines[chunk_start:chunk_end]
                
#                 # For the first chunk of a new sentence, force new slide if current >=3 lines
#                 if chunk_start == 0 and len(current_slide_lines) >= 3:
#                     if current_slide_lines:
#                         self._commit_slide(current_slide_lines)
#                     current_slide_lines = []
                
#                 # Add the chunk to current slide if it fits
#                 potential_lines = current_slide_lines + chunk_lines
#                 if len(potential_lines) > MAX:
#                     if current_slide_lines:
#                         self._commit_slide(current_slide_lines)
#                     current_slide_lines = chunk_lines
#                 else:
#                     current_slide_lines = potential_lines
                
#                 chunk_start += MAX
        
#         if current_slide_lines:
#             self._commit_slide(current_slide_lines)

#     def _commit_slide(self, lines: List[List[WordTimestamp]]):
#         if not lines: return
        
#         # Optional: Remove trailing comma from last word of slide (as in original)
#         if lines and lines[-1]:
#             last_word_obj = lines[-1][-1]
#             text = last_word_obj.display_word
#             if text.endswith(","):
#                 last_word_obj.display_word = text[:-1]

#         # Compute total height and center vertically within usable area
#         total_h = len(lines) * self.line_height
#         start_y = self.margin_y + (self.usable_height - total_h) // 2
        
#         layout = {}
#         curr_y = start_y
        
#         dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
#         space_width = dummy_draw.textlength(" ", font=self.font)
        
#         for line in lines:
#             curr_x = self.margin_x
#             for word in line:
#                 layout[word.id] = (int(curr_x), int(curr_y))
#                 w_len = dummy_draw.textlength(word.display_word, font=self.font)
#                 curr_x += w_len + space_width
#             curr_y += self.line_height
            
#         self.slides.append(lines)
#         self.slide_layouts.append(layout)
        
#         s_start = lines[0][0].start
#         s_end = lines[-1][-1].end
#         self.slide_timings.append((s_start, s_end))

# # --- WORKER ---
# def _generate_frame_batch_worker_v5(batch_data):
#     frame_tasks, gen_data, output_dir, width, height = batch_data
    
#     # 1. Load Fonts (global font size for now; can extend to per-slide if dynamic needed)
#     try:
#         font_bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, gen_data['font_size'])
#         # Try to load a regular version of the same size for the "inactive" state
#         # If you don't have a regular font file, use the bold one for both (just color changes)
#         try:
#             font_reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, gen_data['font_size'])
#         except Exception as e:
#             print("inside exception", e)
#             font_reg = font_bold 
#     except Exception as e:
#         print("exception", e)
#         # raise e
#         font_bold = ImageFont.load_default(gen_data['font_size'])
#         font_reg = font_bold

#     # 2. Animation Settings (Keep fade-in as client likes)
#     C_ACTIVE = (20, 20, 20, 255)        # Black
#     C_FUTURE = (210, 210, 210, 255)    # Light Gray
#     FADE_DURATION = 0.30               # 0.30s transition window

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
                
#                 start = w_data['start']
                
#                 LIFT_PIXELS = 6  # How many pixels the word moves up
                
#                 # LOGIC
#                 y_offset = 0
                
#                 # Case 1: Active (Current Word)
#                 if ts >= start and ts < w_data['end']:
#                     color = (0, 0, 0, 255) # Pure Black
#                     font = font_bold
#                     y_offset = -LIFT_PIXELS # Move UP
                
#                 # Case 2: Transitioning (Just before start)
#                 elif start - FADE_DURATION <= ts < start:
#                     progress = (ts - (start - FADE_DURATION)) / FADE_DURATION
#                     color = interpolate_color(C_FUTURE, C_ACTIVE, progress)
#                     font = font_bold if progress > 0.5 else font_reg
#                     # Animate the lift smoothly
#                     y_offset = int(-LIFT_PIXELS * progress)

#                 # Case 3: Done (Past Word) - Return to ground, stay black
#                 elif ts >= w_data['end']:
#                      color = (0, 0, 0, 255)
#                      font = font_bold
#                      y_offset = 0
                     
#                 # Case 4: Future
#                 else:
#                     color = (180, 180, 180, 255) # Gray
#                     font = font_reg
#                     y_offset = 0

#                 # DRAW
#                 draw.text((coords[0], coords[1] + y_offset), w_data['display_word'], font=font, fill=color)

#                 # draw.text(coords, w_data['display_word'], font=font, fill=color)
            
#         path = output_dir / f"f_{frame_num:06d}.png"
#         img.save(path, optimize=False, compress_level=0)
#         generated.append(str(path))
        
#     return generated

# # --- MAIN ---
# def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
#     logger.info("--- Starting V5 Render (Semantic Grouping + 4-Line Max + Fade-In) ---")
#     temp_frames_dir = None
#     try:
#         audio_clip = AudioFileClip(str(audio_path))
#         duration = audio_clip.duration
#         fps = settings.VIDEO_FPS
#         width = settings.VIDEO_WIDTH
#         height = settings.VIDEO_HEIGHT
#         bg_clip = ImageClip(settings.DEFAULT_BACKGROUND).with_duration(duration)
        
#         gen = FrameGeneratorBigFlow(timestamps_path, width, height)
        
#         total_frames = int(duration * fps)
#         frame_tasks = []
#         current_slide_idx = 0
        
#         for i in range(total_frames):
#             ts = i / fps
#             if current_slide_idx < len(gen.slide_timings):
#                 slide_start, slide_end = gen.slide_timings[current_slide_idx]
#                 # Lookahead: If we are past end, go next
#                 if ts > slide_end and current_slide_idx < len(gen.slide_timings) - 1:
#                     current_slide_idx += 1
#             frame_tasks.append((i, ts, current_slide_idx, 0))

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

#         temp_frames_dir = Path(tempfile.mkdtemp(prefix="v5_frames_"))
#         cpu_count = os.cpu_count() or 4
#         batch_size = max(50, len(frame_tasks) // cpu_count)
#         batches = [frame_tasks[i:i + batch_size] for i in range(0, len(frame_tasks), batch_size)]
        
#         worker_args = [(b, gen_data, temp_frames_dir, width, height) for b in batches]
#         logger.info(f"Rendering {len(frame_tasks)} frames...")
        
#         all_files = []
#         with multiprocessing.Pool(cpu_count) as pool:
#             results = list(tqdm(pool.imap(_generate_frame_batch_worker_v5, worker_args), 
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


##GROK single word ending handler
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

# FONT_HEIGHT_DIVISOR = 9      # Smaller = bigger text. 8 works great for 4-line fill
# LINE_HEIGHT_RATIO = 1.3

# MIN_WORD_DURATION = 0.2
# MAX_SILENCE_GAP = 0.3

# PAUSE_THRESHOLD = 0.8        # Gap > 0.8s = sentence break

# # --- DATA MODELS ---
# class WordTimestamp(BaseModel):
#     word: str
#     start: float
#     end: float
#     display_word: Optional[str] = None
#     id: int = 0 

# # --- SANITIZER ---
# def sanitize_words(raw_words: List[Dict]) -> List[WordTimestamp]:
#     # ... (unchanged, keep your existing sanitizer)
#     # (I'll keep it exactly as you had for safety)
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

#     def split_into_sentences(self, words: List[WordTimestamp]) -> List[List[WordTimestamp]]:
#         sentences = []
#         current = []
#         for i in range(len(words)):
#             current.append(words[i])
#             word = words[i].display_word
#             if word.endswith(('.', '!', '?')):
#                 if i == len(words) - 1 or words[i + 1].display_word[0].isupper():
#                     sentences.append(current)
#                     current = []
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
#     logger.info("--- Starting V6+ Render (Anti-Isolation + Persistent Text + Fixed Unpack) ---")
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

#         temp_frames_dir = Path(tempfile.mkdtemp(prefix="v6_frames_"))
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

TEXT_ALIGN = "left"
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
        
        self.margin_x = int(self.bg_width * MARGIN_PERCENT_X)
        self.margin_y = int(self.bg_height * MARGIN_PERCENT_Y)
        self.usable_height = self.bg_height - 2 * self.margin_y
        self.max_text_width = self.bg_width - (2 * self.margin_x)
        
        self.font_size = int(self.bg_height / FONT_HEIGHT_DIVISOR)
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
        MAX = MAX_LINES_PER_SLIDE
        
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
            x = self.margin_x
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
        width = settings.VIDEO_WIDTH
        height = settings.VIDEO_HEIGHT
        
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





## GEmini with multiple animations
# import json
# import logging
# import os
# import tempfile
# import multiprocessing
# import shutil
# import sys
# from pathlib import Path
# from typing import List, Optional, Dict, Tuple, Any
# import math
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

# # --- DATA MODELS ---
# class WordTimestamp(BaseModel):
#     word: str
#     start: float
#     end: float
#     display_word: Optional[str] = None
#     id: int = 0 

# # --- SANITIZER ---
# def sanitize_words(raw_words: List[Dict]) -> List[WordTimestamp]:
#     # ... (unchanged, keep your existing sanitizer)
#     # (I'll keep it exactly as you had for safety)
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

# def ease_out_expo(x):
#     """Starts fast, slows down gently."""
#     if x == 1: return 1
#     return 1 - math.pow(2, -10 * x)

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

#     def split_into_sentences(self, words: List[WordTimestamp]) -> List[List[WordTimestamp]]:
#         sentences = []
#         current = []
#         for i in range(len(words)):
#             current.append(words[i])
#             word = words[i].display_word
#             if word.endswith(('.', '!', '?')):
#                 if i == len(words) - 1 or words[i + 1].display_word[0].isupper():
#                     sentences.append(current)
#                     current = []
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
# # def _generate_frame_batch_worker_v5(batch_data):
# #     frame_tasks, gen_data, output_dir, width, height = batch_data
    
# #     # 1. Load Fonts
# #     try:
# #         font_bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, gen_data['font_size'])
# #         try:
# #             font_reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, gen_data['font_size'])
# #         except:
# #             font_reg = font_bold 
# #     except:
# #         font_bold = ImageFont.load_default(gen_data['font_size'])
# #         font_reg = font_bold

# #     generated = []
# #     # FIXED: Added the underscore "_" to handle the 4th item in the tuple
# #     for frame_num, ts, slide_idx in frame_tasks:
# #         img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
# #         draw = ImageDraw.Draw(img)
        
# #         # Guard against out-of-bounds index
# #         if slide_idx >= len(gen_data['slides']):
# #              pass
# #         else:
# #             slide_lines = gen_data['slides'][slide_idx]
# #             layout = gen_data['layouts'][slide_idx]
            
# #             for line in slide_lines:
# #                 for w_data in line:
# #                     coords = layout.get(str(w_data['id']))
# #                     if not coords: continue
                    
# #                     start = w_data['start']
                    
# #                     # SETTINGS
# #                     HIGHLIGHT_COLOR = (255, 215, 0, 255) # Gold/Yellow
# #                     TEXT_ON_HIGHLIGHT = (0, 0, 0, 255)   # Black text on yellow
# #                     NORMAL_TEXT = (30, 30, 30, 255)      # Dark Grey normally
# #                     PADDING = 6                          # Padding around the box (Increased slightly)

# #                     # LOGIC
# #                     # Case 1: Currently Speaking (Active)
# #                     if ts >= start and ts <= w_data['end']:
# #                         # A. Draw the Box
# #                         # Calculate size of the word
# #                         bbox = draw.textbbox(coords, w_data['display_word'], font=font_bold)
# #                         # Expand box by padding
# #                         rect_coords = (bbox[0]-PADDING, bbox[1]-PADDING, bbox[2]+PADDING, bbox[3]+PADDING)
# #                         draw.rectangle(rect_coords, fill=HIGHLIGHT_COLOR)
                        
# #                         # B. Set Text settings to draw ON TOP of box
# #                         font = font_bold
# #                         color = TEXT_ON_HIGHLIGHT
                    
# #                     # Case 2: Past words (Stay dark, no highlight)
# #                     elif ts > w_data['end']:
# #                         font = font_bold
# #                         color = NORMAL_TEXT
                    
# #                     # Case 3: Future words (Grayed out)
# #                     else:
# #                         font = font_bold 
# #                         color = (200, 200, 200, 255) 

# #                     # DRAW
# #                     draw.text(coords, w_data['display_word'], font=font, fill=color)
        
# #         path = output_dir / f"f_{frame_num:06d}.png"
# #         img.save(path, optimize=False, compress_level=0)
# #         generated.append(str(path))
        
# #     return generated

# #Show as you go
# # def _generate_frame_batch_worker_v5(batch_data):
# #     frame_tasks, gen_data, output_dir, width, height = batch_data
    
# #     # 1. Load Fonts
# #     try:
# #         font_bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, gen_data['font_size'])
# #         try:
# #             font_reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, gen_data['font_size'])
# #         except:
# #             font_reg = font_bold 
# #     except:
# #         font_bold = ImageFont.load_default(gen_data['font_size'])
# #         font_reg = font_bold

# #     # SETTINGS
# #     C_VISIBLE = (0, 0, 0, 255)      # Fully visible black
# #     C_INVISIBLE = (0, 0, 0, 0)      # Fully transparent
# #     FADE_DURATION = 0.30            # Speed of fade-in

# #     generated = []
# #     # FIXED: Added "_" for tuple unpacking
# #     for frame_num, ts, slide_idx in frame_tasks:
# #         img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
# #         draw = ImageDraw.Draw(img)
        
# #         if slide_idx >= len(gen_data['slides']):
# #              pass
# #         else:
# #             slide_lines = gen_data['slides'][slide_idx]
# #             layout = gen_data['layouts'][slide_idx]
            
# #             for line in slide_lines:
# #                 for w_data in line:
# #                     coords = layout.get(str(w_data['id']))
# #                     if not coords: continue
                    
# #                     start = w_data['start']
                    
# #                     # LOGIC
# #                     # Case 1: Fully Visible (Currently speaking or Past)
# #                     if ts >= start:
# #                         color = C_VISIBLE
# #                         font = font_bold
                    
# #                     # Case 2: Fading In (Just before speaking)
# #                     elif start - FADE_DURATION <= ts < start:
# #                         progress = (ts - (start - FADE_DURATION)) / FADE_DURATION
# #                         # Calculate Alpha (0 to 255)
# #                         alpha = int(255 * progress)
# #                         color = (0, 0, 0, alpha)
# #                         font = font_bold
                    
# #                     # Case 3: Invisible (Future)
# #                     else:
# #                         color = C_INVISIBLE
# #                         font = font_bold 

# #                     # DRAW
# #                     draw.text(coords, w_data['display_word'], font=font, fill=color)
        
# #         path = output_dir / f"f_{frame_num:06d}.png"
# #         img.save(path, optimize=False, compress_level=0)
# #         generated.append(str(path))
        
# #     return generated

# # --- 3d rendering---

# # --- WORKER (SMOOTH LIFT & LAND) ---
# def _generate_frame_batch_worker_v5(batch_data):
#     frame_tasks, gen_data, output_dir, width, height = batch_data
    
#     # 1. Load Fonts
#     try:
#         font_bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, gen_data['font_size'])
#         try:
#             font_reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, gen_data['font_size'])
#         except:
#             font_reg = font_bold 
#     except:
#         font_bold = ImageFont.load_default(gen_data['font_size'])
#         font_reg = font_bold

#     # 2. Animation Settings
#     C_ACTIVE = (0, 0, 0, 255)          # Pure Black
#     C_SHADOW = (220, 220, 220, 255)    # Light Gray Shadow
#     C_FUTURE = (180, 180, 180, 255)    # Dark Gray
    
#     # ANIMATION PARAMETERS
#     FADE_IN_DURATION = 0.35   # How long to fly UP
#     RETURN_DURATION = 0.25    # How long to fly DOWN (The Fix!)
#     LIFT_PIXELS = 14          # Height of flight

#     generated = []
    
#     for frame_num, ts, slide_idx in frame_tasks:
#         img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
#         draw = ImageDraw.Draw(img)
        
#         if slide_idx >= len(gen_data['slides']):
#             pass
#         else:
#             slide_lines = gen_data['slides'][slide_idx]
#             layout = gen_data['layouts'][slide_idx]
            
#             for line in slide_lines:
#                 for w_data in line:
#                     coords = layout.get(str(w_data['id']))
#                     if not coords: continue
                    
#                     start = w_data['start']
#                     end = w_data['end']
                    
#                     x, y = coords

#                     # --- FULL LIFECYCLE ANIMATION ---
                    
#                     # Phase 1: TAKEOFF (Transition to Active)
#                     if start - FADE_IN_DURATION <= ts < start:
#                         # Linear Progress 0 -> 1
#                         p = (ts - (start - FADE_IN_DURATION)) / FADE_IN_DURATION
#                         eased = ease_out_expo(p)
                        
#                         # Color: Future -> Active
#                         color = interpolate_color(C_FUTURE, C_ACTIVE, eased)
#                         font = font_bold if eased > 0.5 else font_reg
                        
#                         # Lift: 0 -> Max
#                         lift = int(LIFT_PIXELS * eased)
#                         draw.text((x, y - lift), w_data['display_word'], font=font, fill=color)

#                     # Phase 2: HOVER (Active Speaking)
#                     elif ts >= start and ts <= end:
#                         # Shadow on ground
#                         draw.text((x, y), w_data['display_word'], font=font_bold, fill=C_SHADOW)
#                         # Text fully lifted
#                         draw.text((x, y - LIFT_PIXELS), w_data['display_word'], font=font_bold, fill=C_ACTIVE)

#                     # Phase 3: LANDING (Transition to Done) - THE GLITCH FIX
#                     elif end < ts <= end + RETURN_DURATION:
#                         # Linear Progress 0 -> 1
#                         p = (ts - end) / RETURN_DURATION
#                         eased = ease_out_expo(p)
                        
#                         # Lift: Max -> 0
#                         # We invert the eased progress (1 -> 0)
#                         lift = int(LIFT_PIXELS * (1 - eased))
                        
#                         draw.text((x, y - lift), w_data['display_word'], font=font_bold, fill=C_ACTIVE)
                        
#                     # Phase 4: SETTLED (Past)
#                     elif ts > end + RETURN_DURATION:
#                          draw.text((x, y), w_data['display_word'], font=font_bold, fill=C_ACTIVE)

#                     # Phase 5: WAITING (Future)
#                     else:
#                         draw.text((x, y), w_data['display_word'], font=font_reg, fill=C_FUTURE)

#         path = output_dir / f"f_{frame_num:06d}.png"
#         img.save(path, optimize=False, compress_level=0)
#         generated.append(str(path))
        
#     return generated

# # --- MAIN RENDER FUNCTION (PERSISTENT SLIDE + CUMULATIVE INDEX) ---
# def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
#     logger.info("--- Starting V6+ Render (Anti-Isolation + Persistent Text + Fixed Unpack) ---")
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

#         temp_frames_dir = Path(tempfile.mkdtemp(prefix="v6_frames_"))
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