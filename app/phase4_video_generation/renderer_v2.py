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
        
#         # Load data grouped by SENTENCES (Segments)
#         self.segments_of_words = self._load_data_grouped(timestamps_path)

#         # --- Founders Style Config ---
#         # 1080p logic:
#         self.font_size = int(self.bg_height / 15) # Good size for paragraph reading
#         self.line_height = int(self.font_size * 1.4) 
        
#         # STRICT LEFT ALIGNMENT (Fixed Margin)
#         self.margin_left = int(self.bg_width * 0.10) # 10% margin
#         self.margin_right = int(self.bg_width * 0.10)
#         self.max_text_width = self.bg_width - self.margin_left - self.margin_right
        
#         # Area safe for text (vertically)
#         self.max_lines_per_slide = 8 # Force up to 8 lines per screen to "fill" it
        
#         self.regular_font, self.bold_font = self._load_fonts(self.font_size)
        
#         # Build "Smart" Slides
#         self.slides, self.slide_layouts, self.slide_timings = self._build_paragraph_slides()

#     def _load_fonts(self, size):
#         try:
#             reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, size)
#             bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
#             return reg, bold
#         except:
#             return ImageFont.load_default(size), ImageFont.load_default(size)

#     def _load_data_grouped(self, path: Path) -> List[List[WordTimestamp]]:
#         """
#         Loads data but keeps it grouped by WHISPER SEGMENTS (Sentences).
#         This is key to preventing the "This" word appearing at the end of a slide.
#         """
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
        
#         grouped_segments = []
        
#         # Use the 'segments' list from Whisper to group words
#         raw_segments = data.get("segments", [])
#         all_words_flat = data.get("words", [])
        
#         # Logic: Map flat words to segments based on time
#         # This ensures we have word-level precision but sentence-level grouping
        
#         current_word_idx = 0
        
#         for seg in raw_segments:
#             seg_start = seg['start']
#             seg_end = seg['end']
#             segment_words = []
            
#             # If segments have their own 'words' key (some Whisper versions)
#             if 'words' in seg:
#                 for w in seg['words']:
#                     clean_w = w['word'].strip()
#                     obj = WordTimestamp(word=clean_w, start=w['start'], end=w['end'], display_word=clean_w)
#                     segment_words.append(obj)
#             else:
#                 # Fallback: grabbing words from the flat list that fit this time window
#                 # giving a small buffer for drift
#                 while current_word_idx < len(all_words_flat):
#                     w = all_words_flat[current_word_idx]
#                     # If word starts before segment ends, it belongs here (mostly)
#                     if w['start'] < seg_end + 0.1: 
#                         clean_w = w['word'].strip()
#                         obj = WordTimestamp(word=clean_w, start=w['start'], end=w['end'], display_word=clean_w)
#                         segment_words.append(obj)
#                         current_word_idx += 1
#                     else:
#                         break
            
#             # Add punctuation from the segment text to the last word if missing
#             if segment_words:
#                 seg_text = seg['text'].strip()
#                 last_char = seg_text[-1] if seg_text else ""
#                 if last_char in ['.', '?', '!'] and segment_words[-1].display_word[-1] not in ['.', '?', '!']:
#                     segment_words[-1].display_word += last_char

#             if segment_words:
#                 grouped_segments.append(segment_words)

#         return grouped_segments

#     def _build_paragraph_slides(self):
#         """
#         Packs multiple segments (sentences) onto a slide until full.
#         NEVER splits a sentence between slides.
#         """
#         slides = []
#         layouts = {}
#         timings = []

#         dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
#         space_width = dummy_draw.textlength(" ", font=self.bold_font)

#         current_slide_segments = []
#         current_slide_line_count = 0
        
#         def calculate_lines_for_segment(segment_words):
#             """Simulates how many lines a sentence will take."""
#             lines = 1
#             current_width = 0
#             for word in segment_words:
#                 w_len = dummy_draw.textlength(word.display_word, font=self.bold_font)
#                 if current_width + space_width + w_len > self.max_text_width:
#                     lines += 1
#                     current_width = w_len
#                 else:
#                     current_width += space_width + w_len
#             return lines

#         def commit_slide(segments_list):
#             if not segments_list: return
            
#             # Flatten segments into one list of words for layout
#             all_words_on_slide = [w for seg in segments_list for w in seg]
            
#             lines = []
#             current_line = []
#             current_line_width = 0
            
#             for word in all_words_on_slide:
#                 w_len = dummy_draw.textlength(word.display_word, font=self.bold_font)
#                 is_overflow = (current_line_width + space_width + w_len > self.max_text_width)
                
#                 if current_line and is_overflow:
#                     lines.append(current_line)
#                     current_line = [word]
#                     current_line_width = w_len
#                 else:
#                     if current_line: current_line_width += space_width
#                     current_line.append(word)
#                     current_line_width += w_len
#             if current_line: lines.append(current_line)

#             # --- LAYOUT CALCULATION ---
#             slide_layout = {}
#             total_h = len(lines) * self.line_height
#             start_y = (self.bg_height - total_h) // 2 # Centered Block
            
#             curr_y = start_y
#             for line in lines:
#                 curr_x = self.margin_left
#                 for word in line:
#                     slide_layout[id(word)] = (curr_x, curr_y)
#                     w_len = dummy_draw.textlength(word.display_word, font=self.bold_font)
#                     curr_x += w_len + space_width
#                 curr_y += self.line_height

#             slides.append(lines)
#             layouts[len(slides)-1] = slide_layout
#             # Timing: Start of first word in first segment to end of last word in last segment
#             s_start = segments_list[0][0].start
#             s_end = segments_list[-1][-1].end
#             timings.append((s_start, s_end))

#         # --- MAIN LOOP: SEGMENT PACKING ---
#         for segment in self.segments_of_words:
#             # 1. How big is this sentence?
#             lines_needed = calculate_lines_for_segment(segment)
            
#             # 2. Does it fit on current slide?
#             if current_slide_line_count + lines_needed > self.max_lines_per_slide:
#                 # Slide is full. Commit it.
#                 commit_slide(current_slide_segments)
#                 # Start new slide with this sentence
#                 current_slide_segments = [segment]
#                 current_slide_line_count = lines_needed
#             else:
#                 # It fits! Add to pile.
#                 current_slide_segments.append(segment)
#                 current_slide_line_count += lines_needed
                
#                 # If we force a new line between sentences (optional, looks cleaner)
#                 # current_slide_line_count += 1 

#         if current_slide_segments:
#             commit_slide(current_slide_segments)

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
#     logger.info("--- Starting V2 Render (Founders Style - Paragraph Mode) ---")
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
#                 if start <= ts <= end + 0.5: 
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
        
#         self.segments_of_words = self._load_data_grouped(timestamps_path)

#         # --- FIXED: Larger Font & Less Lines ---
#         # Changed divisor from 15 -> 11 for larger text
#         self.font_size = int(self.bg_height / 8) 
#         self.line_height = int(self.font_size * 1.25) 
        
#         # Margins
#         self.margin_left = int(self.bg_width * 0.10) 
#         self.margin_right = int(self.bg_width * 0.10)
        
#         # --- FIXED: Top Margin logic ---
#         # We set margin_top exactly equal to margin_left per client request
#         self.margin_top = self.margin_left 
        
#         self.max_text_width = self.bg_width - self.margin_left - self.margin_right
        
#         # --- FIXED: Max 5 Lines ---
#         self.max_lines_per_slide = 5 
        
#         self.regular_font, self.bold_font = self._load_fonts(self.font_size)
        
#         self.slides, self.slide_layouts, self.slide_timings = self._build_paragraph_slides()
        
#     def _load_fonts(self, size):
#         try:
#             reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, size)
#             bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
#             return reg, bold
#         except:
#             return ImageFont.load_default(size), ImageFont.load_default(size)

#     def _load_data_grouped(self, path: Path) -> List[List[WordTimestamp]]:
#         # (This function remains exactly the same as your provided code)
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         raw_segments = data.get("segments", [])
#         all_words_flat = data.get("words", [])
#         current_word_idx = 0
#         grouped_segments = []
        
#         for seg in raw_segments:
#             seg_start = seg['start']
#             seg_end = seg['end']
#             segment_words = []
#             if 'words' in seg:
#                 for w in seg['words']:
#                     clean_w = w['word'].strip()
#                     obj = WordTimestamp(word=clean_w, start=w['start'], end=w['end'], display_word=clean_w)
#                     segment_words.append(obj)
#             else:
#                 while current_word_idx < len(all_words_flat):
#                     w = all_words_flat[current_word_idx]
#                     if w['start'] < seg_end + 0.1: 
#                         clean_w = w['word'].strip()
#                         obj = WordTimestamp(word=clean_w, start=w['start'], end=w['end'], display_word=clean_w)
#                         segment_words.append(obj)
#                         current_word_idx += 1
#                     else:
#                         break
#             if segment_words:
#                 seg_text = seg['text'].strip()
#                 last_char = seg_text[-1] if seg_text else ""
#                 if last_char in ['.', '?', '!'] and segment_words[-1].display_word[-1] not in ['.', '?', '!']:
#                     segment_words[-1].display_word += last_char
#             if segment_words:
#                 grouped_segments.append(segment_words)
#         return grouped_segments
    

#     def _build_paragraph_slides(self):
#         slides = []
#         layouts = {}
#         timings = []

#         dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
#         space_width = dummy_draw.textlength(" ", font=self.bold_font)

#         current_slide_segments = []
#         current_slide_line_count = 0
        
#         def calculate_lines_for_segment(segment_words):
#             lines = 1
#             current_width = 0
#             for word in segment_words:
#                 w_len = dummy_draw.textlength(word.display_word, font=self.bold_font)
#                 if current_width + space_width + w_len > self.max_text_width:
#                     lines += 1
#                     current_width = w_len
#                 else:
#                     current_width += space_width + w_len
#             return lines

#         def commit_slide(segments_list):
#             if not segments_list: return
            
#             all_words_on_slide = [w for seg in segments_list for w in seg]
#             lines = []
#             current_line = []
#             current_line_width = 0
            
#             for word in all_words_on_slide:
#                 w_len = dummy_draw.textlength(word.display_word, font=self.bold_font)
#                 is_overflow = (current_line_width + space_width + w_len > self.max_text_width)
                
#                 if current_line and is_overflow:
#                     lines.append(current_line)
#                     current_line = [word]
#                     current_line_width = w_len
#                 else:
#                     if current_line: current_line_width += space_width
#                     current_line.append(word)
#                     current_line_width += w_len
#             if current_line: lines.append(current_line)

#             # --- FIXED: TOP ALIGNMENT LOGIC ---
#             slide_layout = {}
            
#             # Previously you calculated total_h and centered it.
#             # Now we just start at self.margin_top
#             curr_y = self.margin_top 
            
#             for line in lines:
#                 curr_x = self.margin_left
#                 for word in line:
#                     slide_layout[id(word)] = (curr_x, curr_y)
#                     w_len = dummy_draw.textlength(word.display_word, font=self.bold_font)
#                     curr_x += w_len + space_width
#                 curr_y += self.line_height

#             slides.append(lines)
#             layouts[len(slides)-1] = slide_layout
#             s_start = segments_list[0][0].start
#             s_end = segments_list[-1][-1].end
#             timings.append((s_start, s_end))

#         for segment in self.segments_of_words:
#             lines_needed = calculate_lines_for_segment(segment)
            
#             if current_slide_line_count + lines_needed > self.max_lines_per_slide:
#                 commit_slide(current_slide_segments)
#                 current_slide_segments = [segment]
#                 current_slide_line_count = lines_needed
#             else:
#                 current_slide_segments.append(segment)
#                 current_slide_line_count += lines_needed
                
#         if current_slide_segments:
#             commit_slide(current_slide_segments)

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

# def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
#     """Main V2 Rendering Entry Point."""
#     logger.info("--- Starting V2 Render (Founders Style - Paragraph Mode) ---")
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
        
#         # 2. Map Frames (FIXED TIMING LOGIC)
#         total_frames = int(duration * fps)
#         frame_tasks = []
        
#         # We can pre-calculate which slide belongs to which time window
#         # Because slides are sequential and gapless in this data structure:
        
#         current_slide_idx = 0
        
#         for i in range(total_frames):
#             ts = i / fps
            
#             # Safety check for index
#             if current_slide_idx >= len(gen.slide_timings):
#                 current_slide_idx = len(gen.slide_timings) - 1

#             slide_start, slide_end = gen.slide_timings[current_slide_idx]
            
#             # --- THE FIX: Exact Cutover ---
#             # If current time > current slide end, move to next.
#             # REMOVED the "+ 0.5" buffer.
#             # Also added a while loop just in case a slide is super short (shorter than 1 frame)
#             while ts > slide_end and current_slide_idx < len(gen.slide_timings) - 1:
#                 current_slide_idx += 1
#                 slide_start, slide_end = gen.slide_timings[current_slide_idx]
            
#             frame_tasks.append((i, ts, current_slide_idx, gen.slide_timings[current_slide_idx][0]))

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




# Dynamci working code for V4 "Big Text" Style Renderer

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

# # --- CONFIGURATION: V4 "BIG TEXT" STYLE ---

# # 1. Layout Controls
# TEXT_ALIGN = "left"        # Strictly Left Aligned as requested
# MAX_LINES_PER_SLIDE = 5    # Hard limit: Never more than 5 lines
# MARGIN_PERCENT_X = 0.12    # Side margins (12% looks cleaner for big text)
# MARGIN_PERCENT_Y = 0.10    # Top/Bottom buffer

# # 2. Font Sizing (The "Huge" Look)
# # We calculate font size based on screen height. 
# # Divisor 10 means font height is 1/10th of screen (Very Big).
# # Decrease this number to make font BIGGER. Increase to make smaller.
# FONT_HEIGHT_DIVISOR = 10   
# LINE_HEIGHT_RATIO = 1.3    # Space between lines

# # --- DATA MODELS ---
# class WordTimestamp(BaseModel):
#     word: str
#     start: float
#     end: float
#     display_word: Optional[str] = None
#     id: int = 0 

# # --- UTILS ---
# def interpolate_color(start_color, end_color, progress):
#     return tuple(
#         int(start_color[i] + (end_color[i] - start_color[i]) * progress)
#         for i in range(4)
#     )

# # --- V4 GENERATOR CLASS ---
# class FrameGeneratorBigFlow:
#     def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
#         self.bg_width = bg_width
#         self.bg_height = bg_height
        
#         # 1. Define Safe Area
#         self.margin_x = int(self.bg_width * MARGIN_PERCENT_X)
#         self.max_text_width = self.bg_width - (2 * self.margin_x)
        
#         # 2. Set Fixed Huge Font
#         self.font_size = int(self.bg_height / FONT_HEIGHT_DIVISOR)
#         self.line_height = int(self.font_size * LINE_HEIGHT_RATIO)
        
#         # Load Fonts immediately
#         self.font = self._load_font(self.font_size)
        
#         # 3. Load All Words (Flat List)
#         self.all_words = self._load_data_flat(timestamps_path)
        
#         # 4. Build Slides (Continuous Flow)
#         # List of List of Lines. Each Line is a List of Words.
#         self.slides = []       
#         self.slide_layouts = [] 
#         self.slide_timings = [] 
        
#         self._build_flow_slides()

#     def _load_font(self, size):
#         try:
#             # Using Bold for everything gives that "impact" look from 11.jpg
#             return ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
#         except:
#             return ImageFont.load_default(size)

#     def _load_data_flat(self, path: Path) -> List[WordTimestamp]:
#         """Loads all words into a single continuous list, ignoring sentence breaks."""
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
        
#         raw_words = data.get("words", [])
#         clean_words = []
        
#         for i, w in enumerate(raw_words):
#             word_str = w['word'].strip()
            
#             # Handle Punctuation:
#             # If this word is just punctuation, attach it to previous word
#             if word_str in ['.', ',', '!', '?', ';', ':'] and clean_words:
#                 clean_words[-1].display_word += word_str
#                 # Extend the previous word's end time slightly? No, keep strict timing.
#                 continue
                
#             # Normal word
#             obj = WordTimestamp(
#                 word=word_str,
#                 start=w['start'],
#                 end=w['end'],
#                 display_word=word_str,
#                 id=i
#             )
#             clean_words.append(obj)
            
#         return clean_words

#     def _build_flow_slides(self):
#         """
#         Pours words into lines, and lines into slides.
#         Stops at MAX_LINES_PER_SLIDE and creates a new slide.
#         """
#         dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
#         space_width = dummy_draw.textlength(" ", font=self.font)
        
#         current_slide_lines = []
#         current_line = []
#         current_line_width = 0
        
#         # Iterate through every single word in order
#         for word in self.all_words:
#             w_len = dummy_draw.textlength(word.display_word, font=self.font)
            
#             # 1. Check if word fits on current line
#             if current_line_width + space_width + w_len > self.max_text_width:
#                 # LINE BREAK
#                 current_slide_lines.append(current_line)
#                 current_line = [word]
#                 current_line_width = w_len
                
#                 # 2. Check if Slide is Full (Max Lines Reached)
#                 if len(current_slide_lines) >= MAX_LINES_PER_SLIDE:
#                     # COMMIT SLIDE
#                     self._commit_slide(current_slide_lines)
#                     # Start new slide with the word that caused the overflow
#                     current_slide_lines = [] 
#             else:
#                 # Add to current line
#                 if current_line: 
#                     current_line_width += space_width
#                 current_line.append(word)
#                 current_line_width += w_len

#         # Flush remaining
#         if current_line:
#             current_slide_lines.append(current_line)
#         if current_slide_lines:
#             self._commit_slide(current_slide_lines)

#     def _commit_slide(self, lines: List[List[WordTimestamp]]):
#         """Calculates layout for a batch of lines and saves them."""
#         if not lines: return
        
#         # 1. Vertical Centering
#         total_h = len(lines) * self.line_height
#         # Center the block in the screen
#         start_y = (self.bg_height - total_h) // 2
        
#         layout = {}
#         curr_y = start_y
        
#         for line in lines:
#             curr_x = self.margin_x # Left Align
            
#             for word in line:
#                 layout[word.id] = (int(curr_x), int(curr_y))
#                 w_len = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(word.display_word, font=self.font)
                
#                 # Advance X
#                 space_w = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(" ", font=self.font)
#                 curr_x += w_len + space_w
            
#             curr_y += self.line_height
            
#         # Save Slide Data
#         self.slides.append(lines)
#         self.slide_layouts.append(layout)
        
#         # Timing: Start of first word -> End of last word
#         # Add a tiny buffer to end so it doesn't cut abruptly before next slide
#         s_start = lines[0][0].start
#         s_end = lines[-1][-1].end
#         self.slide_timings.append((s_start, s_end))

# # --- BATCH WORKER (SAME AS BEFORE, JUST FONT HANDLING SIMPLER) ---
# def _generate_frame_batch_worker_v4(batch_data):
#     frame_tasks, gen_data, output_dir, width, height = batch_data
    
#     # Re-load font inside worker
#     try:
#         font = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, gen_data['font_size'])
#         reg_font = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, gen_data['font_size'])
#     except:
#         font = ImageFont.load_default(gen_data['font_size'])
#         reg_font = font

#     # COLORS (Matches 11.jpg style)
#     # Past/Current words: BLACK
#     # Future words: LIGHT GRAY
#     C_ACTIVE = (0, 0, 0, 255)         
#     C_FUTURE = (200, 200, 200, 255)   
    
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
                
#                 # LOGIC: 11.jpg Style
#                 # If time > start of word -> Black (It's been said or is being said)
#                 # If time < start of word -> Gray (Coming up)
                
#                 if ts >= start:
#                     draw.text(coords, w_data['display_word'], font=font, fill=C_ACTIVE)
#                 else:
#                     # Future words
#                     draw.text(coords, w_data['display_word'], font=reg_font, fill=C_FUTURE)
        
#         path = output_dir / f"f_{frame_num:06d}.png"
#         img.save(path, optimize=False, compress_level=0)
#         generated.append(str(path))
        
#     return generated

# # --- MAIN RENDERER ---
# def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
#     logger.info("--- Starting V4 Render (Big Text Flow) ---")
#     temp_frames_dir = None
#     try:
#         # 1. Setup
#         audio_clip = AudioFileClip(str(audio_path))
#         duration = audio_clip.duration
#         fps = settings.VIDEO_FPS
#         width = settings.VIDEO_WIDTH
#         height = settings.VIDEO_HEIGHT
        
#         bg_clip = ImageClip(settings.DEFAULT_BACKGROUND).with_duration(duration)
        
#         # 2. Generator
#         gen = FrameGeneratorBigFlow(timestamps_path, width, height)
        
#         # 3. Frame Mapping
#         total_frames = int(duration * fps)
#         frame_tasks = []
#         current_slide_idx = 0
        
#         for i in range(total_frames):
#             ts = i / fps
            
#             # Simple sequential logic
#             if current_slide_idx < len(gen.slide_timings):
#                 slide_start, slide_end = gen.slide_timings[current_slide_idx]
#                 # If we passed this slide, move to next
#                 if ts > slide_end and current_slide_idx < len(gen.slide_timings) - 1:
#                     current_slide_idx += 1
            
#             frame_tasks.append((i, ts, current_slide_idx, 0))

#         # 4. Serialize for Workers
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

#         # 5. Multiprocess Render
#         temp_frames_dir = Path(tempfile.mkdtemp(prefix="v4_frames_"))
#         cpu_count = os.cpu_count() or 4
#         batch_size = max(50, len(frame_tasks) // cpu_count)
#         batches = [frame_tasks[i:i + batch_size] for i in range(0, len(frame_tasks), batch_size)]
        
#         worker_args = [(b, gen_data, temp_frames_dir, width, height) for b in batches]
#         logger.info(f"Rendering {len(frame_tasks)} frames...")
        
#         all_files = []
#         with multiprocessing.Pool(cpu_count) as pool:
#             results = list(tqdm(pool.imap(_generate_frame_batch_worker_v4, worker_args), 
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

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, CompositeVideoClip, ImageSequenceClip, ImageClip
from pydantic import BaseModel
from tqdm import tqdm

from app.config import settings

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---

# 1. Layout Controls
TEXT_ALIGN = "left"
MAX_LINES_PER_SLIDE = 5 
MARGIN_PERCENT_X = 0.10    # Left/Right Margin (10% is safe)
MARGIN_PERCENT_Y = 0.10    # Top/Bottom Margin

# 2. Font Sizing
# DECREASE this to make font BIGGER. INCREASE to make smaller.
# 10 is standard "Huge". 8 is "Massive". 12 is "Large".
FONT_HEIGHT_DIVISOR = 8   
LINE_HEIGHT_RATIO = 1.4    

# 3. Data Cleaning Settings
MIN_WORD_DURATION = 0.2    # Seconds. Any word shorter than this gets inflated.
MAX_SILENCE_GAP = 0.3      # Seconds. If gap < 0.3s, we bridge it (no flickering).

# --- DATA MODELS ---
class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    display_word: Optional[str] = None
    id: int = 0 

# --- DATA SANITIZER (THE FIX) ---
def sanitize_words(raw_words: List[Dict]) -> List[WordTimestamp]:
    """
    Fixes 0-duration words, overlaps, and jerky silences.
    """
    clean_words = []
    
    # 1. Convert to Objects & Basic Cleanup
    for i, w in enumerate(raw_words):
        word_str = w['word'].strip()
        # Skip empty strings
        if not word_str: 
            continue
            
        # Fix Punctuation: Attach to previous word if it's just a symbol
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

    # 2. Fix Durations & Overlaps
    for i in range(len(clean_words)):
        curr = clean_words[i]
        
        # A. Force Minimum Duration (Fixes the "Flash" bug)
        if (curr.end - curr.start) < MIN_WORD_DURATION:
            curr.end = curr.start + MIN_WORD_DURATION
            
        # B. Fix Overlaps with Next Word
        if i < len(clean_words) - 1:
            next_w = clean_words[i+1]
            # If current ends AFTER next starts, push next start forward
            if curr.end > next_w.start:
                next_w.start = curr.end
                # Ensure next word still has min duration after being pushed
                if (next_w.end - next_w.start) < MIN_WORD_DURATION:
                    next_w.end = next_w.start + MIN_WORD_DURATION

    # 3. Bridge Small Silences (Fixes the "Jerky" bug)
    # If there is a tiny gap between words, extend the current word to touch the next.
    for i in range(len(clean_words) - 1):
        curr = clean_words[i]
        next_w = clean_words[i+1]
        
        gap = next_w.start - curr.end
        if 0 < gap < MAX_SILENCE_GAP:
            # Extend current word to fill the gap
            curr.end = next_w.start

    return clean_words

def interpolate_color(start_color, end_color, progress):
    """Blends two colors based on progress (0.0 to 1.0)."""
    return tuple(
        int(start_color[i] + (end_color[i] - start_color[i]) * progress)
        for i in range(4)
    )
# --- GENERATOR CLASS ---
class FrameGeneratorBigFlow:
    def __init__(self, timestamps_path: Path, bg_width: int, bg_height: int):
        self.bg_width = bg_width
        self.bg_height = bg_height
        
        self.margin_x = int(self.bg_width * MARGIN_PERCENT_X)
        self.max_text_width = self.bg_width - (2 * self.margin_x)
        
        self.font_size = int(self.bg_height / FONT_HEIGHT_DIVISOR)
        self.line_height = int(self.font_size * LINE_HEIGHT_RATIO)
        
        self.font = self._load_font(self.font_size)
        
        # LOAD AND SANITIZE
        self.all_words = self._load_and_clean_data(timestamps_path)
        
        self.slides = []       
        self.slide_layouts = [] 
        self.slide_timings = [] 
        
        self._build_flow_slides()

    def _load_font(self, size):
        try:
            return ImageFont.truetype(settings.DEFAULT_FONT_BOLD, size)
        except:
            return ImageFont.load_default(size)

    def _load_and_clean_data(self, path: Path) -> List[WordTimestamp]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_words = data.get("words", [])
        # APPLY THE SANITIZER
        return sanitize_words(raw_words)

    def _build_flow_slides(self):
        dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        space_width = dummy_draw.textlength(" ", font=self.font)
        
        current_slide_lines = []
        current_line = []
        current_line_width = 0
        
        for word in self.all_words:
            w_len = dummy_draw.textlength(word.display_word, font=self.font)
            
            if current_line_width + space_width + w_len > self.max_text_width:
                current_slide_lines.append(current_line)
                current_line = [word]
                current_line_width = w_len
                
                if len(current_slide_lines) >= MAX_LINES_PER_SLIDE:
                    self._commit_slide(current_slide_lines)
                    current_slide_lines = [] 
            else:
                if current_line: 
                    current_line_width += space_width
                current_line.append(word)
                current_line_width += w_len

        if current_line:
            current_slide_lines.append(current_line)
        if current_slide_lines:
            self._commit_slide(current_slide_lines)

    def _commit_slide(self, lines: List[List[WordTimestamp]]):
        if not lines: return
        
        if lines and lines[-1]:
            last_word_obj = lines[-1][-1]
            text = last_word_obj.display_word
            
            # 2. Check if it ends with a comma
            if text.endswith(","):
                # Remove the comma
                last_word_obj.display_word = text[:-1]


        total_h = len(lines) * self.line_height
        start_y = (self.bg_height - total_h) // 2
        
        layout = {}
        curr_y = start_y
        
        for line in lines:
            curr_x = self.margin_x
            for word in line:
                layout[word.id] = (int(curr_x), int(curr_y))
                w_len = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(word.display_word, font=self.font)
                space_w = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(" ", font=self.font)
                curr_x += w_len + space_w
            curr_y += self.line_height
            
        self.slides.append(lines)
        self.slide_layouts.append(layout)
        
        s_start = lines[0][0].start
        s_end = lines[-1][-1].end
        self.slide_timings.append((s_start, s_end))

# --- WORKER ---
def _generate_frame_batch_worker_v5(batch_data):
    frame_tasks, gen_data, output_dir, width, height = batch_data
    
    # 1. Load Fonts
    try:
        font_bold = ImageFont.truetype(settings.DEFAULT_FONT_BOLD, gen_data['font_size'])
        # Try to load a regular version of the same size for the "inactive" state
        # If you don't have a regular font file, use the bold one for both (just color changes)
        try:
            font_reg = ImageFont.truetype(settings.DEFAULT_FONT_REGULAR, gen_data['font_size'])
        except:
            font_reg = font_bold 
    except:
        font_bold = ImageFont.load_default(gen_data['font_size'])
        font_reg = font_bold

    # 2. Animation Settings
    C_ACTIVE = (0, 0, 0, 255)          # Black
    C_FUTURE = (200, 200, 200, 255)    # Light Gray
    FADE_DURATION = 0.30               # 0.25s transition window

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
                
                start = w_data['start']
                
                # --- THE ANIMATION LOGIC ---
                
                # Case 1: Fully Active (Time is past start)
                if ts >= start:
                    color = C_ACTIVE
                    font = font_bold
                
                # Case 2: Transitioning (Time is within FADE_DURATION before start)
                # e.g., Start is 2.0s. We are at 1.90s. Fade starts at 1.85s.
                elif start - FADE_DURATION <= ts < start:
                    # Calculate progress (0.0 to 1.0)
                    progress = (ts - (start - FADE_DURATION)) / FADE_DURATION
                    
                    # Interpolate Color
                    color = interpolate_color(C_FUTURE, C_ACTIVE, progress)
                    
                    # Switch Font Weight (Switch to bold when 50% darker)
                    font = font_bold if progress > 0.5 else font_reg
                    
                # Case 3: Fully Future
                else:
                    color = C_FUTURE
                    font = font_reg

                draw.text(coords, w_data['display_word'], font=font, fill=color)
        
        path = output_dir / f"f_{frame_num:06d}.png"
        img.save(path, optimize=False, compress_level=0)
        generated.append(str(path))
        
    return generated
# --- MAIN ---
def render_video_v2(audio_path: Path, timestamps_path: Path, output_path: Path):
    logger.info("--- Starting V5 Render (Sanitized Data + Big Flow) ---")
    temp_frames_dir = None
    try:
        audio_clip = AudioFileClip(str(audio_path))
        duration = audio_clip.duration
        fps = settings.VIDEO_FPS
        width = settings.VIDEO_WIDTH
        height = settings.VIDEO_HEIGHT
        bg_clip = ImageClip(settings.DEFAULT_BACKGROUND).with_duration(duration)
        
        gen = FrameGeneratorBigFlow(timestamps_path, width, height)
        
        total_frames = int(duration * fps)
        frame_tasks = []
        current_slide_idx = 0
        
        for i in range(total_frames):
            ts = i / fps
            if current_slide_idx < len(gen.slide_timings):
                slide_start, slide_end = gen.slide_timings[current_slide_idx]
                # Lookahead: If we are past end, go next
                if ts > slide_end and current_slide_idx < len(gen.slide_timings) - 1:
                    current_slide_idx += 1
            frame_tasks.append((i, ts, current_slide_idx, 0))

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

        temp_frames_dir = Path(tempfile.mkdtemp(prefix="v5_frames_"))
        cpu_count = os.cpu_count() or 4
        batch_size = max(50, len(frame_tasks) // cpu_count)
        batches = [frame_tasks[i:i + batch_size] for i in range(0, len(frame_tasks), batch_size)]
        
        worker_args = [(b, gen_data, temp_frames_dir, width, height) for b in batches]
        logger.info(f"Rendering {len(frame_tasks)} frames...")
        
        all_files = []
        with multiprocessing.Pool(cpu_count) as pool:
            results = list(tqdm(pool.imap(_generate_frame_batch_worker_v5, worker_args), 
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