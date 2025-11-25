# File Purpose Documentation

This document explains the purpose of each file in the PDF-to-Video Generation project.

## Project Overview

This project converts PDF books into narrated videos with synchronized text overlays. It consists of a FastAPI backend and a React frontend, processing PDFs through multiple phases: extraction, AI services, audio processing, and video generation.

---

## Root Directory Files

### `run_backend.py`
**Purpose**: Entry point for starting the FastAPI backend server. Configures uvicorn to run the application with hot-reload enabled for development.

### `requirements.txt`
**Purpose**: Python package dependencies file. Lists all required Python packages and their versions for the backend application.

### `README.md`
**Purpose**: Main project documentation. Contains setup instructions, project structure overview, and basic usage information.

### `BACKEND_SETUP.md`
**Purpose**: Detailed setup instructions for the backend environment, including Python environment setup, dependency installation, and configuration.

### `FRONTEND_SETUP.md`
**Purpose**: Detailed setup instructions for the frontend React application, including Node.js setup and npm installation.

### `BATCH_PROCESSING_APPROACH.md`
**Purpose**: Documentation explaining the batch processing approach used in the video generation pipeline.

---

## Backend Application (`app/`)

### Configuration Files

#### `app/__init__.py`
**Purpose**: Python package initialization file for the `app` module.

#### `app/config.py`
**Purpose**: Central configuration management using Pydantic settings. Loads environment variables from `.env` file and defines:
- Project paths (assets, fonts, backgrounds, jobs output)
- API keys (OpenAI, Cartesia, Serper)
- Video settings (FPS, dimensions, codec)
- Text rendering settings (colors, fonts)
- Summary generation settings (model, target words, temperature)

#### `app/logging_config.py`
**Purpose**: Configures Python logging system. Sets up console and file handlers, with optional job-specific log files stored in job directories.

---

## API Layer (`app/api/`)

### `app/api/__init__.py`
**Purpose**: Python package initialization for the API module.

### `app/api/main.py`
**Purpose**: FastAPI application entry point. Defines all REST API endpoints:
- `POST /upload-pdf`: Upload PDF and start video generation
- `POST /generate-video-from-text`: Generate video from text input
- `POST /generate-reels-video`: Generate reels/shorts video (9:16 aspect ratio)
- `POST /generate-summary-video`: Generate summary video from text
- `GET /job-status/{job_id}`: Get job status and metadata
- `GET /download-video/{job_id}`: Download generated video
- `POST /generate-summary`: Generate PDF summary
- `GET /cartesia-voices`: Get available Cartesia voices
- `GET /cartesia-models`: Get available Cartesia models

Also configures CORS middleware for frontend communication.

### `app/api/pipeline_service.py`
**Purpose**: Core orchestration service for the video generation pipeline. Coordinates all phases:
- `run_pipeline()`: Main pipeline for PDF-to-video conversion
- `run_pipeline_from_text()`: Pipeline for text-to-video conversion
- `run_pipeline_for_reels()`: Pipeline for reels/shorts video generation
- `generate_summary_video()`: Pipeline for summary video generation

Handles voice service initialization (OpenAI or Cartesia), audio generation, mastering, and video rendering.

### `app/api/job_service.py`
**Purpose**: Job management service. Tracks job status, metadata, and persistence:
- Creates and updates job metadata
- Loads existing jobs from disk
- Provides thread-safe job status tracking
- Manages job lifecycle (pending → processing → completed/failed)

### `app/api/cartesia_service.py`
**Purpose**: Service for interacting with Cartesia API. Handles:
- Voice listing
- Model listing
- API key validation
- Cartesia-specific configuration

### `app/api/README.md`
**Purpose**: API-specific documentation and notes.

---

## Phase 1: PDF Processing (`app/phase1_pdf_processing/`)

### `app/phase1_pdf_processing/__init__.py`
**Purpose**: Package initialization for PDF processing module.

### `app/phase1_pdf_processing/processor.py`
**Purpose**: Core PDF processing logic. Handles:
- Structured text extraction from PDFs
- First page detection
- Table of contents/index extraction
- Table extraction
- Page range filtering
- Adaptive extraction strategies based on book type

### `app/phase1_pdf_processing/service.py`
**Purpose**: High-level PDF extraction service. Provides a clean interface for:
- Extracting structured content from PDFs
- Organizing output files
- Generating extraction JSON with metadata
- Coordinating processor, image extractor, and text cleaner

### `app/phase1_pdf_processing/image_extractor.py`
**Purpose**: Extracts images from PDF files. Handles image extraction, saving, and metadata tracking for embedded images in PDFs.

### `app/phase1_pdf_processing/text_cleaner.py`
**Purpose**: Text cleaning utilities. Removes unwanted characters, normalizes whitespace, and prepares text for further processing.

### `app/phase1_pdf_processing/utils/__init__.py`
**Purpose**: Package initialization for PDF processing utilities.

### `app/phase1_pdf_processing/utils/pdf_extraction_strategies.py`
**Purpose**: Advanced extraction strategies for different book types. Implements:
- Book structure analysis
- Adaptive index extraction
- Adaptive table extraction
- Book type detection (novel, textbook, etc.)
- Extraction configuration management

---

## Phase 2: AI Services (`app/phase2_ai_services/`)

### `app/phase2_ai_services/__init__.py`
**Purpose**: Package initialization for AI services module.

### `app/phase2_ai_services/openai_client.py`
**Purpose**: OpenAI API client wrapper. Handles:
- Text-to-Speech (TTS) using OpenAI's TTS API
- Speech-to-Text (STT) using Whisper API
- Voice selection (alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer)
- Book genre detection using Serper API
- Audio file generation and saving

### `app/phase2_ai_services/cartesia_client.py`
**Purpose**: Cartesia API client wrapper. Handles:
- Text-to-Speech using Cartesia API
- Voice and model selection
- Audio generation with Cartesia's voice models

### `app/phase2_ai_services/pdf_summarizer.py`
**Purpose**: PDF summarization service. Generates comprehensive summaries of entire PDFs:
- Chunks large PDFs when necessary
- Uses GPT-4o-mini for summarization
- Targets specific word counts (~9000 words for ~1 hour narration)
- Handles token limits and context windows

### `app/phase2_ai_services/book_summary.py`
**Purpose**: Book summary generation utilities. May contain additional book-specific summarization logic.

---

## Phase 3: Audio Processing (`app/phase3_audio_processing/`)

### `app/phase3_audio_processing/__init__.py`
**Purpose**: Package initialization for audio processing module.

### `app/phase3_audio_processing/mastering.py`
**Purpose**: Audio mastering pipeline. Applies professional audio processing:
- Converts audio to high-quality WAV format
- Reduces static noise
- Denoises audio
- Applies mastering effects (EQ, compression, normalization)
- Maintains audio length (no time stretching)
- Uses FFmpeg for processing

---

## Phase 4: Video Generation (`app/phase4_video_generation/`)

### `app/phase4_video_generation/__init__.py`
**Purpose**: Package initialization for video generation module.

### `app/phase4_video_generation/renderer.py`
**Purpose**: Core video rendering engine. Handles:
- Frame generation at specific timestamps
- Text rendering with word-level synchronization
- Slide building and grouping
- Background image handling (including programmatic generation for reels)
- Text layout and alignment
- Word bolding/highlighting
- Slide transition logic
- Video encoding with FFmpeg
- Support for different video dimensions (main video: 1920x1080, reels: 1080x1920)
- Font size and line limit configuration
- Text alignment (center for main, left for reels)

---

## Orchestration (`app/orchestration/`)

### `app/orchestration/__init__.py`
**Purpose**: Package initialization for orchestration module.

### `app/orchestration/chapter_processor.py`
**Purpose**: Chapter-level video processing. Handles:
- Chapter extraction from PDF headings
- Chapter selection and filtering
- Chapter summary generation
- Individual chapter video creation
- Integration with the full pipeline for chapter-specific processing

---

## Frontend (`frontend/`)

### Root Files

#### `frontend/package.json`
**Purpose**: Node.js package configuration. Defines dependencies, scripts, and project metadata for the React application.

#### `frontend/package-lock.json`
**Purpose**: Locked dependency versions for reproducible npm installations.

#### `frontend/README.md`
**Purpose**: Frontend-specific documentation and setup instructions.

### Source Files (`frontend/src/`)

#### `frontend/src/index.js`
**Purpose**: React application entry point. Renders the root `App` component into the DOM.

#### `frontend/src/index.css`
**Purpose**: Global CSS styles for the entire application.

#### `frontend/src/App.js`
**Purpose**: Main React application component. Manages:
- Application state and routing
- View switching (home, upload, job status, summary, reels)
- Component coordination
- Navigation between different sections

#### `frontend/src/App.css`
**Purpose**: CSS styles specific to the main App component.

### Components (`frontend/src/components/`)

#### `frontend/src/components/HomePage.js` & `.css`
**Purpose**: Home page component. Displays the main landing page with options to:
- Generate video from PDF
- Generate summary video
- Generate reels/shorts

#### `frontend/src/components/PDFUpload.js` & `.css`
**Purpose**: PDF upload component. Provides UI for:
- File upload
- Page range selection
- Voice provider selection (OpenAI/Cartesia)
- OpenAI voice selection dropdown
- Cartesia voice/model configuration
- Summary generation option
- Initiating video generation

#### `frontend/src/components/JobStatus.js` & `.css`
**Purpose**: Job status monitoring component. Displays:
- Current job status (pending, processing, completed, failed)
- Progress information
- Download link for completed videos
- Real-time status updates via polling

#### `frontend/src/components/SummaryGeneration.js` & `.css`
**Purpose**: Summary generation component. Handles:
- PDF upload for summary generation
- Summary generation initiation
- Display of generated summary

#### `frontend/src/components/SummaryPrompt.js` & `.css`
**Purpose**: Summary prompt component. May display prompts or instructions related to summary generation.

#### `frontend/src/components/SummaryReview.js` & `.css`
**Purpose**: Summary review component. Allows users to:
- Review generated summaries
- Edit summary text
- Generate video from summary text
- Select voice provider and OpenAI voice
- Configure video generation settings

#### `frontend/src/components/SummaryVideoPrompt.js` & `.css`
**Purpose**: Summary video prompt component. May handle prompts or configuration for summary video generation.

#### `frontend/src/components/ReelsShorts.js` & `.css`
**Purpose**: Reels/shorts generation component. Provides UI for:
- Text input for reels content
- Voice provider selection
- OpenAI voice selection
- Cartesia configuration
- Initiating reels video generation (9:16 aspect ratio)

#### `frontend/src/components/CartesiaConfig.js` & `.css`
**Purpose**: Cartesia configuration component. Handles:
- Cartesia voice selection
- Cartesia model selection
- API configuration for Cartesia TTS

### Services (`frontend/src/services/`)

#### `frontend/src/services/api.js`
**Purpose**: Frontend API service layer. Provides functions for:
- `uploadPDF()`: Upload PDF and start video generation
- `generateVideoFromText()`: Generate video from text
- `generateReelsVideo()`: Generate reels video
- `generateSummary()`: Generate PDF summary
- `getJobStatus()`: Get job status
- `downloadVideo()`: Download generated video
- `getCartesiaVoices()`: Get Cartesia voices
- `getCartesiaModels()`: Get Cartesia models

Uses axios for HTTP requests with interceptors for debugging.

### Public Files (`frontend/public/`)

#### `frontend/public/index.html`
**Purpose**: HTML template for the React application. Contains the root div where the React app is mounted.

---

## Assets (`assets/`)

### `assets/backgrounds/`
**Purpose**: Directory containing background images for video generation:
- White solid color backgrounds in various resolutions
- Custom background images
- Default backgrounds for different video dimensions

### `assets/fonts/`
**Purpose**: Directory containing font files used for text rendering:
- Book Antiqua (regular and bold)
- Georgia
- Playfair Display (regular, medium, bold)

### `assets/pdfs/`
**Purpose**: Directory containing sample PDF files for testing.

---

## Jobs Output (`jobs/`)

**Purpose**: Directory where all generated content is stored. Each job creates a subdirectory with:
- `job_metadata.json`: Job status and metadata
- `*_extraction.json`: PDF extraction results
- `*_full_text.txt`: Extracted text content
- `*_timestamps.json`: Word-level timestamps for synchronization
- `*_raw_audio.mp3`: Raw TTS audio
- `*_processed_audio.mp3`: Mastered audio
- `*_final_video.mp4`: Final generated video
- `*_run.log`: Job execution log
- `images/`: Extracted images from PDF
- `tables/`: Extracted tables from PDF

---

## Scripts (`scripts/`)

### `scripts/run_full_pipeline.py`
**Purpose**: Standalone script for running the complete pipeline. May be used for testing or batch processing outside the API.

### `scripts/test_only_pdf.py`
**Purpose**: Test script for PDF processing functionality only.

### `scripts/test_video_only.py`
**Purpose**: Test script for video generation functionality only.

### `scripts/test_script.txt`
**Purpose**: Test text file for testing text-to-video generation.

---

## Environment Configuration

### `.env` (not in repository)
**Purpose**: Environment variables file containing:
- `OPENAI_API_KEY`: OpenAI API key for TTS/STT
- `CARTESIA_API_KEY`: Cartesia API key (optional)
- `SERPER_API_KEY`: Serper API key for genre detection (optional)
- `VIDEO_FPS`: Video frames per second
- `VIDEO_WIDTH`: Video width in pixels
- `VIDEO_HEIGHT`: Video height in pixels
- `VIDEO_CODEC`: Video codec (e.g., libx264)
- `TEXT_REGULAR_COLOR`: RGBA color for regular text
- `TEXT_BOLD_COLOR`: RGBA color for bold text

---

## Summary

This project follows a modular architecture with clear separation of concerns:
- **Phase 1**: PDF extraction and processing
- **Phase 2**: AI services (TTS, summarization)
- **Phase 3**: Audio mastering
- **Phase 4**: Video rendering
- **API Layer**: RESTful API for frontend communication
- **Frontend**: React-based user interface
- **Orchestration**: Chapter-level processing coordination

Each module is self-contained and can be developed, tested, and maintained independently while working together to create the complete PDF-to-video pipeline.

