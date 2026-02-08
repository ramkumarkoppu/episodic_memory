#!/usr/bin/env python3
"""
Gemini Episodic Memory (GEM)
============================
AI-Powered Assistive Memory for People with Episodic Memory Challenges

Author: Ram Koppu
Date: January 2026

"Where did I leave my keys?" → GEM finds them AND tells you how they got there.

GEM is a wearable memory assistant designed to help people with memory challenges
(Alzheimer's, dementia, age-related memory loss) by implementing episodic memory
(Tulving, 1972) - all 5 dimensions:
  - WHAT: Objects + activities detected via Gemini Vision
  - WHERE: Scene location + position in frame + spatial relationships
  - WHEN: Timestamps for temporal tracking and time-based queries
  - WHO: People detected via audio (names) + visual (descriptions)
  - HOW: Cause-and-effect reasoning via movement history

Key Features for Memory Assistance:
  - Voice search: "Where are my glasses?"
  - Activity recall: "What did I do this morning?" / "Did I take my medication?"
  - People memory: "Who did I meet today?" (audio names + visual descriptions)
  - Smart suggestions: When object not found, suggests likely locations
  - Proactive announcements: "Keys placed on kitchen counter"
  - TTS audio feedback: Speaks results for hands-free operation

Optimized for Raspberry Pi Zero 2W (512MB RAM, 32GB microSD):
  - Hash-based O(1) object lookup (no embeddings, no extra API calls)
  - Atomic writes for crash-safe persistence on microSD
  - Local regex query classification (no API round-trip for common queries)
  - Gemini native box_2d format for accurate bounding box detection
  - Fuzzy time queries: "what did I see this morning?", "last 2 hours"
  - Co-occurrence queries: "what was near my keys?"
  - Temporal decay + retrieval reinforcement (Ebbinghaus forgetting curve)

Future: MedGemma Integration
  - Patient-specific memory profiles
  - Caregiver notifications and daily reports
  - Medication reminder integration
  - HIPAA-compliant data handling for healthcare settings

Quick Start (Raspberry Pi Zero 2W):
    # 1. System packages (camera, display, audio tools)
    sudo apt install -y python3-picamera2 python3-numpy python3-pil alsa-utils

    # 2. Whisplay HAT driver (includes WM8960 audio + LCD + LED + button)
    #    https://docs.pisugar.com/docs/product-wiki/whisplay/overview
    cd ~
    git clone https://github.com/PiSugar/Whisplay.git --depth 1
    cd Whisplay/Driver
    sudo bash install_wm8960_drive.sh
    sudo reboot

    # 3. Python venv with system packages
    python3 -m venv ~/gem-venv --system-site-packages
    source ~/gem-venv/bin/activate
    pip install google-genai json5    # Gemini API + lenient JSON parser

    # 4. Set API key and run
    export GEMINI_API_KEY=your_key_here
    python gem.py hw_test        # Test hardware + API
    python gem.py             # Marathon agent (with LCD/LED)
    python gem.py --headless  # Marathon agent (background, no LCD)
    python gem.py search      # Interactive search (voice + LCD)

    # Run capture in background while using voice search:
    python gem.py --headless &   # Background capture
    python gem.py search         # Voice search uses LCD/mic

Hardware:
    - Raspberry Pi Zero 2W (512MB RAM, 32GB microSD)
    - Arducam IMX708 camera - uses libcamera/picamera2 (built into Pi OS)
    - Whisplay HAT: LCD (ST7789 240x280), WM8960 mic/speaker, RGB LED, button

Environment Variables:
    GEMINI_API_KEY          - Required: API key from aistudio.google.com
    GEM_VISION_MODEL        - Vision model (default: gemini-3-flash-preview)
    GEM_AUDIO_MODEL         - Audio/NLU model (default: gemini-3-flash-preview)
    GEM_ROOT                - Root storage directory (default: ~/gem)
                              Contains: memories/, data/
    GEM_CAPTURE_INTERVAL    - Frame capture interval in seconds (default: 1.0)
    GEM_CHANGE_THRESHOLD    - Scene change threshold 0.0-1.0 (default: 0.15)
    GEM_MIN_INTERVAL        - Min seconds between analyses (default: 5)
    GEM_FORCE_INTERVAL      - Force analysis interval in seconds (default: 30)

    # TTS and Announcement Settings
    GEM_TTS_ENABLED         - Enable spoken audio feedback globally (default: false)
                              Uses Gemini 3 native TTS, requires alsa-utils (aplay)
                              Disabled by default due to onboard speaker noise
    GEM_SEARCH_TTS_ENABLED  - Enable TTS for search results only (default: true)
                              Speaks found object locations aloud
                              Works independently of GEM_TTS_ENABLED
    GEM_ANNOUNCE_ENABLED    - Enable proactive announcements (default: false)
                              Speaks when important objects are placed
    GEM_ANNOUNCE_OBJECTS    - Comma-separated objects to announce (default:
                              phone,keys,wallet,glasses,remote,headphones,watch)
    GEM_ANNOUNCE_COOLDOWN   - Seconds between announcements per object (default: 60)

    # Future: MedGemma Integration (healthcare settings)
    GEM_MEDGEMMA_ENABLED    - Enable MedGemma mode (default: false)
    GEM_MEDGEMMA_MODEL      - MedGemma model ID (default: medgemma-1.5)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
#
# GEM implements EPISODIC MEMORY - the ability to remember WHAT happened, WHERE,
# WHEN, and most importantly HOW (cause and effect).
#
# DATA FLOW (Marathon Agent Mode):
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Camera ──► Frame ──► Change Detection ──► Gemini Vision ──► Objects       │
# │                                                    │                        │
# │                                              ┌─────┴──────────┐             │
# │                                              ▼                ▼             │
# │  TemporalGraph ◄── Movement Detection     MemoryIndex                     │
# │  (cause-effect)                            (hash O(1))                     │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# KEY COMPONENTS:
# 1. Camera (Arducam IMX708)   - Captures JPEG frames
# 2. GeminiClient              - Vision + NLU (native box_2d bounding boxes)
# 3. TemporalGraph             - Tracks object movements (cause-and-effect)
# 4. MemoryIndex               - Fast O(1) object/location lookup
# 5. WhisplayHAT               - LCD display, mic input, button, LED feedback
#
# SEARCH FLOW (object / scene / time / near):
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Voice/Text ──► STT ──► Local Regex Classification                         │
# │                               │                                            │
# │               ┌───────────┬───┴────────┬──────────┐                        │
# │               ▼           ▼            ▼          ▼                        │
# │          MemoryIndex   SceneSearch  TimeWindow  Co-occurrence              │
# │          (hash O(1))   (location)   (fuzzy)    (near/with)                │
# │               │           │            │          │                        │
# │               └─────┬─────┘────────────┘──────────┘                        │
# │                     ▼                                                      │
# │              TemporalGraph + Retrieval Reinforcement                       │
# │              (movements)     (access_count → decay_score)                  │
# │                     │                                                      │
# │                     ▼                                                      │
# │  "Your keys are on the kitchen counter (next to wallet).                   │
# │   You put them there 2h ago when you came back from shopping."             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS AND MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════
#
# We organize imports into three groups:
# 1. Standard library (built into Python)
# 2. Third-party packages (installed via pip)
# 3. Optional hardware-specific imports (may not be available on all systems)
#
# ═══════════════════════════════════════════════════════════════════════════════

# -----------------------------------------------------------------------------
# Standard Library Imports
# These are built into Python and always available
# -----------------------------------------------------------------------------
import re           # Regular expressions for local query classification
import os           # Environment variables, file paths
import sys          # System-specific parameters, exit codes
import io           # In-memory binary streams (for JPEG buffers)
import time         # Timestamps, sleep, performance timing
import math         # Mathematical functions (exponential decay)
import random       # Jitter for exponential backoff retry
import select       # Non-blocking I/O for voice+keyboard input
import tempfile     # Atomic file writes (crash-safe persistence)
import subprocess   # Execute external commands (arecord for audio)
from datetime import datetime, timedelta  # Human-readable timestamps + time math
from pathlib import Path           # Modern file path handling
from dataclasses import dataclass, field  # Clean data class definitions

# -----------------------------------------------------------------------------
# Third-Party Imports
# Install with: pip install google-genai json5
# On Pi: uses system pillow/numpy from apt (python3-pil, python3-numpy)
# -----------------------------------------------------------------------------
import json5                       # JSON parser (superset of json, handles malformed LLM output)
import numpy as np                 # Numerical operations for frame comparison
from PIL import Image, ImageDraw, ImageFont  # Image processing and annotation

# Google Gemini SDK - All AI capabilities come from Gemini 3
from google import genai           # Main Gemini client
from google.genai import types     # Type definitions for API calls

# -----------------------------------------------------------------------------
# Optional Hardware Imports
# These are only available on Raspberry Pi with proper setup
# The system gracefully degrades if hardware is not available
# -----------------------------------------------------------------------------

# Pi Camera - requires: sudo apt install python3-picamera2
# This is the official Raspberry Pi camera library
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True

except ImportError:
    # Running on non-Pi system (development/testing)
    PICAMERA_AVAILABLE = False

# Whisplay HAT driver - requires HAT hardware + driver installation
# Provides LCD display, microphone, button, and LED
try:
    sys.path.append('/home/pi/Whisplay/Driver')
    from WhisPlay import WhisPlayBoard
    WHISPLAY_AVAILABLE = True

except ImportError:
    # HAT not connected or driver not installed
    WHISPLAY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# All configuration is loaded from environment variables with sensible defaults.
# This allows the system to be configured without modifying code.
#
# Configuration is grouped into:
# - Model selection (which Gemini models to use)
# - Storage paths (where to save data)
# - Capture behavior (when to analyze frames)
# - Hardware settings (camera, audio, display)
#
# ═══════════════════════════════════════════════════════════════════════════════

# Configuration is loaded from environment variables
# Set them before running: export GEMINI_API_KEY=your_key

# -----------------------------------------------------------------------------
# Gemini 3 Model Configuration
# -----------------------------------------------------------------------------
# We use gemini-3-flash-preview for real-time performance

VISION_MODEL = os.getenv("GEM_VISION_MODEL", "gemini-3-flash-preview")
# Used for: Object detection with bounding boxes
# Input: JPEG image
# Output: JSON with location, description, objects[]

AUDIO_MODEL = os.getenv("GEM_AUDIO_MODEL", "gemini-3-flash-preview")
# Used for: Speech-to-text, NLU, narrative generation
# Input: WAV audio or text queries
# Output: Transcribed text or generated narratives

# -----------------------------------------------------------------------------
# Storage Path Configuration
# -----------------------------------------------------------------------------
# All data is stored on the local filesystem in two directories:
#   ~/gem/memories/  - Individual memory files (images + metadata)
#   ~/gem/data/      - Shared indexes (temporal graph, memory index)

GEM_ROOT = Path(os.path.expanduser(os.getenv("GEM_ROOT", "~/gem")))

MEMORY_DIR = GEM_ROOT / "memories"
# Format: mem_YYYYMMDD_HHMMSS.jpg (image) + mem_YYYYMMDD_HHMMSS.json (metadata)

DATA_DIR = GEM_ROOT / "data"
# Format: temporal_graph.json, memory_index.json

MAX_MEMORIES = int(os.getenv("GEM_MAX_MEMORIES", "1000"))
# Maximum memories before cleanup (prevents disk from filling up)

# -----------------------------------------------------------------------------
# Capture Behavior Configuration
# -----------------------------------------------------------------------------
# The Marathon Agent doesn't analyze every frame - that would be expensive.
# Instead, it uses a smart triggering system:
# 1. Analyze when scene changes significantly (something moved)
# 2. Analyze periodically even if nothing changed (catch slow changes)
# 3. Enforce minimum interval to prevent API rate limiting

CAPTURE_INTERVAL = float(os.getenv("GEM_CAPTURE_INTERVAL", "1.0"))
# How often to capture frames (seconds)
# Lower = more responsive but more CPU usage

CHANGE_THRESHOLD = float(os.getenv("GEM_CHANGE_THRESHOLD", "0.15"))
# Percentage of pixels that must change to trigger analysis
# 0.15 = 15% of pixels changed (responsive to phone appearing in frame)
# Lower = more sensitive, higher = less sensitive

MIN_ANALYZE_INTERVAL = int(os.getenv("GEM_MIN_INTERVAL", "3"))
# Minimum seconds between analyses (prevents API spam)
# 3 seconds is good for paid tier - responsive but not wasteful

FORCE_ANALYZE_INTERVAL = int(os.getenv("GEM_FORCE_INTERVAL", "30"))
# Force analysis even if scene hasn't changed (catch slow movements)
# Every 30 seconds, analyze regardless of change detection

MIN_CONFIDENCE = float(os.getenv("GEM_MIN_CONFIDENCE", "0.5"))
# Minimum confidence score to accept an object detection (0.0-1.0)
# 0.7 = 70% confidence required - filters out hallucinated objects
# Higher = fewer false positives but might miss some real objects

PIXEL_CHANGE_THRESHOLD = 30
# Per-pixel intensity change threshold (0-255)
# A pixel is "changed" if its intensity differs by more than this amount
# Helps ignore noise and minor lighting changes

# -----------------------------------------------------------------------------
# Hardware Configuration
# -----------------------------------------------------------------------------

CAMERA_INDEX = int(os.getenv("GEM_CAMERA_INDEX", "0"))
# Which camera to use if multiple are connected
# 0 = first camera (usually the only one)

RECORD_DURATION = int(os.getenv("GEM_RECORD_DURATION", "3"))
# How long to record audio for voice queries (seconds)
# 3 seconds is enough for "Where are my keys?"

TTS_ENABLED = os.getenv("GEM_TTS_ENABLED", "false").lower() == "true"
# Enable text-to-speech audio feedback globally (daemon + search)
# Disabled by default due to onboard speaker noise

SEARCH_TTS_ENABLED = os.getenv("GEM_SEARCH_TTS_ENABLED", "true").lower() == "true"
# Enable TTS for search results only (speaks found object locations)
# Enabled by default - search results are spoken aloud
# Set to false if speaker noise is unacceptable

ANNOUNCE_ENABLED = os.getenv("GEM_ANNOUNCE_ENABLED", "false").lower() == "true"
# Proactively announce when objects are placed in new locations
# Helps prevent forgetting where you just put something
# Disabled by default (requires TTS_ENABLED=true)

ANNOUNCE_OBJECTS = set(os.getenv("GEM_ANNOUNCE_OBJECTS",
    "phone,keys,wallet,glasses,remote,headphones,watch").split(","))
# Which objects to announce when placed (important items only)

ANNOUNCE_COOLDOWN = int(os.getenv("GEM_ANNOUNCE_COOLDOWN", "60"))
# Seconds between announcements for the same object (prevent spam)

# Track last announcement time per object to prevent spam
_announcement_cooldowns: dict[str, float] = {}

# -----------------------------------------------------------------------------
# Audio Capture for Complete Episodic Memory (WHO dimension)
# -----------------------------------------------------------------------------
# Captures ambient audio during daemon to enable:
# - "Who did I meet?" queries (extracts names from conversations)
# - "What did they say?" queries (conversation recall)
# - Richer episodic context (WHAT + WHERE + WHEN + WHO + HOW)

AUDIO_CAPTURE_ENABLED = os.getenv("GEM_AUDIO_CAPTURE_ENABLED", "true").lower() == "true"
# Enable audio capture during daemon image captures
# Records short audio clips to capture conversations/introductions

AUDIO_CAPTURE_DURATION = int(os.getenv("GEM_AUDIO_CAPTURE_DURATION", "5"))
# Duration in seconds for each audio capture (default: 5 seconds)
# Longer = more context but more processing/storage

# -----------------------------------------------------------------------------
# Future: MedGemma Integration for Healthcare Settings
# -----------------------------------------------------------------------------
# MedGemma (Google's medical AI) integration is planned for future versions
# to provide enhanced assistance for users with Alzheimer's and dementia.
#
# Planned features:
#   - Patient-specific memory profiles with medical context
#   - Caregiver notifications when user seems confused
#   - Daily activity reports for healthcare providers
#   - Medication reminder integration
#   - HIPAA-compliant data handling
#
# To enable (when available):
#   export GEM_MEDGEMMA_ENABLED=true
#   export GEM_MEDGEMMA_MODEL=medgemma-1.5
#   export GEM_CAREGIVER_EMAIL=caregiver@example.com
#
MEDGEMMA_ENABLED = os.getenv("GEM_MEDGEMMA_ENABLED", "false").lower() == "true"
MEDGEMMA_MODEL = os.getenv("GEM_MEDGEMMA_MODEL", "medgemma-1.5")
# Note: MedGemma features are not yet implemented - these are placeholders

HAT_LCD_WIDTH = 240   # Whisplay HAT LCD width in pixels
HAT_LCD_HEIGHT = 280  # Whisplay HAT LCD height in pixels


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def atomic_write_text(path: Path, data: str):
    """
    Write text to file atomically using temp file + rename.

    On Pi, power loss during write corrupts the file. This function writes to
    a temp file in the same directory, then atomically renames it. If the
    process dies mid-write, the original file remains intact.

    Args:
        path: Target file path
        data: Text content to write
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=path.stem
    )
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))  # Atomic on POSIX
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path, data: bytes):
    """
    Write bytes to file atomically using temp file + rename.

    Same crash-safety guarantee as atomic_write_text, but for binary data
    (JPEG images, etc.).

    Args:
        path: Target file path
        data: Binary content to write
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=path.stem
    )
    try:
        with os.fdopen(tmp_fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3B: LOGGING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
#
# Simple timestamped logging for debugging and monitoring.
#
# ═══════════════════════════════════════════════════════════════════════════════

def log(msg: str):
    """
    Print timestamped log message to stdout.
    
    Format: [HH:MM:SS] message
    
    Args:
        msg: Message to log
        
    Example:
        log("Starting capture daemon")
        # Output: [14:32:15] Starting capture daemon
    """
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")


def log_error(msg: str):
    """
    Print timestamped error message to stderr.
    
    Format: [HH:MM:SS] ❌ message
    
    Args:
        msg: Error message to log
        
    Example:
        log_error("Camera not found")
        # Output: [14:32:15] ❌ Camera not found
    """
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] ❌ {msg}", file=sys.stderr)


def human_time(iso_timestamp: str) -> str:
    """
    Convert ISO timestamp to human-friendly format for people with memory issues.

    Examples:
        "2026-02-02T15:43:00" → "Today at 3:43 PM" (if today)
        "2026-02-01T10:30:00" → "Yesterday at 10:30 AM"
        "2026-01-28T14:00:00" → "Monday at 2:00 PM" (within 7 days)
        "2026-01-15T09:15:00" → "Jan 15 at 9:15 AM" (older)

    Args:
        iso_timestamp: ISO format timestamp (e.g., "2026-02-02T15:43:00")

    Returns:
        Human-friendly string like "Today at 3:43 PM"
    """
    try:
        # Parse the timestamp
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        now = datetime.now()

        # Format time as 12-hour with AM/PM
        time_str = dt.strftime("%-I:%M %p").replace(" 0", " ")  # "3:43 PM"

        # Calculate difference
        diff = now.date() - dt.date()

        if diff.days == 0:
            return f"Today at {time_str}"
        elif diff.days == 1:
            return f"Yesterday at {time_str}"
        elif diff.days < 7:
            day_name = dt.strftime("%A")  # "Monday"
            return f"{day_name} at {time_str}"
        else:
            # Older - show date
            date_str = dt.strftime("%b %-d")  # "Jan 15"
            return f"{date_str} at {time_str}"
    except (ValueError, AttributeError):
        # Fallback if parsing fails
        return iso_timestamp[:16] if iso_timestamp else "Unknown time"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════
#
# Core data classes that represent the building blocks of episodic memory:
#
# 1. BoundingBox - A detected object with its location in the image
# 2. ObjectMovement - A record of an object moving between locations
# 3. TemporalGraph - Tracks all object movements over time (cause and effect)
# 4. Memory - A complete episodic memory record (WHAT, WHERE, WHEN)
#
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BoundingBox:
    """
    Detected object with normalized bounding box coordinates.
    
    COORDINATE SYSTEM:
    ┌─────────────────────────┐
    │ (0,0)              (1,0)│
    │   ┌─────────┐           │
    │   │  Object │           │
    │   │  (x1,y1)┼───(x2,y2) │
    │   └─────────┘           │
    │ (0,1)              (1,1)│
    └─────────────────────────┘
    
    Coordinates are normalized to 0.0-1.0 range where:
    - (0, 0) = top-left corner of image
    - (1, 1) = bottom-right corner of image
    
    This normalization allows consistent handling regardless of image resolution.
    To convert to pixel coordinates: pixel_x = normalized_x * image_width
    
    Attributes:
        name: Object name (e.g., "keys", "wallet", "phone")
        x1: Left edge of bounding box (0.0-1.0)
        y1: Top edge of bounding box (0.0-1.0)
        x2: Right edge of bounding box (0.0-1.0)
        y2: Bottom edge of bounding box (0.0-1.0)
        confidence: Detection confidence score (0.0-1.0)
    """
    name: str           # Object name (e.g., "keys", "wallet")
    x1: float           # Left edge (0.0-1.0)
    y1: float           # Top edge (0.0-1.0)
    x2: float           # Right edge (0.0-1.0)
    y2: float           # Bottom edge (0.0-1.0)
    confidence: float = 1.0  # Detection confidence (default: 100%)
    context: str = ""        # Placement context (e.g., "on table", "in pocket")
    
    def position(self) -> str:
        """
        Get human-readable position description.
        
        Divides the frame into a 3x3 grid and returns the cell name:
        
        ┌────────┬────────┬────────┐
        │top-left│  top   │top-right│
        ├────────┼────────┼────────┤
        │  left  │ center │  right │
        ├────────┼────────┼────────┤
        │bot-left│ bottom │bot-right│
        └────────┴────────┴────────┘
        
        This provides intuitive location descriptions like:
        "Your keys are in the top-left of the frame"
        
        Returns:
            Position string: "top-left", "center", "bottom-right", etc.
        """
        # Calculate center point of bounding box
        cx = (self.x1 + self.x2) / 2  # Center X (0.0-1.0)
        cy = (self.y1 + self.y2) / 2  # Center Y (0.0-1.0)
        
        # Determine horizontal position (left/center/right)
        # Using 0.33 and 0.66 as boundaries for 3 equal columns
        h = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
        
        # Determine vertical position (top/middle/bottom)
        v = "top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle"
        
        # Combine into human-readable string
        if h == "center" and v == "middle":
            return "center"          # Dead center
        elif v == "middle":
            return h                 # "left" or "right"
        elif h == "center":
            return v                 # "top" or "bottom"
        return f"{v}-{h}"           # "top-left", "bottom-right", etc.
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.

        Used when saving memory metadata to disk.

        Returns:
            Dictionary with all bounding box attributes
        """
        d = {
            "name": self.name,
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "confidence": self.confidence
        }
        if self.context:
            d["context"] = self.context
        return d


@dataclass
class ObjectMovement:
    """
    Tracks an object's movement between locations over time.
    
    This is the KEY DATA STRUCTURE for TEMPORAL-SPATIAL understanding.
    It enables cause-and-effect reasoning:
    - "Your keys moved from the kitchen counter to your jacket pocket"
    - "This happened 2 hours ago when you came home from work"
    
    Each ObjectMovement records:
    - WHAT moved (object_name)
    - FROM where (from_location, from_position)
    - TO where (to_location, to_position)
    - WHEN it happened (from_time, to_time)
    - WHICH memories captured it (from_memory_id, to_memory_id)
    
    Attributes:
        object_name: Name of the object that moved (e.g., "keys")
        from_location: Scene description where object was before (e.g., "kitchen counter")
        to_location: Scene description where object is now (e.g., "jacket pocket")
        from_position: Position in frame before (e.g., "center")
        to_position: Position in frame now (e.g., "left")
        from_time: ISO timestamp when object was last seen at old location
        to_time: ISO timestamp when object was seen at new location
        from_memory_id: Memory ID where object was seen before
        to_memory_id: Memory ID where object is now
    """
    object_name: str         # e.g., "keys"
    from_location: str       # e.g., "kitchen counter"
    to_location: str         # e.g., "jacket pocket"
    from_position: str       # e.g., "center"
    to_position: str         # e.g., "left"
    from_time: str           # ISO timestamp (e.g., "2026-01-22T14:30:00")
    to_time: str             # ISO timestamp
    from_memory_id: str      # Memory ID where object was seen before
    to_memory_id: str        # Memory ID where object is now
    
    def duration_str(self) -> str:
        """
        Calculate human-readable duration between sightings.
        
        Converts the time difference into natural language:
        - "2d 5h ago" (days and hours)
        - "3h 45m ago" (hours and minutes)
        - "15m ago" (minutes)
        - "30s ago" (seconds)
        
        Returns:
            Human-readable duration string
        """
        try:
            # Parse ISO timestamps
            from_dt = datetime.fromisoformat(self.from_time)
            to_dt = datetime.fromisoformat(self.to_time)
            delta = to_dt - from_dt
            
            # Convert to most appropriate unit
            if delta.days > 0:
                # More than a day: show days and hours
                return f"{delta.days}d {delta.seconds // 3600}h ago"
            elif delta.seconds >= 3600:
                # More than an hour: show hours and minutes
                hours = delta.seconds // 3600
                minutes = (delta.seconds % 3600) // 60
                return f"{hours}h {minutes}m ago"
            elif delta.seconds >= 60:
                # More than a minute: show minutes
                return f"{delta.seconds // 60}m ago"
            else:
                # Less than a minute: show seconds
                return f"{delta.seconds}s ago"
        
        except Exception:
            return "unknown"
    
    def to_narrative(self) -> str:
        """
        Generate a natural language description of this movement.
        
        Example output:
        "keys moved from kitchen counter (center) to jacket pocket (left), 2h ago"
        
        Returns:
            Human-readable movement description
        """
        return (
            f"{self.object_name} moved from {self.from_location} ({self.from_position}) "
            f"to {self.to_location} ({self.to_position}), {self.duration_str()}"
        )


class TemporalGraph:
    """
    In-memory graph tracking object movements over time.
    
    This is the CORE DATA STRUCTURE for CAUSE-AND-EFFECT reasoning.
    It answers questions like:
    - "Where was object X before it was here?"
    - "What path did object X take?"
    - "When did object X leave location Y?"
    
    DESIGN DECISIONS:
    
    1. In-Memory Storage:
       We keep the graph in memory rather than a database because:
       - Queries are O(1) for last location lookup
       - No network/disk latency
       - Small memory footprint (movements are just references)
       
    2. Hash Index (self.by_object):
       Maps object names to their last known location.
       Enables instant lookup: "Where are my keys?" → O(1)
       
    3. Movement History (self.movements):
       Keeps last 100 movements per object.
       Enables narrative generation: "Your keys were on the counter,
       then moved to your bag, then to the desk."
       
    4. Persistence:
       Saved to JSON file for marathon agent continuity.
       When the daemon restarts, it loads the previous state.
    
    ALGORITHM FOR MOVEMENT DETECTION:
    
    When we see an object in a new frame:
    1. Check if we've seen this object before (in self.last_seen)
    2. If yes, compare current location/position with previous
    3. If different, record as a movement
    4. Update last_seen to current location
    
    Attributes:
        last_seen: Dict mapping object_name → (location, position, timestamp, memory_id)
        movements: Dict mapping object_name → List[ObjectMovement] (newest first)
        total_movements: Counter for statistics
        start_time: When the graph was created (for uptime tracking)
    """
    
    def __init__(self):
        """Initialize empty temporal graph."""
        # O(1) lookup: object_name → (location, position, timestamp, memory_id)
        self.last_seen: dict[str, tuple] = {}

        # Movement history: object_name → list of ObjectMovement (newest first)
        self.movements: dict[str, list[ObjectMovement]] = {}

        # Track attached objects (glasses on face, watch on wrist, etc.)
        # Maps object_name → (timestamp, location) when last seen attached
        self.attached_objects: dict[str, tuple[str, str]] = {}

        # Statistics for monitoring
        self.total_movements = 0
        self.start_time = datetime.now().isoformat()
    
    def update(self, obj_name: str, location: str, position: str,
               timestamp: str, memory_id: str) -> ObjectMovement | None:
        """
        Update object location and detect if it moved.
        
        ALGORITHM:
        1. Normalize object name to lowercase (case-insensitive matching)
        2. Check if we've seen this object before
        3. If yes, compare locations:
           - Same location AND position → No movement, just update timestamp
           - Different location OR position → Record movement
        4. Update last_seen with current information
        5. Return movement if detected, None otherwise
        
        Args:
            obj_name: Name of the object (will be lowercased)
            location: Scene description (e.g., "kitchen counter")
            position: Position in frame (e.g., "center")
            timestamp: ISO timestamp of this sighting
            memory_id: ID of the memory where this object was seen
            
        Returns:
            ObjectMovement if the object moved, None if first sighting or same location
        """
        obj_name = obj_name.lower()  # Normalize for consistent matching
        movement = None

        # Check if we've seen this object before
        if obj_name in self.last_seen:
            # Unpack previous sighting
            prev_loc, prev_pos, prev_time, prev_mem = self.last_seen[obj_name]

            # Normalize locations to reduce false positives from Gemini's
            # inconsistent naming (e.g., "indoor room" vs "indoor office" vs "office")
            def normalize_location(loc: str) -> str:
                """Extract core location words, ignore modifiers."""
                loc = loc.lower().strip()
                # Remove common prefixes that Gemini adds inconsistently
                for prefix in ["indoor ", "outdoor ", "inside ", "the "]:
                    loc = loc.replace(prefix, "")
                loc = loc.strip()
                # Remove generic words that add no useful location info
                # These are words Gemini uses when it can't identify a specific location
                generic_words = ["room", "space", "area", "indoor", "outdoor", "inside", "office", "workspace"]
                if loc in generic_words or loc == "":
                    return "generic"
                # Remove these words if they appear as suffixes
                for suffix in generic_words:
                    if loc.endswith(" " + suffix):
                        loc = loc[:-len(suffix)-1].strip()
                # After cleanup, check again
                if loc in generic_words or loc == "":
                    return "generic"
                return loc or "generic"

            # Check if positions are significantly different
            # Positions like "bottom" vs "bottom-left" are close enough to ignore
            def positions_different(pos1: str, pos2: str) -> bool:
                """Return True only if positions are significantly different."""
                pos1, pos2 = pos1.lower(), pos2.lower()
                if pos1 == pos2:
                    return False
                # Extract primary direction (top/bottom/center, left/right/center)
                def primary(p):
                    if "top" in p: return "top"
                    if "bottom" in p: return "bottom"
                    return "center"
                def secondary(p):
                    if "left" in p: return "left"
                    if "right" in p: return "right"
                    return "center"
                # Only count as different if BOTH primary directions changed
                # (e.g., top-left → bottom-right is different, but center → center-left is not)
                return primary(pos1) != primary(pos2) and secondary(pos1) != secondary(pos2)

            norm_prev = normalize_location(prev_loc)
            norm_curr = normalize_location(location)

            # Detect movement: significant location OR position change
            # This reduces false positives from Gemini's inconsistent naming
            location_changed = norm_prev != norm_curr
            position_changed = positions_different(prev_pos, position)

            if location_changed or position_changed:
                # Create movement record
                movement = ObjectMovement(
                    object_name=obj_name,
                    from_location=prev_loc,
                    to_location=location,
                    from_position=prev_pos,
                    to_position=position,
                    from_time=prev_time,
                    to_time=timestamp,
                    from_memory_id=prev_mem,
                    to_memory_id=memory_id
                )
                
                # Add to movement history (newest first)
                if obj_name not in self.movements:
                    self.movements[obj_name] = []
                self.movements[obj_name].insert(0, movement)
                
                # Keep only last 100 movements per object to bound memory usage
                self.movements[obj_name] = self.movements[obj_name][:100]
                
                self.total_movements += 1
                log(f"[TEMPORAL] {movement.to_narrative()}")
        
        # Update last seen (whether new or moved)
        self.last_seen[obj_name] = (location, position, timestamp, memory_id)
        
        return movement
    
    def get_history(self, obj_name: str, limit: int = 10) -> list[ObjectMovement]:
        """
        Get movement history for an object (newest first).
        
        Args:
            obj_name: Object name to look up
            limit: Maximum number of movements to return
            
        Returns:
            List of ObjectMovement records, newest first
        """
        return self.movements.get(obj_name.lower(), [])[:limit]
    
    def get_last_location(self, obj_name: str) -> tuple | None:
        """
        Get last known location of an object.
        
        Args:
            obj_name: Object name to look up
            
        Returns:
            Tuple of (location, position, timestamp, memory_id) or None if never seen
        """
        return self.last_seen.get(obj_name.lower())
    
    def generate_narrative(self, obj_name: str, limit: int = 5) -> str:
        """
        Generate a causal narrative explaining object's journey.
        
        This is NOT simple object detection - it's TEMPORAL REASONING
        that explains cause and effect over time.
        
        Example output:
        "📍 Movement history for 'keys':
          1. keys moved from desk (left) to kitchen counter (center), 2h ago
          2. keys moved from jacket pocket (right) to desk (left), 5h ago
          
          ➡️  Currently at: kitchen counter (center)"
        
        Args:
            obj_name: Object to generate narrative for
            limit: Maximum number of movements to include
            
        Returns:
            Multi-line narrative string
        """
        history = self.get_history(obj_name, limit)
        
        if not history:
            # No movement history - just report current location
            last = self.get_last_location(obj_name)
            if last:
                return f"Your {obj_name} is at {last[0]} ({last[1]}). First seen {last[2][:16]}."
            return f"I haven't seen your {obj_name} yet."
        
        # Build narrative from movement history
        lines = [f"📍 Movement history for '{obj_name}':"]
        for i, m in enumerate(history):
            lines.append(f"  {i+1}. {m.to_narrative()}")
        
        # Add current location
        last = self.get_last_location(obj_name)
        if last:
            lines.append(f"\n  ➡️  Currently at: {last[0]} ({last[1]})")
        
        return "\n".join(lines)

    def mark_attached(self, obj_name: str, timestamp: str, location: str):
        """
        Mark an object as currently attached to the person.

        Called when we see glasses on face, watch on wrist, etc.
        This allows us to detect when they're removed later.

        Args:
            obj_name: Object name (e.g., "glasses")
            timestamp: When we saw it attached
            location: Scene location (for context)
        """
        obj_name = obj_name.lower()
        self.attached_objects[obj_name] = (timestamp, location)

    def check_removed_attached(self, current_objects: list[str],
                                current_time: str,
                                timeout_seconds: int = 30) -> list[str]:
        """
        Check if any previously-attached objects have been removed.

        If an object was attached (e.g., glasses on face) and we haven't
        seen it for timeout_seconds, it was likely removed and placed somewhere.

        Args:
            current_objects: List of object names currently visible (attached or not)
            current_time: Current timestamp (ISO format)
            timeout_seconds: How long before we consider object removed

        Returns:
            List of object names that were removed (no longer attached)
        """
        removed = []
        current_set = {o.lower() for o in current_objects}
        current_dt = datetime.fromisoformat(current_time.replace('Z', '+00:00'))

        for obj_name, (last_time, last_loc) in list(self.attached_objects.items()):
            # Skip if we still see this object
            if obj_name in current_set:
                continue

            # Check if enough time has passed
            try:
                last_dt = datetime.fromisoformat(last_time.replace('Z', '+00:00'))
                elapsed = (current_dt - last_dt).total_seconds()
                if elapsed >= timeout_seconds:
                    removed.append(obj_name)
                    # Remove from attached tracking
                    del self.attached_objects[obj_name]
            except (ValueError, TypeError):
                pass

        return removed

    def get_attached_status(self, obj_name: str) -> tuple[str, str] | None:
        """
        Get when/where an object was last seen attached.

        Args:
            obj_name: Object name

        Returns:
            (timestamp, location) or None if not tracked
        """
        return self.attached_objects.get(obj_name.lower())

    def save(self, path: Path):
        """
        Persist temporal graph to disk for marathon agent continuity.
        
        This enables the agent to maintain memory across restarts.
        The file is human-readable JSON for debugging.
        
        Args:
            path: Path to save JSON file
        """
        data = {
            "start_time": self.start_time,
            "total_movements": self.total_movements,
            "last_seen": {k: list(v) for k, v in self.last_seen.items()},
            "movements": {
                k: [vars(m) for m in v]
                for k, v in self.movements.items()
            },
            "attached_objects": {k: list(v) for k, v in self.attached_objects.items()}
        }
        atomic_write_text(path, json5.dumps(data, indent=2))

    def load(self, path: Path):
        """
        Load temporal graph from disk (marathon agent resume).
        
        Called on startup to restore previous state.
        
        Args:
            path: Path to JSON file
        """
        if not path.exists():
            return
        
        try:
            data = json5.loads(path.read_text())
            self.start_time = data.get("start_time", self.start_time)
            self.total_movements = data.get("total_movements", 0)
            
            # Restore last_seen dictionary
            self.last_seen = {k: tuple(v) for k, v in data.get("last_seen", {}).items()}
            
            # Restore movement history (reconstruct ObjectMovement objects)
            self.movements = {
                k: [ObjectMovement(**m) for m in v]
                for k, v in data.get("movements", {}).items()
            }

            # Restore attached objects tracking
            self.attached_objects = {
                k: tuple(v) for k, v in data.get("attached_objects", {}).items()
            }

            log(f"[TEMPORAL] Loaded graph: {len(self.last_seen)} objects, {self.total_movements} movements")
        except Exception as e:
            log_error(f"Failed to load temporal graph: {e}")


@dataclass
class Memory:
    """
    Episodic memory record implementing Tulving's (1972) 5 dimensions.

    This is the core data structure that implements complete episodic memory:

    - WHAT: Objects detected in the scene (self.objects) + activities (self.activities)
    - WHERE: Scene location and object positions (self.location, bbox.position())
    - WHEN: Timestamp of capture (self.timestamp)
    - WHO: People present (self.people) from audio transcription
    - HOW: Movement tracking via TemporalGraph (external)

    Each memory is stored as:
    - JPEG image file (mem_YYYYMMDD_HHMMSS.jpg)
    - JSON metadata file (mem_YYYYMMDD_HHMMSS.json)
    - JSON index entry (for object/activity search)

    Attributes:
        id: Unique identifier (format: "mem_YYYYMMDD_HHMMSS")
        timestamp: ISO format datetime string
        location: Scene description from Gemini (e.g., "kitchen counter")
        description: Detailed scene description from Gemini
        objects: List of detected objects with bounding boxes
        activities: List of detected activities (e.g., "taking medication")
        image_path: Path to saved JPEG file
        image_data: Raw JPEG bytes (for display without disk read)
    """
    id: str                  # Unique ID: "mem_YYYYMMDD_HHMMSS"
    timestamp: str           # ISO format: "2026-01-22T14:30:00"
    location: str            # Scene description: "kitchen counter"
    description: str         # Full scene description
    objects: list[BoundingBox] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)        # Detected activities: ["taking medication"]
    image_path: str = ""     # Path to saved JPEG
    image_data: bytes = b""  # Raw JPEG bytes (for display)
    tags: list[str] = field(default_factory=list)              # Scene tags: ["kitchen", "cooking"]
    relationships: list[str] = field(default_factory=list)     # Spatial: ["keys on desk"]
    # Audio/Conversation episodic memory (WHO dimension)
    audio_transcript: str = ""                                  # Transcribed speech from scene
    people: list[str] = field(default_factory=list)            # Extracted names: ["John", "Sarah"]
    conversation_context: str = ""                              # Summary of what was discussed
    # Visual person detection (WHO dimension - visual)
    persons: list[dict] = field(default_factory=list)          # Visual persons: [{"description": "man in blue shirt", "context": "sitting at desk"}]
    
    def find_object(self, name: str) -> BoundingBox | None:
        """
        Find object by name with case-insensitive partial matching.
        
        Supports partial matching so "key" matches "car keys", "house keys", etc.
        
        Args:
            name: Object name to search for (case-insensitive)
            
        Returns:
            BoundingBox if found, None otherwise
            
        Example:
            memory.find_object("key")  # Matches "keys", "car keys", etc.
        """
        name_lower = name.lower()
        for obj in self.objects:
            # Partial match: "key" in "car keys" = True
            if name_lower in obj.name.lower():
                return obj
        return None
    
    def object_names(self) -> list[str]:
        """
        Get list of all detected object names.
        
        Returns:
            List of object name strings
            
        Example:
            ["keys", "wallet", "phone", "coffee mug"]
        """
        return [obj.name for obj in self.objects]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: FILE STORAGE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Simple filesystem-based storage for images and metadata.
#
# DESIGN DECISIONS:
#
# 1. Separate Files for Image and Metadata:
#    - Images are binary (JPEG) - efficient, compact
#    - Metadata is JSON - human-readable, debuggable
#    - Allows inspecting metadata without loading images
#
# 2. Predictable Naming:
#    - Format: mem_YYYYMMDD_HHMMSS.{jpg,json}
#    - Sortable by filename = sortable by time
#    - No collisions (one memory per second max)
#
# 3. JSON Index for Search:
#    - Object names indexed in memory_index.json
#    - Enables fast O(1) object lookup
#    - Filesystem is source of truth, JSON index is secondary
#
# ═══════════════════════════════════════════════════════════════════════════════

def init_storage() -> None:
    """
    Create storage directories if they don't exist.
    ...
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log(f"📁 Storage: {MEMORY_DIR}")


def save_image(mem_id: str, data: bytes) -> str:
    """
    Save JPEG image to disk.
    
    Args:
        mem_id: Memory ID for filename (e.g., "mem_20260122_143000")
        data: Raw JPEG bytes from camera
        
    Returns:
        Absolute path to saved file
    """
    path = MEMORY_DIR / f"{mem_id}.jpg"
    atomic_write_bytes(path, data)
    return str(path)


def save_metadata(memory: Memory) -> None:
    """
    Save memory metadata to JSON file.
    
    Stored separately from image for efficient metadata queries.
    The JSON is human-readable for debugging.
    
    Args:
        memory: Memory object to save
    """
    path = MEMORY_DIR / f"{memory.id}.json"
    atomic_write_text(path, json5.dumps({
        "id": memory.id,
        "timestamp": memory.timestamp,
        "location": memory.location,
        "description": memory.description,
        "objects": [o.to_dict() for o in memory.objects],
        "activities": memory.activities,  # WHAT dimension: detected activities
        "image_path": memory.image_path,
        "tags": memory.tags,
        "relationships": memory.relationships,
        # Episodic memory: WHO dimension (audio/conversation)
        "audio_transcript": memory.audio_transcript,
        "people": memory.people,
        "conversation_context": memory.conversation_context,
        # Episodic memory: WHO dimension (visual persons)
        "persons": memory.persons
    }, indent=2))


def load_memory(mem_id: str) -> Memory | None:
    """
    Load memory from disk by ID.
    
    Loads both the JSON metadata and the JPEG image data.
    
    Args:
        mem_id: Memory ID to load
        
    Returns:
        Memory object with image data loaded, or None if not found
    """
    path = MEMORY_DIR / f"{mem_id}.json"
    if not path.exists():
        return None
    
    try:
        # Load JSON metadata
        data = json5.loads(path.read_text())
        
        # Reconstruct Memory object
        memory = Memory(
            id=data["id"],
            timestamp=data["timestamp"],
            location=data["location"],
            description=data["description"],
            objects=[BoundingBox(**o) for o in data.get("objects", [])],
            activities=data.get("activities", []),  # WHAT dimension: detected activities
            image_path=data.get("image_path", ""),
            tags=data.get("tags", []),
            relationships=data.get("relationships", []),
            # Episodic memory: WHO dimension (audio/conversation)
            audio_transcript=data.get("audio_transcript", ""),
            people=data.get("people", []),
            conversation_context=data.get("conversation_context", ""),
            # Episodic memory: WHO dimension (visual persons)
            persons=data.get("persons", [])
        )
        
        # Load image data for display (guard against empty/missing paths)
        if memory.image_path:
            img_path = Path(memory.image_path)
            if img_path.exists() and img_path.stat().st_size > 0:
                memory.image_data = img_path.read_bytes()

        return memory
    except Exception:
        return None


def cleanup_old_memories(index: 'MemoryIndex' = None) -> int:
    """
    Delete lowest-scoring memories when exceeding MAX_MEMORIES limit.

    Uses temporal decay + retrieval reinforcement (Ebbinghaus forgetting curve)
    to decide which memories to forget. Frequently searched memories survive
    longer, mimicking how human recall strengthens memory traces.

    If no index is available, falls back to oldest-first deletion.

    Args:
        index: Optional MemoryIndex to score memories and remove deleted entries

    Returns:
        Number of memories deleted
    """
    # Get all memory files sorted by name (oldest first due to timestamp in name)
    memory_files = sorted(MEMORY_DIR.glob("mem_*.json"))

    if len(memory_files) <= MAX_MEMORIES:
        return 0

    # Calculate how many to delete
    to_delete = len(memory_files) - MAX_MEMORIES
    deleted = 0

    if index:
        # --- DECAY-BASED CLEANUP (Ebbinghaus forgetting curve) ---
        # Instead of simply deleting oldest memories, we score each memory
        # using decay_score() which combines:
        #   1. Recency (exponential decay, 7-day half-life)
        #   2. Retrieval boost (+0.1 per search access, max +1.0)
        # This means a 30-day-old memory that's been searched 10 times
        # (score ~1.05) survives over a 7-day-old memory never searched
        # (score ~0.5). This mimics how human recall strengthens traces.
        scored = []
        for json_path in memory_files:
            mem_id = json_path.stem
            score = index.decay_score(mem_id)
            scored.append((score, json_path))
        # Sort ascending — lowest scores (old + never recalled) forgotten first
        scored.sort(key=lambda x: x[0])
        candidates = [path for _, path in scored[:to_delete]]
    else:
        # Fallback without index: simple oldest-first (FIFO)
        candidates = memory_files[:to_delete]

    # Delete candidate memories from disk and all in-memory indexes
    for json_path in candidates:
        try:
            mem_id = json_path.stem
            jpg_path = MEMORY_DIR / f"{mem_id}.jpg"

            # Delete both files together. If one fails, the other is still
            # cleaned up — orphans are harmless and will be caught next cycle.
            json_path.unlink(missing_ok=True)
            jpg_path.unlink(missing_ok=True)

            # Remove from in-memory indexes to stay consistent with disk.
            # The by_object hash maps object_name → {mem_id_1, mem_id_2, ...}.
            # We must remove this mem_id from every object set that references it,
            # otherwise searches would return deleted memories.
            # We use .discard() (not .remove()) because it won't error if missing.
            if index and mem_id in index.memories:
                meta = index.memories.get(mem_id, {})
                for obj in meta.get('objects', '').split(','):
                    obj = obj.strip().lower()
                    if obj and obj in index.by_object:
                        index.by_object[obj].discard(mem_id)
                del index.memories[mem_id]
            # Remove access log entry to prevent unbounded growth
            if index:
                index.access_log.pop(mem_id, None)

            deleted += 1
        except Exception:
            pass

    if deleted > 0:
        log(f"[CLEANUP] Forgot {deleted} low-score memories (limit: {MAX_MEMORIES})")

    return deleted


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CAMERA
# ═══════════════════════════════════════════════════════════════════════════════
#
# Pi Camera interface using Picamera2 library.
#
# DESIGN DECISIONS:
#
# 1. Resolution: 640x480
#    - Good balance of quality and processing speed
#    - Sufficient for object detection
#    - Keeps JPEG file sizes reasonable (~50KB)
#
# 2. Format: RGB888
#    - Standard format for image processing
#    - Compatible with PIL for annotation
#    - Easy conversion to JPEG
#
# 3. Frame Difference Detection:
#    - Grayscale comparison for speed
#    - Per-pixel threshold to ignore noise
#    - Returns percentage of changed pixels
#
# ═══════════════════════════════════════════════════════════════════════════════

class Camera:
    """
    Raspberry Pi camera interface using Picamera2.
    
    Captures frames at 640x480 resolution for balance of quality and speed.
    Provides both JPEG bytes (for storage/API) and numpy array (for change detection).
    
    Usage:
        camera = Camera()
        jpeg_bytes, frame_array = camera.capture()
        camera.close()
    """
    
    def __init__(self):
        """
        Initialize camera with optimal settings for GEM.
        
        Raises:
            RuntimeError: If picamera2 not installed or no camera found
        """
        if not PICAMERA_AVAILABLE:
            raise RuntimeError(
                "picamera2 not installed.\n"
                "Run: sudo apt install python3-picamera2"
            )
        
        # Check available cameras
        cameras = Picamera2.global_camera_info()
        if not cameras:
            raise RuntimeError("No camera found! Check ribbon cable connection.")
        
        if CAMERA_INDEX >= len(cameras):
            raise RuntimeError(f"Camera {CAMERA_INDEX} not found")
        
        # Initialize camera
        self.camera = Picamera2(camera_num=CAMERA_INDEX)
        
        # Configure for 640x480 RGB
        # This resolution balances quality and processing speed
        config = self.camera.create_still_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.camera.configure(config)
        self.camera.start()
        
        model = cameras[CAMERA_INDEX].get('Model', 'unknown')
        log(f"📷 Camera: {model}")
    
    def capture(self) -> tuple:
        """
        Capture a frame from the camera.
        
        Returns both formats needed by the system:
        - JPEG bytes for storage and API transmission
        - Numpy array for frame change detection
        
        Returns:
            tuple: (jpeg_bytes, numpy_array)
        """
        # Capture raw frame as numpy array
        frame = self.camera.capture_array()
        
        # Convert to JPEG for efficient storage
        # Using PIL for the conversion
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)  # 75% quality = good balance
        
        return buf.getvalue(), frame
    
    def close(self):
        """Release camera resources."""
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception:
                pass


def frame_difference(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Calculate percentage of changed pixels between two frames.
    
    Used to detect scene changes and trigger analysis.
    Only analyzes when something significant has changed,
    saving API calls and processing time.
    
    ALGORITHM:
    1. Convert both frames to grayscale (faster comparison)
    2. Calculate absolute difference per pixel
    3. Count pixels that changed by more than threshold
    4. Return ratio of changed pixels to total pixels
    
    Args:
        f1: Previous frame (numpy array, RGB or grayscale)
        f2: Current frame (numpy array, RGB or grayscale)
        
    Returns:
        Float 0.0-1.0 representing percentage of changed pixels
        1.0 = 100% changed (or invalid input, forces analysis)
    """
    # Handle edge cases: no previous frame or size mismatch
    if f1 is None or f2 is None or f1.shape != f2.shape:
        return 1.0  # Force analysis on first frame or size change
    
    # Convert to grayscale for efficient comparison
    # Averaging RGB channels is fast and effective
    if len(f1.shape) == 3:
        f1 = np.mean(f1, axis=2).astype(np.uint8)
        f2 = np.mean(f2, axis=2).astype(np.uint8)
    
    # Calculate absolute difference per pixel
    # Using int16 to avoid overflow when subtracting uint8
    diff = np.abs(f1.astype(np.int16) - f2.astype(np.int16))
    
    # Count pixels that changed significantly
    # PIXEL_CHANGE_THRESHOLD (default 30) filters out noise
    changed = np.sum(diff > PIXEL_CHANGE_THRESHOLD)
    
    # Return ratio of changed pixels
    return changed / f1.size


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: IMAGE ANNOTATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# Draws bounding boxes on images for visualization.
#
# When the user searches for an object, we show them the image with:
# - Green box around the searched object (highlighted)
# - Blue boxes around other detected objects
# - Labels above each box
#
# ═══════════════════════════════════════════════════════════════════════════════

def annotate_image(memory: Memory, highlight: str = "",
                   info_text: str = "") -> bytes:
    """
    Draw bounding boxes on image with labels and info banner.

    Creates a visual result showing where objects are located.
    The searched object is highlighted in green, others in blue.
    Info text (like movement history) is shown as a banner at the bottom.

    Args:
        memory: Memory containing image and detected objects
        highlight: Object name to highlight in green (the search target)
        info_text: Text to display at bottom (location, movement, etc.)

    Returns:
        Annotated JPEG bytes, or original image on error

    Visual Example:
        ┌────────────────────────────────┐
        │  [keys]                        │
        │  ┌─────┐                       │
        │  │GREEN│  ← Searched object    │
        │  └─────┘                       │
        │           [wallet]             │
        │           ┌──────┐             │
        │           │ BLUE │ ← Other     │
        │           └──────┘             │
        ├────────────────────────────────┤
        │ 📍 kitchen   📌 counter        │
        │ moved from living room, 5m ago │
        └────────────────────────────────┘
    """
    if not memory.image_data:
        return b""
    
    try:
        # Load image from bytes
        img = Image.open(io.BytesIO(memory.image_data))
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        # Try to load a nice font, fall back to default
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
            )
        except Exception:
            font = ImageFont.load_default()
        
        # Draw each bounding box
        for obj in memory.objects:
            # Determine color: green for searched object, blue for others
            is_match = highlight.lower() in obj.name.lower() if highlight else False
            color = (0, 255, 0) if is_match else (0, 150, 255)  # Green or Blue
            thickness = 3 if is_match else 2  # Thicker for searched object
            
            # Convert normalized coordinates (0.0-1.0) to pixel coordinates
            x1, y1 = int(obj.x1 * w), int(obj.y1 * h)
            x2, y2 = int(obj.x2 * w), int(obj.y2 * h)
            
            # Draw rectangle with thickness (PIL doesn't have line width)
            for i in range(thickness):
                draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)
            
            # Draw label background and text
            label = obj.name
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # Position label above box (or below if no room)
            ly = y1 - th - 4 if y1 > th + 4 else y2 + 2
            draw.rectangle([x1, ly, x1 + tw + 4, ly + th + 4], fill=color)
            draw.text((x1 + 2, ly + 2), label, fill=(0, 0, 0), font=font)

        # Draw info banner at bottom if provided
        if info_text:
            # Use larger font for info text (readable on small LCD)
            try:
                small_font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
                )
            except Exception:
                small_font = font

            # Split text into lines that fit the image width
            lines = []
            for line in info_text.split('\n'):
                # Wrap long lines
                words = line.split()
                current_line = ""
                for word in words:
                    test = current_line + " " + word if current_line else word
                    bbox = draw.textbbox((0, 0), test, font=small_font)
                    if bbox[2] - bbox[0] < w - 10:
                        current_line = test
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)

            # Calculate banner height (larger for readability)
            line_height = 20
            banner_height = len(lines) * line_height + 10
            banner_y = h - banner_height

            # Draw semi-transparent black background
            overlay = Image.new('RGBA', (w, banner_height), (0, 0, 0, 200))
            img = img.convert('RGBA')
            img.paste(overlay, (0, banner_y), overlay)
            draw = ImageDraw.Draw(img)

            # Draw text lines
            y = banner_y + 5
            for line in lines:
                draw.text((8, y), line, fill=(255, 255, 255), font=small_font)
                y += line_height

            # Convert back to RGB for JPEG
            img = img.convert('RGB')

        # Convert back to JPEG bytes
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
        
    except Exception:
        return memory.image_data  # Return original on error


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: WHISPLAY HAT
# ═══════════════════════════════════════════════════════════════════════════════
#
# Hardware interface for the Whisplay HAT add-on board.
#
# The HAT provides:
# - 240×280 LCD display (ST7789 controller)
# - WM8960 microphone for voice input
# - RGB LED for status indication
# - Tactile button for triggering search
#
# All methods gracefully degrade if hardware is not available,
# allowing development/testing on non-Pi systems.
#
# ═══════════════════════════════════════════════════════════════════════════════

class WhisplayHAT:
    """
    Interface for Whisplay HAT hardware components.
    
    Provides unified access to:
    - LCD display (show images and text)
    - Microphone (record voice queries)
    - LED (status indication)
    - Button (trigger search)
    
    All methods are safe to call even if hardware is not present -
    they will simply do nothing (graceful degradation).
    
    Usage:
        hat = WhisplayHAT()
        hat.display_text("Ready!")
        if hat.button_pressed():
            audio = hat.record_audio()
        hat.cleanup()
    """
    
    def __init__(self, headless: bool = False):
        """
        Initialize HAT components.

        Args:
            headless: If True, skip LCD/LED but still enable mic for audio capture.
                      This allows daemon to capture audio in background mode.

        Attempts to initialize the WhisPlayBoard driver.
        If not available, methods will be no-ops.
        """
        self.board = None
        self.mic = False
        self.headless = headless

        # Always check for microphone (uses ALSA, independent of board)
        self.mic = self._check_mic()

        # In headless mode, skip board init (LCD/LED) but keep mic
        if headless:
            log("[HAT] Headless mode (mic only, no LCD/LED)")
            return

        # Try to initialize WhisPlayBoard for LCD/LED
        if WHISPLAY_AVAILABLE:
            try:
                # Clean up any stale GPIO allocation from previous runs
                # (fixes "GPIO not allocated" error after Ctrl+C)
                # Also monkey-patch PWM.__del__ to suppress cleanup errors
                self._patch_gpio_cleanup()

                self.board = WhisPlayBoard()
                self.board.set_backlight(80)       # 80% brightness
                self.board.set_rgb(0, 100, 255)   # Blue = ready
                log("[HAT] Initialized (LCD + LED + mic)")
            except Exception as e:
                log_error(f"HAT init failed: {e}")

    def _patch_gpio_cleanup(self):
        """
        Patch GPIO library to suppress PWM cleanup errors.

        The lgpio library throws TypeError when GPIO.cleanup() is called
        before PWM objects are destroyed. This patches PWM.__del__ to
        suppress those harmless errors.
        """
        try:
            import RPi.GPIO as GPIO
            GPIO.setwarnings(False)

            # Monkey-patch PWM.__del__ to suppress cleanup errors
            if hasattr(GPIO, 'PWM') and hasattr(GPIO.PWM, '__del__'):
                original_del = getattr(GPIO.PWM, '__del__')
                def safe_del(pwm_self):
                    try:
                        original_del(pwm_self)
                    except (TypeError, Exception):
                        pass  # Suppress "NoneType & int" errors
                setattr(GPIO.PWM, '__del__', safe_del)

            GPIO.cleanup()
        except Exception:
            pass
    
    def _check_mic(self) -> bool:
        """
        Check if microphone is available via ALSA.
        
        Returns:
            True if microphone detected, False otherwise
        """
        try:
            # Use arecord -l to list audio capture devices
            r = subprocess.run(
                ["arecord", "-l"],
                capture_output=True, text=True, timeout=5
            )
            # If "card" appears in output, we have audio devices
            return "card" in r.stdout.lower()
        except Exception:
            return False
    
    def _to_rgb565(self, img: Image.Image) -> bytes:
        """
        Convert PIL Image to RGB565 format for LCD using NumPy vectorization.

        The ST7789 LCD controller expects 16-bit color in RGB565 format:
        - 5 bits for red (0-31)
        - 6 bits for green (0-63)
        - 5 bits for blue (0-31)

        This implementation uses NumPy for ~10x faster conversion compared
        to Python loops (67,200 pixels at 240x280).

        Args:
            img: PIL Image in RGB mode

        Returns:
            Bytes in RGB565 format (big-endian)
        """
        # Convert to NumPy array for vectorized operations
        arr = np.array(img, dtype=np.uint16)

        # Pack 24-bit RGB (8+8+8) into 16-bit RGB565 (5+6+5):
        #
        #   8-bit RGB:    RRRRRRRR GGGGGGGG BBBBBBBB  (24 bits)
        #   16-bit RGB565: RRRRRGGG GGGBBBBB           (16 bits)
        #
        # Red:   Keep top 5 bits (& 0xF8 = 11111000), shift left 8 → bits 15-11
        # Green: Keep top 6 bits (& 0xFC = 11111100), shift left 3 → bits 10-5
        # Blue:  Keep top 5 bits by shifting right 3                → bits 4-0
        r = (arr[:, :, 0] & 0xF8) << 8   # Red:   8-bit → 5-bit, placed at bits 15-11
        g = (arr[:, :, 1] & 0xFC) << 3   # Green: 8-bit → 6-bit, placed at bits 10-5
        b = arr[:, :, 2] >> 3            # Blue:  8-bit → 5-bit, placed at bits 4-0

        # Combine all channels with bitwise OR and output as big-endian uint16
        rgb565 = (r | g | b).astype('>u2')  # '>u2' = big-endian unsigned 16-bit
        return rgb565.tobytes()
    
    def display_image(self, image_bytes: bytes, info_text: str = ""):
        """
        Display image on LCD with optional info text overlay.

        Automatically resizes and centers the image to fit the display.
        Info text is drawn AFTER scaling so it's always readable on LCD.

        Args:
            image_bytes: JPEG or PNG image data
            info_text: Optional text to overlay at bottom (location, position, etc.)
        """
        if not self.board:
            return

        try:
            # Load and convert to RGB
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # Resize to fit LCD while maintaining aspect ratio
            img.thumbnail((HAT_LCD_WIDTH, HAT_LCD_HEIGHT), Image.Resampling.LANCZOS)

            # Center on black background
            bg = Image.new('RGB', (HAT_LCD_WIDTH, HAT_LCD_HEIGHT), (0, 0, 0))
            x = (HAT_LCD_WIDTH - img.width) // 2
            y = (HAT_LCD_HEIGHT - img.height) // 2
            bg.paste(img, (x, y))

            # Draw info text overlay at bottom (AFTER scaling, so always readable)
            if info_text:
                draw = ImageDraw.Draw(bg)
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
                    )
                except Exception:
                    font = ImageFont.load_default()

                # Split into lines and draw at bottom
                lines = info_text.split('\n')[:3]  # Max 3 lines
                line_height = 22
                banner_height = len(lines) * line_height + 6
                banner_y = HAT_LCD_HEIGHT - banner_height

                # Semi-transparent black banner
                for by in range(banner_y, HAT_LCD_HEIGHT):
                    for bx in range(HAT_LCD_WIDTH):
                        r, g, b = bg.getpixel((bx, by))
                        bg.putpixel((bx, by), (r // 3, g // 3, b // 3))

                # Draw text
                ty = banner_y + 3
                for line in lines:
                    draw.text((6, ty), line, fill=(255, 255, 255), font=font)
                    ty += line_height

            # Send to display
            self.board.draw_image(0, 0, HAT_LCD_WIDTH, HAT_LCD_HEIGHT, self._to_rgb565(bg))
        except Exception as e:
            log_error(f"display_image failed: {e}")
    
    def display_text(self, text: str, color=(255, 255, 255)):
        """
        Display centered text on LCD.
        
        Supports multiple lines separated by newline characters.
        Text is automatically centered horizontally and distributed vertically.
        
        Args:
            text: Text to display (use \\n for line breaks)
            color: RGB tuple for text color (default: white)
        """
        if not self.board:
            return
            
        try:
            # Create black background
            img = Image.new('RGB', (HAT_LCD_WIDTH, HAT_LCD_HEIGHT), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Load font
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
                )
            except Exception:
                font = ImageFont.load_default()
            
            # Draw each line centered
            y = 60  # Start 60 pixels from top
            for line in text.split('\n'):
                bbox = draw.textbbox((0, 0), line, font=font)
                x = (HAT_LCD_WIDTH - (bbox[2] - bbox[0])) // 2
                draw.text((x, y), line, fill=color, font=font)
                y += 35  # 35 pixels between lines
            
            # Send to display
            self.board.draw_image(0, 0, HAT_LCD_WIDTH, HAT_LCD_HEIGHT, self._to_rgb565(img))

        except Exception as e:
            log_error(f"display_text failed: {e}")

    def record_audio(self, duration: int = RECORD_DURATION) -> bytes | None:
        """
        Record audio from WM8960 microphone.
        
        Uses ALSA's arecord utility to capture audio.
        LED turns red during recording to provide visual feedback.
        
        Args:
            duration: Recording duration in seconds (default: 3)
            
        Returns:
            WAV audio bytes (16kHz, 16-bit, mono) or None on error
        """
        if not self.mic:
            return None
        
        log(f"🎤 Recording {duration}s...")
        
        # Visual feedback: red LED during recording
        if self.board:
            self.board.set_rgb(255, 0, 0)  # Red
        
        try:
            # Record using ALSA
            # WM8960 is card 0 on Pi with Whisplay HAT
            result = subprocess.run([
                "arecord",
                "-D", "plughw:0,0",   # WM8960 device (card 0)
                "-f", "S16_LE",       # 16-bit signed little-endian
                "-r", "16000",        # 16kHz sample rate (good for speech)
                "-c", "1",            # Mono (single channel)
                "-t", "wav",          # WAV format with header
                "-d", str(duration),  # Duration in seconds
                "-q",                 # Quiet (no progress output)
                "-"                   # Output to stdout
            ], capture_output=True, timeout=duration + 5)
            
            # Reset LED to blue
            if self.board:
                self.board.set_rgb(0, 100, 255)
            
            # Check if we got valid audio (WAV files are at least 44 bytes)
            if result.returncode == 0 and len(result.stdout) > 10000:
                log(f"   Recorded {len(result.stdout)//1024}KB")
                return result.stdout
                
        except Exception:
            if self.board:
                self.board.set_rgb(0, 100, 255)
        
        return None

    def speak(self, text: str, blocking: bool = False, gemini: 'GeminiClient | None' = None, for_search: bool = False):
        """
        Speak text aloud using Gemini 3 TTS.

        Uses Gemini 3's native text-to-speech for natural voice quality.
        This is IN ADDITION to LCD display - provides redundant feedback
        for users with vision problems or when not looking at screen.

        Args:
            text: Text to speak (keep short for responsiveness)
            blocking: If True, wait for speech to complete (default: False)
            gemini: GeminiClient instance for TTS (required)
            for_search: If True, use SEARCH_TTS_ENABLED setting (default: False)
        """
        # Check appropriate TTS setting
        tts_enabled = SEARCH_TTS_ENABLED if for_search else TTS_ENABLED
        if not tts_enabled or not gemini:
            return

        # Clean text for speech (remove special characters)
        clean_text = text.replace('\n', ' ').replace('📍', '').replace('📌', '')
        clean_text = clean_text.replace('🕐', '').replace('📋', '').strip()

        if not clean_text:
            return

        # Limit length for responsiveness
        if len(clean_text) > 200:
            clean_text = clean_text[:197] + "..."

        try:
            audio_data = gemini.text_to_speech(clean_text)
            if audio_data:
                # Save to temp file and play with aplay
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_data)
                    temp_path = f.name

                if blocking:
                    subprocess.run(["aplay", "-D", "plughw:0,0", "-q", temp_path], timeout=30, capture_output=True)
                    os.unlink(temp_path)
                else:
                    # Play and cleanup in background
                    subprocess.Popen(
                        f"aplay -D plughw:0,0 -q {temp_path} && rm {temp_path}",
                        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
        except Exception as e:
            log_error(f"Gemini TTS failed: {e}")

    def button_pressed(self) -> bool:
        """
        Check if button is currently pressed.

        Returns:
            True if button is pressed, False otherwise
        """
        return self.board.button_pressed() if self.board else False
    
    def cleanup(self):
        """
        Turn off LED, backlight, and release GPIO.

        Should be called when shutting down to avoid "GPIO not allocated"
        errors on subsequent runs.
        """
        if self.board:
            try:
                self.board.set_rgb(0, 0, 0)      # LED off
                self.board.set_backlight(0)      # Backlight off
            except Exception:
                pass

            # Stop PWM explicitly before GPIO cleanup to avoid
            # "TypeError: unsupported operand type(s) for &" errors
            try:
                # Access internal PWM objects and stop them gracefully
                if hasattr(self.board, '_backlight_pwm'):
                    self.board._backlight_pwm.stop()
                if hasattr(self.board, 'backlight'):
                    self.board.backlight.stop()
            except Exception:
                pass

            # Clear reference to board to help garbage collection
            self.board = None

        # Suppress stderr during GPIO cleanup to hide PWM __del__ errors
        # (These are harmless but ugly - the lgpio library doesn't handle
        # cleanup order gracefully)
        try:
            # Redirect stderr to /dev/null during cleanup
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stderr = os.dup(2)
            os.dup2(devnull, 2)
            os.close(devnull)

            import RPi.GPIO as GPIO
            GPIO.cleanup()

            # Restore stderr
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: GEMINI 3 CLIENT
# ═══════════════════════════════════════════════════════════════════════════════
#
# Unified client for all Gemini 3 API operations.
# This is the BRAIN of GEM - all intelligence comes from here.
#
# CAPABILITIES:
# 1. Vision - Analyze images, detect objects with bounding boxes
# 2. Audio STT - Transcribe voice queries
# 3. NLU - Extract object names from natural language
# 4. Narrative - Generate causal explanations
# 5. TTS - Text-to-speech for audio feedback
#
# OPTIMIZATIONS:
# - thinking_level=MINIMAL: Skip extended reasoning for faster response
# - Streaming API: Get first token faster
#
# ═══════════════════════════════════════════════════════════════════════════════

# API Rate Limit Retry Configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 16.0     # seconds

# API call tracking
_api_call_count = 0
_api_call_start_time = None


def retry_api_call(func, *args, max_retries: int = MAX_RETRIES, **kwargs):
    """
    Retry an API call with exponential backoff on rate limit errors.

    Handles 429 (rate limit) and 503 (service unavailable) errors
    by waiting and retrying with exponential backoff + jitter.

    Args:
        func: The API function to call
        *args: Positional arguments to pass to func
        max_retries: Maximum retry attempts (default: 3)
        **kwargs: Keyword arguments to pass to func

    Returns:
        The result of func(*args, **kwargs)

    Raises:
        The original exception if all retries are exhausted
    """
    global _api_call_count, _api_call_start_time

    # Initialize start time on first call
    if _api_call_start_time is None:
        _api_call_start_time = time.time()

    last_exception = None
    backoff = INITIAL_BACKOFF

    for attempt in range(max_retries + 1):
        try:
            _api_call_count += 1
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit or overload errors
            is_retryable = (
                "429" in error_str or
                "rate" in error_str or
                "quota" in error_str or
                "503" in error_str or
                "overloaded" in error_str or
                "resource exhausted" in error_str
            )

            if not is_retryable or attempt >= max_retries:
                raise

            last_exception = e
            # Add jitter: backoff * (0.5 to 1.5)
            sleep_time = backoff * (0.5 + random.random())
            sleep_time = min(sleep_time, MAX_BACKOFF)

            log(f"[RETRY] API rate limited, waiting {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(sleep_time)
            backoff *= 2  # Exponential backoff

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


def get_api_stats() -> tuple[int, float]:
    """Get API call statistics: (call_count, calls_per_minute)."""
    global _api_call_count, _api_call_start_time
    if _api_call_start_time is None:
        return (0, 0.0)
    elapsed = time.time() - _api_call_start_time
    calls_per_min = (_api_call_count / elapsed * 60) if elapsed > 0 else 0.0
    return (_api_call_count, calls_per_min)


# Vision prompt - zero-shot object detection for any item
# Detects ANY object in scene, not limited to predefined list
VISION_PROMPT = """Analyze this image for EPISODIC MEMORY. Detect objects, activities, AND people. Return JSON:
{"location":"scene","description":"scene description","tags":["tag1"],"activities":["activity1"],"objects":[{"label":"obj","box_2d":[ymin,xmin,ymax,xmax],"confidence":0.9,"context":"on table","attached":false}],"persons":[{"description":"person description","box_2d":[ymin,xmin,ymax,xmax],"context":"what they are doing"}],"relationships":["keys on desk"]}

ZERO-SHOT DETECTION: Identify ANY object - everyday items, electronics, food, drinks, medication, documents, clothing, tools, containers, etc.

PERSON DETECTION (WHO dimension): For each person visible, provide:
- description: Visual appearance (e.g., "man in blue shirt", "woman with glasses", "child in red jacket")
- box_2d: Bounding box coordinates
- context: What they are doing or where they are (e.g., "sitting at desk", "standing by door", "holding phone")
Do NOT include "person" in the objects list - use the separate "persons" array instead.

ACTIVITY DETECTION: Identify activities happening in the scene. Look for:
- Taking/consuming: "taking medication", "drinking coffee", "eating breakfast"
- Placing/moving: "putting keys on counter", "placing phone on desk"
- Using/working: "reading book", "typing on laptop", "watching TV"
- Wearing/removing: "putting on glasses", "taking off watch"
- Entering/leaving: "entering kitchen", "leaving room"

box_2d: normalized 0-1000 coordinates [ymin, xmin, ymax, xmax].
label: specific object name (e.g., "pill bottle", "coffee mug", "car keys").
confidence: detection confidence 0.0-1.0.
context: where/how object is positioned (e.g., "on kitchen counter", "in hand").
attached: TRUE only if WORN on body (glasses on face, watch on wrist, jewelry). FALSE for handheld items (phone in hand, cup, book, keys).
activities: list of activities detected (e.g., ["taking medication", "drinking water"]).
persons: list of people with visual descriptions (for WHO dimension).
tags: scene category words (e.g., "kitchen", "desk", "bedroom").
relationships: spatial relations between objects.

Scan the ENTIRE image thoroughly. Report every object, person, AND activity you can identify."""


class GeminiClient:
    """
    Unified Gemini 3 API client for all AI operations.
    
    This class encapsulates ALL interaction with Google's Gemini 3 API.
    It provides methods for:
    
    1. Vision Analysis:
       - Input: JPEG image
       - Output: Objects with bounding boxes, scene description

    2. Audio Transcription:
       - Input: WAV audio
       - Output: Transcribed text

    3. Query NLU:
       - Input: Natural language query ("Where are my car keys?")
       - Output: Object name ("keys")

    4. Narrative Generation:
       - Input: Object name, movement history, current location
       - Output: Causal explanation of object's journey

    5. Text-to-Speech:
       - Input: Text string
       - Output: WAV audio bytes for spoken feedback
    
    OPTIMIZATION NOTES:
    
    All methods use thinking_level=MINIMAL to skip extended reasoning.
    This reduces latency with minimal quality loss
    for our simple extraction tasks.
    
    Usage:
        gemini = GeminiClient()
        result = gemini.analyze_image(jpeg_bytes)
        transcript = gemini.transcribe_audio(wav_bytes)
    """
    
    def __init__(self):
        """
        Initialize Gemini client with API key.
        
        Raises:
            ValueError: If GEMINI_API_KEY environment variable is not set
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set!\n"
                "Get key at: https://aistudio.google.com/apikey\n"
                "Then: export GEMINI_API_KEY=your_key"
            )
        
        # Initialize the Gemini client
        self.client = genai.Client(api_key=api_key)
        
        log(f"[GEMINI] Connected")
        log(f"   Vision: {VISION_MODEL}")
        log(f"   Audio/NLU: {AUDIO_MODEL}")
    
    # -------------------------------------------------------------------------
    # Vision: Object Detection with Bounding Boxes
    # -------------------------------------------------------------------------
    
    def analyze_image(self, image_data: bytes) -> dict:
        """
        Analyze image using Gemini 3 native vision with streaming.

        This is the CORE CAPABILITY of GEM. Gemini 3's vision model
        identifies objects and their locations in the image.

        OPTIMIZATIONS (Gemini 3):
        - thinking_level=MINIMAL: Fastest response, minimal reasoning
        - response_mime_type=json: Structured output, no parsing errors
        - Streaming: Get first token faster

        Args:
            image_data: JPEG image bytes

        Returns:
            dict with keys:
            - location: Scene description (e.g., "kitchen counter")
            - description: Detailed scene description
            - objects: List of {name, x1, y1, x2, y2, confidence}
        """
        def _do_vision_call():
            # Use streaming for faster first token
            chunks = []
            for chunk in self.client.models.generate_content_stream(
                model=VISION_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=VISION_PROMPT),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/jpeg",
                                    data=image_data
                                )
                            )
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    # GEMINI 3 OPTIMIZATION: minimal thinking for faster response
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL
                    )
                )
            ):
                if chunk.text:
                    chunks.append(chunk.text)
            return "".join(chunks).strip()

        try:
            text = retry_api_call(_do_vision_call)
            if not text:
                return {"location": "unknown", "description": "", "objects": []}
            # Debug: show raw response to diagnose detection issues
            log(f"[VISION RAW] {text[:300]}...")
            result = self._parse_vision_json(text)
            log(f"[VISION] {len(result['objects'])} objects detected")
            if result['objects']:
                obj_summary = ", ".join(f"{o.get('label','?')}({o.get('confidence',0):.0%})" for o in result['objects'][:5])
                log(f"[VISION] Objects: {obj_summary}")
            return result

        except Exception as e:
            log_error(f"Vision failed: {e}")
            return {"location": "unknown", "description": "", "objects": []}

    def _parse_vision_json(self, text: str) -> dict:
        """
        Robust JSON parser for Gemini vision responses using json5.

        json5 handles common LLM JSON issues automatically:
        - Trailing commas
        - Unquoted keys
        - Single quotes
        - Comments
        - And more

        Falls back to regex extraction if json5 also fails.
        """
        default = {"location": "unknown", "description": "", "objects": []}

        if not text or not text.strip():
            return default

        text = str(text).strip()

        # Clean markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        # Sanitize control characters that break any parser
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        # Try json5 (lenient parser handles most LLM quirks)
        try:
            result = json5.loads(text)
            return self._normalize_vision_result(result)
        except Exception:
            pass

        # Last resort: regex extraction of key fields
        extracted = default.copy()

        loc_match = re.search(r'"location"\s*:\s*"([^"]*)"', text)
        if loc_match:
            extracted["location"] = loc_match.group(1)

        desc_match = re.search(r'"description"\s*:\s*"([^"]*)"', text)
        if desc_match:
            extracted["description"] = desc_match.group(1)

        obj_names = re.findall(r'"name"\s*:\s*"([^"]*)"', text)
        if obj_names:
            extracted["objects"] = [{"name": name, "box_2d": [0, 0, 1000, 1000]} for name in obj_names]

        return extracted

    def _normalize_vision_result(self, result) -> dict:
        """Normalize parsed JSON to expected format."""
        default = {"location": "unknown", "description": "", "objects": []}

        # Handle list responses
        if isinstance(result, list):
            if result and isinstance(result[0], dict) and "location" in result[0]:
                result = result[0]
            else:
                return {"objects": result, "location": "unknown", "description": ""}

        # Ensure it's a dict
        if not isinstance(result, dict):
            return default

        # Ensure required keys
        result.setdefault("objects", [])
        result.setdefault("location", "unknown")
        result.setdefault("description", "")

        return result
    
    # -------------------------------------------------------------------------
    # Audio: Speech-to-Text (Gemini native)
    # -------------------------------------------------------------------------
    
    def transcribe_audio(self, audio_data: bytes) -> str | None:
        """
        Transcribe audio using Gemini's native audio understanding with streaming.

        Gemini 3 can directly process audio without a separate STT service.

        OPTIMIZATIONS:
        - thinking_level=MINIMAL: Fastest transcription
        - Streaming: Get first words faster

        Args:
            audio_data: WAV bytes (16kHz, 16-bit, mono)

        Returns:
            Transcribed text, or None on error
        """
        def _do_transcribe():
            # Domain-specific prompt for memory assistance queries
            stt_prompt = """Transcribe the speech in this audio. If no speech, reply: [silence]
Context: User is asking about finding objects or recalling activities. They may ask about ANY item - everyday objects, medication, food, documents, electronics, clothing. Common phrases: "where is my", "where did I put", "have you seen", "find my", "did I take", "what did I", "when did I"."""
            chunks = []
            for chunk in self.client.models.generate_content_stream(
                model=AUDIO_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=stt_prompt),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="audio/wav",
                                    data=audio_data
                                )
                            )
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL
                    )
                )
            ):
                if chunk.text:
                    chunks.append(chunk.text)
            return "".join(chunks).strip()

        try:
            transcript = retry_api_call(_do_transcribe)
            log(f"[STT] \"{transcript}\"")
            return transcript

        except Exception as e:
            log_error(f"Transcription failed: {e}")
            return None

    def extract_people_from_transcript(self, transcript: str) -> tuple[list[str], str]:
        """
        Extract people's names and conversation context from a transcript.

        This enables the WHO dimension of episodic memory - tracking people
        encountered during the day for "who did I meet?" type queries.

        OPTIMIZATIONS:
        - thinking_level=MINIMAL: Fast extraction
        - JSON output: Structured response
        - Short prompt: Minimize tokens

        Args:
            transcript: Transcribed speech text

        Returns:
            Tuple of (list of names, conversation context summary)
            e.g. (["John", "Sarah"], "discussed lunch plans")
        """
        if not transcript or transcript.strip() == "[silence]":
            return [], ""

        def _do_extract():
            prompt = f'''Extract people and context from this conversation transcript:
"{transcript}"

Return JSON with:
- "people": list of names mentioned or introduced (empty list if none)
- "context": brief summary of what was discussed (empty string if unclear)

Example: {{"people": ["John", "Dr. Smith"], "context": "discussed appointment time"}}
If no names found: {{"people": [], "context": "..."}}'''

            chunks = []
            for chunk in self.client.models.generate_content_stream(
                model=AUDIO_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=prompt)]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL
                    )
                )
            ):
                if chunk.text:
                    chunks.append(chunk.text)
            return "".join(chunks).strip()

        try:
            result_text = retry_api_call(_do_extract)
            if not result_text:
                return [], ""

            # Parse JSON response
            result = json5.loads(result_text)
            people = result.get("people", [])
            context = result.get("context", "")

            # Ensure people is a list of strings
            if isinstance(people, list):
                people = [str(p).strip() for p in people if p]
            else:
                people = []

            if people:
                log(f"[PEOPLE] Extracted: {', '.join(people)}")
            if context:
                log(f"[CONTEXT] {context[:50]}...")

            return people, str(context)

        except Exception as e:
            log_error(f"People extraction failed: {e}")
            return [], ""

    def text_to_speech(self, text: str) -> bytes | None:
        """
        Convert text to speech using Gemini's audio generation.

        Uses Gemini 3's native TTS capability for natural-sounding speech.
        Returns WAV audio bytes that can be played via aplay.

        Args:
            text: Text to convert to speech

        Returns:
            WAV audio bytes, or None on error
        """
        if not text or len(text.strip()) == 0:
            return None

        # Limit text length for reasonable audio duration
        if len(text) > 200:
            text = text[:197] + "..."

        def _do_tts():
            response = self.client.models.generate_content(
                model=AUDIO_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"Read this aloud naturally: {text}")]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Kore"  # Natural female voice
                            )
                        )
                    )
                )
            )
            # Extract audio data from response
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        return part.inline_data.data
            return None

        try:
            audio_data = retry_api_call(_do_tts)
            if audio_data:
                log(f"[TTS] Generated {len(audio_data)//1024}KB audio")
                return audio_data
            return None
        except Exception as e:
            log_error(f"Gemini TTS failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # NLU: Natural Language Query Understanding
    # -------------------------------------------------------------------------
    
    def extract_object(self, query: str) -> str:
        """
        Extract object name from natural language query with streaming.

        OPTIMIZATIONS:
        - Shorter prompt
        - thinking_level=MINIMAL
        - max_output_tokens=10 (object names are very short)

        Also normalizes synonyms to canonical names (eyeglasses→glasses, mobile→phone)
        so search matches stored objects consistently.

        Args:
            query: Natural language question

        Returns:
            Extracted object name suitable for search (normalized)
        """
        def _do_extract():
            chunks = []
            prompt = f'''Extract object from: "{query}"
Return ONLY the normalized object name. Use these standard names:
- glasses (not eyeglasses/spectacles/specs)
- phone (not mobile/cellphone/smartphone)
- bowl (not breakfast bowl/cereal bowl/soup bowl)
- cup (not coffee cup/tea cup)
- mug (not coffee mug)
- plate (not dish/dinner plate)
- keys (not key/keychain)
- laptop (not notebook/computer)
- headphones (not earphones/earbuds)
- remote (not tv remote/controller)
- wallet (not purse)
- pen (not pencil/marker)
- charger (not charging cable/power adapter)'''
            for chunk in self.client.models.generate_content_stream(
                model=AUDIO_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=prompt)]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=10,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL
                    )
                )
            ):
                if chunk.text:
                    chunks.append(chunk.text)
            return "".join(chunks).strip()

        try:
            raw_obj = retry_api_call(_do_extract)
            obj = raw_obj.lower().split('\n')[0]
            obj = obj.replace('"', '').replace("'", "")
            log(f"[NLU] \"{query}\" → \"{obj}\"")
            return obj

        except Exception as e:
            log_error(f"NLU failed: {e}")
            return query.split()[-1] if query else ""
    
    def understand_query(self, query: str) -> dict:
        """
        Use Gemini 3 NLU to fully understand a user query.

        This is the CORE NLU for episodic memory queries. Gemini 3 understands:
        - Query intent (object/scene/time/person/near/activity)
        - Entity extraction (object name, person name, location, time expression)
        - Time parsing (converts "this morning" to actual datetime range)
        - Synonym normalization (eyeglasses → glasses)

        This replaces hardcoded regex patterns with Gemini's language understanding.

        Args:
            query: Natural language query (e.g., "where did I put my keys this morning?")

        Returns:
            dict with keys:
            - type: "object" | "scene" | "time" | "person" | "near" | "activity" | "vqa"
            - entity: extracted entity (object/person/location/activity name)
            - time_start: ISO datetime string if time mentioned (or None)
            - time_end: ISO datetime string if time mentioned (or None)
            - query: original query
        """
        now = datetime.now()

        def _do_understand():
            prompt = f'''Analyze this episodic memory query. Current time: {now.strftime("%Y-%m-%d %H:%M")}

Query: "{query}"

Classify the query type and extract relevant information. Return JSON:

{{
  "type": "object|scene|time|person|near|activity|vqa",
  "entity": "extracted name or empty string",
  "question": "the visual question if type=vqa, else null",
  "placed": true or false,
  "time_start": "ISO datetime or null",
  "time_end": "ISO datetime or null"
}}

Query types:
- "object": looking for a physical item's LOCATION (where are my keys, find my wallet)

IMPORTANT "placed" field (for object queries):
- placed=true: User wants to know where they PUT DOWN/LEFT the object (not in hand)
  Examples: "where did I LEAVE my keys?", "where did I PUT my phone?", "where did I set down my glasses?"
- placed=false: User just wants to find the object (any location including in hand)
  Examples: "where are my keys?", "find my phone", "where is my wallet?"
- "activity": asking if an ACTION was performed (did I take medication, did I eat, did I lock the door)
- "scene": asking about a location/place (what was on the kitchen counter)
- "time": asking about activities in a time period (what did I do, what happened)
- "person": asking about people (who did I meet, did I see someone)
- "near": asking about co-located objects (what was near/with something)
- "vqa": asking about VISUAL PROPERTIES of an object (what color, how many, what brand, what size)

IMPORTANT: Use "vqa" type for questions about visual properties like color, size, count, brand, material.
Use "activity" type for "did I..." questions about ACTIONS (take, eat, drink, do, lock, turn off, etc.)
Use "object" type for "where is..." questions about LOCATIONS.

Entity extraction:
- For object: extract the item name, normalize synonyms (eyeglasses→glasses, mobile→phone)
- For activity: extract the action+object (e.g., "taking medication", "eating breakfast", "locking door")
- For person: extract person's name, or empty string for "who did I meet?"
- For scene: extract the location name
- For time: extract what they're looking for (empty if just asking about activities)
- For near: extract the reference object
- For vqa: extract the object name, and put the full question in "question" field

Time parsing (convert relative to absolute using current time {now.strftime("%Y-%m-%d %H:%M")}):
- "this morning" → today 06:00 to 12:00
- "this afternoon" → today 12:00 to 18:00
- "this evening/tonight" → today 18:00 to 23:59
- "yesterday" → yesterday 00:00 to 23:59
- "last hour" → 1 hour ago to now
- "last N hours" → N hours ago to now
- If no time mentioned, set both to null

Examples:
- "where are my keys?" → {{"type":"object","entity":"keys","question":null,"placed":false,"time_start":null,"time_end":null}}
- "where did I leave my keys?" → {{"type":"object","entity":"keys","question":null,"placed":true,"time_start":null,"time_end":null}}
- "where did I put my phone?" → {{"type":"object","entity":"phone","question":null,"placed":true,"time_start":null,"time_end":null}}
- "find my wallet" → {{"type":"object","entity":"wallet","question":null,"placed":false,"time_start":null,"time_end":null}}
- "did I take my medication this morning?" → {{"type":"activity","entity":"taking medication","question":null,"placed":false,"time_start":"{now.strftime('%Y-%m-%d')} 06:00","time_end":"{now.strftime('%Y-%m-%d')} 12:00"}}
- "did I eat breakfast?" → {{"type":"activity","entity":"eating breakfast","question":null,"placed":false,"time_start":null,"time_end":null}}
- "did I lock the door?" → {{"type":"activity","entity":"locking door","question":null,"placed":false,"time_start":null,"time_end":null}}
- "who did I meet today?" → {{"type":"person","entity":"","question":null,"placed":false,"time_start":"{now.strftime('%Y-%m-%d')} 00:00","time_end":"{now.isoformat()}"}}
- "what was on the kitchen counter?" → {{"type":"scene","entity":"kitchen counter","question":null,"placed":false,"time_start":null,"time_end":null}}
- "what was near my wallet?" → {{"type":"near","entity":"wallet","question":null,"placed":false,"time_start":null,"time_end":null}}
- "what color is the chair?" → {{"type":"vqa","entity":"chair","question":"what color is the chair?","placed":false,"time_start":null,"time_end":null}}
- "how many boxes are there?" → {{"type":"vqa","entity":"boxes","question":"how many boxes are there?","placed":false,"time_start":null,"time_end":null}}
- "what brand is the laptop?" → {{"type":"vqa","entity":"laptop","question":"what brand is the laptop?","placed":false,"time_start":null,"time_end":null}}'''

            chunks = []
            for chunk in self.client.models.generate_content_stream(
                model=AUDIO_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=prompt)]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL
                    )
                )
            ):
                if chunk.text:
                    chunks.append(chunk.text)
            return "".join(chunks).strip()

        try:
            result_text = retry_api_call(_do_understand)
            result = json5.loads(result_text)

            # Ensure required fields
            query_type = result.get("type", "object")
            entity = result.get("entity", "")
            time_start = result.get("time_start")
            time_end = result.get("time_end")

            log(f"[NLU] \"{query}\" → type={query_type}, entity=\"{entity}\"")
            if time_start:
                log(f"   Time: {time_start} to {time_end}")

            return {
                "type": query_type,
                "entity": entity.lower().strip() if entity else "",
                "time_start": time_start,
                "time_end": time_end,
                "query": query
            }

        except Exception as e:
            log_error(f"Query understanding failed: {e}")
            # Fallback: treat as object query with full query as entity
            return {
                "type": "object",
                "entity": query.split()[-1].lower() if query else "",
                "time_start": None,
                "time_end": None,
                "query": query
            }

    def suggest_locations(self, obj_name: str, context: str = "") -> list[str]:
        """
        Use Gemini to suggest where an object might be when not found.

        Leverages Gemini's world knowledge about common object locations.
        This is smarter than hardcoded suggestions because it considers context.

        Args:
            obj_name: Object being searched for
            context: Optional context (e.g., last known location, time of day)

        Returns:
            List of 3-5 suggested locations to check
        """
        def _do_suggest():
            prompt = f'''Where might someone find their {obj_name}?

Context: {context if context else "No additional context"}

List 3-5 common locations where people typically leave or store this item.
Return JSON array of location names, most likely first.

Example for "keys": ["front door hook", "kitchen counter", "coat pocket", "desk", "car"]'''

            chunks = []
            for chunk in self.client.models.generate_content_stream(
                model=AUDIO_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=prompt)]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL
                    )
                )
            ):
                if chunk.text:
                    chunks.append(chunk.text)
            return "".join(chunks).strip()

        try:
            result_text = retry_api_call(_do_suggest)
            suggestions = json5.loads(result_text)
            if isinstance(suggestions, list):
                log(f"[SUGGEST] {obj_name} → {suggestions[:3]}")
                return suggestions[:5]
            return []
        except Exception as e:
            log_error(f"Suggestion failed: {e}")
            return []

    def answer_visual_question(self, image_data: bytes, question: str) -> str:
        """
        Answer a visual question about an image (VQA).

        This enables answering questions about visual properties of objects
        in stored memories, such as:
        - "What color is the chair?"
        - "How many boxes are there?"
        - "What brand is the laptop?"

        Args:
            image_data: JPEG image bytes
            question: The visual question to answer

        Returns:
            Natural language answer to the question
        """
        def _do_vqa():
            chunks = []
            for chunk in self.client.models.generate_content_stream(
                model=VISION_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                            types.Part(text=f'''Answer this question about the image:

Question: {question}

Provide a direct, concise answer. If you cannot determine the answer from the image, say "Cannot determine from image."''')
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=100,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL
                    )
                )
            ):
                if chunk.text:
                    chunks.append(chunk.text)
            return "".join(chunks).strip()

        try:
            answer = retry_api_call(_do_vqa)
            if answer:
                log(f"[VQA] {question} → {answer}")
                return answer
            return "Unable to answer question about image."
        except Exception as e:
            log_error(f"VQA failed: {e}")
            return "Unable to answer question about image."

    # -------------------------------------------------------------------------
    # Temporal Reasoning: Causal Narrative Generation
    # -------------------------------------------------------------------------

    def generate_causal_narrative(self, obj_name: str, movements: list,
                                   current_location: str, current_position: str) -> str:
        """
        Generate a causal narrative explaining object's journey with streaming.

        OPTIMIZATIONS:
        - thinking_level=LOW (needs some reasoning for cause-and-effect)
        - Shorter prompt
        - max_output_tokens=100

        Args:
            obj_name: Object being searched for
            movements: List of ObjectMovement records (newest first)
            current_location: Current scene description
            current_position: Current position in frame

        Returns:
            Natural language causal narrative
        """
        if not movements:
            return f"Your {obj_name} is at the {current_position} of your {current_location}."

        # Build compact movement summary
        movement_text = "; ".join([
            f"{m.from_location}→{m.to_location} ({m.to_time[:16]})"
            for m in movements[:3]  # Last 3 movements only
        ])

        def _do_narrative():
            chunks = []
            for chunk in self.client.models.generate_content_stream(
                model=AUDIO_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"""Object: {obj_name}
Now: {current_location} ({current_position})
History: {movement_text}

Reply in 2 sentences: where it is and how it got there.""")]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=100,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                    )
                )
            ):
                if chunk.text:
                    chunks.append(chunk.text)
            return "".join(chunks).strip()

        try:
            return retry_api_call(_do_narrative)

        except Exception as e:
            log_error(f"Narrative generation failed: {e}")
            return f"Your {obj_name} is at the {current_position} of your {current_location}."

    def generate_activity_summary(self, memories: list, time_period: str) -> str:
        """
        Generate a natural activity narrative from memories.

        For users with memory problems, this helps answer "What did I do this morning?"
        with a friendly, easy-to-understand summary of activities.

        Args:
            memories: List of Memory objects from a time period
            time_period: Human-readable time ("this morning", "yesterday")

        Returns:
            Natural language narrative summarizing activities
        """
        if not memories:
            return f"I don't have any memories from {time_period}."

        # Build compact summary of memories for the prompt
        # Limit to 10 memories to keep prompt size reasonable on Pi Zero
        memory_summaries = []
        for mem in memories[:10]:
            time_str = mem.timestamp[11:16] if len(mem.timestamp) > 16 else mem.timestamp
            objs = ", ".join(mem.object_names()[:4]) or "no objects"
            memory_summaries.append(f"- {time_str} at {mem.location}: {objs}")

        memories_text = "\n".join(memory_summaries)

        def _do_summary():
            chunks = []
            for chunk in self.client.models.generate_content_stream(
                model=AUDIO_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"""These are memory snapshots from {time_period}:
{memories_text}

Summarize the person's activities in 2-3 simple sentences.
Focus on what they were doing based on objects and locations.
Use friendly, easy-to-understand language.""")]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=120,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                    )
                )
            ):
                if chunk.text:
                    chunks.append(chunk.text)
            return "".join(chunks).strip()

        try:
            result = retry_api_call(_do_summary)
            if result:
                return result
            # Fallback if empty response
            locations = list(set(m.location for m in memories[:5]))
            return f"During {time_period}, you were at: {', '.join(locations)}."
        except Exception as e:
            log_error(f"Activity summary failed: {e}")
            # Fallback: simple list of locations
            locations = list(set(m.location for m in memories[:5]))
            return f"During {time_period}, you were at: {', '.join(locations)}."



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: MEMORY INDEX
# ═══════════════════════════════════════════════════════════════════════════════
#
# JSON-based memory index with O(1) object lookup, time-based search,
# co-occurrence search, and retrieval reinforcement.
#
# Search capabilities:
#   - find_by_object():     O(1) hash lookup + fuzzy plural/compound matching
#   - find_by_location():   Partial string match on scene location
#   - find_by_time():       Datetime window filtering (fuzzy time expressions)
#   - find_cooccurrence():  Objects seen in the same memory frame
#
# Memory lifecycle:
#   - record_access():      Track search hits (retrieval reinforcement)
#   - decay_score():        Score = recency + retrieval boost (Ebbinghaus)
#   - Used by cleanup_old_memories() to decide what to forget
#
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryIndex:
    """
    JSON-based memory index with O(1) object and person lookup.

    DATA STRUCTURE:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  by_object (Hash Index)           memories (Metadata Cache)         │
    │  ─────────────────────            ─────────────────────             │
    │  "keys"    → {mem_001, mem_005}   mem_001 → {location, objects...}  │
    │  "wallet"  → {mem_002, mem_003}   mem_002 → {location, objects...}  │
    │  "phone"   → {mem_001, mem_004}   mem_003 → {location, objects...}  │
    │                                                                      │
    │  by_person (Hash Index) - WHO dimension of episodic memory          │
    │  ───────────────────────                                             │
    │  "john"    → {mem_002, mem_005}   (memories where John was present) │
    │  "sarah"   → {mem_003}            (memories where Sarah was present)│
    └─────────────────────────────────────────────────────────────────────┘

    ALGORITHM FOR SEARCH:
    1. Query: "keys"
    2. Lookup in by_object hash: O(1) → {mem_001, mem_005}
    3. Return newest memory ID (sorted by timestamp)
    4. Load full memory from disk if needed

    ALGORITHM FOR PARTIAL MATCH:
    1. Query: "key" (not exact match)
    2. Iterate by_object keys, find "keys" contains "key"
    3. Return matching memory IDs

    WHY NOT USE A DATABASE?
    - Pi Zero 2W has limited RAM (512MB)
    - JSON file is simple and human-readable for debugging
    - Hash index provides O(1) lookup for exact matches
    - Partial match is O(n) but n is small (typically <1000 objects)

    Attributes:
        by_object: Hash index mapping object names to memory IDs (O(1) lookup)
        by_person: Hash index mapping person names to memory IDs (WHO dimension)
        by_activity: Hash index mapping activities to memory IDs (WHAT dimension)
        memories: Cached metadata for quick access without disk reads
        access_log: Tracks which memories are accessed (for cleanup priority)
        index_file: Path to JSON persistence file
    """

    def __init__(self, gemini: GeminiClient):
        """
        Initialize memory index and load existing data.

        Args:
            gemini: GeminiClient instance (for potential embedding operations)
        """
        self.gemini = gemini

        # Hash index: object_name → set of memory IDs containing that object
        # Enables O(1) lookup: "Where are my keys?" → instantly find memories
        self.by_object: dict[str, set] = {}

        # Hash index: person_name → set of memory IDs where person was present
        # Enables O(1) lookup for WHO dimension: "Who did I meet?" queries
        self.by_person: dict[str, set] = {}

        # Hash index: activity → set of memory IDs where activity occurred
        # Enables O(1) lookup: "Did I take medication?" → find memories with that activity
        self.by_activity: dict[str, set] = {}

        # Metadata cache: memory_id → {location, objects, timestamp, ...}
        # Avoids disk reads for common operations
        self.memories: dict[str, dict] = {}

        # Access tracking: memory_id → {access_count, last_accessed}
        # Used for smart cleanup - frequently accessed memories survive longer
        self.access_log: dict[str, dict] = {}

        # Persistence file path
        self.index_file = MEMORY_DIR / "memory_index.json"

        # Load existing index from disk
        self._load()
        log(f"[INDEX] {len(self.memories)} memories loaded")

    def _load(self):
        """Load existing memories from JSON index file, or scan directory."""
        # Try loading from index file first
        if self.index_file.exists():
            try:
                data = json5.loads(self.index_file.read_text())
                self.memories = data.get("memories", {})
                self.access_log = data.get("access_log", {})
                # Rebuild object index
                for mem_id, meta in self.memories.items():
                    for obj in meta.get('objects', '').split(','):
                        obj = obj.strip().lower()
                        if obj:
                            self.by_object.setdefault(obj, set()).add(mem_id)
                    # Rebuild person index (WHO dimension - audio names)
                    for person in meta.get('people', '').split(','):
                        person = person.strip().lower()
                        if person:
                            self.by_person.setdefault(person, set()).add(mem_id)
                    # Rebuild visual person index (WHO dimension - visual)
                    for person_desc in meta.get('persons', '').split(';'):
                        person_desc = person_desc.strip().lower()
                        if person_desc:
                            self.by_person.setdefault(person_desc, set()).add(mem_id)
                    # Rebuild activity index (WHAT dimension)
                    for activity in meta.get('activities', '').split(','):
                        activity = activity.strip().lower()
                        if activity:
                            self.by_activity.setdefault(activity, set()).add(mem_id)
                if self.memories:
                    return  # Index loaded successfully
            except Exception as e:
                log_error(f"Failed to load index: {e}")

        # Fallback: scan directory for memory JSON files
        # This handles the case where marathon is still running and hasn't saved yet
        self._rebuild_from_files()

    def _rebuild_from_files(self):
        """Rebuild index by scanning memory directory for JSON files."""
        if not MEMORY_DIR.exists():
            return

        for path in MEMORY_DIR.glob("mem_*.json"):
            try:
                data = json5.loads(path.read_text())
                mem_id = data.get("id", path.stem)
                if mem_id in self.memories:
                    continue  # Already in index

                # Add to index
                objs = [o.get("name", "") for o in data.get("objects", [])]
                people = data.get("people", [])
                activities = data.get("activities", [])
                persons = data.get("persons", [])
                # Extract visual person descriptions
                person_descs = [p.get("description", "") for p in persons if isinstance(p, dict)]
                self.memories[mem_id] = {
                    "timestamp": data.get("timestamp", ""),
                    "location": data.get("location", "unknown"),
                    "objects": ",".join(objs),
                    "people": ",".join(people) if people else "",
                    "persons": ";".join(person_descs) if person_descs else "",
                    "activities": ",".join(activities) if activities else "",
                    "image_path": str(MEMORY_DIR / f"{mem_id}.jpg"),
                    "tags": ",".join(data.get("tags", [])),
                    "relationships": ";".join(data.get("relationships", [])),
                    "description": data.get("description", ""),
                    "audio_transcript": data.get("audio_transcript", ""),
                    "conversation_context": data.get("conversation_context", "")
                }

                # Update object index
                for obj in objs:
                    obj = obj.strip().lower()
                    if obj:
                        self.by_object.setdefault(obj, set()).add(mem_id)

                # Update person index (WHO dimension - audio names)
                for person in people:
                    person = person.strip().lower()
                    if person:
                        self.by_person.setdefault(person, set()).add(mem_id)

                # Update visual person index (WHO dimension - visual)
                for person_desc in person_descs:
                    person_desc = person_desc.strip().lower()
                    if person_desc:
                        self.by_person.setdefault(person_desc, set()).add(mem_id)

                # Update activity index (WHAT dimension)
                for activity in activities:
                    activity = activity.strip().lower()
                    if activity:
                        self.by_activity.setdefault(activity, set()).add(mem_id)
            except Exception:
                pass  # Skip malformed files

    def _save(self):
        """Save memory index to JSON file."""
        try:
            atomic_write_text(self.index_file, json5.dumps({
                "memories": self.memories,
                "access_log": self.access_log
            }, indent=2))
        except Exception as e:
            log_error(f"Failed to save index: {e}")

    def add(self, memory: Memory, save_now: bool = False):
        """
        Add memory to the index.

        Args:
            memory: Memory to add
            save_now: If True, persist to disk immediately.
                      If False, call save() later to batch writes.
        """
        # Extract visual person descriptions for indexing
        person_descs = [p.get("description", "") for p in memory.persons if isinstance(p, dict)]
        meta = {
            "timestamp": memory.timestamp,
            "location": memory.location,
            "objects": ",".join(memory.object_names()),
            "activities": ",".join(memory.activities) if memory.activities else "",
            "people": ",".join(memory.people) if memory.people else "",
            "persons": ";".join(person_descs) if person_descs else "",
            "image_path": memory.image_path,
            "tags": ",".join(memory.tags) if memory.tags else "",
            "relationships": ";".join(memory.relationships) if memory.relationships else "",
            "description": memory.description,
            "audio_transcript": memory.audio_transcript,
            "conversation_context": memory.conversation_context
        }

        # Update hash index for objects (WHAT dimension)
        self.memories[memory.id] = meta
        for obj in memory.objects:
            self.by_object.setdefault(obj.name.lower(), set()).add(memory.id)

        # Update hash index for activities (WHAT dimension - actions)
        for activity in memory.activities:
            self.by_activity.setdefault(activity.lower(), set()).add(memory.id)

        # Update hash index for people (WHO dimension - audio names)
        for person in memory.people:
            self.by_person.setdefault(person.lower(), set()).add(memory.id)

        # Update hash index for visual persons (WHO dimension - visual)
        # Index both descriptions and names (if linked)
        for person_info in memory.persons:
            # Index by name if available (linked from audio)
            if person_info.get("name"):
                self.by_person.setdefault(person_info["name"].lower(), set()).add(memory.id)
            # Also index by description keywords (e.g., "blue shirt", "glasses")
            desc = person_info.get("description", "")
            if desc:
                # Index the full description for "person in blue shirt" queries
                self.by_person.setdefault(desc.lower(), set()).add(memory.id)

        # Optionally persist to disk (for batched writes, call save() separately)
        if save_now:
            self._save()
        log(f"[INDEX] Added {memory.id}")

    def save(self):
        """Persist index to disk. Call periodically for batched writes."""
        self._save()

    def reload(self):
        """Reload index from disk to pick up new memories from daemon."""
        self.by_object.clear()
        self.by_activity.clear()
        self.by_person.clear()
        self.memories.clear()
        self._load()
        # Also scan for new JSON files not yet in the index
        # (daemon saves individual files immediately but batches index writes)
        self._rebuild_from_files()
        log(f"[INDEX] Reloaded: {len(self.memories)} memories")

    # -------------------------------------------------------------------------
    # Retrieval Reinforcement (Ebbinghaus forgetting curve)
    # -------------------------------------------------------------------------
    # Human memory is strengthened by recall — each time you remember
    # something, the memory trace gets stronger. We model this by tracking
    # how often each memory is returned in search results (access_count).
    # During cleanup, memories with higher access counts survive longer.
    # -------------------------------------------------------------------------

    def record_access(self, mem_ids: list[str]):
        """
        Record that memories were accessed during a search.

        Retrieval reinforcement: memories that are searched for more often
        get a higher decay score and survive cleanup longer, mimicking how
        human memory strengthens through recall.

        Args:
            mem_ids: List of memory IDs that were returned in a search
        """
        now = datetime.now().isoformat()
        for mem_id in mem_ids:
            # Initialize access entry if this is the first retrieval
            if mem_id not in self.access_log:
                self.access_log[mem_id] = {"access_count": 0, "last_accessed": now}
            # Increment counter and update timestamp on each retrieval
            self.access_log[mem_id]["access_count"] += 1
            self.access_log[mem_id]["last_accessed"] = now

    def decay_score(self, mem_id: str) -> float:
        """
        Calculate a retention score for a memory based on temporal decay
        and retrieval reinforcement (Ebbinghaus forgetting curve).

        Score = recency_component + retrieval_boost_component

        - Recency: e^(-0.693 * age_days / 7) → exponential decay, half-life 7 days
          - Day 0: 1.0, Day 7: 0.5, Day 14: 0.25, Day 30: ~0.05
        - Retrieval boost: access_count * 0.1, capped at 1.0
          - 0 searches: +0.0, 5 searches: +0.5, 10+ searches: +1.0

        Example scores:
          - 1-day-old memory, never searched:  ~0.91
          - 7-day-old memory, never searched:  ~0.50
          - 7-day-old memory, searched 5 times: ~1.00
          - 30-day-old memory, never searched: ~0.05 (likely forgotten)
          - 30-day-old memory, searched 10 times: ~1.05 (survives!)

        Args:
            mem_id: Memory ID to score

        Returns:
            float score (0.0 = should forget, higher = keep)
        """
        now = datetime.now()

        # --- Component 1: Recency (exponential decay) ---
        # Newer memories are more valuable. Uses e^(-lambda * t) where
        # lambda = ln(2)/half_life. Half-life of 7 days means a memory
        # loses half its recency score every week.
        ts_str = self.memories.get(mem_id, {}).get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
            age_days = (now - ts).total_seconds() / 86400.0
        except (ValueError, TypeError):
            age_days = 30.0  # Assume old if no valid timestamp
        recency = math.exp(-0.693 * age_days / 7.0)  # ln(2) ≈ 0.693

        # --- Component 2: Retrieval boost ---
        # Each search hit adds +0.1 to the score (max +1.0).
        # This means a memory searched 10+ times is as "strong" as
        # a brand-new memory, even if it's weeks old.
        access = self.access_log.get(mem_id, {})
        retrieval_boost = min(access.get("access_count", 0) * 0.1, 1.0)

        return recency + retrieval_boost

    def search(self, query: str, n: int = 5) -> list[str]:
        """Search by object name (partial match)."""
        return self.find_by_object(query)[:n]

    # Synonym dictionary: maps alternative names to canonical names
    OBJECT_SYNONYMS = {
        # Eyewear
        "eyeglasses": "glasses",
        "eye glasses": "glasses",
        "spectacles": "glasses",
        "specs": "glasses",
        "reading glasses": "glasses",
        "sunglasses": "glasses",
        # Phone variants
        "mobile": "phone",
        "mobile phone": "phone",
        "cell phone": "phone",
        "cellphone": "phone",
        "smartphone": "phone",
        "iphone": "phone",
        "android": "phone",
        # Drinkware
        "mug": "cup",
        "coffee mug": "cup",
        "coffee cup": "cup",
        "tea cup": "cup",
        # Keys
        "car keys": "keys",
        "house keys": "keys",
        "key": "keys",
        # Remote
        "tv remote": "remote",
        "remote control": "remote",
        # Wallet
        "purse": "wallet",
        "billfold": "wallet",
        # Laptop/computer
        "notebook": "laptop",
        "macbook": "laptop",
        "computer": "laptop",
        # Headphones
        "earphones": "headphones",
        "earbuds": "headphones",
        "airpods": "headphones",
    }

    # Build reverse mapping: canonical -> all synonyms
    SYNONYM_GROUPS = {}
    for syn, canonical in OBJECT_SYNONYMS.items():
        if canonical not in SYNONYM_GROUPS:
            SYNONYM_GROUPS[canonical] = {canonical}
        SYNONYM_GROUPS[canonical].add(syn)
    # Also add entries pointing from canonical to itself
    for canonical in set(OBJECT_SYNONYMS.values()):
        if canonical not in SYNONYM_GROUPS:
            SYNONYM_GROUPS[canonical] = {canonical}

    def find_by_object(self, name: str) -> list[str]:
        """Fast O(1) lookup by object name with controlled fuzzy matching and synonym support."""
        name = name.lower().strip()

        # Expand search terms to include synonyms
        search_terms = {name}
        # If name is a known synonym, get the canonical form
        if name in self.OBJECT_SYNONYMS:
            canonical = self.OBJECT_SYNONYMS[name]
            search_terms.add(canonical)
            # Also add all other synonyms in the group
            if canonical in self.SYNONYM_GROUPS:
                search_terms.update(self.SYNONYM_GROUPS[canonical])
        # If name is a canonical form, get all its synonyms
        if name in self.SYNONYM_GROUPS:
            search_terms.update(self.SYNONYM_GROUPS[name])

        # Try exact match first for all search terms (O(1) hash lookup)
        matches = set()
        for term in search_terms:
            if term in self.by_object:
                matches.update(self.by_object[term])

        if matches:
            return sorted(matches, reverse=True)

        # Controlled fuzzy match: only allow plural/singular variants and
        # compound word matches to avoid false positives (e.g. "car" should
        # NOT match "card" or "cartoon", but "key" SHOULD match "keys").
        for key, ids in self.by_object.items():
            for term in search_terms:
                # Allow plural/singular: "key"↔"keys", "glass"↔"glasses"
                if key.rstrip("s") == term.rstrip("s"):
                    matches.update(ids)
                # Allow compound words: "water bottle" matches "bottle"
                elif term in key.split() or key in term.split():
                    matches.update(ids)

        return sorted(matches, reverse=True)

    def find_by_person(self, name: str) -> list[str]:
        """
        Fast O(1) lookup by person name for WHO dimension of episodic memory.

        Enables queries like "who did I meet?" or "did I see John?"

        Args:
            name: Person name to search for (case-insensitive)
                  If empty string, returns all memories with people

        Returns:
            List of memory IDs where person was present, newest first
        """
        name = name.lower().strip()

        # If no name specified, return all memories that have people
        if not name:
            matches = set()
            for person_name, mem_ids in self.by_person.items():
                matches.update(mem_ids)
            return sorted(matches, reverse=True)

        # Try exact match first (O(1) hash lookup)
        if name in self.by_person:
            return sorted(self.by_person[name], reverse=True)

        # Fuzzy match: partial name matching
        # "john" matches "john smith", "dr. john", etc.
        matches = set()
        for key, ids in self.by_person.items():
            # Allow partial match: "john" matches "john smith"
            if name in key or key in name:
                matches.update(ids)
            # Allow first name match: "john" matches "john"
            if name.split()[0] == key.split()[0]:
                matches.update(ids)

        return sorted(matches, reverse=True)

    def get_all_people(self) -> list[str]:
        """
        Get list of all people encountered in memory.

        Useful for "who did I meet today?" type queries where we need
        to enumerate all people in the time window.

        Returns:
            List of unique person names encountered
        """
        return list(self.by_person.keys())

    def find_by_activity(self, activity: str) -> list[str]:
        """
        Fast lookup by activity for WHAT dimension (actions) of episodic memory.

        Enables queries like "did I take my medication?" or "did I eat breakfast?"
        Uses fuzzy matching to find related activities.

        Args:
            activity: Activity to search for (e.g., "taking medication", "medication")
                      If empty string, returns all memories with activities

        Returns:
            List of memory IDs where activity occurred, newest first
        """
        activity = activity.lower().strip()

        # If no activity specified, return all memories that have activities
        if not activity:
            matches = set()
            for act_name, mem_ids in self.by_activity.items():
                matches.update(mem_ids)
            return sorted(matches, reverse=True)

        # Try exact match first (O(1) hash lookup)
        if activity in self.by_activity:
            return sorted(self.by_activity[activity], reverse=True)

        # Fuzzy match: partial activity matching
        # "medication" matches "taking medication", "medication reminder", etc.
        matches = set()
        for key, ids in self.by_activity.items():
            # Allow partial match: "medication" matches "taking medication"
            if activity in key or key in activity:
                matches.update(ids)
            # Allow keyword match: any word in activity matches
            activity_words = set(activity.split())
            key_words = set(key.split())
            if activity_words & key_words:  # Intersection
                matches.update(ids)

        return sorted(matches, reverse=True)

    def get_all_activities(self) -> list[str]:
        """
        Get list of all activities detected in memory.

        Useful for "what did I do today?" type queries.

        Returns:
            List of unique activities detected
        """
        return list(self.by_activity.keys())

    def __len__(self):
        """Return number of memories in index."""
        return len(self.memories)

    def find_by_location(self, location: str, n: int = 10) -> list[str]:
        """
        Find memories by scene location (partial match).

        Enables scene-level queries like "what was on the kitchen counter?"

        Args:
            location: Location to search for (case-insensitive partial match)
            n: Maximum results to return

        Returns:
            List of memory IDs matching the location, newest first
        """
        location = location.lower()
        matches = []
        for mem_id, meta in self.memories.items():
            mem_loc = meta.get("location", "").lower()
            if location in mem_loc or mem_loc in location:
                matches.append(mem_id)
        return sorted(matches, reverse=True)[:n]

    # -------------------------------------------------------------------------
    # Time-Based Search (Tulving's WHEN dimension)
    # -------------------------------------------------------------------------
    # Human episodic memory is often recalled by time context:
    # "What did I do this morning?", "Where were my keys yesterday?"
    # This method filters memories by a datetime window produced by
    # parse_time_entity() from fuzzy natural language expressions.
    # -------------------------------------------------------------------------

    def find_by_time(self, start: datetime, end: datetime, n: int = 20) -> list[str]:
        """
        Find memories within a time window.

        Enables temporal queries like "what did I see this morning?"

        Args:
            start: Window start (inclusive)
            end: Window end (inclusive)
            n: Maximum results to return

        Returns:
            List of memory IDs within the time window, newest first
        """
        matches = []
        for mem_id, meta in self.memories.items():
            ts_str = meta.get("timestamp", "")
            if not ts_str:
                continue
            try:
                # Parse ISO timestamp and check if it falls within the window
                ts = datetime.fromisoformat(ts_str)
                if start <= ts <= end:
                    matches.append(mem_id)
            except (ValueError, TypeError):
                # Skip memories with invalid timestamps
                continue
        # Return newest first (memory IDs embed timestamps, so reverse sort works)
        return sorted(matches, reverse=True)[:n]

    # -------------------------------------------------------------------------
    # Co-occurrence Search (spatial context)
    # -------------------------------------------------------------------------
    # Objects are rarely alone — keys are next to wallets, phones are near
    # chargers. Co-occurrence search leverages this by finding all objects
    # that appeared in the same memory frame as the queried entity.
    # No extra storage needed — uses the existing objects CSV in metadata.
    # -------------------------------------------------------------------------

    def find_cooccurrence(self, entity: str, n: int = 10) -> list[tuple[str, list[str]]]:
        """
        Find objects that appeared in the same memory as the given entity.

        Enables queries like "what was near my keys?"

        Args:
            entity: Object name to find co-occurrences for
            n: Maximum results to return

        Returns:
            List of (memory_id, [co-occurring object names]) tuples, newest first
        """
        entity_lower = entity.lower().strip()
        # First, find all memories containing the queried entity
        mem_ids = self.find_by_object(entity_lower)
        results = []
        for mem_id in mem_ids[:n]:
            meta = self.memories.get(mem_id, {})
            # Split the comma-separated object list from metadata
            all_objs = [o.strip() for o in meta.get("objects", "").split(",") if o.strip()]
            # Exclude the queried entity itself (including plural/singular variants)
            others = [o for o in all_objs if o.lower() != entity_lower
                       and o.lower().rstrip("s") != entity_lower.rstrip("s")]
            if others:
                results.append((mem_id, others))
        return results



def parse_time_entity(entity: str) -> tuple[datetime, datetime]:
    """
    Parse a fuzzy time expression into a (start, end) datetime window.

    Human episodic memory is often anchored to natural time boundaries:
    "this morning", "yesterday", "last hour". This function maps those
    fuzzy expressions to precise datetime ranges for memory filtering.

    Supports natural language time references commonly used in episodic recall:
    - "this morning" → 6:00 AM to 12:00 PM today
    - "this afternoon" → 12:00 PM to 6:00 PM today
    - "this evening" / "tonight" → 6:00 PM to midnight today
    - "today" → midnight to now
    - "yesterday" → midnight to midnight yesterday
    - "last hour" / "past hour" → 1 hour ago to now
    - "last night" → 8:00 PM yesterday to 6:00 AM today
    - "last N hours" / "past N hours" → N hours ago to now
    - "last N minutes" / "past N minutes" → N minutes ago to now

    Args:
        entity: Fuzzy time expression from classify_query

    Returns:
        (start, end) datetime tuple defining the time window
    """
    now = datetime.now()
    # Midnight today — anchor point for day-relative expressions
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    entity = entity.lower().strip()

    # --- Relative numeric expressions: "last N hours", "past 30 minutes" ---
    # These must be checked first because they use regex with \d+ groups,
    # and would be missed by the keyword-based checks below.
    m = re.match(r'(?:last|past)\s+(\d+)\s+hours?', entity)
    if m:
        hours = int(m.group(1))
        return now - timedelta(hours=hours), now

    m = re.match(r'(?:last|past)\s+(\d+)\s+minutes?', entity)
    if m:
        minutes = int(m.group(1))
        return now - timedelta(minutes=minutes), now

    # --- Day-part expressions: fixed time boundaries ---
    # Morning/afternoon/evening use conventional boundaries that align
    # with how people naturally divide their day.
    if entity == "this morning":
        # 6 AM to noon
        return today_start.replace(hour=6), today_start.replace(hour=12)
    elif entity == "this afternoon":
        # Noon to 6 PM
        return today_start.replace(hour=12), today_start.replace(hour=18)
    elif entity in ("this evening", "tonight"):
        # 6 PM to end of day
        return today_start.replace(hour=18), today_start.replace(hour=23, minute=59, second=59)
    elif entity == "today":
        # Midnight to current time
        return today_start, now
    elif entity == "yesterday":
        # Full previous day (midnight to midnight)
        yesterday = today_start - timedelta(days=1)
        return yesterday, today_start
    elif entity in ("last hour", "past hour"):
        # Rolling 60-minute window ending now
        return now - timedelta(hours=1), now
    elif entity == "last night":
        # 8 PM yesterday to 6 AM today (sleep period)
        yesterday = today_start - timedelta(days=1)
        return yesterday.replace(hour=20), today_start.replace(hour=6)

    # Fallback: if expression is unrecognized, default to last 24 hours
    return now - timedelta(hours=24), now


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: CORE PIPELINE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# The main processing pipelines that tie everything together:
#
# 1. analyze_and_store(): Capture → Vision → Temporal → Store → Index
# 2. find_object(): Query → O(1) Lookup / Time / Co-occurrence → Narrative
# 3. parse_time_entity(): Fuzzy time expression → (start, end) datetime window
#
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_and_store(gemini: GeminiClient, index: MemoryIndex,
                      temporal: TemporalGraph, image_data: bytes,
                      capture_ts: datetime | None = None,
                      audio_data: bytes | None = None) -> tuple[Memory, bool]:
    """
    Full capture-to-storage pipeline with audio for episodic memory.

    This is the MARATHON AGENT pipeline. It doesn't just detect objects -
    it tracks their movements over time for CAUSE-AND-EFFECT reasoning.
    With audio capture, it also records WHO was present (people's names
    from conversations) for complete episodic memory.

    PIPELINE:
    1. Send image to Gemini Vision → get objects with bounding boxes
    2. Create Memory record with timestamp (from CAPTURE time, not analysis time)
    3. If audio provided: transcribe and extract people's names (WHO dimension)
    4. Update TemporalGraph → detect if any objects moved
    5. Save image to disk (JPEG)
    6. Save metadata to disk (JSON) - includes audio transcript and people
    7. Add to JSON index (for object and person lookup)
    8. Persist temporal graph (for marathon continuity)

    Args:
        gemini: Gemini client for vision analysis
        index: Memory index for semantic search
        temporal: Temporal graph for movement tracking
        image_data: JPEG image bytes from camera
        capture_ts: When the frame was captured (important for accurate tracking!)
                    If None, uses current time (less accurate)
        audio_data: Optional WAV audio bytes captured during the scene

    Returns:
        Tuple of (Memory, was_saved): Memory object and whether it was saved
    """
    # STEP 1: Analyze image with Gemini Vision
    # NOTE: This API call may take 14-40+ seconds on free tier
    analysis = gemini.analyze_image(image_data)

    # STEP 2: Create memory record with CAPTURE timestamp (not analysis completion)
    # This is critical for accurate movement tracking - we want to know when
    # the object was SEEN, not when the API finished processing
    ts = capture_ts or datetime.now()
    memory = Memory(
        id=f"mem_{ts.strftime('%Y%m%d_%H%M%S')}",
        timestamp=ts.isoformat(),
        location=analysis["location"],
        description=analysis["description"],
        image_data=image_data,
        activities=analysis.get("activities", []),  # WHAT dimension: detected activities
        tags=analysis.get("tags", []),
        relationships=analysis.get("relationships", [])
    )

    # Log detected activities (important for "did I take my medication?" queries)
    if memory.activities:
        log(f"[ACTIVITIES] {', '.join(memory.activities)}")

    # STEP 2.5: Process audio for WHO dimension of episodic memory
    # If audio was captured, transcribe it and extract people's names
    if audio_data and AUDIO_CAPTURE_ENABLED:
        try:
            # Transcribe speech from the scene
            transcript = gemini.transcribe_audio(audio_data)
            if transcript and transcript != "[silence]":
                memory.audio_transcript = transcript
                # Extract names and conversation context
                people, context = gemini.extract_people_from_transcript(transcript)
                memory.people = people
                memory.conversation_context = context
                if people:
                    log(f"[WHO] People in scene: {', '.join(people)}")
        except Exception as e:
            log_error(f"Audio processing failed: {e}")

    # STEP 2.6: Process visual persons for WHO dimension
    # Extract person descriptions from vision analysis (complements audio names)
    persons = analysis.get("persons", [])
    if persons:
        # Store visual person detections
        memory.persons = [
            {
                "description": p.get("description", "person"),
                "context": p.get("context", ""),
                "box_2d": p.get("box_2d", [])
            }
            for p in persons if isinstance(p, dict)
        ]
        person_descs = [p.get("description", "person") for p in persons]
        log(f"[WHO-VISUAL] {len(persons)} person(s): {', '.join(person_descs)}")

        # If we have audio names AND visual persons, link them
        # (assumes first name matches first person when counts align)
        if memory.people and len(memory.people) == len(memory.persons):
            for i, name in enumerate(memory.people):
                memory.persons[i]["name"] = name
            log(f"[WHO-LINKED] Names linked to visual persons")

    # STEP 3: Process detected objects and update temporal graph
    movements_detected = 0
    useful_object_count = 0  # Track non-attached objects

    for obj in analysis["objects"]:
        # Parse bounding box from Gemini's native box_2d format.
        # Gemini returns [ymin, xmin, ymax, xmax] on a 0-1000 scale.
        #   e.g., [120, 50, 450, 300] means:
        #     ymin=120/1000=0.12, xmin=50/1000=0.05,
        #     ymax=450/1000=0.45, xmax=300/1000=0.30
        # NOTE: Gemini uses Y-first order (ymin, xmin, ymax, xmax), but
        # our BoundingBox uses X-first order (x1, y1, x2, y2) - so we swap.
        # We clamp to 0.0-1.0 because model can return out-of-range values.
        box = obj.get("box_2d", [0, 0, 1000, 1000])
        if len(box) == 4:
            # Divide by 1000 to convert from 0-1000 → 0.0-1.0
            ymin, xmin, ymax, xmax = [float(v) / 1000.0 for v in box]
        else:
            # Fallback: full image if box format is invalid
            ymin, xmin, ymax, xmax = 0.0, 0.0, 1.0, 1.0
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 1.0))))

        # Filter out low-confidence detections (likely hallucinations)
        if confidence < MIN_CONFIDENCE:
            obj_name = obj.get("label", obj.get("name", "?"))
            log(f"[VISION] Skipping low-confidence detection: {obj_name} ({confidence:.0%})")
            continue

        bbox = BoundingBox(
            name=obj.get("label", obj.get("name", "?")),
            x1=max(0.0, min(1.0, xmin)),   # Clamp left edge to valid range
            y1=max(0.0, min(1.0, ymin)),   # Clamp top edge to valid range
            x2=max(0.0, min(1.0, xmax)),   # Clamp right edge to valid range
            y2=max(0.0, min(1.0, ymax)),   # Clamp bottom edge to valid range
            confidence=confidence,
            context=obj.get("context", "")
        )
        memory.objects.append(bbox)

        # Check if Gemini marked this object as attached to person
        # (glasses on face, watch on wrist, headphones worn - these move with the person)
        # Gemini decides based on world knowledge, not hardcoded phrases
        is_attached = obj.get("attached", False)

        if is_attached:
            # Mark as attached for removal detection later
            temporal.mark_attached(bbox.name, memory.timestamp, memory.location)
            # Still record last seen, but don't track movement
            temporal.last_seen[bbox.name] = (memory.location, bbox.position(),
                                             memory.timestamp, memory.id)
            continue

        # This is a useful (non-attached) object
        useful_object_count += 1

        # Update temporal graph - this detects movements (CAUSE AND EFFECT)
        movement = temporal.update(
            obj_name=bbox.name,
            location=memory.location,
            position=bbox.position(),
            timestamp=memory.timestamp,
            memory_id=memory.id
        )
        if movement:
            movements_detected += 1

            # Proactive announcement: tell user when important object is placed
            # This helps prevent forgetting where you just put something
            if ANNOUNCE_ENABLED and bbox.name.lower() in ANNOUNCE_OBJECTS:
                obj_lower = bbox.name.lower()
                now = time.time()
                last_announced = _announcement_cooldowns.get(obj_lower, 0)

                # Only announce if cooldown has passed
                if now - last_announced > ANNOUNCE_COOLDOWN:
                    _announcement_cooldowns[obj_lower] = now
                    announcement = f"{bbox.name} placed on {memory.location}"
                    log(f"[ANNOUNCE] {announcement}")

                    # TTS: Speak the announcement using Gemini TTS (non-blocking)
                    try:
                        if TTS_ENABLED:
                            audio_data = gemini.text_to_speech(announcement)
                            if audio_data:
                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                    f.write(audio_data)
                                    temp_path = f.name
                                # Play and cleanup in background
                                subprocess.Popen(
                                    f"aplay -D plughw:0,0 -q {temp_path} && rm {temp_path}",
                                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                                )
                    except Exception:
                        pass  # TTS failure shouldn't block memory capture

    # Check if any previously-attached objects have been removed
    # (e.g., glasses that were on face but now not seen at all)
    all_visible = [obj.get("label", obj.get("name", "")) for obj in analysis["objects"]]
    removed_attached = temporal.check_removed_attached(all_visible, memory.timestamp, timeout_seconds=30)
    for removed_obj in removed_attached:
        log(f"[REMOVED] {removed_obj} was removed (was attached, now not visible)")
        # When the object reappears NOT attached, it will be saved as a normal memory

    # Check if there are any useful (non-attached) objects
    if useful_object_count == 0:
        log(f"[SKIP] No useful objects (only attached: {', '.join(memory.object_names())})")
        return (memory, False)  # Return (memory, was_saved=False)

    # STEP 4 & 5: Save to filesystem
    memory.image_path = save_image(memory.id, image_data)
    save_metadata(memory)

    # STEP 6: Add to search index (don't persist yet - batched in daemon loop)
    index.add(memory, save_now=False)

    # NOTE: Temporal graph and index are saved periodically by the daemon
    # loop to reduce disk I/O. Also saved on graceful shutdown.

    # Log results
    log(f"[MEMORY] {memory.id}")
    log(f"   📍 {memory.location}")
    log(f"   📦 {', '.join(memory.object_names()[:5])}")
    if memory.tags:
        log(f"   🏷️  {', '.join(memory.tags[:3])}")
    if memory.relationships:
        log(f"   🔗 {'; '.join(memory.relationships[:2])}")
    if movements_detected > 0:
        log(f"   🔄 {movements_detected} object(s) moved")

    return (memory, True)  # Return (memory, was_saved=True)


def find_object(index: MemoryIndex, temporal: TemporalGraph,
                query: str, query_type: str = "object",
                entity: str = "", gemini: 'GeminiClient | None' = None,
                question: 'str | None' = None,
                placed: bool = False) -> tuple:
    """
    Search for object/person using hash lookup + location/time/co-occurrence search.

    Args:
        placed: If True, user asked "where did I LEAVE/PUT X" - filter out "in hand" results
                and find where the object was actually placed on a surface.

    Returns:
    1. Current location (Memory with image) or list of memories (time/person queries)
    2. Movement history (how it got there) or co-occurrence list or people list

    SEARCH STRATEGY (Tulving's episodic memory dimensions):
    - Object queries (WHAT): O(1) hash lookup by object name
    - Scene queries (WHERE): Location string matching
    - Time queries (WHEN): Fuzzy time window filtering
    - Near queries: Co-occurrence lookup (objects seen together)
    - Person queries (WHO): O(1) hash lookup by person name
    - All: record access for retrieval reinforcement

    Args:
        index: Memory index for hash-based search
        temporal: Temporal graph for movement history
        query: Search query (object name or natural language)
        query_type: Query classification (object|scene|time|near|person)
        entity: Extracted search entity

    Returns:
        tuple: (Memory or None, List[ObjectMovement]) for object/scene
               (List[Memory], []) for time queries
               (Memory or None, List[tuple]) for near queries
               (List[Memory], List[str]) for person queries (memories, people list)
    """
    search_entity = entity or query

    # The search dispatches to one of four strategies based on query_type.
    # Each strategy also records access for retrieval reinforcement, so
    # memories returned in search results get a higher decay_score and
    # survive cleanup longer (Ebbinghaus forgetting curve).

    # --- PATH 1: TIME QUERY ---
    # "this morning", "last 2 hours", "yesterday"
    # Converts fuzzy time expression to datetime window, then filters.
    # Returns a LIST of Memory objects (unlike other paths which return one).
    if query_type == "time":
        start, end = parse_time_entity(search_entity)
        mem_ids = index.find_by_time(start, end)
        if mem_ids:
            # Reinforce top 5 results (user is recalling this time period)
            index.record_access(mem_ids[:5])
        # Load up to 10 memories (cap to avoid loading too many into RAM)
        memories = [load_memory(mid) for mid in mem_ids[:10]]
        memories = [m for m in memories if m is not None]
        return memories, []

    # --- PATH 2: NEAR / CO-OCCURRENCE QUERY ---
    # "what was near my keys?", "what was with my phone?"
    # Finds the entity itself AND all objects that shared a memory frame.
    # Returns (Memory, List[(mem_id, [co-occurring objects])]).
    if query_type == "near":
        cooccurrences = index.find_cooccurrence(search_entity)
        mem_ids = index.find_by_object(search_entity)
        memory = load_memory(mem_ids[0]) if mem_ids else None
        if mem_ids:
            index.record_access(mem_ids[:1])
        return memory, cooccurrences

    # --- PATH 2.5: PERSON QUERY (WHO dimension) ---
    # "who did I meet?", "did I see John?", "who was I with?"
    # Implements Tulving's WHO dimension of episodic memory.
    # Returns (List[Memory], List[str]) where second item is people found.
    if query_type == "person":
        mem_ids = index.find_by_person(search_entity)
        if mem_ids:
            index.record_access(mem_ids[:5])
        # Load memories and collect all people mentioned
        memories = [load_memory(mid) for mid in mem_ids[:10]]
        memories = [m for m in memories if m is not None]
        # Collect unique people from these memories
        people_found = set()
        for mem in memories:
            if mem.people:
                people_found.update(mem.people)
        # If searching for specific person, filter to just that person
        if search_entity:
            people_found = {p for p in people_found if search_entity.lower() in p.lower()}
        return memories, list(people_found)

    # --- PATH 2.75: ACTIVITY QUERY (WHAT dimension - actions) ---
    # "did I take my medication?", "did I eat breakfast?", "did I lock the door?"
    # Searches for detected activities to answer "did I..." questions.
    # Returns (List[Memory], List[str]) where second item is activities found.
    if query_type == "activity":
        mem_ids = index.find_by_activity(search_entity)
        if mem_ids:
            index.record_access(mem_ids[:5])
        # Load memories with matching activities
        memories = [load_memory(mid) for mid in mem_ids[:10]]
        memories = [m for m in memories if m is not None]
        # Collect matching activities from these memories
        activities_found = set()
        for mem in memories:
            if mem.activities:
                for act in mem.activities:
                    if not search_entity or search_entity.lower() in act.lower():
                        activities_found.add(act)
        return memories, list(activities_found)

    # --- PATH 2.8: VQA QUERY (Visual Question Answering) ---
    # "what color is the chair?", "how many boxes?", "what brand is the laptop?"
    # Finds the object, then asks Gemini Vision to answer the question about it.
    # Returns (Memory, str) where second item is the VQA answer.
    if query_type == "vqa":
        if not gemini:
            return None, "VQA requires Gemini client."
        # First find the object in memory
        mem_ids = index.find_by_object(search_entity)
        if mem_ids:
            index.record_access(mem_ids[:1])
            memory = load_memory(mem_ids[0])
            if memory and memory.image_data:
                # Ask Gemini Vision the question about this image
                vqa_question = question if question else query
                answer = gemini.answer_visual_question(memory.image_data, vqa_question)
                return memory, answer
        return None, "Object not found in memory."

    # --- PATH 3: SCENE QUERY ---
    # "what was on the kitchen counter?"
    # Searches by location string match. No movement history since we're
    # asking about a place, not an object.
    if query_type == "scene":
        mem_ids = index.find_by_location(search_entity)
        if mem_ids:
            index.record_access(mem_ids[:1])
            memory = load_memory(mem_ids[0])
            return memory, []
        # Fall through to object search if no location match

    # --- PATH 4: OBJECT QUERY (default) ---
    # "where are my keys?", "find my wallet", "keys"
    # Core O(1) hash lookup + temporal movement history for cause-and-effect.
    # This is the primary episodic memory use case.
    mem_ids = index.find_by_object(search_entity)

    # HUMAN-LIKE EPISODIC MEMORY: When user asks "where did I LEAVE my X?"
    # they want to know where they PUT IT DOWN, not where they were HOLDING it.
    # Filter out "in hand" memories when placed=True.
    if placed and mem_ids:
        IN_HAND_CONTEXTS = ["in hand", "held", "holding", "carrying", "gripping"]
        placed_mem_ids = []
        in_hand_memory = None  # Keep track of "in hand" memory for fallback

        for mid in mem_ids:
            mem = load_memory(mid)
            if mem:
                obj = mem.find_object(search_entity)
                if obj and obj.context:
                    ctx_lower = obj.context.lower()
                    if any(hint in ctx_lower for hint in IN_HAND_CONTEXTS):
                        if in_hand_memory is None:
                            in_hand_memory = mem  # Save for fallback
                        continue  # Skip "in hand" memories
                placed_mem_ids.append(mid)

        if placed_mem_ids:
            mem_ids = placed_mem_ids
        elif in_hand_memory:
            # Only "in hand" memories found - return special marker for renderer
            movements = temporal.get_history(search_entity) if temporal else []
            return in_hand_memory, ("only_in_hand", movements)

    if mem_ids:
        index.record_access(mem_ids[:1])
    memory = load_memory(mem_ids[0]) if mem_ids else None
    # Get movement history from TemporalGraph for HOW reasoning
    movements = temporal.get_history(search_entity) if temporal else []

    return memory, movements


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Command-line interface for the GEM system:
#
# - python gem.py          → Run capture daemon (Marathon Agent)
# - python gem.py search   → Interactive search mode
# - python gem.py list     → List stored memories
# - python gem.py hw_test     → Test hardware and API
# - python gem.py --help   → Show documentation
#
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_daemon(headless: bool = False):
    """
    Run the Marathon Agent capture daemon.

    This is the MAIN MODE of GEM. It runs continuously, capturing frames,
    detecting objects, and tracking their movements over time.

    Args:
        headless: If True, run without HAT (no LCD/LED). This allows
                  search mode to use the HAT while daemon runs in background.

    BEHAVIOR:
    - Captures frames every CAPTURE_INTERVAL seconds
    - Analyzes when scene changes by >= CHANGE_THRESHOLD
    - Analyzes at least every FORCE_ANALYZE_INTERVAL seconds
    - Saves memories and updates temporal graph
    - Runs until Ctrl+C

    MARATHON AGENT FEATURES:
    - Runs for hours/days without supervision
    - Persists state across restarts (temporal graph saved to disk)
    - Self-correcting (re-analyzes on scene changes)
    """
    init_storage()
    
    print()
    log("=" * 60)
    log("  GEM Marathon Agent (Gemini 3)")
    log("  Episodic memory for Pi Zero 2W")
    log("=" * 60)

    # Initialize Gemini client
    try:
        gemini = GeminiClient()
    except ValueError as e:
        log_error(str(e))
        sys.exit(1)

    # Initialize memory index (JSON-based hash lookup)
    index = MemoryIndex(gemini)

    # Initialize temporal graph (cause-and-effect tracking)
    # Load from disk to resume marathon across restarts
    temporal = TemporalGraph()
    temporal.load(DATA_DIR / "temporal_graph.json")

    # Initialize camera
    try:
        camera = Camera()
    except Exception as e:
        log_error(f"Camera: {e}")
        sys.exit(1)

    # Initialize HAT
    # In headless mode: mic enabled for audio capture, LCD/LED disabled
    # This allows search mode to use LCD while daemon captures in background
    hat = WhisplayHAT(headless=headless)
    if hat.board:
        mem_count = len(index)
        hat.display_text(f"GEM Marathon\n\nGemini 3\n{mem_count} memories", color=(0, 255, 0))

    log(f"[MARATHON] Running autonomously (Ctrl+C to stop)")
    log(f"   Objects tracked: {len(temporal.last_seen)}")
    log(f"   Total movements: {temporal.total_movements}")
    print()

    # State for capture loop
    prev_frame = None
    last_analyze = 0
    captures_this_session = 0

    try:
        while True:
            # Capture frame
            jpeg, frame = camera.capture()

            # Calculate frame difference
            change = frame_difference(prev_frame, frame)
            now = time.time()

            # ANALYSIS DECISION LOGIC:
            #
            # We don't analyze every frame (too expensive for Gemini API).
            # Instead, we use two trigger conditions:
            #
            # Condition 1 - REACTIVE (scene changed):
            #   change >= CHANGE_THRESHOLD (default 15%) AND
            #   time since last analysis >= MIN_ANALYZE_INTERVAL (default 5s)
            #   → Captures when objects move, person enters room, etc.
            #   → MIN_ANALYZE_INTERVAL prevents API spam during rapid changes
            #
            # Condition 2 - PERIODIC (nothing changed):
            #   time since last analysis >= FORCE_ANALYZE_INTERVAL (default 30s)
            #   → Ensures we don't miss static scenes for too long
            #   → Catches objects placed while camera wasn't looking
            #
            analyze = False
            reason = ""

            if change >= CHANGE_THRESHOLD and (now - last_analyze) >= MIN_ANALYZE_INTERVAL:
                analyze = True
                reason = f"change={change*100:.0f}%"
            elif (now - last_analyze) >= FORCE_ANALYZE_INTERVAL:
                analyze = True
                reason = "periodic"

            if analyze:
                log(f"[CAPTURE] {reason}")

                # Visual feedback: white LED during processing
                if hat and hat.board:
                    hat.board.set_rgb(255, 255, 255)

                # Capture audio for episodic memory WHO dimension
                # This records ambient speech for name/people extraction
                audio_data = None
                if AUDIO_CAPTURE_ENABLED and hat and hat.mic:
                    audio_data = hat.record_audio(duration=AUDIO_CAPTURE_DURATION)
                    if audio_data:
                        log(f"[AUDIO] Captured {len(audio_data)//1024}KB")

                # Run full pipeline with CAPTURE timestamp
                # Pass the actual capture time, not when Gemini finishes analyzing
                capture_time = datetime.fromtimestamp(now)
                memory, was_saved = analyze_and_store(
                    gemini, index, temporal, jpeg, capture_time, audio_data
                )
                last_analyze = now

                # If only attached objects detected, apply short cooldown to reduce API calls
                # 5 seconds balances API savings vs responsiveness
                if not was_saved:
                    last_analyze = now + 5  # Skip next 5 seconds of potential triggers
                else:
                    captures_this_session += 1

                    # Periodic save: persist index and graph every 10 captures
                    # This reduces microSD I/O while ensuring data isn't lost
                    if captures_this_session % 10 == 0:
                        index.save()
                        temporal.save(DATA_DIR / "temporal_graph.json")
                        cleanup_old_memories(index)
                        log(f"[SAVE] Persisted index and temporal graph")

                    # Visual feedback: green LED when done
                    if hat and hat.board:
                        hat.board.set_rgb(0, 255, 0)
                        hat.display_text(
                            f"Captured!\n\n{', '.join(memory.object_names()[:3])}\n\n"
                            f"{len(index)} memories",
                            color=(0, 255, 0)
                        )

            prev_frame = frame
            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        print()
        log("[MARATHON] Agent stopped")
        log(f"   Session captures: {captures_this_session}")
        log(f"   Total objects: {len(temporal.last_seen)}")
        log(f"   Total movements: {temporal.total_movements}")
    finally:
        # Persist all data on shutdown
        index.save()
        temporal.save(DATA_DIR / "temporal_graph.json")
        camera.close()
        if hat:
            hat.cleanup()


def cmd_search():
    """
    Interactive search mode with episodic memory query types.

    FEATURES:
    - Object queries: "where are my keys?" → hash lookup + movement history
    - Scene queries: "what was on the kitchen counter?" → location search
    - Time queries: "what did I see this morning?" → fuzzy time window
    - Near queries: "what was near my keys?" → co-occurrence lookup
    - Voice input via Whisplay HAT button
    - Causal narratives: "You put them there 2h ago"
    - Retrieval reinforcement: searched memories survive cleanup longer
    - Visual display: bounding box overlay on HAT LCD
    """
    init_storage()

    print()
    log("=" * 60)
    log("  GEM Search (Gemini 3)")
    log("  Object + Scene + Time + Co-occurrence")
    log("=" * 60)

    # Initialize Gemini client
    try:
        gemini = GeminiClient()
    except ValueError as e:
        log_error(str(e))
        sys.exit(1)

    # Initialize search components
    index = MemoryIndex(gemini)
    temporal = TemporalGraph()
    temporal.load(DATA_DIR / "temporal_graph.json")

    if len(index) == 0:
        log("No memories! Run: python gem.py")
        return

    # Initialize HAT
    hat = WhisplayHAT()
    voice = hat.board and hat.mic

    print()
    log(f"[SEARCH] {len(index)} memories, {len(temporal.last_seen)} objects tracked")
    log(f"   Movements: {temporal.total_movements}")
    if voice:
        log("   Press HAT button for voice")
    log("   Type query or 'quit'")
    log("   Examples: 'where are my keys?', 'this morning', 'what was near my phone?'")
    print()

    if hat.board:
        hat.display_text("Ready!\n\nGEM Search", color=(100, 200, 255))

    last_voice_time = 0  # Cooldown timer to prevent double-triggers
    VOICE_COOLDOWN = 2.0  # Minimum seconds between voice triggers

    try:
        while True:
            # Dual-mode input: voice via HAT button press, or keyboard text.
            # Voice path: button pressed → record audio → transcribe via Gemini STT.
            # Keyboard path: type query and press Enter.
            #
            # Uses non-blocking input check so button press works anytime.

            raw_query = None

            if voice:
                # Non-blocking input loop that checks both keyboard and button
                print("\n🔍 Search (or press button for voice): ", end="", flush=True)

                while raw_query is None:
                    # Check for button press (with debouncing and cooldown)
                    if hat.button_pressed():
                        # Debounce: wait for button release before proceeding
                        while hat.button_pressed():
                            time.sleep(0.05)
                        time.sleep(0.1)  # Extra debounce delay

                        # Cooldown: ignore if triggered too recently
                        if time.time() - last_voice_time < VOICE_COOLDOWN:
                            continue

                        last_voice_time = time.time()
                        print("\n[VOICE]")
                        if hat.board:
                            hat.display_text("Listening...", color=(255, 100, 100))
                            hat.board.set_rgb(255, 100, 100)

                        audio = hat.record_audio()
                        if audio:
                            raw_query = gemini.transcribe_audio(audio)
                            if raw_query:
                                log(f"[VOICE] \"{raw_query}\"")
                            else:
                                log("[VOICE] No speech detected")
                                if hat.board:
                                    hat.display_text("No speech\ndetected\n\nTry again", color=(255, 150, 50))
                                    time.sleep(1.5)
                                print("🔍 Search (or press button for voice): ", end="", flush=True)
                                continue
                        else:
                            log("[VOICE] Recording failed")
                            print("🔍 Search (or press button for voice): ", end="", flush=True)
                            continue
                        break

                    # Check for keyboard input (non-blocking)
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        line = sys.stdin.readline()
                        if line:
                            raw_query = line.strip()
                        break
            else:
                # No voice available, just use blocking keyboard input
                raw_query = input("\n🔍 Search: ").strip()

            if not raw_query:
                continue

            if raw_query.lower() in ['quit', 'exit', 'q']:
                break

            # Reload index to pick up new memories from daemon
            index.reload()
            temporal.load(DATA_DIR / "temporal_graph.json")

            # Use Gemini 3 NLU to understand the query
            # This replaces hardcoded regex patterns with true language understanding
            if hat.board:
                hat.display_text(f"Understanding:\n{raw_query[:30]}", color=(255, 255, 0))
                hat.board.set_rgb(255, 255, 0)

            # Gemini understands: query type, entity, and time range (if any)
            understanding = gemini.understand_query(raw_query)
            query_type = understanding["type"]
            entity = understanding["entity"]
            time_start = understanding.get("time_start")
            time_end = understanding.get("time_end")
            vqa_question = understanding.get("question")  # For VQA queries
            placed = understanding.get("placed", False)  # "leave/put" vs "find"

            # ─────────────────────────────────────────────────────────
            # SEARCH: Dispatch to find_object which routes by query_type
            # Returns vary by type:
            #   time → (List[Memory], [])
            #   near → (Memory|None, List[(mem_id, [objects])])
            #   person → (List[Memory], List[str])
            #   object/scene → (Memory|None, List[ObjectMovement])
            # ─────────────────────────────────────────────────────────

            # For time queries, use Gemini-parsed time range directly
            if query_type == "time" and time_start and time_end:
                start = datetime.fromisoformat(time_start)
                end = datetime.fromisoformat(time_end)
                mem_ids = index.find_by_time(start, end)
                if mem_ids:
                    index.record_access(mem_ids[:5])
                memories = [load_memory(mid) for mid in mem_ids[:10]]
                result = [m for m in memories if m is not None]
                extra = []
            else:
                result, extra = find_object(
                    index, temporal, raw_query,
                    query_type=query_type,
                    entity=entity,
                    gemini=gemini,
                    question=vqa_question,
                    placed=placed
                )

            # ─────────────────────────────────────────────────────────
            # RESULT RENDERING: Four paths, one per query type.
            # Each path formats the output for terminal + HAT LCD.
            # ─────────────────────────────────────────────────────────

            # --- RENDERER 1: TIME QUERY ---
            # Shows a chronological list of memories from the time window
            # PLUS a natural language activity summary for episodic recall.
            # Example: "this morning" → narrative + 5 memories with timestamps.
            if query_type == "time" and isinstance(result, list):
                if result:
                    # Generate natural activity summary (helps users with memory problems)
                    print()
                    narrative = gemini.generate_activity_summary(result, entity)
                    log(f"[ACTIVITY] {entity}")
                    log(f"   {narrative}")

                    print()
                    log(f"[DETAILS] {len(result)} memories:")
                    # Show up to 5 memories with their key details
                    for mem in result[:5]:
                        objs = ", ".join(mem.object_names()[:4])
                        log(f"   🕐 {mem.timestamp[:16]}  📍 {mem.location}")
                        log(f"      📦 {objs}")
                    if len(result) > 5:
                        log(f"   ... +{len(result) - 5} more")

                    if hat.board:
                        # Show abbreviated narrative on LCD
                        short_narrative = narrative[:80] + "..." if len(narrative) > 80 else narrative
                        hat.display_text(
                            f"{entity}:\n\n{short_narrative}",
                            color=(100, 200, 255)
                        )
                        hat.board.set_rgb(0, 255, 0)

                    # TTS: Speak the activity summary
                    hat.speak(narrative, gemini=gemini, for_search=True)
                else:
                    log(f"[TIME] No memories from '{entity}'")
                    if hat.board:
                        hat.display_text(f"No memories:\n{entity}", color=(255, 100, 100))
                        hat.board.set_rgb(255, 0, 0)

            # --- RENDERER 2: NEAR / CO-OCCURRENCE QUERY ---
            # Shows a deduplicated list of objects seen in the same frame.
            # Example: "what was near my keys?" → wallet, phone, coffee mug
            elif query_type == "near":
                memory = result
                cooccurrences = extra
                if memory or cooccurrences:
                    print()
                    log(f"[NEAR] Objects seen with '{entity}':")
                    # Deduplicate co-occurring objects across multiple memories
                    # using lowercase key → original case value to preserve display
                    seen_with = {}
                    for mem_id, others in cooccurrences[:5]:
                        for o in others:
                            seen_with[o.lower()] = o
                    if seen_with:
                        for obj_name in list(seen_with.values())[:10]:
                            log(f"   📦 {obj_name}")
                    else:
                        log(f"   (no co-occurring objects found)")

                    # Also show the entity's last known location
                    if memory:
                        log(f"   📍 Last seen: {memory.location}")
                        log(f"   🕐 {memory.timestamp[:16]}")

                    if hat.board:
                        items = ", ".join(list(seen_with.values())[:3])
                        hat.display_text(
                            f"Near {entity}:\n{items}",
                            color=(100, 255, 200)
                        )
                        hat.board.set_rgb(0, 255, 0)
                else:
                    log(f"[NOT FOUND] '{entity}'")
                    # Check if object was recently seen attached
                    attached_info = temporal.get_attached_status(entity)
                    if attached_info:
                        last_time, _ = attached_info
                        log(f"   ℹ️  Last seen: on you at {last_time[:16]}")
                        if hat.board:
                            hat.display_text(f"Not found:\n{entity}\n\nLast: on you\n{last_time[11:16]}", color=(255, 150, 50))
                            hat.board.set_rgb(255, 100, 0)
                    elif hat.board:
                        hat.display_text(f"Not found:\n{entity}", color=(255, 100, 100))
                        hat.board.set_rgb(255, 0, 0)

            # --- RENDERER 2.5: PERSON QUERY (WHO dimension) ---
            # Shows people encountered in the memory time window.
            # Example: "who did I meet this morning?" → John, Sarah
            # This is Tulving's WHO dimension of episodic memory.
            elif query_type == "person":
                memories = result if isinstance(result, list) else []
                people = extra if isinstance(extra, list) else []

                # Also collect visual persons from memories
                visual_persons = []
                for mem in memories:
                    for vp in getattr(mem, 'persons', []):
                        desc = vp.get('description', '')
                        name = vp.get('name', '')
                        if name and name not in people:
                            visual_persons.append(f"{name} ({desc})")
                        elif desc and desc not in [p.lower() for p in people]:
                            visual_persons.append(desc)

                has_people = people or visual_persons

                if has_people:
                    print()
                    log(f"[WHO] People encountered:")
                    # Show audio-extracted names first
                    for person in people:
                        log(f"   👤 {person} (from conversation)")
                    # Show visual persons
                    for vp in visual_persons[:5]:
                        log(f"   👁️ {vp} (visual)")
                    # Show context from memories if available
                    for mem in memories[:3]:
                        if mem.conversation_context:
                            log(f"   💬 {mem.conversation_context}")
                            log(f"      📍 {mem.location} @ {mem.timestamp[:16]}")
                        elif mem.persons:
                            # Show visual context if no audio context
                            for vp in mem.persons[:2]:
                                ctx = vp.get('context', '')
                                if ctx:
                                    log(f"   👁️ {vp.get('description', 'person')}: {ctx}")
                            log(f"      📍 {mem.location} @ {mem.timestamp[:16]}")

                    if hat.board:
                        # Combine audio names and visual descriptions for display
                        all_people = people + visual_persons[:3]
                        people_str = ", ".join(all_people[:3])
                        if len(all_people) > 3:
                            people_str += f" +{len(all_people)-3}"
                        hat.display_text(
                            f"People:\n\n{people_str}",
                            color=(200, 150, 255)
                        )
                        hat.board.set_rgb(0, 255, 0)

                    # TTS: Speak the people found
                    speech_people = people if people else visual_persons[:3]
                    if speech_people:
                        speech = f"You saw {', '.join(speech_people[:3])}"
                        hat.speak(speech, gemini=gemini, for_search=True)
                else:
                    if entity:
                        log(f"[WHO] No memory of meeting '{entity}'")
                        msg = f"No memory of:\n{entity}"
                    else:
                        log(f"[WHO] No people found in memories")
                        msg = "No people found\nin memories"
                    if hat.board:
                        hat.display_text(msg, color=(255, 100, 100))
                        hat.board.set_rgb(255, 0, 0)

            # --- RENDERER 2.75: ACTIVITY QUERY (did I...) ---
            # Shows whether an activity was detected in memory.
            # Example: "did I take my medication?" → "Yes, taking medication at 8:30 AM"
            # This answers "did I..." questions about actions.
            elif query_type == "activity":
                memories = result if isinstance(result, list) else []
                activities = extra if isinstance(extra, list) else []

                if activities or memories:
                    print()
                    if activities:
                        log(f"[YES!] Activity detected:")
                        for act in activities[:3]:
                            log(f"   ✓ {act}")
                    # Show when and where
                    for mem in memories[:3]:
                        log(f"   📍 {mem.location} @ {mem.timestamp[:16]}")
                        if mem.activities:
                            log(f"      Activities: {', '.join(mem.activities[:3])}")

                    if hat.board:
                        # Show confirmation with first activity
                        act_str = activities[0] if activities else "activity detected"
                        loc_str = memories[0].location if memories else ""
                        time_str = memories[0].timestamp[11:16] if memories else ""
                        hat.display_text(
                            f"YES!\n\n{act_str}\n\n{loc_str}\n{time_str}",
                            color=(100, 255, 100)
                        )
                        hat.board.set_rgb(0, 255, 0)

                    # TTS: Confirm the activity
                    if activities and memories:
                        speech = f"Yes, I saw {activities[0]} at {memories[0].timestamp[11:16]}"
                        hat.speak(speech, gemini=gemini, for_search=True)
                else:
                    log(f"[NO] No memory of '{entity}'")
                    if hat.board:
                        hat.display_text(f"No memory of:\n\n{entity}", color=(255, 100, 100))
                        hat.board.set_rgb(255, 0, 0)
                    # TTS: Say we didn't see it
                    hat.speak(f"I don't have a memory of {entity}", gemini=gemini, for_search=True)

            # --- RENDERER 2.9: VQA QUERY (Visual Question Answering) ---
            # Shows the answer to a visual question about an object.
            # Example: "what color is the chair?" → "The chair is white"
            # Enhances episodic memory by answering WHAT details about remembered objects.
            elif query_type == "vqa":
                memory = result if not isinstance(result, list) else None
                vqa_answer = extra if isinstance(extra, str) else ""

                if memory and vqa_answer:
                    print()
                    log(f"[VQA] {vqa_question}")
                    log(f"   💡 {vqa_answer}")
                    log(f"   📍 {memory.location} @ {human_time(memory.timestamp)}")

                    if hat.board:
                        # Show the answer prominently
                        hat.display_text(
                            f"Answer:\n\n{vqa_answer[:60]}",
                            color=(150, 200, 255)
                        )
                        hat.board.set_rgb(0, 255, 0)

                    # TTS: Speak the answer
                    hat.speak(vqa_answer, gemini=gemini, for_search=True)
                else:
                    error_msg = vqa_answer if vqa_answer else f"Could not find '{entity}' in memory"
                    log(f"[VQA] {error_msg}")
                    if hat.board:
                        hat.display_text(f"Not found:\n\n{entity}", color=(255, 100, 100))
                        hat.board.set_rgb(255, 0, 0)
                    hat.speak(error_msg, gemini=gemini, for_search=True)

            # --- RENDERER 2.95: "ONLY IN HAND" FALLBACK ---
            # When user asks "where did I LEAVE X?" but we only have memories
            # of them HOLDING it, provide helpful feedback instead of wrong answer.
            # This is human-like: "I saw you holding your phone, but I didn't see you put it down."
            elif (result and not isinstance(result, list) and
                  isinstance(extra, tuple) and len(extra) == 2 and extra[0] == "only_in_hand"):
                memory = result  # type: ignore
                movements = extra[1]  # The actual movement list
                obj = memory.find_object(entity)

                print()
                log(f"[ONLY HOLDING]")
                log(f"   🖐️ I saw you holding your {entity}")
                log(f"   📍 {memory.location} @ {human_time(memory.timestamp)}")
                log(f"   ❓ I haven't seen you put it down yet")

                if movements:
                    print()
                    log("[MOVEMENT HISTORY]")
                    for i, m in enumerate(movements[:3]):
                        log(f"   {i+1}. {m.to_narrative()}")

                if hat.board:
                    hat.display_text(
                        f"Saw you holding:\n{entity}\n\nNot seen\nput down",
                        color=(255, 200, 100)
                    )
                    hat.board.set_rgb(255, 150, 0)

                # TTS: Helpful message
                speech = f"I saw you holding your {entity} at {memory.location}, but I haven't seen you put it down yet."
                hat.speak(speech, gemini=gemini, for_search=True)

            # --- RENDERER 3: OBJECT / SCENE QUERY (core episodic recall) ---
            # Shows: location, position, timestamp, context, movement history,
            # causal narrative (via Gemini), and annotated bounding box image.
            # This is the primary "where did I leave my keys?" flow.
            elif result and not isinstance(result, list):
                memory = result  # type: ignore
                movements = extra  # type: ignore
                obj = memory.find_object(entity)

                print()
                log(f"[FOUND!]")
                log(f"   📍 {memory.location}")
                log(f"   📌 {obj.position() if obj else '?'}")
                log(f"   🕐 {human_time(memory.timestamp)}")
                if obj and obj.context:
                    log(f"   📋 {obj.context}")

                # Show location info on LCD first (before image)
                if hat.board:
                    conf_str = f" ({obj.confidence:.0%})" if obj and obj.confidence < 1.0 else ""
                    loc_display = f"Found: {entity}{conf_str}\n\n{memory.location}\n{obj.position() if obj else ''}"
                    log(f"   [LCD] Displaying: {loc_display.replace(chr(10), ' | ')}")
                    hat.display_text(loc_display, color=(0, 255, 0))
                    time.sleep(2)  # Let user read location info

                # Show movement history (TEMPORAL REASONING / HOW dimension)
                # This is what distinguishes episodic from semantic memory:
                # not just WHERE something is, but HOW it got there.
                if movements:
                    print()
                    log("[MOVEMENT HISTORY]")
                    for i, m in enumerate(movements[:3]):
                        log(f"   {i+1}. {m.to_narrative()}")

                    # Generate causal narrative using Gemini (thinking_level=LOW)
                    # Example: "Your keys are on the kitchen counter. You put
                    # them there 2h ago when you came back from shopping."
                    print()
                    narrative = gemini.generate_causal_narrative(
                        entity,
                        movements,
                        memory.location,
                        obj.position() if obj else "unknown"
                    )
                    log(f"[NARRATIVE] {narrative}")

                # Draw bounding boxes on the memory image: green for the searched
                # object, blue for others. Save annotated JPEG to result.jpg
                # (overwritten each search) and display on HAT LCD if available.
                # Include location and movement info as text banner at bottom.
                info_lines = [f"Location: {memory.location}", f"Position: {obj.position() if obj else '?'}"]
                if movements:
                    # Add most recent movement
                    info_lines.append(movements[0].to_narrative())
                info_text = "\n".join(info_lines)
                annotated = annotate_image(memory, entity, info_text)
                if annotated:
                    path = MEMORY_DIR / "result.jpg"
                    path.write_bytes(annotated)
                    log(f"   🖼️  {path}")

                    if hat.board:
                        # Pass info_text to draw AFTER scaling (readable on small LCD)
                        conf_pct = f" {obj.confidence:.0%}" if obj and obj.confidence < 1.0 else ""
                        lcd_info = f"{memory.location}{conf_pct}\n{obj.position() if obj else '?'}"
                        hat.display_image(annotated, info_text=lcd_info)
                        hat.board.set_rgb(0, 255, 0)

                    # TTS: Speak result (in addition to LCD display)
                    speech = f"Found your {entity} on the {memory.location}"
                    if obj and obj.context:
                        speech += f", {obj.context}"
                    hat.speak(speech, gemini=gemini, for_search=True)
                else:
                    log(f"   ⚠️  No image data (image_data={len(memory.image_data) if memory.image_data else 0} bytes)")

            # --- RENDERER 4: NOT FOUND ---
            else:
                log(f"[NOT FOUND] '{entity}'")

                # Check if object was recently seen attached (worn/held)
                attached_info = temporal.get_attached_status(entity)
                if attached_info:
                    last_time, last_loc = attached_info
                    log(f"   ℹ️  Last seen: on you at {last_time[:16]}")
                    if hat.board:
                        hat.display_text(f"Not found:\n{entity}\n\nLast: on you\n{last_time[11:16]}", color=(255, 150, 50))
                        hat.board.set_rgb(255, 100, 0)
                    # TTS for attached object
                    hat.speak(f"I haven't seen your {entity} recently. Last seen on you at {last_time[11:16]}", gemini=gemini, for_search=True)
                else:
                    # Use Gemini's world knowledge to suggest where to look
                    # This is smarter than hardcoded suggestions - Gemini knows
                    # common locations for any object type
                    suggestions = gemini.suggest_locations(entity)

                    if suggestions:
                        log(f"[SUGGESTIONS] Gemini suggests trying:")
                        for s in suggestions[:3]:
                            log(f"   → {s}")

                        # LCD: Show top suggestion
                        top_suggestion = suggestions[0]
                        if hat.board:
                            hat.display_text(f"Not found:\n{entity}\n\nTry:\n{top_suggestion}", color=(255, 200, 100))
                            hat.board.set_rgb(255, 150, 0)

                        # TTS: Speak suggestion
                        hat.speak(f"I haven't seen your {entity}. Try looking {top_suggestion}", gemini=gemini, for_search=True)
                    else:
                        if hat.board:
                            hat.display_text(f"Not found:\n{entity}", color=(255, 100, 100))
                            hat.board.set_rgb(255, 0, 0)
                        # TTS for not found
                        hat.speak(f"I haven't seen your {entity}", gemini=gemini, for_search=True)

            # Persist access_log after each query so retrieval reinforcement
            # data survives crashes (important on battery-powered wearable)
            index.save()

            # Wait for button press to dismiss results (or timeout after 30s)
            # This gives user time to view the result on the LCD
            if hat.board:
                log("   (Press button to continue)")
                wait_start = time.time()
                while time.time() - wait_start < 30:  # 30 second timeout
                    if hat.button_pressed():
                        # Debounce: wait for button release
                        while hat.button_pressed():
                            time.sleep(0.05)
                        time.sleep(0.2)  # Extra delay to prevent triggering next voice input
                        last_voice_time = time.time()  # Reset cooldown
                        break
                    time.sleep(0.1)
                hat.display_text("Ready!\n\nGEM Search", color=(100, 200, 255))
                hat.board.set_rgb(0, 100, 255)
            else:
                time.sleep(2)  # No HAT, just brief pause

    except KeyboardInterrupt:
        print()
        log("[SEARCH] Stopped")
    finally:
        index.save()
        hat.cleanup()


def cmd_list():
    """
    List stored memories with temporal statistics.

    Shows recent memories, movement statistics, and most active objects.
    """
    init_storage()

    # Get all memory files (exclude memory_index.json)
    files = sorted(MEMORY_DIR.glob("mem_*.json"), reverse=True)

    # Load temporal graph for statistics
    temporal = TemporalGraph()
    temporal.load(DATA_DIR / "temporal_graph.json")

    if not files:
        log("No memories. Run: python gem.py")
        return

    print()
    log("=" * 60)
    log(f"  GEM Memory Store")
    log("=" * 60)
    log(f"📚 {len(files)} memories")
    log(f"🔄 {temporal.total_movements} object movements tracked")
    log(f"📦 {len(temporal.last_seen)} unique objects seen")
    log("=" * 60)

    # Show recent memories (with tags)
    for path in files[:20]:
        try:
            data = json5.loads(path.read_text())
            mem_id = data.get("id", path.stem)
            loc = data.get("location", "?")
            objs = [o.get("name", "?") for o in data.get("objects", [])[:4]]
            ts = data.get("timestamp", "")[:16]
            tags = data.get("tags", [])

            print(f"\n  {mem_id}")
            print(f"    📍 {loc}")
            print(f"    📦 {', '.join(objs)}")
            if tags:
                print(f"    🏷️  {', '.join(tags[:3])}")
            print(f"    🕐 {ts}")
        except Exception:
            pass

    if len(files) > 20:
        print(f"\n  ... +{len(files) - 20} more")

    # Show most active objects
    if temporal.movements:
        print()
        log("Most active objects (by movements):")
        sorted_objs = sorted(
            temporal.movements.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        for obj_name, moves in sorted_objs:
            print(f"    {obj_name}: {len(moves)} movements")

    print()


def cmd_hw_test():
    """
    Test all hardware components and Gemini connection.
    
    Runs through each component to verify the system is working:
    1. Gemini API connection
    2. Camera capture → LCD display
    3. Button + LED
    4. Microphone → Gemini STT
    """
    init_storage()
    
    print()
    log("=" * 50)
    log("  GEM Hardware Test ")
    log("=" * 50)
    print()
    
    results = {}
    hat = WhisplayHAT()
    
    # Test 1: Gemini API
    log("[TEST 1/4] Gemini 3 API")
    log("-" * 40)
    gemini = None
    try:
        gemini = GeminiClient()
        log("   ✅ Client initialized")

        # Actually test the API with a real call (optimized for speed)
        log("   Testing API connection...")
        response = gemini.client.models.generate_content(
            model=VISION_MODEL,
            contents="Reply: OK",
            config=types.GenerateContentConfig(
                max_output_tokens=5,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL
                )
            )
        )

        if response and response.text:
            results["gemini"] = True
            log(f"   ✅ API responded: {response.text.strip()}")
        else:
            results["gemini"] = False
            log("   ❌ API returned empty response")

    except Exception as e:
        results["gemini"] = False
        log(f"   ❌ {e}")

    print()
    

    # Test 2: Camera + LCD
    log("[TEST 2/4] Camera → LCD")
    log("-" * 40)
    try:
        if PICAMERA_AVAILABLE:
            camera = Camera()
            jpeg, _ = camera.capture()
            camera.close()
            
            path = MEMORY_DIR / "test_camera.jpg"
            path.write_bytes(jpeg)
            log(f"   📷 {path} ({len(jpeg)//1024}KB)")
            
            if hat.board:
                hat.display_image(jpeg)
            
            results["camera"] = True
            log("   ✅ Camera OK")
        else:
            log("   ⚠️  picamera2 not installed")
            results["camera"] = False

    except Exception as e:
        results["camera"] = False
        log(f"   ❌ {e}")

    print()
    time.sleep(2)


    # Test 3: Button + LED
    log("[TEST 3/4] Button + LED")
    log("-" * 40)
    if hat.board:
        hat.display_text("Press\nbutton!", color=(255, 255, 0))
        log("   🔘 Waiting 5s...")
        
        pressed = False
        start = time.time()
        flash = False
        
        while time.time() - start < 5:
            if hat.button_pressed():
                pressed = True
                break

            flash = not flash
            hat.board.set_rgb(255, 255, 0) if flash else hat.board.set_rgb(0, 0, 0)
            time.sleep(0.15)
        
        if pressed:
            hat.board.set_rgb(0, 255, 0)
            results["button"] = True
            log("   ✅ Button OK")
        else:
            results["button"] = False
            log("   ⚠️  No press (LED works)")
    
    else:
        results["button"] = False
        log("   ⚠️  HAT not available")
    
    print()
    time.sleep(1)


    # Test 4: Mic → Gemini STT (+ Speaker if TTS enabled)
    test_name = "Mic → Speaker → Gemini STT" if TTS_ENABLED else "Mic → Gemini STT"
    log(f"[TEST 4/4] {test_name}")
    log("-" * 40)

    if hat.mic and gemini:
        if hat.board:
            hat.display_text("Recording\n3s...\n\nSpeak!", color=(255, 100, 100))

        audio = hat.record_audio(3)

        if audio and len(audio) > 10000:
            path = MEMORY_DIR / "test_audio.wav"
            path.write_bytes(audio)
            log(f"   💾 {path} ({len(audio)//1024}KB)")

            # Test speaker: playback the recording (only if TTS enabled)
            if TTS_ENABLED:
                if hat.board:
                    hat.display_text("Playing back...\n\n(Speaker test)", color=(255, 200, 100))
                    hat.board.set_rgb(0, 0, 255)  # Blue during playback

                log("   🔊 Playing back recording...")
                try:
                    subprocess.run(
                        ["aplay", "-D", "plughw:0,0", "-q", str(path)],
                        timeout=10, capture_output=True
                    )
                    log("   ✅ Speaker playback OK")
                    results["speaker"] = True
                except Exception as e:
                    log(f"   ⚠️  Speaker playback failed: {e}")
                    results["speaker"] = False
            else:
                log("   ⏭️  Speaker test skipped (TTS disabled)")
                results["speaker"] = True  # Mark as passed (not applicable)

            if hat.board:
                hat.display_text("Transcribing...", color=(100, 200, 255))
                hat.board.set_rgb(255, 255, 255)  # White during transcription

            transcript = gemini.transcribe_audio(audio)

            if transcript:
                results["mic_stt"] = True
                log(f"   📝 \"{transcript}\"")
                log("   ✅ Mic → Gemini STT OK")

                if hat.board:
                    hat.display_text(f"Heard:\n{transcript[:30]}", color=(0, 255, 0))
            else:
                results["mic_stt"] = False
                log("   ⚠️  Transcription empty")

        else:
            results["mic_stt"] = False
            results["speaker"] = True if not TTS_ENABLED else False
            log("   ❌ Recording failed")

    else:
        results["mic_stt"] = False
        results["speaker"] = True if not TTS_ENABLED else False
        log("   ⚠️  Mic or Gemini not available")
    
    print()
    
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    log("=" * 50)
    log(f"  Results: {passed}/{total} passed")
    log("=" * 50)
    
    if hat.board:
        time.sleep(2)
        color = (0, 255, 0) if passed == total else (255, 255, 0)
        hat.display_text(f"Tests Done\n\n{passed}/{total} OK", color=color)
    
    time.sleep(3)
    hat.cleanup()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point - parse command and run.
    
    Commands:
        (none)    - Run capture daemon (Marathon Agent)
        search    - Interactive search mode
        list      - List stored memories
        test      - Test hardware components
        --help    - Show documentation
    """
    if len(sys.argv) >= 2:
        cmd = sys.argv[1].lower()
        
        if cmd in ["-h", "--help", "help"]:
            print(__doc__)
            return
        
        if cmd == "list":
            cmd_list()
            return
        
        if cmd in ["hw_test", "test"]:  # hw_test preferred, test for backwards compat
            cmd_hw_test()
            return
        
        if cmd == "search":
            cmd_search()
            return

        if cmd in ["--headless", "headless", "-H"]:
            # Run daemon without HAT (allows search to use LCD/mic)
            cmd_daemon(headless=True)
            return

    # Default: run capture daemon with HAT
    cmd_daemon(headless=False)


if __name__ == "__main__":
    main()
