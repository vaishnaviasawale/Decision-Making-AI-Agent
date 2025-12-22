"""
Configuration settings for the Decision Making Agent.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "amazon.csv"

# LLM Configuration
CLAUDE_API_KEY = os.getenv(
    "CLAUDE_API_KEY",
    "",
)

MODEL_NAME = os.getenv("MODEL_NAME", "")
TEMPERATURE = float(os.getenv("TEMPERATURE", ""))

# Agent Configuration
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"
