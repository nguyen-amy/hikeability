"""Vercel Python serverless entry point.

Vercel's @vercel/python runtime auto-detects files under /api and exposes
the WSGI `app` symbol as the function handler.
"""
import sys
from pathlib import Path

# Make the parent directory importable so `from app import create_app` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app

app = create_app()
