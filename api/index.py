"""Vercel Python serverless entry point.

Vercel's @vercel/python runtime auto-detects files under /api and exposes
the WSGI `app` symbol as the function handler. We add the project root to
sys.path so we can import the app package that lives at the repo root.
"""
import os
import sys

# Repo root is one level up from /api; add it to sys.path so `from app import ...` works.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app import create_app

app = create_app()
