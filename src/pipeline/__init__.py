"""Zoom recordings to syllabus-organised, transcribed lectures on Google Drive.

See docs/ARCHITECTURE.md for the full design. The short version:

    ingest -> media -> asr -> classify -> name -> place -> publish

Every stage is idempotent and every transition is persisted, so a worker can
die at any point and the job resumes from its last committed state.
"""

__version__ = "0.1.0"
