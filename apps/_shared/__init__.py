"""Shared infrastructure for the federation specialist apps."""
from .specialist_factory import build_specialist_app, SpecialistConfig

__all__ = ["build_specialist_app", "SpecialistConfig"]
