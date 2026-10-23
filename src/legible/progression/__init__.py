"""Phase 5: mastery-gated, compounding difficulty shared by every learning module."""

from .curves import Curve
from .engine import Outcome, TrackSpec, TrackState, apply_session
from .tracks import TRACKS, caption_mode

__all__ = ["TRACKS", "Curve", "Outcome", "TrackSpec", "TrackState", "apply_session", "caption_mode"]
