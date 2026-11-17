"""Convert frame/sample trigger scores into stable temporal event intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Event:
    start_sec: float
    end_sec: float
    peak_score: float
    method: str


def scores_to_events(rows: list[tuple[float, float]], threshold: float, method: str, min_active: int = 2, min_inactive: int = 2) -> list[Event]:
    """Apply hysteresis so a one-frame score spike does not create a saved clip."""
    events: list[Event] = []
    active_count = inactive_count = 0
    start: float | None = None
    peak = 0.0
    last_time = 0.0
    for time_sec, score in rows:
        last_time = time_sec
        if score >= threshold:
            active_count += 1
            inactive_count = 0
            peak = max(peak, score)
            if start is None and active_count >= min_active:
                start = time_sec
        else:
            inactive_count += 1
            active_count = 0
            if start is not None and inactive_count >= min_inactive:
                events.append(Event(start, time_sec, peak, method))
                start, peak = None, 0.0
    if start is not None:
        events.append(Event(start, last_time, peak, method))
    return events


def expand_and_merge(events: list[Event], duration: float, pre: float, post: float, gap: float) -> list[Event]:
    """Add pre/post context and merge nearby events without exceeding video bounds."""
    expanded = [Event(max(0.0, e.start_sec - pre), min(duration, e.end_sec + post), e.peak_score, e.method) for e in events]
    if not expanded:
        return []
    merged = [expanded[0]]
    for current in expanded[1:]:
        previous = merged[-1]
        if current.start_sec - previous.end_sec <= gap:
            previous.end_sec = max(previous.end_sec, current.end_sec)
            previous.peak_score = max(previous.peak_score, current.peak_score)
        else:
            merged.append(current)
    return merged


def events_as_dicts(events: list[Event]) -> list[dict[str, object]]:
    return [asdict(event) for event in events]

