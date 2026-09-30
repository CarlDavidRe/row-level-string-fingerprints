from __future__ import annotations

from time import perf_counter


def format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def log_progress(message: str, started_at: float | None = None) -> None:
    prefix = ""
    if started_at is not None:
        prefix = f"[{format_duration(perf_counter() - started_at)}] "
    print(prefix + message, flush=True)
