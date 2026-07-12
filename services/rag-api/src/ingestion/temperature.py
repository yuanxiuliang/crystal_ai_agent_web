from __future__ import annotations


def coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def format_number(value: float) -> str:
    if abs(value - round(value)) < 0.01:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_temperature_c(value: object) -> str | None:
    number = coerce_float(value)
    if number is None:
        return None
    return f"{format_number(number)} C"


def format_duration_hours(value: object) -> str | None:
    hours = coerce_float(value)
    if hours is None:
        return None
    formatted_hours = f"{format_number(hours)} h"
    if hours >= 24:
        days = hours / 24
        return f"{formatted_hours} ({format_number(days)} days)"
    return formatted_hours
