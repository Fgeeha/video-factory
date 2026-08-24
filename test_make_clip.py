#!/usr/bin/env python3
"""Assert-based self-check for the pure logic in make_clip.py (no ffmpeg/piper needed).

Usage: uv run test_make_clip.py
"""
from make_clip import build_zoompan_filter, split_durations, _srt_ts


def test_split_durations_even():
    durations = split_durations(3, 9.0)
    assert len(durations) == 3
    assert abs(sum(durations) - 9.0) < 0.05


def test_split_durations_given():
    durations = split_durations(2, 9.0, given=[3.0, 6.0])
    assert durations == [3.0, 6.0]


def test_split_durations_mismatch_raises():
    try:
        split_durations(2, 9.0, given=[1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on length mismatch")


def test_zoompan_filter_contains_frame_count():
    f = build_zoompan_filter(duration=2.0)
    assert "d=60" in f  # 2s * 30fps
    assert "zoompan" in f


def test_srt_timestamp_format():
    assert _srt_ts(0) == "00:00:00,000"
    assert _srt_ts(65.5) == "00:01:05,500"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok: {name}")
    print("all tests passed")
