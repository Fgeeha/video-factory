#!/usr/bin/env python3
"""Assert-based self-check for the pure logic in make_clip.py (no ffmpeg/piper needed).

Usage: uv run test_make_clip.py
"""
from make_clip import build_zoompan_filter, split_durations, _srt_ts
from comfy_client import build_zimage_workflow, build_wan_t2v_workflow


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


def test_zimage_workflow_wires_prompt_and_size():
    wf = build_zimage_workflow("a truck", width=768, height=1344, seed=42)["prompt"]
    assert wf["4"]["inputs"]["prompt"] == "a truck"
    assert wf["6"]["inputs"]["width"] == 768
    assert wf["6"]["inputs"]["height"] == 1344
    assert wf["7"]["inputs"]["seed"] == 42


def test_wan_t2v_workflow_splits_steps_across_high_low_noise():
    wf = build_wan_t2v_workflow("a truck", width=480, height=832, length=33, seed=7)["prompt"]
    assert wf["11"]["inputs"]["length"] == 33
    high = wf["12"]["inputs"]
    low = wf["13"]["inputs"]
    assert (high["start_at_step"], high["end_at_step"]) == (0, 2)
    assert (low["start_at_step"], low["end_at_step"]) == (2, 4)
    assert high["add_noise"] == "enable" and low["add_noise"] == "disable"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok: {name}")
    print("all tests passed")
