"""Deterministic schedule and event decisions. No PsychoPy or device access."""

from dataclasses import dataclass, replace
import random
import math


def validate_hardware(h):
    names = [h["trigger_key"], *h["response_keys"], h["arm_key"], h["abort_key"]]
    if len(set(names)) != len(names):
        raise ValueError("Trigger, response, arm and abort keys must be distinct")
    if h["keyboard_backend"] != "ptb":
        raise ValueError("This example requires the PTB asynchronous keyboard backend")
    if len(h["response_keys"]) != 2 or h["expected_tr_s"] <= 0:
        raise ValueError("Two response keys and a positive expected TR are required")


def make_schedule(spec, participant):
    rng = random.Random(spec["seed"] + participant)
    n = spec["trials_per_condition"]
    if not isinstance(n, int) or n <= 0 or n % 2:
        raise ValueError("Use a positive even trial count per condition")
    intervals = spec["trial_intervals_s"]
    durations = [
        spec[k]
        for k in [
            "stimulus_duration_s",
            "response_window_s",
            "initial_fixation_s",
            "final_fixation_s",
            "grating_size_height",
            "grating_cycles",
        ]
    ]
    if len(intervals) != 2 or any(
        not math.isfinite(v) or v <= 0 for v in durations + intervals
    ):
        raise ValueError(
            "Use two positive intervals and positive finite durations and stimulus sizes"
        )
    if max(spec["stimulus_duration_s"], spec["response_window_s"]) >= min(intervals):
        raise ValueError("Stimulus and response window must fit before the next trial")
    if (
        set(spec["contrasts"]) != {"high", "low"}
        or not 0 < spec["contrasts"]["low"] < spec["contrasts"]["high"] <= 1
    ):
        raise ValueError("Use high and low contrast with 0 < low < high <= 1")
    cells = [
        (c, o)
        for c in spec["contrasts"]
        for o in spec["orientations_deg"]
        for _ in range(n // 2)
    ]
    if n % 2 or len(spec["orientations_deg"]) != 2:
        raise ValueError("Use an even trial count and two orientations")
    rng.shuffle(cells)
    intervals = spec["trial_intervals_s"] * (len(cells) // 2)
    rng.shuffle(intervals)
    onset = spec["initial_fixation_s"]
    rows = []
    for i, ((condition, ori), interval) in enumerate(zip(cells, intervals)):
        rows.append(
            dict(
                trial=i,
                condition=condition,
                orientation=ori,
                contrast=spec["contrasts"][condition],
                planned_onset=onset,
                duration=spec["stimulus_duration_s"],
                response_window=spec["response_window_s"],
                response_index=(int(ori > 0) + participant % 2) % 2,
            )
        )
        onset += interval
    return rows, onset + spec["final_fixation_s"]


@dataclass(frozen=True)
class State:
    phase: str = "ready"
    armed_at: float | None = None
    zero: float | None = None
    last_trigger: float | None = None
    missing_reported: bool = False
    ready_at: float = 0.0


def route(state, events, h, now):
    """Route every key-down once. Equal-time start precedes a response."""
    records = []
    order = {h["abort_key"]: 0, h["arm_key"]: 1, h["trigger_key"]: 2}
    for key, timestamp in sorted(events, key=lambda e: (e[1], order.get(e[0], 3))):
        category = "ignored"
        if timestamp < state.ready_at:
            category = "stale_input"
        elif state.phase in ("aborted", "timeout"):
            category = "after_terminal"
        elif key == h["abort_key"]:
            state = replace(state, phase="aborted")
            category = "abort"
        elif key == h["arm_key"] and state.phase == "ready":
            state = replace(state, phase="armed", armed_at=timestamp)
            category = "arm"
        elif key == h["trigger_key"]:
            if state.armed_at is None or timestamp < state.armed_at:
                category = "stale_trigger"
            elif state.zero is None:
                state = replace(
                    state, phase="running", zero=timestamp, last_trigger=timestamp
                )
                category = "start"
            else:
                interval = timestamp - state.last_trigger
                category = (
                    "duplicate_trigger"
                    if interval < h["duplicate_trigger_threshold_s"]
                    else "trigger"
                )
                if category == "trigger":
                    if (
                        abs(interval - h["expected_tr_s"])
                        > h["trigger_interval_tolerance_s"]
                    ):
                        category = "trigger_interval_warning"
                    state = replace(
                        state, last_trigger=timestamp, missing_reported=False
                    )
        elif key in h["response_keys"]:
            category = (
                "response"
                if state.zero is not None and timestamp >= state.zero
                else "early_response"
            )
        records.append(
            dict(key=key, timestamp=timestamp, category=category, retrieved_at=now)
        )
    if state.phase == "armed" and now - state.armed_at >= h["start_timeout_s"]:
        state = replace(state, phase="timeout")
    if (
        state.phase == "running"
        and not state.missing_reported
        and now - state.last_trigger > 1.5 * h["expected_tr_s"]
    ):
        records.append(
            dict(key="", timestamp=now, category="missing_trigger", retrieved_at=now)
        )
        state = replace(state, missing_reported=True)
    return state, records


def score_responses(schedule, flips, records, keys):
    """Use displayed onset and a fixed response window, independent of poll time."""
    used = set()
    result = []
    for trial in schedule:
        i = trial["trial"]
        if i not in flips or "offset" not in flips[i]:
            continue
        onset = flips[i]["onset"]
        candidates = [
            (j, e)
            for j, e in enumerate(records)
            if e["category"] == "response"
            and j not in used
            and onset <= e["timestamp"] < onset + trial["response_window"]
        ]
        row = dict(
            trial,
            flip_onset=onset,
            flip_offset=flips[i]["offset"],
            response="n/a",
            rt="n/a",
            correct="n/a",
        )
        if candidates:
            j, event = min(candidates, key=lambda x: x[1]["timestamp"])
            used.add(j)
            row.update(
                response=event["key"],
                rt=event["timestamp"] - onset,
                correct=int(event["key"] == keys[trial["response_index"]]),
            )
        result.append(row)
    return result
