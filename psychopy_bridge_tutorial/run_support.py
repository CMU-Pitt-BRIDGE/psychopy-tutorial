"""Input collection and saved records for the Builder lesson.

The experiment's routines and visual components remain in starter.psyexp.
"""

import atexit
import csv
import json
import os
from pathlib import Path
import time
from dataclasses import replace

from trial_core import State, make_schedule, route, score_responses, validate_hardware


def write_table(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class Session:
    def __init__(self, win, participant):
        from psychopy import logging
        from psychopy.hardware.keyboard import Keyboard

        self.win, self.clock = win, logging.defaultClock
        root = Path(__file__).resolve().parent
        self.spec = json.loads((root / "task.json").read_text())
        self.hardware = dict(
            arm_key="space",
            abort_key="escape",
            keyboard_backend="ptb",
            start_timeout_s=60,
            duplicate_trigger_threshold_s=0.1,
            trigger_interval_tolerance_s=0.05,
            **json.loads((root / "hardware.json").read_text()),
        )
        validate_hardware(self.hardware)
        self.trials, self.end = make_schedule(self.spec, participant)
        for i, row in enumerate(self.trials):
            next_onset = (
                self.trials[i + 1]["planned_onset"]
                if i + 1 < len(self.trials)
                else self.end - self.spec["final_fixation_s"]
            )
            row["interval_s"] = next_onset - row["planned_onset"]
        self.kb = Keyboard(backend="ptb", clock=self.clock, bufferSize=10000)
        if self.kb.getBackend() != "ptb":
            raise RuntimeError("This lesson requires the PTB keyboard backend.")
        self.state = State(ready_at=self.clock.getTime())
        self.records, self.frames, self.flips = [], [], {}
        self.current = self.stimulus = None
        self.status, self.saved = "incomplete", False
        self.demo = os.environ.get("BRIDGE_DEMO") == "1"
        self.abort_at = float(os.environ.get("BRIDGE_ABORT_S", "inf"))
        self.last_demo_pulse = -1
        self.responded = set()
        self.out = root / "data" / f"p{participant:03d}_{time.time_ns()}"
        self.out.mkdir(parents=True)
        write_table(self.out / "schedule.tsv", self.trials)
        (self.out / "settings.json").write_text(
            json.dumps(
                dict(
                    task=self.spec,
                    hardware=self.hardware,
                    participant=participant,
                    software_demo=self.demo,
                ),
                indent=2,
            )
        )
        (self.out / "status.json").write_text('{"status":"incomplete"}')
        atexit.register(self.save)

    def next_time(self):
        if self.state.zero is None:
            return float("-inf")
        return self.win.getFutureFlipTime(clock=self.clock) - self.state.zero

    def capture_flips(self):
        if self.current is None:
            return
        start = getattr(self.stimulus, "tStartRefresh", None)
        stop = getattr(self.stimulus, "tStopRefresh", None)
        if start is not None:
            self.flips[self.current] = dict(onset=start)
            if stop is not None:
                self.flips[self.current]["offset"] = stop

    def frame_stamp(self):
        now = self.win._frameTime
        if self.frames and now == self.frames[-1]["timestamp"]:
            return
        last = self.frames[-1]["timestamp"] if self.frames else None
        self.frames.append(
            dict(timestamp=now, interval="n/a" if last is None else now - last)
        )

    def poll(self):
        self.capture_flips()
        now = self.clock.getTime()
        allowed = {
            self.hardware["trigger_key"],
            *self.hardware["response_keys"],
            "space",
            "escape",
        }
        events = [
            (e.name, e.tDown)
            for e in self.kb.getKeys(waitRelease=False, clear=True)
            if e.name in allowed
        ]
        # Only the explicit desktop demo injects input. These are not hardware tests.
        if self.demo:
            if self.state.phase == "ready" and now - self.state.ready_at > 1:
                events.append(("space", now))
            elif self.state.phase == "armed" and now - self.state.armed_at > 1:
                events.append((self.hardware["trigger_key"], now))
            if self.state.zero is not None:
                elapsed = now - self.state.zero
                pulse = int(elapsed / self.hardware["expected_tr_s"])
                if pulse > self.last_demo_pulse:
                    self.last_demo_pulse = pulse
                    if pulse:
                        events.append((self.hardware["trigger_key"], now))
                if elapsed >= self.abort_at:
                    events.append(("escape", now))
                if self.current in self.flips and self.current not in self.responded:
                    if now - self.flips[self.current]["onset"] > 0.45:
                        self.responded.add(self.current)
                        if self.current % 3:
                            key = self.hardware["response_keys"][
                                self.trials[self.current]["response_index"]
                            ]
                            events.append((key, now))
        self.state, rows = route(self.state, events, self.hardware, now)
        self.records.extend(rows)
        if self.state.phase in ("aborted", "timeout"):
            self.status = self.state.phase
            self.save()
            self.win.close()
            from psychopy import core

            core.quit()
        self.win.callOnFlip(self.frame_stamp)

    def begin_trial(self, trial, stimulus):
        self.current, self.stimulus = int(trial), stimulus

    def complete(self):
        self.capture_flips()
        self.status = "software_demo_complete" if self.demo else "complete"
        self.run_end = self.clock.getTime()
        self.state = replace(self.state, phase="complete", ended_at=self.run_end)
        self.save()

    def save(self):
        if self.saved:
            return
        self.capture_flips()
        saved_at = self.clock.getTime()
        end_time = self.state.ended_at
        if end_time is None:
            end_time = saved_at
        rows = score_responses(
            self.trials,
            self.flips,
            self.records,
            self.hardware["response_keys"],
            end_time,
        )
        for row in rows:
            row["onset"] = row["flip_onset"] - self.state.zero
            row["offset"] = row["flip_offset"] - self.state.zero
            row["onset_error"] = row["onset"] - row["planned_onset"]
        write_table(self.out / "trials.tsv", rows)
        write_table(self.out / "inputs.tsv", self.records)
        write_table(self.out / "frames.tsv", self.frames)
        (self.out / "flips.json").write_text(json.dumps(self.flips, indent=2))
        (self.out / "status.json").write_text(
            json.dumps(
                dict(
                    status=self.status,
                    start=self.state.zero,
                    end=end_time,
                    saved_at=saved_at,
                    software_demo=self.demo,
                ),
                indent=2,
            )
        )
        self.saved = True
