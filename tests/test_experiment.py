import json
from pathlib import Path
import unittest
from psychopy_bridge_tutorial.trial_core import (
    State,
    make_schedule,
    route,
    score_responses,
    validate_hardware,
)

ROOT = Path(__file__).resolve().parents[1] / "psychopy_bridge_tutorial"
SPEC = json.loads((ROOT / "task.json").read_text())
H = dict(
    trigger_key="5",
    response_keys=["1", "2"],
    arm_key="space",
    abort_key="escape",
    keyboard_backend="ptb",
    expected_tr_s=2,
    start_timeout_s=60,
    duplicate_trigger_threshold_s=0.1,
    trigger_interval_tolerance_s=0.05,
)


class ExperimentTests(unittest.TestCase):
    def test_invalid_settings(self):
        for spec in [
            dict(SPEC, trials_per_condition=0),
            dict(SPEC, trials_per_condition=15),
            dict(SPEC, response_window_s=6),
            dict(SPEC, trial_intervals_s=[4, 6, 8]),
        ]:
            with self.assertRaises(ValueError):
                make_schedule(spec, 0)

    def test_shorter_window_same_schedule(self):
        original, end = make_schedule(SPEC, 0)
        changed, new_end = make_schedule(dict(SPEC, response_window_s=0.5), 0)
        self.assertEqual(
            [r["planned_onset"] for r in original],
            [r["planned_onset"] for r in changed],
        )
        self.assertEqual(end, new_end)
        self.assertTrue(all(r["response_window"] == 0.5 for r in changed))

    def test_balanced_schedule(self):
        rows, end = make_schedule(SPEC, 0)
        self.assertEqual((len(rows), end), (32, 176))
        self.assertEqual(
            {
                (c, o): sum(r["condition"] == c and r["orientation"] == o for r in rows)
                for c in ["high", "low"]
                for o in [-45, 45]
            },
            {("high", -45): 8, ("high", 45): 8, ("low", -45): 8, ("low", 45): 8},
        )
        self.assertEqual(make_schedule(SPEC, 0), make_schedule(SPEC, 0))
        intervals = [
            b["planned_onset"] - a["planned_onset"] for a, b in zip(rows, rows[1:])
        ] + [end - SPEC["final_fixation_s"] - rows[-1]["planned_onset"]]
        self.assertEqual(sorted(intervals), [4.0] * 16 + [6.0] * 16)

    def test_button_mapping(self):
        even = make_schedule(SPEC, 0)[0]
        odd = make_schedule(SPEC, 1)[0]
        for rows, parity in [(even, 0), (odd, 1)]:
            self.assertTrue(
                all(
                    r["response_index"] == (int(r["orientation"] > 0) + parity) % 2
                    for r in rows
                )
            )

    def test_conflicting_keys(self):
        with self.assertRaises(ValueError):
            validate_hardware(dict(H, trigger_key="1"))

    def test_start_timestamp(self):
        state, events = route(
            State(ready_at=8), [("5", 9), ("space", 10), ("1", 11), ("5", 11)], H, 11.2
        )
        self.assertEqual(state.zero, 11)
        self.assertEqual(
            [e["category"] for e in events],
            ["stale_trigger", "arm", "start", "response"],
        )
        self.assertEqual(events[-1]["timestamp"], 11)
        self.assertEqual(events[-1]["retrieved_at"], 11.2)

    def test_later_triggers(self):
        state = State("running", armed_at=0, zero=1, last_trigger=1)
        state, events = route(state, [("5", 1.01), ("5", 5)], H, 5.1)
        self.assertEqual(state.zero, 1)
        self.assertEqual(
            [e["category"] for e in events],
            ["duplicate_trigger", "trigger_interval_warning"],
        )

    def test_abort_and_timeout(self):
        state, _ = route(State(), [("escape", 1)], H, 1)
        self.assertEqual(state.phase, "aborted")
        state, _ = route(State("armed", armed_at=1), [], H, 61)
        self.assertEqual(state.phase, "timeout")

    def test_response_scoring(self):
        rows = make_schedule(SPEC, 0)[0][:2]
        flips = {
            0: {"onset": 100.0, "offset": 101.0},
            1: {"onset": 106.0, "offset": 107.0},
        }
        records = [
            dict(category="response", key=k, timestamp=t)
            for k, t in [("1", 99.9), ("2", 100.4), ("1", 100.6), ("1", 107.5)]
        ]
        scored = score_responses(rows, flips, records, H["response_keys"], 108)
        self.assertEqual(scored[0]["response"], "2")
        self.assertAlmostEqual(scored[0]["rt"], 0.4)
        self.assertEqual(
            (scored[1]["response"], scored[1]["rt"], scored[1]["correct"]),
            ("n/a", "n/a", "n/a"),
        )

    def test_trial_without_offset(self):
        rows = make_schedule(SPEC, 0)[0][:1]
        self.assertEqual(
            score_responses(rows, {0: {"onset": 100}}, [], H["response_keys"], 100.5),
            [],
        )

    def test_abort_timestamp(self):
        trial = dict(trial=0, response_window=1.5, response_index=0)
        for abort_at, complete in [(1.2, False), (1.5, True), (1.6, True)]:
            with self.subTest(abort_at=abort_at):
                state, records = route(
                    State("running", armed_at=0, zero=0, last_trigger=0),
                    [("escape", abort_at), ("1", abort_at + 0.01)],
                    H,
                    1.7,
                )
                self.assertEqual(state.ended_at, abort_at)
                self.assertEqual(records[-1]["category"], "after_terminal")
                row = score_responses(
                    [trial],
                    {0: dict(onset=0, offset=1)},
                    records,
                    H["response_keys"],
                    state.ended_at,
                )[0]
                self.assertEqual(row["response_window_complete"], complete)
                self.assertEqual(row["rt"], "n/a")


if __name__ == "__main__":
    unittest.main()
