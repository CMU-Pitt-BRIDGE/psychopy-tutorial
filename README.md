# PsychoPy fMRI tutorial

A Builder experiment for reporting whether a striped pattern tilts left or right. It compares low and high contrast across 32 trials in a 176-second schedule.

## Run the experiment

1. Open PsychoPy Standalone 2026.2.5. This tutorial was tested with the macOS arm64 post1 distribution.
2. Open `psychopy_bridge_tutorial/starter.psyexp`. Keep the adjacent Python and JSON files together.
3. Click Run. Press Space when ready, then 5 to simulate the scanner's start signal.
4. Press 1 for left tilt or 2 for right tilt. Escape stops the run and saves partial data.
5. Open the new run folder under `psychopy_bridge_tutorial/data/`.

Change the participant ID in Builder's Experiment Settings, under Experiment info. The default is 000. Odd IDs reverse the button mapping. Follow the keys displayed on the instruction screen.

At the scanner, run `check_keys.py` in PsychoPy Coder to check the actual button names. Enter them in `hardware.json`. Before collecting participant data, check scanner synchronization, stimulus timing and display geometry on the installed setup.

## Try a shorter response window

Each pattern has a planned duration of 1 second. Responses count from its recorded onset until 1.5 seconds later. The first allowed response before that deadline is scored.

In `task.json`, change `response_window_s` from 1.5 to 0.5. Predict what will happen to missing responses and reaction times. Run again and inspect the results. The pattern's duration stays the same. Restore 1.5 afterward.

Repeating the task also adds practice and fatigue. This exercise demonstrates a deadline. A study comparing deadlines needs an order or group assignment that separates those effects. The 32-trial example is not a power calculation.

## Edit the task

| File             | Edit                                                      |
| ---------------- | --------------------------------------------------------- |
| `starter.psyexp` | Builder routines and visual components                    |
| `task.json`      | Contrast, stimulus duration, response window and schedule |
| `hardware.json`  | Trigger key, response keys and expected pulse interval    |

JSON changes take effect on the next run. After Builder edits, regenerate `starter.py` using Write Python. `trial_core.py` contains scheduling and scoring; `run_support.py` contains keyboard collection and data saving.

## Read the results

| File                            | Contents                                                                                              |
| ------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `trials.tsv`                    | Stimulus times, scored response, reaction time in seconds, correctness and `response_window_complete` |
| `inputs.tsv`                    | Key-down timestamps, retrieval times and event categories                                             |
| `flips.json`                    | Recorded stimulus onset and offset flips                                                              |
| `frames.tsv`                    | Software frame intervals                                                                              |
| `settings.json`, `schedule.tsv` | Configuration and schedule used for the run                                                           |
| `status.json`                   | Run status, start, end and save times                                                                 |

Reaction time is key-down time minus onset flip time. Missing responses use `n/a`, not zero. A correct response can occur before an interrupted response window. A trial without a recorded stimulus offset has no row in `trials.tsv`; check `flips.json` before assuming the pattern never appeared.

The first accepted trigger after Space defines task zero. Later pulses do not restart the schedule. For Brain Imaging Data Structure, or BIDS, event files, express event onsets relative to the acquisition start of the first stored image volume. Software timestamps do not measure physical light or button latency.

## Development

Run tests with `python -m unittest discover -s tests`. Use Standalone's Python to run `build_starter.py` when rebuilding the Builder experiment and generated script.

`BRIDGE_DEMO=1` injects synthetic inputs. Add `BRIDGE_ABORT_S=20` to stop a demo early. These modes test software behavior, not human performance or scanner hardware.
