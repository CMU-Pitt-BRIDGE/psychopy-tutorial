# Checked environment and limits

The workshop uses PsychoPy Standalone 2026.2.5, macOS arm64 post1, with its bundled Python 3.11.9, PsychToolbox 3.0.22.2 keyboard backend, pyglet 1.5.27 window backend and NumPy 2.2.6. The starter rejects a keyboard-backend fallback. Additional bundled plugins are not required by this lesson.

The deterministic tests cover condition and interval balance, counterbalancing, event timestamp preservation, stale and duplicate input, start timeout, abort, response boundaries and partial trials. A desktop software run exercises the generated Builder script with synthetic pulses, responses and intentional misses. A separate software stop exercises partial output. These inputs are explicitly marked as a software demonstration in the saved settings and status.

Before using the task in a study, establish the actual trigger and response key names, pulse-to-volume correspondence and stored-volume reference. Measure display timing and geometry on the participant setup. Determine acceptable onset and duration error from the scientific question. Review software frame intervals and condition-dependent delays. Software timestamps alone do not establish physical light, sound or button latency. This example is not a power calculation or scanner timing qualification.

Eye tracking and audio are outside the starter. They can be introduced after the basic visual task and its saved data are understood.

## Repository check on 12 November 2026

I ran all 11 deterministic tests with the bundled Python 3.11.9 from PsychoPy Standalone 2026.2.5. All passed. I also compiled the committed `starter.psyexp` with that version of Builder. The generated Python matched the committed `starter.py` after excluding the generated header date and trailing whitespace. `ruff check .` passed for the authored code. I did not repeat the graphical run or test scanner hardware during this check.

I checked the response-window definition and saved-row descriptions against `trial_core.score_responses` and `Session.save`. I replaced "completed trial row" with "saved trial row" because a row can contain `response_window_complete = False`. This is a factual terminology correction. I standardized "response opportunity" to "response window" for consistency with `response_window_s` and the workshop. This is a wording change, not a claim that "opportunity" is a retired term. I found no policy-driven term retirements.

For key-down timestamps, I checked the [PsychoPy 2026.2.5 keyboard implementation](https://github.com/psychopy/psychopy/blob/2026.2.5/psychopy/hardware/keyboard.py). The PTB path computes `tDown` relative to `logging.defaultClock`, the clock used by this starter. For the image-time reference, I checked the [BIDS event-onset definition](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html). These checks concern software records and definitions. They do not measure physical stimulus onset or input latency.
