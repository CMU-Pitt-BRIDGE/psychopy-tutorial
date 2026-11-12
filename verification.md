# Checked environment and limits

The workshop uses PsychoPy Standalone 2026.2.5, macOS arm64 post1, with its bundled Python 3.11.9, PsychToolbox 3.0.22.2 keyboard backend, pyglet 1.5.27 window backend and NumPy 2.2.6. The starter rejects a keyboard-backend fallback. Additional bundled plugins are not required by this lesson.

The deterministic tests cover condition and interval balance, counterbalancing, event timestamp preservation, stale and duplicate input, start timeout, abort, response boundaries and partial trials. A desktop software run exercises the generated Builder script with synthetic pulses, responses and intentional misses. A separate software stop exercises partial output. These inputs are explicitly marked as a software demonstration in the saved settings and status.

Before using the task in a study, establish the actual trigger and response key names, pulse-to-volume correspondence and stored-volume reference. Measure display timing and geometry on the participant setup. Determine acceptable onset and duration error from the scientific question. Review software frame intervals and condition-dependent delays. Software timestamps alone do not establish physical light, sound or button latency. This example is not a power calculation or scanner timing qualification.

Eye tracking and audio are outside the starter. They can be introduced after the basic visual task and its saved data are understood.
