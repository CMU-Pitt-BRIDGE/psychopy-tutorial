"""Run separately in PsychoPy Coder to observe the installed input key names."""

from psychopy import core, visual
from psychopy.hardware.keyboard import Keyboard

win = visual.Window(size=(1000, 600), units="height", color=[-0.5] * 3)
clock = core.Clock()
keys = Keyboard(backend="ptb", clock=clock)
message = visual.TextStim(
    win,
    text="Press the actual trigger and response buttons.\n\nEscape closes this check.",
    height=0.04,
)
try:
    while True:
        events = keys.getKeys(waitRelease=False, clear=True)
        for event in events:
            print(event.name, event.tDown)
            if event.name == "escape":
                core.quit()
            message.text = f"Key name: {event.name}\nKey-down time: {event.tDown:.3f} s\n\nEscape closes this check."
        message.draw()
        win.flip()
finally:
    win.close()
