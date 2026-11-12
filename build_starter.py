"""Rebuild the Builder source and its generated Python in PsychoPy 2026.2.5."""

from pathlib import Path
from psychopy.experiment import Experiment
from psychopy.experiment.routines import Routine
from psychopy.experiment.components.code import CodeComponent
from psychopy.experiment.components.text import TextComponent
from psychopy.experiment.components.grating import GratingComponent
from psychopy.experiment.loops import TrialHandler, LoopInitiator, LoopTerminator

root = Path(__file__).resolve().parent / "psychopy_bridge_tutorial"
exp = Experiment()
for key, value in {
    "expName": "BRIDGE visual discrimination",
    "Enable Escape": False,
    "Full-screen window": False,
    "Window size (pixels)": [1920, 1080],
    "Show info dlg": False,
    "Experiment info": {"participant": "000"},
    "Units": "height",
    "color": [-0.5, -0.5, -0.5],
    "winType": "pyglet",
    "keyboardBackend": "PsychToolbox",
    "Audio lib": "ptb",
    "Save log file": False,
    "Save wide csv file": False,
    "Save psydat file": False,
}.items():
    if key in exp.settings.params:
        exp.settings.params[key].val = value


def routine(name, message, code, begin="", end=""):
    r = Routine(name=name, exp=exp)
    c = CodeComponent(
        exp,
        name,
        name=name + "_input",
        codeType="Py",
        beginExp="from run_support import Session" if name == "instructions" else "",
        beginRoutine=begin,
        eachFrame="session.poll()\n" + code,
        endRoutine=end,
        endExperiment="session.save()" if name == "instructions" else "",
    )
    r.addComponent(c)
    text = TextComponent(
        exp,
        name,
        name=name + "_text",
        text=message,
        letterHeight=0.035,
        startVal=0,
        stopVal="",
    )
    r.addComponent(text)
    exp.addRoutine(name, r)
    return r


intro = routine(
    "instructions",
    "Report the tilt: left or right.\n\nPress Space when ready.\nEscape stops the experiment.",
    "continueRoutine = session.state.phase == 'ready'",
    'session = Session(win, int(expInfo["participant"]))\n'
    'instructions_text.text = "Report the tilt: left or right.\\n" + '
    '"Left = " + session.hardware["response_keys"][int(expInfo["participant"])%2] + '
    '"; right = " + session.hardware["response_keys"][(1+int(expInfo["participant"])%2)%2] + '
    '"\\n\\nPress Space when ready. Escape stops the experiment."',
)
wait = routine(
    "scanner_start",
    "Waiting for scanner",
    "continueRoutine = session.state.zero is None or baseline_fixation.tStartRefresh is None",
)
next(c for c in wait if c.name == "scanner_start_text").params[
    "startType"
].val = "condition"
next(c for c in wait if c.name == "scanner_start_text").params[
    "startVal"
].val = "session.state.zero is None"
next(c for c in wait if c.name == "scanner_start_text").params[
    "stopType"
].val = "condition"
next(c for c in wait if c.name == "scanner_start_text").params[
    "stopVal"
].val = "session.state.zero is not None"
fix = TextComponent(
    exp,
    "scanner_start",
    name="baseline_fixation",
    text="+",
    letterHeight=0.04,
    startType="condition",
    startVal="session.state.zero is not None",
    stopVal="",
)
wait.addComponent(fix)
trial = routine(
    "trial",
    "+",
    "continueRoutine = session.next_time() < planned_onset + interval_s",
    'session.begin_trial(thisTrial["trial"], grating)',
    "session.capture_flips()",
)
grating = GratingComponent(
    exp,
    "trial",
    name="grating",
    image="sin",
    mask="gauss",
    sf='session.spec["grating_cycles"]/session.spec["grating_size_height"]',
    size='$session.spec["grating_size_height"]',
    ori="orientation",
    contrast="contrast",
    startType="condition",
    startVal="session.next_time() >= planned_onset",
    stopType="condition",
    stopVal="session.next_time() >= planned_onset + duration",
)
for key in ["sf", "size", "ori", "contrast"]:
    grating.params[key].updates = "set every repeat"
trial.addComponent(grating)
final = routine(
    "final_fixation",
    "+",
    "continueRoutine = session.next_time() < session.end",
    end="session.complete()",
)
thanks = routine("thanks", "Thank you", "continueRoutine = t < 2")
intro_code = next(c for c in intro if c.name == "instructions_input")
intro_code.params["End Routine"].val = "grating.draw()\nwin.clearBuffer()"
exp.flow.extend([intro, wait])
loop = TrialHandler(
    exp,
    name="trials",
    loopType="sequential",
    nReps=1,
    conditionsFile='$str(session.out / "schedule.tsv")',
    isTrials=True,
)
exp.flow.extend([LoopInitiator(loop), trial, LoopTerminator(loop), final, thanks])
# Store the interval in each condition once; the same schedule is used for logging.
# It is added by Session immediately after make_schedule, not as a second design.
exp.saveToXML(str(root / "starter.psyexp"))
script = exp.writeScript(expPath=str(root / "starter.py"))
(root / "starter.py").write_text(
    "\n".join(line.rstrip() for line in script.splitlines()) + "\n",
    encoding="utf-8-sig",
)
print("Built starter.psyexp and its generated starter.py")
