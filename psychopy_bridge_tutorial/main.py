#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Mon Nov 18 18:53:24 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'main'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/bridge-center/bridge/psychopy-bridge-tutorial/psychopy_bridge_tutorial/main.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('adv_intro') is None:
        # initialise adv_intro
        adv_intro = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='adv_intro',
        )
    # create speaker 'encoding_audio'
    deviceManager.addDevice(
        deviceName='encoding_audio',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=4.0
    )
    if deviceManager.getDevice('facttest_response') is None:
        # initialise facttest_response
        facttest_response = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='facttest_response',
        )
    if deviceManager.getDevice('source_response') is None:
        # initialise source_response
        source_response = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='source_response',
        )
    if deviceManager.getDevice('exit_response') is None:
        # initialise exit_response
        exit_response = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='exit_response',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "IntroText" ---
    intro_text = visual.TextStim(win=win, name='intro_text',
        text='Welcome to BRIDGE!\n\nThis is PsychoPy',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    adv_intro = keyboard.Keyboard(deviceName='adv_intro')
    
    # --- Initialize components for Routine "Encoding" ---
    encoding_image = visual.ImageStim(
        win=win,
        name='encoding_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    encoding_audio = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='encoding_audio',    name='encoding_audio'
    )
    encoding_audio.setVolume(10.0)
    encoding_text = visual.TextStim(win=win, name='encoding_text',
        text='',
        font='Arial',
        pos=(0, .3), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "FactTest" ---
    facttest_image = visual.ImageStim(
        win=win,
        name='facttest_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, .2), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    facttest_question = visual.TextStim(win=win, name='facttest_question',
        text='',
        font='Open Sans',
        pos=(0, -.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    facttest_options = visual.TextStim(win=win, name='facttest_options',
        text='',
        font='Open Sans',
        pos=(0, -.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    facttest_response = keyboard.Keyboard(deviceName='facttest_response')
    
    # --- Initialize components for Routine "SourceTest" ---
    source_text = visual.TextStim(win=win, name='source_text',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    source_response = keyboard.Keyboard(deviceName='source_response')
    
    # --- Initialize components for Routine "Exit" ---
    text = visual.TextStim(win=win, name='text',
        text='Thanks for participating!\n\nClick the space bar to quit.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    exit_response = keyboard.Keyboard(deviceName='exit_response')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "IntroText" ---
    # create an object to store info about Routine IntroText
    IntroText = data.Routine(
        name='IntroText',
        components=[intro_text, adv_intro],
    )
    IntroText.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for adv_intro
    adv_intro.keys = []
    adv_intro.rt = []
    _adv_intro_allKeys = []
    # store start times for IntroText
    IntroText.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    IntroText.tStart = globalClock.getTime(format='float')
    IntroText.status = STARTED
    thisExp.addData('IntroText.started', IntroText.tStart)
    IntroText.maxDuration = None
    # keep track of which components have finished
    IntroTextComponents = IntroText.components
    for thisComponent in IntroText.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "IntroText" ---
    IntroText.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_text* updates
        
        # if intro_text is starting this frame...
        if intro_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_text.frameNStart = frameN  # exact frame index
            intro_text.tStart = t  # local t and not account for scr refresh
            intro_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            intro_text.status = STARTED
            intro_text.setAutoDraw(True)
        
        # if intro_text is active this frame...
        if intro_text.status == STARTED:
            # update params
            pass
        
        # *adv_intro* updates
        waitOnFlip = False
        
        # if adv_intro is starting this frame...
        if adv_intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            adv_intro.frameNStart = frameN  # exact frame index
            adv_intro.tStart = t  # local t and not account for scr refresh
            adv_intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(adv_intro, 'tStartRefresh')  # time at next scr refresh
            # update status
            adv_intro.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(adv_intro.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(adv_intro.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if adv_intro.status == STARTED and not waitOnFlip:
            theseKeys = adv_intro.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _adv_intro_allKeys.extend(theseKeys)
            if len(_adv_intro_allKeys):
                adv_intro.keys = _adv_intro_allKeys[0].name  # just the first key pressed
                adv_intro.rt = _adv_intro_allKeys[0].rt
                adv_intro.duration = _adv_intro_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            IntroText.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in IntroText.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "IntroText" ---
    for thisComponent in IntroText.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for IntroText
    IntroText.tStop = globalClock.getTime(format='float')
    IntroText.tStopRefresh = tThisFlipGlobal
    thisExp.addData('IntroText.stopped', IntroText.tStop)
    # check responses
    if adv_intro.keys in ['', [], None]:  # No response was made
        adv_intro.keys = None
    thisExp.addData('adv_intro.keys',adv_intro.keys)
    if adv_intro.keys != None:  # we had a response
        thisExp.addData('adv_intro.rt', adv_intro.rt)
        thisExp.addData('adv_intro.duration', adv_intro.duration)
    thisExp.nextEntry()
    # the Routine "IntroText" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('stim/stim.csv'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Encoding" ---
        # create an object to store info about Routine Encoding
        Encoding = data.Routine(
            name='Encoding',
            components=[encoding_image, encoding_audio, encoding_text],
        )
        Encoding.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        encoding_image.setImage('stim/'+animal_image)
        encoding_audio.setSound('stim/'+question_audio, secs=6, hamming=True)
        encoding_audio.setVolume(10.0, log=False)
        encoding_audio.seek(0)
        encoding_text.setText(animal_name)
        # store start times for Encoding
        Encoding.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Encoding.tStart = globalClock.getTime(format='float')
        Encoding.status = STARTED
        thisExp.addData('Encoding.started', Encoding.tStart)
        Encoding.maxDuration = None
        # keep track of which components have finished
        EncodingComponents = Encoding.components
        for thisComponent in Encoding.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Encoding" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        Encoding.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 6.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *encoding_image* updates
            
            # if encoding_image is starting this frame...
            if encoding_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                encoding_image.frameNStart = frameN  # exact frame index
                encoding_image.tStart = t  # local t and not account for scr refresh
                encoding_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(encoding_image, 'tStartRefresh')  # time at next scr refresh
                # update status
                encoding_image.status = STARTED
                encoding_image.setAutoDraw(True)
            
            # if encoding_image is active this frame...
            if encoding_image.status == STARTED:
                # update params
                pass
            
            # if encoding_image is stopping this frame...
            if encoding_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encoding_image.tStartRefresh + 6-frameTolerance:
                    # keep track of stop time/frame for later
                    encoding_image.tStop = t  # not accounting for scr refresh
                    encoding_image.tStopRefresh = tThisFlipGlobal  # on global time
                    encoding_image.frameNStop = frameN  # exact frame index
                    # update status
                    encoding_image.status = FINISHED
                    encoding_image.setAutoDraw(False)
            
            # *encoding_audio* updates
            
            # if encoding_audio is starting this frame...
            if encoding_audio.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                encoding_audio.frameNStart = frameN  # exact frame index
                encoding_audio.tStart = t  # local t and not account for scr refresh
                encoding_audio.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                encoding_audio.status = STARTED
                encoding_audio.play(when=win)  # sync with win flip
            
            # if encoding_audio is stopping this frame...
            if encoding_audio.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encoding_audio.tStartRefresh + 6-frameTolerance or encoding_audio.isFinished:
                    # keep track of stop time/frame for later
                    encoding_audio.tStop = t  # not accounting for scr refresh
                    encoding_audio.tStopRefresh = tThisFlipGlobal  # on global time
                    encoding_audio.frameNStop = frameN  # exact frame index
                    # update status
                    encoding_audio.status = FINISHED
                    encoding_audio.stop()
            
            # *encoding_text* updates
            
            # if encoding_text is starting this frame...
            if encoding_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                encoding_text.frameNStart = frameN  # exact frame index
                encoding_text.tStart = t  # local t and not account for scr refresh
                encoding_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(encoding_text, 'tStartRefresh')  # time at next scr refresh
                # update status
                encoding_text.status = STARTED
                encoding_text.setAutoDraw(True)
            
            # if encoding_text is active this frame...
            if encoding_text.status == STARTED:
                # update params
                pass
            
            # if encoding_text is stopping this frame...
            if encoding_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encoding_text.tStartRefresh + 6-frameTolerance:
                    # keep track of stop time/frame for later
                    encoding_text.tStop = t  # not accounting for scr refresh
                    encoding_text.tStopRefresh = tThisFlipGlobal  # on global time
                    encoding_text.frameNStop = frameN  # exact frame index
                    # update status
                    encoding_text.status = FINISHED
                    encoding_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[encoding_audio]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Encoding.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Encoding.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Encoding" ---
        for thisComponent in Encoding.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Encoding
        Encoding.tStop = globalClock.getTime(format='float')
        Encoding.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Encoding.stopped', Encoding.tStop)
        encoding_audio.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Encoding.maxDurationReached:
            routineTimer.addTime(-Encoding.maxDuration)
        elif Encoding.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-6.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler2(
        name='trials_2',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('stim/stim.csv'), 
        seed=None, 
    )
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "FactTest" ---
        # create an object to store info about Routine FactTest
        FactTest = data.Routine(
            name='FactTest',
            components=[facttest_image, facttest_question, facttest_options, facttest_response],
        )
        FactTest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        facttest_image.setImage('stim/'+animal_image)
        facttest_question.setText(question)
        facttest_options.setText(letter_a + choice_a + '\n' + letter_b + choice_b)
        # create starting attributes for facttest_response
        facttest_response.keys = []
        facttest_response.rt = []
        _facttest_response_allKeys = []
        # store start times for FactTest
        FactTest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        FactTest.tStart = globalClock.getTime(format='float')
        FactTest.status = STARTED
        thisExp.addData('FactTest.started', FactTest.tStart)
        FactTest.maxDuration = None
        # keep track of which components have finished
        FactTestComponents = FactTest.components
        for thisComponent in FactTest.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "FactTest" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        FactTest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *facttest_image* updates
            
            # if facttest_image is starting this frame...
            if facttest_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                facttest_image.frameNStart = frameN  # exact frame index
                facttest_image.tStart = t  # local t and not account for scr refresh
                facttest_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(facttest_image, 'tStartRefresh')  # time at next scr refresh
                # update status
                facttest_image.status = STARTED
                facttest_image.setAutoDraw(True)
            
            # if facttest_image is active this frame...
            if facttest_image.status == STARTED:
                # update params
                pass
            
            # if facttest_image is stopping this frame...
            if facttest_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > facttest_image.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    facttest_image.tStop = t  # not accounting for scr refresh
                    facttest_image.tStopRefresh = tThisFlipGlobal  # on global time
                    facttest_image.frameNStop = frameN  # exact frame index
                    # update status
                    facttest_image.status = FINISHED
                    facttest_image.setAutoDraw(False)
            
            # *facttest_question* updates
            
            # if facttest_question is starting this frame...
            if facttest_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                facttest_question.frameNStart = frameN  # exact frame index
                facttest_question.tStart = t  # local t and not account for scr refresh
                facttest_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(facttest_question, 'tStartRefresh')  # time at next scr refresh
                # update status
                facttest_question.status = STARTED
                facttest_question.setAutoDraw(True)
            
            # if facttest_question is active this frame...
            if facttest_question.status == STARTED:
                # update params
                pass
            
            # if facttest_question is stopping this frame...
            if facttest_question.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > facttest_question.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    facttest_question.tStop = t  # not accounting for scr refresh
                    facttest_question.tStopRefresh = tThisFlipGlobal  # on global time
                    facttest_question.frameNStop = frameN  # exact frame index
                    # update status
                    facttest_question.status = FINISHED
                    facttest_question.setAutoDraw(False)
            
            # *facttest_options* updates
            
            # if facttest_options is starting this frame...
            if facttest_options.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                facttest_options.frameNStart = frameN  # exact frame index
                facttest_options.tStart = t  # local t and not account for scr refresh
                facttest_options.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(facttest_options, 'tStartRefresh')  # time at next scr refresh
                # update status
                facttest_options.status = STARTED
                facttest_options.setAutoDraw(True)
            
            # if facttest_options is active this frame...
            if facttest_options.status == STARTED:
                # update params
                pass
            
            # if facttest_options is stopping this frame...
            if facttest_options.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > facttest_options.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    facttest_options.tStop = t  # not accounting for scr refresh
                    facttest_options.tStopRefresh = tThisFlipGlobal  # on global time
                    facttest_options.frameNStop = frameN  # exact frame index
                    # update status
                    facttest_options.status = FINISHED
                    facttest_options.setAutoDraw(False)
            
            # *facttest_response* updates
            waitOnFlip = False
            
            # if facttest_response is starting this frame...
            if facttest_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                facttest_response.frameNStart = frameN  # exact frame index
                facttest_response.tStart = t  # local t and not account for scr refresh
                facttest_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(facttest_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'facttest_response.started')
                # update status
                facttest_response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(facttest_response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(facttest_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if facttest_response is stopping this frame...
            if facttest_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > facttest_response.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    facttest_response.tStop = t  # not accounting for scr refresh
                    facttest_response.tStopRefresh = tThisFlipGlobal  # on global time
                    facttest_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'facttest_response.stopped')
                    # update status
                    facttest_response.status = FINISHED
                    facttest_response.status = FINISHED
            if facttest_response.status == STARTED and not waitOnFlip:
                theseKeys = facttest_response.getKeys(keyList=['a','b'], ignoreKeys=["escape"], waitRelease=False)
                _facttest_response_allKeys.extend(theseKeys)
                if len(_facttest_response_allKeys):
                    facttest_response.keys = _facttest_response_allKeys[-1].name  # just the last key pressed
                    facttest_response.rt = _facttest_response_allKeys[-1].rt
                    facttest_response.duration = _facttest_response_allKeys[-1].duration
                    # was this correct?
                    if (facttest_response.keys == str(correct_answer)) or (facttest_response.keys == correct_answer):
                        facttest_response.corr = 1
                    else:
                        facttest_response.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                FactTest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FactTest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "FactTest" ---
        for thisComponent in FactTest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for FactTest
        FactTest.tStop = globalClock.getTime(format='float')
        FactTest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('FactTest.stopped', FactTest.tStop)
        # check responses
        if facttest_response.keys in ['', [], None]:  # No response was made
            facttest_response.keys = None
            # was no response the correct answer?!
            if str(correct_answer).lower() == 'none':
               facttest_response.corr = 1;  # correct non-response
            else:
               facttest_response.corr = 0;  # failed to respond (incorrectly)
        # store data for trials_2 (TrialHandler)
        trials_2.addData('facttest_response.keys',facttest_response.keys)
        trials_2.addData('facttest_response.corr', facttest_response.corr)
        if facttest_response.keys != None:  # we had a response
            trials_2.addData('facttest_response.rt', facttest_response.rt)
            trials_2.addData('facttest_response.duration', facttest_response.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if FactTest.maxDurationReached:
            routineTimer.addTime(-FactTest.maxDuration)
        elif FactTest.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "SourceTest" ---
        # create an object to store info about Routine SourceTest
        SourceTest = data.Routine(
            name='SourceTest',
            components=[source_text, source_response],
        )
        SourceTest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        source_text.setText('Did a male or female voice tell you this fact?\n\na. Male\nb. Female')
        # create starting attributes for source_response
        source_response.keys = []
        source_response.rt = []
        _source_response_allKeys = []
        # store start times for SourceTest
        SourceTest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        SourceTest.tStart = globalClock.getTime(format='float')
        SourceTest.status = STARTED
        thisExp.addData('SourceTest.started', SourceTest.tStart)
        SourceTest.maxDuration = None
        # keep track of which components have finished
        SourceTestComponents = SourceTest.components
        for thisComponent in SourceTest.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SourceTest" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        SourceTest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *source_text* updates
            
            # if source_text is starting this frame...
            if source_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                source_text.frameNStart = frameN  # exact frame index
                source_text.tStart = t  # local t and not account for scr refresh
                source_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(source_text, 'tStartRefresh')  # time at next scr refresh
                # update status
                source_text.status = STARTED
                source_text.setAutoDraw(True)
            
            # if source_text is active this frame...
            if source_text.status == STARTED:
                # update params
                pass
            
            # if source_text is stopping this frame...
            if source_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > source_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    source_text.tStop = t  # not accounting for scr refresh
                    source_text.tStopRefresh = tThisFlipGlobal  # on global time
                    source_text.frameNStop = frameN  # exact frame index
                    # update status
                    source_text.status = FINISHED
                    source_text.setAutoDraw(False)
            
            # *source_response* updates
            waitOnFlip = False
            
            # if source_response is starting this frame...
            if source_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                source_response.frameNStart = frameN  # exact frame index
                source_response.tStart = t  # local t and not account for scr refresh
                source_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(source_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'source_response.started')
                # update status
                source_response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(source_response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(source_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if source_response is stopping this frame...
            if source_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > source_response.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    source_response.tStop = t  # not accounting for scr refresh
                    source_response.tStopRefresh = tThisFlipGlobal  # on global time
                    source_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'source_response.stopped')
                    # update status
                    source_response.status = FINISHED
                    source_response.status = FINISHED
            if source_response.status == STARTED and not waitOnFlip:
                theseKeys = source_response.getKeys(keyList=['a','b'], ignoreKeys=["escape"], waitRelease=False)
                _source_response_allKeys.extend(theseKeys)
                if len(_source_response_allKeys):
                    source_response.keys = _source_response_allKeys[-1].name  # just the last key pressed
                    source_response.rt = _source_response_allKeys[-1].rt
                    source_response.duration = _source_response_allKeys[-1].duration
                    # was this correct?
                    if (source_response.keys == str(correct_answer)) or (source_response.keys == correct_answer):
                        source_response.corr = 1
                    else:
                        source_response.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                SourceTest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in SourceTest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SourceTest" ---
        for thisComponent in SourceTest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for SourceTest
        SourceTest.tStop = globalClock.getTime(format='float')
        SourceTest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('SourceTest.stopped', SourceTest.tStop)
        # check responses
        if source_response.keys in ['', [], None]:  # No response was made
            source_response.keys = None
            # was no response the correct answer?!
            if str(correct_answer).lower() == 'none':
               source_response.corr = 1;  # correct non-response
            else:
               source_response.corr = 0;  # failed to respond (incorrectly)
        # store data for trials_2 (TrialHandler)
        trials_2.addData('source_response.keys',source_response.keys)
        trials_2.addData('source_response.corr', source_response.corr)
        if source_response.keys != None:  # we had a response
            trials_2.addData('source_response.rt', source_response.rt)
            trials_2.addData('source_response.duration', source_response.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if SourceTest.maxDurationReached:
            routineTimer.addTime(-SourceTest.maxDuration)
        elif SourceTest.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_2'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Exit" ---
    # create an object to store info about Routine Exit
    Exit = data.Routine(
        name='Exit',
        components=[text, exit_response],
    )
    Exit.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for exit_response
    exit_response.keys = []
    exit_response.rt = []
    _exit_response_allKeys = []
    # store start times for Exit
    Exit.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Exit.tStart = globalClock.getTime(format='float')
    Exit.status = STARTED
    thisExp.addData('Exit.started', Exit.tStart)
    Exit.maxDuration = None
    # keep track of which components have finished
    ExitComponents = Exit.components
    for thisComponent in Exit.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Exit" ---
    Exit.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.tStopRefresh = tThisFlipGlobal  # on global time
                text.frameNStop = frameN  # exact frame index
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # *exit_response* updates
        waitOnFlip = False
        
        # if exit_response is starting this frame...
        if exit_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exit_response.frameNStart = frameN  # exact frame index
            exit_response.tStart = t  # local t and not account for scr refresh
            exit_response.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(exit_response, 'tStartRefresh')  # time at next scr refresh
            # update status
            exit_response.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(exit_response.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(exit_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if exit_response is stopping this frame...
        if exit_response.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > exit_response.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                exit_response.tStop = t  # not accounting for scr refresh
                exit_response.tStopRefresh = tThisFlipGlobal  # on global time
                exit_response.frameNStop = frameN  # exact frame index
                # update status
                exit_response.status = FINISHED
                exit_response.status = FINISHED
        if exit_response.status == STARTED and not waitOnFlip:
            theseKeys = exit_response.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _exit_response_allKeys.extend(theseKeys)
            if len(_exit_response_allKeys):
                exit_response.keys = _exit_response_allKeys[-1].name  # just the last key pressed
                exit_response.rt = _exit_response_allKeys[-1].rt
                exit_response.duration = _exit_response_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Exit.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Exit.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Exit" ---
    for thisComponent in Exit.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Exit
    Exit.tStop = globalClock.getTime(format='float')
    Exit.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Exit.stopped', Exit.tStop)
    # check responses
    if exit_response.keys in ['', [], None]:  # No response was made
        exit_response.keys = None
    thisExp.addData('exit_response.keys',exit_response.keys)
    if exit_response.keys != None:  # we had a response
        thisExp.addData('exit_response.rt', exit_response.rt)
        thisExp.addData('exit_response.duration', exit_response.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Exit.maxDurationReached:
        routineTimer.addTime(-Exit.maxDuration)
    elif Exit.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
