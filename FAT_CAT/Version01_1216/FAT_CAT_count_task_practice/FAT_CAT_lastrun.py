#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on 十二月 16, 2025, at 18:43
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
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, parallel
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
expName = 'Voice_indentity_task'  # from the Builder filename that created this script
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
_fullScr = True
_winSize = [1707, 960]
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
        originPath='E:\\Post_log\\FAT_IEEG\\Exp\\Exp_preparation\\FAT_CAT\\Program_psychopy\\program\\no_trigger\\Version_01_1216\\FAT_CAT_count_task_practice\\FAT_CAT_lastrun.py',
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
            logging.getLevel('info')
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
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
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
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
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
    if deviceManager.getDevice('instruction_key_resp') is None:
        # initialise instruction_key_resp
        instruction_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction_key_resp',
        )
    if deviceManager.getDevice('start_key_resp') is None:
        # initialise start_key_resp
        start_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start_key_resp',
        )
    # create speaker 'trial_start_sound'
    deviceManager.addDevice(
        deviceName='trial_start_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'trial_word_sound'
    deviceManager.addDevice(
        deviceName='trial_word_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('count_key_resp') is None:
        # initialise count_key_resp
        count_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='count_key_resp',
        )
    if deviceManager.getDevice('free_response_key_resp') is None:
        # initialise free_response_key_resp
        free_response_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='free_response_key_resp',
        )
    if deviceManager.getDevice('Control_count_key_resp') is None:
        # initialise Control_count_key_resp
        Control_count_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Control_count_key_resp',
        )
    if deviceManager.getDevice('controlled_response_key_resp') is None:
        # initialise controlled_response_key_resp
        controlled_response_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='controlled_response_key_resp',
        )
    # create speaker 'Controlled_category_sound'
    deviceManager.addDevice(
        deviceName='Controlled_category_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('Control_count_key_resp_2') is None:
        # initialise Control_count_key_resp_2
        Control_count_key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Control_count_key_resp_2',
        )
    if deviceManager.getDevice('controlled_response_key_resp_2') is None:
        # initialise controlled_response_key_resp_2
        controlled_response_key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='controlled_response_key_resp_2',
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
    
    # --- Initialize components for Routine "Stimuli_assignment" ---
    # Run 'Begin Experiment' code from assign_code
    import pandas as pd
    import random
    import sounddevice as sd
    from psychopy import prefs
    
    prefs.hardware['audioBufferSecs'] = 180
    prefs.saveUserPrefs()
    
    stimulus_file = os.path.join(os.getcwd(), 'stimulus', 'Stimulus_practice.xlsx')
    category_file = os.path.join(os.getcwd(), 'stimulus', 'Category_instructions.xlsx')
    
    stimulus_df = pd.read_excel(stimulus_file)
    category_df = pd.read_excel(category_file)
    print(f"成功读取刺激材料文件，共{len(stimulus_df)}个刺激")
    print(f"成功读取类别指导语文件，共{len(category_df)}个类别")
    
    def create_stimulus_conditions(stimulus_df, category_df):
        """
        根据word_category创建free和controlled条件的刺激材料
        """
        # 定义6个类别
        categories = ['Nature', 'Objects', 'States', 'People', 'Animals/Plants', 'Actions']
        
        for idx, row in stimulus_df.iterrows():
            other_categories = [cat for cat in categories if cat != row['word_category']]
        # 随机选择一个类别作为controlled_category
            stimulus_df.at[idx, 'controlled_category'] = random.choice(other_categories)
            if idx<2:
                stimulus_df.at[idx, 'condition'] = 'free'
            else:
                stimulus_df.at[idx, 'condition'] = 'controlled'
                
        # 合并Category_instructions信息
        # 首先为每个刺激添加对应的指导语信息
        stimulus_df = stimulus_df.merge(
            category_df, 
            on='word_category', 
            how='left',
            suffixes=('', '_from_category')
        )
        
        # 然后为controlled条件的刺激添加controlled_category对应的指导语信息
        # 重命名category_df的列，以便与controlled_category合并
        controlled_category_df = category_df.copy()
        controlled_category_df.rename(columns={
            'word_category': 'controlled_category',
            'Instructions': 'controlled_Instructions',
            'Instructions_audiofile': 'controlled_Instructions_audiofile',
            'Instructions_audiofile_path': 'controlled_Instructions_audiofile_path'
        }, inplace=True)
        
        # 合并controlled_category对应的指导语信息
        stimulus_df = stimulus_df.merge(
            controlled_category_df,
            on='controlled_category',
            how='left'
        )
        stimulus_df.to_csv('trial_df.csv')
        
        return stimulus_df
    
    # 执行刺激材料创建
    final_stimuli_df = create_stimulus_conditions(stimulus_df, category_df)
    print(f"\n数据形状: {final_stimuli_df.shape}")
    print(f"Free条件数量: {len(final_stimuli_df[final_stimuli_df['condition'] == 'free'])}")
    print(f"Controlled条件数量: {len(final_stimuli_df[final_stimuli_df['condition'] == 'controlled'])}")
    
    trials_list = []
    for idx, row in final_stimuli_df.iterrows():
        trial = {
            'wordID': row['wordID'],
            'word_list': row['word_list'],
            'word_category': row['word_category'],
            'word_list_audiofile': row.get('word_list_audiofile', ''),
            'word_list_audiofile_path': row.get('word_list_audiofile_path', ''),
            'condition': row['condition'],
            'Instructions': row.get('Instructions', ''),
            'Instructions_audiofile': row.get('Instructions_audiofile', ''),
            'Instructions_audiofile_path': row.get('Instructions_audiofile_path', ''),
            'controlled_category': row.get('controlled_category', ''),
            'controlled_Instructions': row.get('controlled_Instructions', ''),
            'controlled_Instructions_audiofile': row.get('controlled_Instructions_audiofile', ''),
            'controlled_Instructions_audiofile_path': row.get('controlled_Instructions_audiofile_path', '')
        }
        trials_list.append(trial)
    print(f"\n成功创建试验列表，共{len(trials_list)}个试验")
    
    random.shuffle(trials_list)
    
    # 打印试验列表摘要
    print("\n试验列表摘要:")
    for i, trial in enumerate(trials_list):
        print(f"试验 {i+1}: ID={trial['wordID']}, word={trial['word_list']}, 条件={trial['condition']}, 类别={trial['word_category']}", end="")
        print(f", 控制类别={trial['controlled_category']}")
        print(f", 控制音频文件={trial['controlled_Instructions_audiofile_path']}")
        print()
    
    thisExp.trial_counter = 0
    thisExp.ifControlTrial = None
    thisExp.responseCount = 0
    thisExp.randomCount = 0
    # Run 'Begin Experiment' code from instruction_assign_code
    instruction_file = os.path.join(os.getcwd(), 'stimulus', 'instructions.xlsx')
    
    try:
        instruction_df = pd.read_excel(instruction_file)
        print(f"成功读取刺激材料文件，共{len(instruction_df)}个刺激")
    except Exception as e:
        print(f"读取文件失败: {e}")
        # 创建示例数据
        instruction_df = pd.DataFrame({
            'category': range(1, 5),  
        })
    
    instructions_list = []
    for idx, row in instruction_df.iterrows():
        instruction = {
            'image_path': row['image_path'],
        }
        instructions_list.append(instruction)
    print(f"\n成功创建指导语列表，共{len(instructions_list)}个指导语图片")
    
    thisExp.instruction_counter = 0
    
    # --- Initialize components for Routine "instruction" ---
    # Run 'Begin Experiment' code from instruction_code
    currentInstrImage = []
    instruction_image = visual.ImageStim(
        win=win,
        name='instruction_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.78, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    instruction_key_resp = keyboard.Keyboard(deviceName='instruction_key_resp')
    
    # --- Initialize components for Routine "start" ---
    start_image = visual.ImageStim(
        win=win,
        name='start_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.78, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    start_key_resp = keyboard.Keyboard(deviceName='start_key_resp')
    
    # --- Initialize components for Routine "fixation" ---
    text = visual.TextStim(win=win, name='text',
        text='＋',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial_text" ---
    # Run 'Begin Experiment' code from trial_code
    currentWord = []
    currentWordAudiofilePath = []
    trial_word_text = visual.TextStim(win=win, name='trial_word_text',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    trial_start_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='trial_start_sound',    name='trial_start_sound'
    )
    trial_start_sound.setVolume(1.0)
    trial_word_sound = sound.Sound(
        'A', 
        secs=2, 
        stereo=True, 
        hamming=True, 
        speaker='trial_word_sound',    name='trial_word_sound'
    )
    trial_word_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "free_response_talk" ---
    count_text = visual.TextStim(win=win, name='count_text',
        text=None,
        font='Arial',
        pos=(-0.4, 0.4), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    free_association_text = visual.TextStim(win=win, name='free_association_text',
        text='请进行自由联想',
        font='Arial',
        pos=(0,0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    count_key_resp = keyboard.Keyboard(deviceName='count_key_resp')
    free_response_key_resp = keyboard.Keyboard(deviceName='free_response_key_resp')
    
    # --- Initialize components for Routine "Controlled_response_talk" ---
    Controlled_association_text_free = visual.TextStim(win=win, name='Controlled_association_text_free',
        text='请进行自由联想',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Control_count_text = visual.TextStim(win=win, name='Control_count_text',
        text=None,
        font='Arial',
        pos=(-0.4, 0.4), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    Control_count_key_resp = keyboard.Keyboard(deviceName='Control_count_key_resp')
    controlled_response_key_resp = keyboard.Keyboard(deviceName='controlled_response_key_resp')
    
    # --- Initialize components for Routine "control" ---
    Control_category_text = visual.TextStim(win=win, name='Control_category_text',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Controlled_category_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='Controlled_category_sound',    name='Controlled_category_sound'
    )
    Controlled_category_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "after_control_response" ---
    Control_count_text_2 = visual.TextStim(win=win, name='Control_count_text_2',
        text=None,
        font='Arial',
        pos=(-0.4, 0.4), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Controlled_association_text_free_4 = visual.TextStim(win=win, name='Controlled_association_text_free_4',
        text='请继续进行自由联想',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    Control_count_key_resp_2 = keyboard.Keyboard(deviceName='Control_count_key_resp_2')
    controlled_response_key_resp_2 = keyboard.Keyboard(deviceName='controlled_response_key_resp_2')
    
    # --- Initialize components for Routine "rest_within" ---
    rest_text = visual.TextStim(win=win, name='rest_text',
        text='rest',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
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
    
    # --- Prepare to start Routine "Stimuli_assignment" ---
    # create an object to store info about Routine Stimuli_assignment
    Stimuli_assignment = data.Routine(
        name='Stimuli_assignment',
        components=[],
    )
    Stimuli_assignment.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Stimuli_assignment
    Stimuli_assignment.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Stimuli_assignment.tStart = globalClock.getTime(format='float')
    Stimuli_assignment.status = STARTED
    thisExp.addData('Stimuli_assignment.started', Stimuli_assignment.tStart)
    Stimuli_assignment.maxDuration = None
    # keep track of which components have finished
    Stimuli_assignmentComponents = Stimuli_assignment.components
    for thisComponent in Stimuli_assignment.components:
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
    
    # --- Run Routine "Stimuli_assignment" ---
    Stimuli_assignment.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
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
            Stimuli_assignment.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Stimuli_assignment.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Stimuli_assignment" ---
    for thisComponent in Stimuli_assignment.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Stimuli_assignment
    Stimuli_assignment.tStop = globalClock.getTime(format='float')
    Stimuli_assignment.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Stimuli_assignment.stopped', Stimuli_assignment.tStop)
    thisExp.nextEntry()
    # the Routine "Stimuli_assignment" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    instruction_loop = data.TrialHandler2(
        name='instruction_loop',
        nReps=4.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(instruction_loop)  # add the loop to the experiment
    thisInstruction_loop = instruction_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisInstruction_loop.rgb)
    if thisInstruction_loop != None:
        for paramName in thisInstruction_loop:
            globals()[paramName] = thisInstruction_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisInstruction_loop in instruction_loop:
        currentLoop = instruction_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisInstruction_loop.rgb)
        if thisInstruction_loop != None:
            for paramName in thisInstruction_loop:
                globals()[paramName] = thisInstruction_loop[paramName]
        
        # --- Prepare to start Routine "instruction" ---
        # create an object to store info about Routine instruction
        instruction = data.Routine(
            name='instruction',
            components=[instruction_image, instruction_key_resp],
        )
        instruction.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from instruction_code
        instruction_index = thisExp.instruction_counter
        
        current_instruction = instructions_list[instruction_index]
        currentInstrImage = current_instruction['image_path']
        
        print(f"current_instruction : {current_instruction}")
        
        
        # 更新计数器
        thisExp.instruction_counter += 1
        instruction_image.setImage(currentInstrImage)
        # create starting attributes for instruction_key_resp
        instruction_key_resp.keys = []
        instruction_key_resp.rt = []
        _instruction_key_resp_allKeys = []
        # store start times for instruction
        instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        instruction.tStart = globalClock.getTime(format='float')
        instruction.status = STARTED
        thisExp.addData('instruction.started', instruction.tStart)
        instruction.maxDuration = None
        # keep track of which components have finished
        instructionComponents = instruction.components
        for thisComponent in instruction.components:
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
        
        # --- Run Routine "instruction" ---
        # if trial has changed, end Routine now
        if isinstance(instruction_loop, data.TrialHandler2) and thisInstruction_loop.thisN != instruction_loop.thisTrial.thisN:
            continueRoutine = False
        instruction.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *instruction_image* updates
            
            # if instruction_image is starting this frame...
            if instruction_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instruction_image.frameNStart = frameN  # exact frame index
                instruction_image.tStart = t  # local t and not account for scr refresh
                instruction_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instruction_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instruction_image.started')
                # update status
                instruction_image.status = STARTED
                instruction_image.setAutoDraw(True)
            
            # if instruction_image is active this frame...
            if instruction_image.status == STARTED:
                # update params
                pass
            
            # *instruction_key_resp* updates
            waitOnFlip = False
            
            # if instruction_key_resp is starting this frame...
            if instruction_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instruction_key_resp.frameNStart = frameN  # exact frame index
                instruction_key_resp.tStart = t  # local t and not account for scr refresh
                instruction_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instruction_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instruction_key_resp.started')
                # update status
                instruction_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(instruction_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(instruction_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if instruction_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = instruction_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _instruction_key_resp_allKeys.extend(theseKeys)
                if len(_instruction_key_resp_allKeys):
                    instruction_key_resp.keys = _instruction_key_resp_allKeys[-1].name  # just the last key pressed
                    instruction_key_resp.rt = _instruction_key_resp_allKeys[-1].rt
                    instruction_key_resp.duration = _instruction_key_resp_allKeys[-1].duration
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
                instruction.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instruction.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instruction" ---
        for thisComponent in instruction.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for instruction
        instruction.tStop = globalClock.getTime(format='float')
        instruction.tStopRefresh = tThisFlipGlobal
        thisExp.addData('instruction.stopped', instruction.tStop)
        # check responses
        if instruction_key_resp.keys in ['', [], None]:  # No response was made
            instruction_key_resp.keys = None
        instruction_loop.addData('instruction_key_resp.keys',instruction_key_resp.keys)
        if instruction_key_resp.keys != None:  # we had a response
            instruction_loop.addData('instruction_key_resp.rt', instruction_key_resp.rt)
            instruction_loop.addData('instruction_key_resp.duration', instruction_key_resp.duration)
        # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 4.0 repeats of 'instruction_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "start" ---
    # create an object to store info about Routine start
    start = data.Routine(
        name='start',
        components=[start_image, start_key_resp],
    )
    start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    start_image.setImage('image/start.jpg')
    # create starting attributes for start_key_resp
    start_key_resp.keys = []
    start_key_resp.rt = []
    _start_key_resp_allKeys = []
    # store start times for start
    start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start.tStart = globalClock.getTime(format='float')
    start.status = STARTED
    thisExp.addData('start.started', start.tStart)
    start.maxDuration = None
    # keep track of which components have finished
    startComponents = start.components
    for thisComponent in start.components:
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
    
    # --- Run Routine "start" ---
    start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *start_image* updates
        
        # if start_image is starting this frame...
        if start_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_image.frameNStart = frameN  # exact frame index
            start_image.tStart = t  # local t and not account for scr refresh
            start_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_image.started')
            # update status
            start_image.status = STARTED
            start_image.setAutoDraw(True)
        
        # if start_image is active this frame...
        if start_image.status == STARTED:
            # update params
            pass
        
        # *start_key_resp* updates
        waitOnFlip = False
        
        # if start_key_resp is starting this frame...
        if start_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_key_resp.frameNStart = frameN  # exact frame index
            start_key_resp.tStart = t  # local t and not account for scr refresh
            start_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_key_resp.started')
            # update status
            start_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = start_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _start_key_resp_allKeys.extend(theseKeys)
            if len(_start_key_resp_allKeys):
                start_key_resp.keys = _start_key_resp_allKeys[-1].name  # just the last key pressed
                start_key_resp.rt = _start_key_resp_allKeys[-1].rt
                start_key_resp.duration = _start_key_resp_allKeys[-1].duration
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
            start.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start" ---
    for thisComponent in start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start
    start.tStop = globalClock.getTime(format='float')
    start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('start.stopped', start.tStop)
    # check responses
    if start_key_resp.keys in ['', [], None]:  # No response was made
        start_key_resp.keys = None
    thisExp.addData('start_key_resp.keys',start_key_resp.keys)
    if start_key_resp.keys != None:  # we had a response
        thisExp.addData('start_key_resp.rt', start_key_resp.rt)
        thisExp.addData('start_key_resp.duration', start_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_loop = data.TrialHandler2(
        name='trials_loop',
        nReps=4.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials_loop)  # add the loop to the experiment
    thisTrials_loop = trials_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
    if thisTrials_loop != None:
        for paramName in thisTrials_loop:
            globals()[paramName] = thisTrials_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_loop in trials_loop:
        currentLoop = trials_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
        if thisTrials_loop != None:
            for paramName in thisTrials_loop:
                globals()[paramName] = thisTrials_loop[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[text],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation
        fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation.tStart = globalClock.getTime(format='float')
        fixation.status = STARTED
        thisExp.addData('fixation.started', fixation.tStart)
        fixation.maxDuration = None
        # keep track of which components have finished
        fixationComponents = fixation.components
        for thisComponent in fixation.components:
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
        
        # --- Run Routine "fixation" ---
        # if trial has changed, end Routine now
        if isinstance(trials_loop, data.TrialHandler2) and thisTrials_loop.thisN != trials_loop.thisTrial.thisN:
            continueRoutine = False
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
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
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
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
                if tThisFlipGlobal > text.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
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
                fixation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation
        fixation.tStop = globalClock.getTime(format='float')
        fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation.stopped', fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation.maxDurationReached:
            routineTimer.addTime(-fixation.maxDuration)
        elif fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "trial_text" ---
        # create an object to store info about Routine trial_text
        trial_text = data.Routine(
            name='trial_text',
            components=[trial_word_text, trial_start_sound, trial_word_sound],
        )
        trial_text.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from trial_code
        # 获取当前trial索引
        trial_index = thisExp.trial_counter
        thisExp.responseCount = 0
        
        current_trial = trials_list[trial_index]
        currentWord = current_trial['word_list']
        currentWordID = current_trial['wordID']
        currentCategory = current_trial['word_category']
        currentWordAudiofile = current_trial['word_list_audiofile']
        currentWordAudiofilePath = current_trial['word_list_audiofile_path']
        currentcondition = current_trial['condition']
        
        # controlled条件使用controlled_category的指导语
        currentControlInstr = current_trial['controlled_Instructions']
        currentControlInstrAudio = current_trial['controlled_Instructions_audiofile']
        currentControlInstrAudioPath = current_trial['controlled_Instructions_audiofile_path']
        currentControlledCategory = current_trial['controlled_category']
        
        #print(f"current_trial : {current_trial}")
        print(f"currentWord : {currentWord}")
        print(f"currentcondition : {currentcondition}")
        print(f"currentControlledCategory : {currentControlledCategory}")
        
        
        # 更新计数器
        thisExp.trial_counter += 1
        thisExp.ifControlTrial = currentcondition
        
        thisExp.addData('word_list', currentWord)
        thisExp.addData('wordID', currentWordID)
        thisExp.addData('word_category', currentCategory)
        thisExp.addData('condition', currentcondition)
        thisExp.addData('controlled_category', currentControlledCategory)
        thisExp.addData('word_list_audiofile', currentWordAudiofile)
        thisExp.addData('word_list_audiofile_path', currentWordAudiofilePath)
        thisExp.addData('Instructions', currentControlInstr)
        thisExp.addData('Instructions_audiofile', currentControlInstrAudio)
        thisExp.addData('Instructions_audiofile_path', currentControlInstrAudioPath)
        
        trial_word_text.text = currentWord
        
        ## trial开始时设置为高电平
        #thisExp.output_task.write(np.array([True], dtype=bool).tolist())
        #print("触发器信号: 起始词朗读开始高电平")
        trial_start_sound.setSound('stimulus/audiofile/start/暖心学姐_普通话_接下来我会给您一个起始词，您需__AI生成.wav', secs=6, hamming=True)
        trial_start_sound.setVolume(1.0, log=False)
        trial_start_sound.seek(0)
        trial_word_sound.setSound(currentWordAudiofilePath, secs=2, hamming=True)
        trial_word_sound.setVolume(1.0, log=False)
        trial_word_sound.seek(0)
        # store start times for trial_text
        trial_text.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_text.tStart = globalClock.getTime(format='float')
        trial_text.status = STARTED
        thisExp.addData('trial_text.started', trial_text.tStart)
        trial_text.maxDuration = None
        # keep track of which components have finished
        trial_textComponents = trial_text.components
        for thisComponent in trial_text.components:
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
        
        # --- Run Routine "trial_text" ---
        # if trial has changed, end Routine now
        if isinstance(trials_loop, data.TrialHandler2) and thisTrials_loop.thisN != trials_loop.thisTrial.thisN:
            continueRoutine = False
        trial_text.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 8.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *trial_word_text* updates
            
            # if trial_word_text is starting this frame...
            if trial_word_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_word_text.frameNStart = frameN  # exact frame index
                trial_word_text.tStart = t  # local t and not account for scr refresh
                trial_word_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_word_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_word_text.started')
                # update status
                trial_word_text.status = STARTED
                trial_word_text.setAutoDraw(True)
            
            # if trial_word_text is active this frame...
            if trial_word_text.status == STARTED:
                # update params
                pass
            
            # if trial_word_text is stopping this frame...
            if trial_word_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trial_word_text.tStartRefresh + 8-frameTolerance:
                    # keep track of stop time/frame for later
                    trial_word_text.tStop = t  # not accounting for scr refresh
                    trial_word_text.tStopRefresh = tThisFlipGlobal  # on global time
                    trial_word_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_word_text.stopped')
                    # update status
                    trial_word_text.status = FINISHED
                    trial_word_text.setAutoDraw(False)
            
            # *trial_start_sound* updates
            
            # if trial_start_sound is starting this frame...
            if trial_start_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_start_sound.frameNStart = frameN  # exact frame index
                trial_start_sound.tStart = t  # local t and not account for scr refresh
                trial_start_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('trial_start_sound.started', tThisFlipGlobal)
                # update status
                trial_start_sound.status = STARTED
                trial_start_sound.play(when=win)  # sync with win flip
            
            # if trial_start_sound is stopping this frame...
            if trial_start_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trial_start_sound.tStartRefresh + 6-frameTolerance or trial_start_sound.isFinished:
                    # keep track of stop time/frame for later
                    trial_start_sound.tStop = t  # not accounting for scr refresh
                    trial_start_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    trial_start_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_start_sound.stopped')
                    # update status
                    trial_start_sound.status = FINISHED
                    trial_start_sound.stop()
            
            # *trial_word_sound* updates
            
            # if trial_word_sound is starting this frame...
            if trial_word_sound.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                # keep track of start time/frame for later
                trial_word_sound.frameNStart = frameN  # exact frame index
                trial_word_sound.tStart = t  # local t and not account for scr refresh
                trial_word_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('trial_word_sound.started', tThisFlipGlobal)
                # update status
                trial_word_sound.status = STARTED
                trial_word_sound.play(when=win)  # sync with win flip
            
            # if trial_word_sound is stopping this frame...
            if trial_word_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trial_word_sound.tStartRefresh + 2-frameTolerance or trial_word_sound.isFinished:
                    # keep track of stop time/frame for later
                    trial_word_sound.tStop = t  # not accounting for scr refresh
                    trial_word_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    trial_word_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_word_sound.stopped')
                    # update status
                    trial_word_sound.status = FINISHED
                    trial_word_sound.stop()
            
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
                    playbackComponents=[trial_start_sound, trial_word_sound]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial_text.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_text.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_text" ---
        for thisComponent in trial_text.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_text
        trial_text.tStop = globalClock.getTime(format='float')
        trial_text.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_text.stopped', trial_text.tStop)
        # Run 'End Routine' code from trial_code
        thisExp.randomCount = random.randint(4, 7)
        print(f"random_number:{thisExp.randomCount}")
        
        thisExp.addData('randomCount_control_loc', thisExp.randomCount)
        
        #thisExp.output_task.write(np.array([False], dtype=bool).tolist())
        #print("触发器信号: 起始词朗读结束低电平")
        trial_start_sound.pause()  # ensure sound has stopped at end of Routine
        trial_word_sound.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if trial_text.maxDurationReached:
            routineTimer.addTime(-trial_text.maxDuration)
        elif trial_text.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-8.000000)
        
        # --- Prepare to start Routine "free_response_talk" ---
        # create an object to store info about Routine free_response_talk
        free_response_talk = data.Routine(
            name='free_response_talk',
            components=[count_text, free_association_text, count_key_resp, free_response_key_resp],
        )
        free_response_talk.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from count_code
        thisExp.responseCount = 0
        
        # 创建一个简单的标志来跟踪按键状态
        key_pressed = False
        
        #thisExp.output_task.write(np.array([True], dtype=bool).tolist())
        #print("触发器信号: response 的 routine 开始高电平")
        #
        # create starting attributes for count_key_resp
        count_key_resp.keys = []
        count_key_resp.rt = []
        _count_key_resp_allKeys = []
        # create starting attributes for free_response_key_resp
        free_response_key_resp.keys = []
        free_response_key_resp.rt = []
        _free_response_key_resp_allKeys = []
        # store start times for free_response_talk
        free_response_talk.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        free_response_talk.tStart = globalClock.getTime(format='float')
        free_response_talk.status = STARTED
        thisExp.addData('free_response_talk.started', free_response_talk.tStart)
        free_response_talk.maxDuration = None
        # skip Routine free_response_talk if its 'Skip if' condition is True
        free_response_talk.skipped = continueRoutine and not (thisExp.ifControlTrial == 'controlled')
        continueRoutine = free_response_talk.skipped
        # keep track of which components have finished
        free_response_talkComponents = free_response_talk.components
        for thisComponent in free_response_talk.components:
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
        
        # --- Run Routine "free_response_talk" ---
        # if trial has changed, end Routine now
        if isinstance(trials_loop, data.TrialHandler2) and thisTrials_loop.thisN != trials_loop.thisTrial.thisN:
            continueRoutine = False
        free_response_talk.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from count_code
            # 简化按键检测逻辑
            current_keys = count_key_resp.getKeys(keyList=['1'], waitRelease=True)
            count_text.text = str(thisExp.responseCount + 1)
            
            #thisExp.output_task.write(np.array([True], dtype=bool).tolist())
            ##print("触发器信号: response 的 每一帧高电平")
            
            if current_keys:
                # 只有当检测到新的按键时才增加计数
                thisExp.responseCount += 1
                print(f"计数增加到: {thisExp.responseCount}")
                thisExp.addData('responseCount', thisExp.responseCount)
                thisExp.addData('count_key_resp.keys', count_key_resp.keys)
                thisExp.addData('count_key_resp.rt', count_key_resp.rt)
                # 清空按键缓冲区
                count_key_resp.keys = []
                count_key_resp.rt = []
                count_text.text = str(thisExp.responseCount + 1)
                
            #    thisExp.output_task.write(np.array([False], dtype=bool).tolist())
            #    print("触发器信号: 按键计数低电平")
                
            #    if trigger_on:
            #        sendtrigger(1000+thisExp.trigger_n+1)
            #        thisExp.addData('triggerbox', triggerbox)
            #        thisExp.trigger_n+=1
            #        print(f"thisExp.trigger_n:{thisExp.trigger_n}")
            
            
            
            # *count_text* updates
            
            # if count_text is starting this frame...
            if count_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                count_text.frameNStart = frameN  # exact frame index
                count_text.tStart = t  # local t and not account for scr refresh
                count_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(count_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'count_text.started')
                # update status
                count_text.status = STARTED
                count_text.setAutoDraw(True)
            
            # if count_text is active this frame...
            if count_text.status == STARTED:
                # update params
                pass
            
            # *free_association_text* updates
            
            # if free_association_text is starting this frame...
            if free_association_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                free_association_text.frameNStart = frameN  # exact frame index
                free_association_text.tStart = t  # local t and not account for scr refresh
                free_association_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(free_association_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'free_association_text.started')
                # update status
                free_association_text.status = STARTED
                free_association_text.setAutoDraw(True)
            
            # if free_association_text is active this frame...
            if free_association_text.status == STARTED:
                # update params
                pass
            
            # *count_key_resp* updates
            waitOnFlip = False
            
            # if count_key_resp is starting this frame...
            if count_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                count_key_resp.frameNStart = frameN  # exact frame index
                count_key_resp.tStart = t  # local t and not account for scr refresh
                count_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(count_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'count_key_resp.started')
                # update status
                count_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(count_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(count_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if count_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = count_key_resp.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=False)
                _count_key_resp_allKeys.extend(theseKeys)
                if len(_count_key_resp_allKeys):
                    count_key_resp.keys = _count_key_resp_allKeys[-1].name  # just the last key pressed
                    count_key_resp.rt = _count_key_resp_allKeys[-1].rt
                    count_key_resp.duration = _count_key_resp_allKeys[-1].duration
            
            # *free_response_key_resp* updates
            waitOnFlip = False
            
            # if free_response_key_resp is starting this frame...
            if free_response_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                free_response_key_resp.frameNStart = frameN  # exact frame index
                free_response_key_resp.tStart = t  # local t and not account for scr refresh
                free_response_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(free_response_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'free_response_key_resp.started')
                # update status
                free_response_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(free_response_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(free_response_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if free_response_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = free_response_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _free_response_key_resp_allKeys.extend(theseKeys)
                if len(_free_response_key_resp_allKeys):
                    free_response_key_resp.keys = _free_response_key_resp_allKeys[-1].name  # just the last key pressed
                    free_response_key_resp.rt = _free_response_key_resp_allKeys[-1].rt
                    free_response_key_resp.duration = _free_response_key_resp_allKeys[-1].duration
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
                free_response_talk.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in free_response_talk.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "free_response_talk" ---
        for thisComponent in free_response_talk.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for free_response_talk
        free_response_talk.tStop = globalClock.getTime(format='float')
        free_response_talk.tStopRefresh = tThisFlipGlobal
        thisExp.addData('free_response_talk.stopped', free_response_talk.tStop)
        # check responses
        if count_key_resp.keys in ['', [], None]:  # No response was made
            count_key_resp.keys = None
        trials_loop.addData('count_key_resp.keys',count_key_resp.keys)
        if count_key_resp.keys != None:  # we had a response
            trials_loop.addData('count_key_resp.rt', count_key_resp.rt)
            trials_loop.addData('count_key_resp.duration', count_key_resp.duration)
        # check responses
        if free_response_key_resp.keys in ['', [], None]:  # No response was made
            free_response_key_resp.keys = None
        trials_loop.addData('free_response_key_resp.keys',free_response_key_resp.keys)
        if free_response_key_resp.keys != None:  # we had a response
            trials_loop.addData('free_response_key_resp.rt', free_response_key_resp.rt)
            trials_loop.addData('free_response_key_resp.duration', free_response_key_resp.duration)
        # the Routine "free_response_talk" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Controlled_response_talk" ---
        # create an object to store info about Routine Controlled_response_talk
        Controlled_response_talk = data.Routine(
            name='Controlled_response_talk',
            components=[Controlled_association_text_free, Control_count_text, Control_count_key_resp, controlled_response_key_resp],
        )
        Controlled_response_talk.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from Control_count_code
        thisExp.responseCount = 0
        # 创建一个简单的标志来跟踪按键状态
        key_pressed = False
        
        #thisExp.output_task.write(np.array([True], dtype=bool).tolist())
        #print("触发器信号: response 的 routine 开始高电平")
        
        
        # create starting attributes for Control_count_key_resp
        Control_count_key_resp.keys = []
        Control_count_key_resp.rt = []
        _Control_count_key_resp_allKeys = []
        # create starting attributes for controlled_response_key_resp
        controlled_response_key_resp.keys = []
        controlled_response_key_resp.rt = []
        _controlled_response_key_resp_allKeys = []
        # store start times for Controlled_response_talk
        Controlled_response_talk.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Controlled_response_talk.tStart = globalClock.getTime(format='float')
        Controlled_response_talk.status = STARTED
        thisExp.addData('Controlled_response_talk.started', Controlled_response_talk.tStart)
        # skip Routine Controlled_response_talk if its 'Skip if' condition is True
        Controlled_response_talk.skipped = continueRoutine and not ((thisExp.ifControlTrial == 'free') | (thisExp.responseCount >= thisExp.randomCount))
        continueRoutine = Controlled_response_talk.skipped
        # keep track of which components have finished
        Controlled_response_talkComponents = Controlled_response_talk.components
        for thisComponent in Controlled_response_talk.components:
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
        
        # --- Run Routine "Controlled_response_talk" ---
        # if trial has changed, end Routine now
        if isinstance(trials_loop, data.TrialHandler2) and thisTrials_loop.thisN != trials_loop.thisTrial.thisN:
            continueRoutine = False
        Controlled_response_talk.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on condition)
            if bool(thisExp.responseCount == thisExp.randomCount):
                continueRoutine = False
            # Run 'Each Frame' code from Control_count_code
            # 简化按键检测逻辑
            current_keys = Control_count_key_resp.getKeys(keyList=['1'], waitRelease=True)
            Control_count_text.text = str(thisExp.responseCount + 1)
            
            #thisExp.output_task.write(np.array([True], dtype=bool).tolist())
            ##print("触发器信号: response 的 每一帧高电平")
            
            if current_keys:
                # 只有当检测到新的按键时才增加计数
                thisExp.responseCount += 1
                print(f"thisExp.responseCount:{thisExp.responseCount}")
                thisExp.addData('responseCount', thisExp.responseCount)
                thisExp.addData('Control_count_key_resp.keys', Control_count_key_resp.keys)
                thisExp.addData('Control_count_key_resp.rt', Control_count_key_resp.rt)
                # 清空按键缓冲区
                Control_count_key_resp.keys = []
                Control_count_key_resp.rt = []
                Control_count_text.text = str(thisExp.responseCount + 1)
                
            #    thisExp.output_task.write(np.array([False], dtype=bool).tolist())
            #    print("触发器信号: 按键计数低电平")
                
            #    if trigger_on:
            #        sendtrigger(1000+thisExp.trigger_n+1)
            #        thisExp.addData('triggerbox', triggerbox)
            #        thisExp.trigger_n+=1
            #        print(f"thisExp.trigger_n:{thisExp.trigger_n}")
            #
            
            # *Controlled_association_text_free* updates
            
            # if Controlled_association_text_free is starting this frame...
            if Controlled_association_text_free.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Controlled_association_text_free.frameNStart = frameN  # exact frame index
                Controlled_association_text_free.tStart = t  # local t and not account for scr refresh
                Controlled_association_text_free.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Controlled_association_text_free, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Controlled_association_text_free.started')
                # update status
                Controlled_association_text_free.status = STARTED
                Controlled_association_text_free.setAutoDraw(True)
            
            # if Controlled_association_text_free is active this frame...
            if Controlled_association_text_free.status == STARTED:
                # update params
                pass
            
            # *Control_count_text* updates
            
            # if Control_count_text is starting this frame...
            if Control_count_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Control_count_text.frameNStart = frameN  # exact frame index
                Control_count_text.tStart = t  # local t and not account for scr refresh
                Control_count_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Control_count_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Control_count_text.started')
                # update status
                Control_count_text.status = STARTED
                Control_count_text.setAutoDraw(True)
            
            # if Control_count_text is active this frame...
            if Control_count_text.status == STARTED:
                # update params
                pass
            
            # *Control_count_key_resp* updates
            waitOnFlip = False
            
            # if Control_count_key_resp is starting this frame...
            if Control_count_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Control_count_key_resp.frameNStart = frameN  # exact frame index
                Control_count_key_resp.tStart = t  # local t and not account for scr refresh
                Control_count_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Control_count_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Control_count_key_resp.started')
                # update status
                Control_count_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(Control_count_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(Control_count_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if Control_count_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = Control_count_key_resp.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=False)
                _Control_count_key_resp_allKeys.extend(theseKeys)
                if len(_Control_count_key_resp_allKeys):
                    Control_count_key_resp.keys = _Control_count_key_resp_allKeys[-1].name  # just the last key pressed
                    Control_count_key_resp.rt = _Control_count_key_resp_allKeys[-1].rt
                    Control_count_key_resp.duration = _Control_count_key_resp_allKeys[-1].duration
            
            # *controlled_response_key_resp* updates
            waitOnFlip = False
            
            # if controlled_response_key_resp is starting this frame...
            if controlled_response_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                controlled_response_key_resp.frameNStart = frameN  # exact frame index
                controlled_response_key_resp.tStart = t  # local t and not account for scr refresh
                controlled_response_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(controlled_response_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'controlled_response_key_resp.started')
                # update status
                controlled_response_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(controlled_response_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(controlled_response_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if controlled_response_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = controlled_response_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _controlled_response_key_resp_allKeys.extend(theseKeys)
                if len(_controlled_response_key_resp_allKeys):
                    controlled_response_key_resp.keys = _controlled_response_key_resp_allKeys[-1].name  # just the last key pressed
                    controlled_response_key_resp.rt = _controlled_response_key_resp_allKeys[-1].rt
                    controlled_response_key_resp.duration = _controlled_response_key_resp_allKeys[-1].duration
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
                Controlled_response_talk.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Controlled_response_talk.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Controlled_response_talk" ---
        for thisComponent in Controlled_response_talk.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Controlled_response_talk
        Controlled_response_talk.tStop = globalClock.getTime(format='float')
        Controlled_response_talk.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Controlled_response_talk.stopped', Controlled_response_talk.tStop)
        # check responses
        if Control_count_key_resp.keys in ['', [], None]:  # No response was made
            Control_count_key_resp.keys = None
        trials_loop.addData('Control_count_key_resp.keys',Control_count_key_resp.keys)
        if Control_count_key_resp.keys != None:  # we had a response
            trials_loop.addData('Control_count_key_resp.rt', Control_count_key_resp.rt)
            trials_loop.addData('Control_count_key_resp.duration', Control_count_key_resp.duration)
        # check responses
        if controlled_response_key_resp.keys in ['', [], None]:  # No response was made
            controlled_response_key_resp.keys = None
        trials_loop.addData('controlled_response_key_resp.keys',controlled_response_key_resp.keys)
        if controlled_response_key_resp.keys != None:  # we had a response
            trials_loop.addData('controlled_response_key_resp.rt', controlled_response_key_resp.rt)
            trials_loop.addData('controlled_response_key_resp.duration', controlled_response_key_resp.duration)
        # the Routine "Controlled_response_talk" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "control" ---
        # create an object to store info about Routine control
        control = data.Routine(
            name='control',
            components=[Control_category_text, Controlled_category_sound],
        )
        control.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from control_code
        Control_category_text.text = currentControlInstr
        print("播放control")
        print(f"currentControlInstr:{currentControlInstr}")
        print(f"currentControlInstrAudioPath:{currentControlInstrAudioPath}")
        Controlled_category_sound.setSound(currentControlInstrAudioPath, secs=9, hamming=True)
        Controlled_category_sound.setVolume(1.0, log=False)
        Controlled_category_sound.seek(0)
        # store start times for control
        control.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        control.tStart = globalClock.getTime(format='float')
        control.status = STARTED
        thisExp.addData('control.started', control.tStart)
        control.maxDuration = None
        # skip Routine control if its 'Skip if' condition is True
        control.skipped = continueRoutine and not ((thisExp.ifControlTrial == 'free') | (thisExp.responseCount != thisExp.randomCount))
        continueRoutine = control.skipped
        # keep track of which components have finished
        controlComponents = control.components
        for thisComponent in control.components:
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
        
        # --- Run Routine "control" ---
        # if trial has changed, end Routine now
        if isinstance(trials_loop, data.TrialHandler2) and thisTrials_loop.thisN != trials_loop.thisTrial.thisN:
            continueRoutine = False
        control.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 9.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Control_category_text* updates
            
            # if Control_category_text is starting this frame...
            if Control_category_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Control_category_text.frameNStart = frameN  # exact frame index
                Control_category_text.tStart = t  # local t and not account for scr refresh
                Control_category_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Control_category_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Control_category_text.started')
                # update status
                Control_category_text.status = STARTED
                Control_category_text.setAutoDraw(True)
            
            # if Control_category_text is active this frame...
            if Control_category_text.status == STARTED:
                # update params
                pass
            
            # if Control_category_text is stopping this frame...
            if Control_category_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Control_category_text.tStartRefresh + 9-frameTolerance:
                    # keep track of stop time/frame for later
                    Control_category_text.tStop = t  # not accounting for scr refresh
                    Control_category_text.tStopRefresh = tThisFlipGlobal  # on global time
                    Control_category_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Control_category_text.stopped')
                    # update status
                    Control_category_text.status = FINISHED
                    Control_category_text.setAutoDraw(False)
            
            # *Controlled_category_sound* updates
            
            # if Controlled_category_sound is starting this frame...
            if Controlled_category_sound.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Controlled_category_sound.frameNStart = frameN  # exact frame index
                Controlled_category_sound.tStart = t  # local t and not account for scr refresh
                Controlled_category_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('Controlled_category_sound.started', tThisFlipGlobal)
                # update status
                Controlled_category_sound.status = STARTED
                Controlled_category_sound.play(when=win)  # sync with win flip
            
            # if Controlled_category_sound is stopping this frame...
            if Controlled_category_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Controlled_category_sound.tStartRefresh + 9-frameTolerance or Controlled_category_sound.isFinished:
                    # keep track of stop time/frame for later
                    Controlled_category_sound.tStop = t  # not accounting for scr refresh
                    Controlled_category_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    Controlled_category_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Controlled_category_sound.stopped')
                    # update status
                    Controlled_category_sound.status = FINISHED
                    Controlled_category_sound.stop()
            
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
                    playbackComponents=[Controlled_category_sound]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                control.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in control.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "control" ---
        for thisComponent in control.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for control
        control.tStop = globalClock.getTime(format='float')
        control.tStopRefresh = tThisFlipGlobal
        thisExp.addData('control.stopped', control.tStop)
        Controlled_category_sound.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if control.maxDurationReached:
            routineTimer.addTime(-control.maxDuration)
        elif control.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-9.000000)
        
        # --- Prepare to start Routine "after_control_response" ---
        # create an object to store info about Routine after_control_response
        after_control_response = data.Routine(
            name='after_control_response',
            components=[Control_count_text_2, Controlled_association_text_free_4, Control_count_key_resp_2, controlled_response_key_resp_2],
        )
        after_control_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from after_control_code
        #thisExp.output_task.write(np.array([True], dtype=bool).tolist())
        #print("触发器信号: after control response 的 routine 开始高电平")
        # create starting attributes for Control_count_key_resp_2
        Control_count_key_resp_2.keys = []
        Control_count_key_resp_2.rt = []
        _Control_count_key_resp_2_allKeys = []
        # create starting attributes for controlled_response_key_resp_2
        controlled_response_key_resp_2.keys = []
        controlled_response_key_resp_2.rt = []
        _controlled_response_key_resp_2_allKeys = []
        # store start times for after_control_response
        after_control_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        after_control_response.tStart = globalClock.getTime(format='float')
        after_control_response.status = STARTED
        thisExp.addData('after_control_response.started', after_control_response.tStart)
        after_control_response.maxDuration = None
        # skip Routine after_control_response if its 'Skip if' condition is True
        after_control_response.skipped = continueRoutine and not ((thisExp.ifControlTrial == 'free') | (thisExp.responseCount < thisExp.randomCount))
        continueRoutine = after_control_response.skipped
        # keep track of which components have finished
        after_control_responseComponents = after_control_response.components
        for thisComponent in after_control_response.components:
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
        
        # --- Run Routine "after_control_response" ---
        # if trial has changed, end Routine now
        if isinstance(trials_loop, data.TrialHandler2) and thisTrials_loop.thisN != trials_loop.thisTrial.thisN:
            continueRoutine = False
        after_control_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from after_control_code
            # 简化按键检测逻辑
            current_keys = Control_count_key_resp_2.getKeys(keyList=['1'], waitRelease=True)
            Control_count_text_2.text = str(thisExp.responseCount + 1)
            
            #thisExp.output_task.write(np.array([True], dtype=bool).tolist())
            ##print("触发器信号: response 的 每一帧高电平")
            
            if current_keys:
                # 只有当检测到新的按键时才增加计数
                thisExp.responseCount += 1
                print(f"thisExp.responseCount:{thisExp.responseCount}")
                thisExp.addData('responseCount', thisExp.responseCount)
                thisExp.addData('Control_count_key_resp_2.keys', Control_count_key_resp_2.keys)
                thisExp.addData('Control_count_key_resp_2.rt', Control_count_key_resp_2.rt)
                # 清空按键缓冲区
                Control_count_key_resp_2.keys = []
                Control_count_key_resp_2.rt = []
                Control_count_text_2.text = str(thisExp.responseCount + 1)
                
            #    thisExp.output_task.write(np.array([False], dtype=bool).tolist())
            #    print("触发器信号: 按键计数低电平")
            #    
            #    if trigger_on:
            #        sendtrigger(1000+thisExp.trigger_n+1)
            #        thisExp.addData('triggerbox', triggerbox)
            #        thisExp.trigger_n+=1
            #        print(f"thisExp.trigger_n:{thisExp.trigger_n}")
            
            # *Control_count_text_2* updates
            
            # if Control_count_text_2 is starting this frame...
            if Control_count_text_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Control_count_text_2.frameNStart = frameN  # exact frame index
                Control_count_text_2.tStart = t  # local t and not account for scr refresh
                Control_count_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Control_count_text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Control_count_text_2.started')
                # update status
                Control_count_text_2.status = STARTED
                Control_count_text_2.setAutoDraw(True)
            
            # if Control_count_text_2 is active this frame...
            if Control_count_text_2.status == STARTED:
                # update params
                pass
            
            # *Controlled_association_text_free_4* updates
            
            # if Controlled_association_text_free_4 is starting this frame...
            if Controlled_association_text_free_4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                Controlled_association_text_free_4.frameNStart = frameN  # exact frame index
                Controlled_association_text_free_4.tStart = t  # local t and not account for scr refresh
                Controlled_association_text_free_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Controlled_association_text_free_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Controlled_association_text_free_4.started')
                # update status
                Controlled_association_text_free_4.status = STARTED
                Controlled_association_text_free_4.setAutoDraw(True)
            
            # if Controlled_association_text_free_4 is active this frame...
            if Controlled_association_text_free_4.status == STARTED:
                # update params
                pass
            
            # *Control_count_key_resp_2* updates
            waitOnFlip = False
            
            # if Control_count_key_resp_2 is starting this frame...
            if Control_count_key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Control_count_key_resp_2.frameNStart = frameN  # exact frame index
                Control_count_key_resp_2.tStart = t  # local t and not account for scr refresh
                Control_count_key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Control_count_key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Control_count_key_resp_2.started')
                # update status
                Control_count_key_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(Control_count_key_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(Control_count_key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if Control_count_key_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = Control_count_key_resp_2.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=False)
                _Control_count_key_resp_2_allKeys.extend(theseKeys)
                if len(_Control_count_key_resp_2_allKeys):
                    Control_count_key_resp_2.keys = _Control_count_key_resp_2_allKeys[-1].name  # just the last key pressed
                    Control_count_key_resp_2.rt = _Control_count_key_resp_2_allKeys[-1].rt
                    Control_count_key_resp_2.duration = _Control_count_key_resp_2_allKeys[-1].duration
            
            # *controlled_response_key_resp_2* updates
            waitOnFlip = False
            
            # if controlled_response_key_resp_2 is starting this frame...
            if controlled_response_key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                controlled_response_key_resp_2.frameNStart = frameN  # exact frame index
                controlled_response_key_resp_2.tStart = t  # local t and not account for scr refresh
                controlled_response_key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(controlled_response_key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'controlled_response_key_resp_2.started')
                # update status
                controlled_response_key_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(controlled_response_key_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(controlled_response_key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if controlled_response_key_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = controlled_response_key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _controlled_response_key_resp_2_allKeys.extend(theseKeys)
                if len(_controlled_response_key_resp_2_allKeys):
                    controlled_response_key_resp_2.keys = _controlled_response_key_resp_2_allKeys[-1].name  # just the last key pressed
                    controlled_response_key_resp_2.rt = _controlled_response_key_resp_2_allKeys[-1].rt
                    controlled_response_key_resp_2.duration = _controlled_response_key_resp_2_allKeys[-1].duration
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
                after_control_response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in after_control_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "after_control_response" ---
        for thisComponent in after_control_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for after_control_response
        after_control_response.tStop = globalClock.getTime(format='float')
        after_control_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('after_control_response.stopped', after_control_response.tStop)
        # check responses
        if Control_count_key_resp_2.keys in ['', [], None]:  # No response was made
            Control_count_key_resp_2.keys = None
        trials_loop.addData('Control_count_key_resp_2.keys',Control_count_key_resp_2.keys)
        if Control_count_key_resp_2.keys != None:  # we had a response
            trials_loop.addData('Control_count_key_resp_2.rt', Control_count_key_resp_2.rt)
            trials_loop.addData('Control_count_key_resp_2.duration', Control_count_key_resp_2.duration)
        # check responses
        if controlled_response_key_resp_2.keys in ['', [], None]:  # No response was made
            controlled_response_key_resp_2.keys = None
        trials_loop.addData('controlled_response_key_resp_2.keys',controlled_response_key_resp_2.keys)
        if controlled_response_key_resp_2.keys != None:  # we had a response
            trials_loop.addData('controlled_response_key_resp_2.rt', controlled_response_key_resp_2.rt)
            trials_loop.addData('controlled_response_key_resp_2.duration', controlled_response_key_resp_2.duration)
        # the Routine "after_control_response" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "rest_within" ---
        # create an object to store info about Routine rest_within
        rest_within = data.Routine(
            name='rest_within',
            components=[rest_text],
        )
        rest_within.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from rest_code
        # 生成0到1秒之间的随机持续时间
        rest_duration = random.uniform(0.2, 0.4)
        # store start times for rest_within
        rest_within.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        rest_within.tStart = globalClock.getTime(format='float')
        rest_within.status = STARTED
        thisExp.addData('rest_within.started', rest_within.tStart)
        rest_within.maxDuration = None
        # keep track of which components have finished
        rest_withinComponents = rest_within.components
        for thisComponent in rest_within.components:
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
        
        # --- Run Routine "rest_within" ---
        # if trial has changed, end Routine now
        if isinstance(trials_loop, data.TrialHandler2) and thisTrials_loop.thisN != trials_loop.thisTrial.thisN:
            continueRoutine = False
        rest_within.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rest_text* updates
            
            # if rest_text is starting this frame...
            if rest_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rest_text.frameNStart = frameN  # exact frame index
                rest_text.tStart = t  # local t and not account for scr refresh
                rest_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rest_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rest_text.started')
                # update status
                rest_text.status = STARTED
                rest_text.setAutoDraw(True)
            
            # if rest_text is active this frame...
            if rest_text.status == STARTED:
                # update params
                pass
            
            # if rest_text is stopping this frame...
            if rest_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rest_text.tStartRefresh + rest_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    rest_text.tStop = t  # not accounting for scr refresh
                    rest_text.tStopRefresh = tThisFlipGlobal  # on global time
                    rest_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rest_text.stopped')
                    # update status
                    rest_text.status = FINISHED
                    rest_text.setAutoDraw(False)
            
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
                rest_within.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rest_within.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rest_within" ---
        for thisComponent in rest_within.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for rest_within
        rest_within.tStop = globalClock.getTime(format='float')
        rest_within.tStopRefresh = tThisFlipGlobal
        thisExp.addData('rest_within.stopped', rest_within.tStop)
        # Run 'End Routine' code from rest_code
        #thisExp.output_task.write(np.array([False], dtype=bool).tolist())
        #print("触发器信号: trial结束 的 低电平")
        # the Routine "rest_within" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 4.0 repeats of 'trials_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
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
