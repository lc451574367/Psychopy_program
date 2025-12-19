#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on 十一月 20, 2025, at 16:24
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
        originPath='E:\\Post_log\\Voice_Imagery\\Exp\\Exp_preparation\\Voice imagery task\\Program_psychopy\\Voice_listening_task\\Sub02\\Voice_listening_task_lastrun.py',
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
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
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
    if deviceManager.getDevice('instr_formal1_key_resp') is None:
        # initialise instr_formal1_key_resp
        instr_formal1_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_formal1_key_resp',
        )
    # create speaker 'formal_1_sound'
    deviceManager.addDevice(
        deviceName='formal_1_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('selection_key_resp') is None:
        # initialise selection_key_resp
        selection_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='selection_key_resp',
        )
    if deviceManager.getDevice('rest_key_resp') is None:
        # initialise rest_key_resp
        rest_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='rest_key_resp',
        )
    # create speaker 'formal1_run2_sound'
    deviceManager.addDevice(
        deviceName='formal1_run2_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'formal3_sound'
    deviceManager.addDevice(
        deviceName='formal3_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'formal4_sound'
    deviceManager.addDevice(
        deviceName='formal4_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
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
    
    stimulus_file = os.path.join(os.getcwd(), 'stimulus', 'Stimulus_info.xlsx')
    roleName_file = os.path.join(os.getcwd(), 'stimulus', 'RoleName.xlsx')
    
    try:
        stimulus_df = pd.read_excel(stimulus_file)
        rolename_df = pd.read_excel(roleName_file)
        print(f"成功读取刺激材料文件，共{len(stimulus_df)}个刺激")
    except Exception as e:
        print(f"读取刺激材料文件失败: {e}")
        # 创建示例数据（416个刺激）
        stimulus_df = pd.DataFrame({
            'sentenceID': range(1, 417),  # 修改为416个刺激
            'sentences_list': [f'句子_{i}' for i in range(1, 417)]
        })
        
    roleName = {row['Role']: [row['Name'], row['EnglishName']] for _, row in rolename_df.iterrows()}
    thisExp.roleName = roleName
    # 将416个刺激随机分成4组，每组104个
    all_stimuli = stimulus_df.copy()
    all_indices = list(all_stimuli.index)
    random.shuffle(all_indices)  # 随机打乱
    
    # 分成4组，每组104个
    group_size = len(all_stimuli) // 4
    group_indices = [
        all_indices[:group_size],
        all_indices[group_size:group_size*2],
        all_indices[group_size*2:group_size*3],
        all_indices[group_size*3:]
    ]
    
    # 将所有组的数据存储在group_stimuli列表中
    group_stimuli = []  # 注意这里是单数形式
    for indices in group_indices:
        group_data = all_stimuli.loc[indices].reset_index(drop=True)
        group_stimuli.append(group_data)
    
    print(f"总刺激数量: {len(all_stimuli)}")
    for i, group in enumerate(group_stimuli):
        print(f"第{i+1}组刺激数量: {len(group)}")
    #    print(group)
    
    # 创建一个包含4个子列表的大数组来存储所有试验
    all_trials = [[] for _ in range(4)]
    
    for group_idx, group_data in enumerate(group_stimuli):  # 这里修正为 group_stimuli
        for idx, row in group_data.iterrows():
            trial = {
                'sentenceID': row['sentenceID'],
                'sentence_list': row['sentences_list'],
                'category': row['category'],
                'mid_category': row['mid_category'],
                'sub_category': row['sub_category'],
                'condition': row['condition'],
                'Stimulus_path': row['Stimulus_path'],
                'StimulusID': row['StimulusID'],
                'group': group_idx + 1  # 组号从1开始
            }
            all_trials[group_idx].append(trial)
        
        # 随机化当前组内的试验顺序
        random.shuffle(all_trials[group_idx])
    
    print(f"第一组刺激随机化: {len(all_trials[0])}")
    print(f"第二组刺激随机化: {len(all_trials[1])}")
    print(f"第三组刺激随机化: {len(all_trials[2])}")
    print(f"第四组刺激随机化: {len(all_trials[3])}")
    
    # 或者直接保存整个数组
    thisExp.all_trials = all_trials
    
    # 创建trial_counter数组
    thisExp.trial_counter = [0] * 4
    print(f"trial_counter数组已初始化: {thisExp.trial_counter}")
    
    thisExp.attention_trials_all = [None] * 4
    thisExp.current_attention_index_all = [0] * 4
    thisExp.show_attention_all = [False] * 4
    
    # --- Initialize components for Routine "instruction" ---
    instruction_image = visual.ImageStim(
        win=win,
        name='instruction_image', 
        image='image/instr_start.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.78, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    instruction_key_resp = keyboard.Keyboard(deviceName='instruction_key_resp')
    
    # --- Initialize components for Routine "instr_formal1" ---
    instr_formal1_image = visual.ImageStim(
        win=win,
        name='instr_formal1_image', 
        image='image/instr_formal_part1.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.78, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    instr_formal1_key_resp = keyboard.Keyboard(deviceName='instr_formal1_key_resp')
    
    # --- Initialize components for Routine "fixation" ---
    text = visual.TextStim(win=win, name='text',
        text='＋',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "formal_run1_trial" ---
    # Run 'Begin Experiment' code from formal_1_trial_code
    currentfilepath = []
    
    formal_1_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='formal_1_sound',    name='formal_1_sound'
    )
    formal_1_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "whose_voice" ---
    # Run 'Begin Experiment' code from selection_formal1_code
    # 在selection_formal1_2 routine的Begin Experiment标签中添加以下代码
    import random
    
    # 定义选项位置映射（固定位置）
    option_positions = {
        '1': (-0.5, 0),  # 选项1位置（左）
        '2': (-0.1, 0),     # 选项2位置（中）
        '3': (0.3, 0),    # 选项3位置（右）
        '4': (0.7, 0)    # 选项4位置（右）
    }
    
    
    # 定义正确答案映射 - 对应角色名称
    correct_answers = {
        'familiar': 'familiar',
        'unfamiliar': 'unfamiliar', 
        'celebrity': 'celebrity',
        'lab': 'lab'
    }
    option1_text = visual.TextStim(win=win, name='option1_text',
        text=None,
        font='Arial',
        pos=[option_positions['1']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    option2_text = visual.TextStim(win=win, name='option2_text',
        text=None,
        font='Arial',
        pos=[option_positions['2']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    option3_text = visual.TextStim(win=win, name='option3_text',
        text=None,
        font='Arial',
        pos=[option_positions['3']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    option4_text = visual.TextStim(win=win, name='option4_text',
        text=None,
        font='Arial',
        pos=[option_positions['4']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    selection_key_resp = keyboard.Keyboard(deviceName='selection_key_resp')
    
    # --- Initialize components for Routine "feedback" ---
    feedback_display = visual.TextStim(win=win, name='feedback_display',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "content" ---
    content_text = visual.TextStim(win=win, name='content_text',
        text='刚才那个人说了什么？',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "rest_within" ---
    rest_text = visual.TextStim(win=win, name='rest_text',
        text='rest',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "rest" ---
    rest_image = visual.ImageStim(
        win=win,
        name='rest_image', 
        image='image/rest.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.78, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    rest_key_resp = keyboard.Keyboard(deviceName='rest_key_resp')
    
    # --- Initialize components for Routine "run2_code_routine" ---
    
    # --- Initialize components for Routine "fixation" ---
    text = visual.TextStim(win=win, name='text',
        text='＋',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "formal_run2_trial" ---
    # Run 'Begin Experiment' code from formal2_trial_code
    # 在每个run的正式trial的Begin Experiment中
    run_index = 1  # 第一个run为0，第二个为1，以此类推
    attention_trials = random.sample(range(104), 2)
    attention_trials.sort()
    thisExp.attention_trials_all[run_index] = attention_trials
    thisExp.current_attention_index_all[run_index] = 0
    thisExp.show_attention_all[run_index] = False
    formal1_run2_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='formal1_run2_sound',    name='formal1_run2_sound'
    )
    formal1_run2_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "whose_voice" ---
    # Run 'Begin Experiment' code from selection_formal1_code
    # 在selection_formal1_2 routine的Begin Experiment标签中添加以下代码
    import random
    
    # 定义选项位置映射（固定位置）
    option_positions = {
        '1': (-0.5, 0),  # 选项1位置（左）
        '2': (-0.1, 0),     # 选项2位置（中）
        '3': (0.3, 0),    # 选项3位置（右）
        '4': (0.7, 0)    # 选项4位置（右）
    }
    
    
    # 定义正确答案映射 - 对应角色名称
    correct_answers = {
        'familiar': 'familiar',
        'unfamiliar': 'unfamiliar', 
        'celebrity': 'celebrity',
        'lab': 'lab'
    }
    option1_text = visual.TextStim(win=win, name='option1_text',
        text=None,
        font='Arial',
        pos=[option_positions['1']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    option2_text = visual.TextStim(win=win, name='option2_text',
        text=None,
        font='Arial',
        pos=[option_positions['2']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    option3_text = visual.TextStim(win=win, name='option3_text',
        text=None,
        font='Arial',
        pos=[option_positions['3']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    option4_text = visual.TextStim(win=win, name='option4_text',
        text=None,
        font='Arial',
        pos=[option_positions['4']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    selection_key_resp = keyboard.Keyboard(deviceName='selection_key_resp')
    
    # --- Initialize components for Routine "feedback" ---
    feedback_display = visual.TextStim(win=win, name='feedback_display',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "content" ---
    content_text = visual.TextStim(win=win, name='content_text',
        text='刚才那个人说了什么？',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "rest_within" ---
    rest_text = visual.TextStim(win=win, name='rest_text',
        text='rest',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "rest" ---
    rest_image = visual.ImageStim(
        win=win,
        name='rest_image', 
        image='image/rest.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.78, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    rest_key_resp = keyboard.Keyboard(deviceName='rest_key_resp')
    
    # --- Initialize components for Routine "run3_code_routine" ---
    
    # --- Initialize components for Routine "fixation" ---
    text = visual.TextStim(win=win, name='text',
        text='＋',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "formal_run3_trial" ---
    # Run 'Begin Experiment' code from formal2_run3_trial_code
    # 在每个run的正式trial的Begin Experiment中
    run_index = 2  # 第一个run为0，第二个为1，以此类推
    attention_trials = random.sample(range(104), 2)
    attention_trials.sort()
    thisExp.attention_trials_all[run_index] = attention_trials
    thisExp.current_attention_index_all[run_index] = 0
    thisExp.show_attention_all[run_index] = False
    formal3_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='formal3_sound',    name='formal3_sound'
    )
    formal3_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "whose_voice" ---
    # Run 'Begin Experiment' code from selection_formal1_code
    # 在selection_formal1_2 routine的Begin Experiment标签中添加以下代码
    import random
    
    # 定义选项位置映射（固定位置）
    option_positions = {
        '1': (-0.5, 0),  # 选项1位置（左）
        '2': (-0.1, 0),     # 选项2位置（中）
        '3': (0.3, 0),    # 选项3位置（右）
        '4': (0.7, 0)    # 选项4位置（右）
    }
    
    
    # 定义正确答案映射 - 对应角色名称
    correct_answers = {
        'familiar': 'familiar',
        'unfamiliar': 'unfamiliar', 
        'celebrity': 'celebrity',
        'lab': 'lab'
    }
    option1_text = visual.TextStim(win=win, name='option1_text',
        text=None,
        font='Arial',
        pos=[option_positions['1']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    option2_text = visual.TextStim(win=win, name='option2_text',
        text=None,
        font='Arial',
        pos=[option_positions['2']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    option3_text = visual.TextStim(win=win, name='option3_text',
        text=None,
        font='Arial',
        pos=[option_positions['3']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    option4_text = visual.TextStim(win=win, name='option4_text',
        text=None,
        font='Arial',
        pos=[option_positions['4']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    selection_key_resp = keyboard.Keyboard(deviceName='selection_key_resp')
    
    # --- Initialize components for Routine "feedback" ---
    feedback_display = visual.TextStim(win=win, name='feedback_display',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "content" ---
    content_text = visual.TextStim(win=win, name='content_text',
        text='刚才那个人说了什么？',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "rest_within" ---
    rest_text = visual.TextStim(win=win, name='rest_text',
        text='rest',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "rest" ---
    rest_image = visual.ImageStim(
        win=win,
        name='rest_image', 
        image='image/rest.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.78, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    rest_key_resp = keyboard.Keyboard(deviceName='rest_key_resp')
    
    # --- Initialize components for Routine "run4_code_routine" ---
    
    # --- Initialize components for Routine "fixation" ---
    text = visual.TextStim(win=win, name='text',
        text='＋',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "formal_run4_trial" ---
    # Run 'Begin Experiment' code from formal2_run4_trial_code
    # 在每个run的正式trial的Begin Experiment中
    run_index = 3  # 第一个run为0，第二个为1，以此类推
    attention_trials = random.sample(range(104), 2)
    attention_trials.sort()
    thisExp.attention_trials_all[run_index] = attention_trials
    thisExp.current_attention_index_all[run_index] = 0
    thisExp.show_attention_all[run_index] = False
    formal4_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='formal4_sound',    name='formal4_sound'
    )
    formal4_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "whose_voice" ---
    # Run 'Begin Experiment' code from selection_formal1_code
    # 在selection_formal1_2 routine的Begin Experiment标签中添加以下代码
    import random
    
    # 定义选项位置映射（固定位置）
    option_positions = {
        '1': (-0.5, 0),  # 选项1位置（左）
        '2': (-0.1, 0),     # 选项2位置（中）
        '3': (0.3, 0),    # 选项3位置（右）
        '4': (0.7, 0)    # 选项4位置（右）
    }
    
    
    # 定义正确答案映射 - 对应角色名称
    correct_answers = {
        'familiar': 'familiar',
        'unfamiliar': 'unfamiliar', 
        'celebrity': 'celebrity',
        'lab': 'lab'
    }
    option1_text = visual.TextStim(win=win, name='option1_text',
        text=None,
        font='Arial',
        pos=[option_positions['1']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    option2_text = visual.TextStim(win=win, name='option2_text',
        text=None,
        font='Arial',
        pos=[option_positions['2']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    option3_text = visual.TextStim(win=win, name='option3_text',
        text=None,
        font='Arial',
        pos=[option_positions['3']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    option4_text = visual.TextStim(win=win, name='option4_text',
        text=None,
        font='Arial',
        pos=[option_positions['4']], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    selection_key_resp = keyboard.Keyboard(deviceName='selection_key_resp')
    
    # --- Initialize components for Routine "feedback" ---
    feedback_display = visual.TextStim(win=win, name='feedback_display',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "content" ---
    content_text = visual.TextStim(win=win, name='content_text',
        text='刚才那个人说了什么？',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "rest_within" ---
    rest_text = visual.TextStim(win=win, name='rest_text',
        text='rest',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "end" ---
    end_image = visual.ImageStim(
        win=win,
        name='end_image', 
        image='image/end.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.78, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
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
    
    # --- Prepare to start Routine "instruction" ---
    # create an object to store info about Routine instruction
    instruction = data.Routine(
        name='instruction',
        components=[instruction_image, instruction_key_resp],
    )
    instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
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
    thisExp.addData('instruction_key_resp.keys',instruction_key_resp.keys)
    if instruction_key_resp.keys != None:  # we had a response
        thisExp.addData('instruction_key_resp.rt', instruction_key_resp.rt)
        thisExp.addData('instruction_key_resp.duration', instruction_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instr_formal1" ---
    # create an object to store info about Routine instr_formal1
    instr_formal1 = data.Routine(
        name='instr_formal1',
        components=[instr_formal1_image, instr_formal1_key_resp],
    )
    instr_formal1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_formal1_key_resp
    instr_formal1_key_resp.keys = []
    instr_formal1_key_resp.rt = []
    _instr_formal1_key_resp_allKeys = []
    # store start times for instr_formal1
    instr_formal1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr_formal1.tStart = globalClock.getTime(format='float')
    instr_formal1.status = STARTED
    thisExp.addData('instr_formal1.started', instr_formal1.tStart)
    instr_formal1.maxDuration = None
    # keep track of which components have finished
    instr_formal1Components = instr_formal1.components
    for thisComponent in instr_formal1.components:
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
    
    # --- Run Routine "instr_formal1" ---
    instr_formal1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr_formal1_image* updates
        
        # if instr_formal1_image is starting this frame...
        if instr_formal1_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_formal1_image.frameNStart = frameN  # exact frame index
            instr_formal1_image.tStart = t  # local t and not account for scr refresh
            instr_formal1_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_formal1_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_formal1_image.started')
            # update status
            instr_formal1_image.status = STARTED
            instr_formal1_image.setAutoDraw(True)
        
        # if instr_formal1_image is active this frame...
        if instr_formal1_image.status == STARTED:
            # update params
            pass
        
        # *instr_formal1_key_resp* updates
        waitOnFlip = False
        
        # if instr_formal1_key_resp is starting this frame...
        if instr_formal1_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_formal1_key_resp.frameNStart = frameN  # exact frame index
            instr_formal1_key_resp.tStart = t  # local t and not account for scr refresh
            instr_formal1_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_formal1_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_formal1_key_resp.started')
            # update status
            instr_formal1_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_formal1_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_formal1_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_formal1_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr_formal1_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_formal1_key_resp_allKeys.extend(theseKeys)
            if len(_instr_formal1_key_resp_allKeys):
                instr_formal1_key_resp.keys = _instr_formal1_key_resp_allKeys[-1].name  # just the last key pressed
                instr_formal1_key_resp.rt = _instr_formal1_key_resp_allKeys[-1].rt
                instr_formal1_key_resp.duration = _instr_formal1_key_resp_allKeys[-1].duration
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
            instr_formal1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_formal1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr_formal1" ---
    for thisComponent in instr_formal1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr_formal1
    instr_formal1.tStop = globalClock.getTime(format='float')
    instr_formal1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr_formal1.stopped', instr_formal1.tStop)
    # check responses
    if instr_formal1_key_resp.keys in ['', [], None]:  # No response was made
        instr_formal1_key_resp.keys = None
    thisExp.addData('instr_formal1_key_resp.keys',instr_formal1_key_resp.keys)
    if instr_formal1_key_resp.keys != None:  # we had a response
        thisExp.addData('instr_formal1_key_resp.rt', instr_formal1_key_resp.rt)
        thisExp.addData('instr_formal1_key_resp.duration', instr_formal1_key_resp.duration)
    # Run 'End Routine' code from run1_code
    # 在每个run的正式trial的Begin Experiment中
    run_index = 0  # 第一个run为0，第二个为1，以此类推
    attention_trials = random.sample(range(104), 2)
    attention_trials.sort()
    thisExp.attention_trials_all[run_index] = attention_trials
    thisExp.current_attention_index_all[run_index] = 0
    thisExp.show_attention_all[run_index] = False
    
    print(f"show_attention_all: {thisExp.show_attention_all}")
    thisExp.nextEntry()
    # the Routine "instr_formal1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    formal1_1_loop_trials = data.TrialHandler2(
        name='formal1_1_loop_trials',
        nReps=104.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(formal1_1_loop_trials)  # add the loop to the experiment
    thisFormal1_1_loop_trial = formal1_1_loop_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisFormal1_1_loop_trial.rgb)
    if thisFormal1_1_loop_trial != None:
        for paramName in thisFormal1_1_loop_trial:
            globals()[paramName] = thisFormal1_1_loop_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisFormal1_1_loop_trial in formal1_1_loop_trials:
        currentLoop = formal1_1_loop_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisFormal1_1_loop_trial.rgb)
        if thisFormal1_1_loop_trial != None:
            for paramName in thisFormal1_1_loop_trial:
                globals()[paramName] = thisFormal1_1_loop_trial[paramName]
        
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
        if isinstance(formal1_1_loop_trials, data.TrialHandler2) and thisFormal1_1_loop_trial.thisN != formal1_1_loop_trials.thisTrial.thisN:
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
        
        # --- Prepare to start Routine "formal_run1_trial" ---
        # create an object to store info about Routine formal_run1_trial
        formal_run1_trial = data.Routine(
            name='formal_run1_trial',
            components=[formal_1_sound],
        )
        formal_run1_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from formal_1_trial_code
        # 获取当前trial索引
        run_index = 0  # 根据当前run设置
        trial_index = thisExp.trial_counter[run_index]
        
        # 检查是否需要显示attention check
        thisExp.show_attention_all[run_index] = False
        if (thisExp.current_attention_index_all[run_index] < len(thisExp.attention_trials_all[run_index]) and 
            trial_index == thisExp.attention_trials_all[run_index][thisExp.current_attention_index_all[run_index]]):
            thisExp.show_attention_all[run_index] = True
            thisExp.current_attention_index_all[run_index] += 1
        
        current_trial = thisExp.all_trials[run_index][trial_index]
        currentSentence = current_trial['sentence_list']
        currentSentenceID = current_trial['sentenceID']
        currentCategory = current_trial['category']
        currentMidCategory = current_trial['mid_category']
        currentSubCategory = current_trial['sub_category']
        currentcondition = current_trial['condition']
        currentfilepath = current_trial['Stimulus_path']
        currentStimulusID = current_trial['StimulusID']
        
        print(f"sentence : {currentSentence}")
        
        # 更新计数器
        thisExp.trial_counter[run_index] += 1
        thisExp.correctAnswer = correct_answers.get(currentcondition, '未知')
        print(f"Voice : {thisExp.correctAnswer}")
        
        thisExp.addData('sentence_list', currentSentence)
        thisExp.addData('sentenceID', currentSentenceID)
        thisExp.addData('category', currentCategory)
        thisExp.addData('mid_category', currentMidCategory)
        thisExp.addData('sub_category', currentSubCategory)
        thisExp.addData('condition', currentcondition)
        thisExp.addData('Stimulus_path', currentfilepath)
        thisExp.addData('StimulusID', currentStimulusID)
        thisExp.addData('filepath', currentfilepath)
        thisExp.addData('correct_answer', thisExp.correctAnswer)
        thisExp.addData('has_attention_check', thisExp.show_attention_all[run_index])  # 记录是否有attention check
        formal_1_sound.setSound(currentfilepath, secs=3, hamming=True)
        formal_1_sound.setVolume(1.0, log=False)
        formal_1_sound.seek(0)
        # store start times for formal_run1_trial
        formal_run1_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal_run1_trial.tStart = globalClock.getTime(format='float')
        formal_run1_trial.status = STARTED
        thisExp.addData('formal_run1_trial.started', formal_run1_trial.tStart)
        formal_run1_trial.maxDuration = None
        # keep track of which components have finished
        formal_run1_trialComponents = formal_run1_trial.components
        for thisComponent in formal_run1_trial.components:
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
        
        # --- Run Routine "formal_run1_trial" ---
        # if trial has changed, end Routine now
        if isinstance(formal1_1_loop_trials, data.TrialHandler2) and thisFormal1_1_loop_trial.thisN != formal1_1_loop_trials.thisTrial.thisN:
            continueRoutine = False
        formal_run1_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *formal_1_sound* updates
            
            # if formal_1_sound is starting this frame...
            if formal_1_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal_1_sound.frameNStart = frameN  # exact frame index
                formal_1_sound.tStart = t  # local t and not account for scr refresh
                formal_1_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('formal_1_sound.started', tThisFlipGlobal)
                # update status
                formal_1_sound.status = STARTED
                formal_1_sound.play(when=win)  # sync with win flip
            
            # if formal_1_sound is stopping this frame...
            if formal_1_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > formal_1_sound.tStartRefresh + 3-frameTolerance or formal_1_sound.isFinished:
                    # keep track of stop time/frame for later
                    formal_1_sound.tStop = t  # not accounting for scr refresh
                    formal_1_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    formal_1_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'formal_1_sound.stopped')
                    # update status
                    formal_1_sound.status = FINISHED
                    formal_1_sound.stop()
            
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
                    playbackComponents=[formal_1_sound]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                formal_run1_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal_run1_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal_run1_trial" ---
        for thisComponent in formal_run1_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal_run1_trial
        formal_run1_trial.tStop = globalClock.getTime(format='float')
        formal_run1_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal_run1_trial.stopped', formal_run1_trial.tStop)
        formal_1_sound.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if formal_run1_trial.maxDurationReached:
            routineTimer.addTime(-formal_run1_trial.maxDuration)
        elif formal_run1_trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "whose_voice" ---
        # create an object to store info about Routine whose_voice
        whose_voice = data.Routine(
            name='whose_voice',
            components=[option1_text, option2_text, option3_text, option4_text, selection_key_resp],
        )
        whose_voice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from selection_formal1_code
        # selection_formal1_2 routine的Begin Routine部分
        # 随机分配三个身份到三个按键位置
        voice_types = ['familiar', 'celebrity', 'unfamiliar', 'lab']
        random.shuffle(voice_types)  # 随机打乱顺序
        
        # 创建本次试验的按键与身份映射
        key_to_voice = {
            '1': voice_types[0],
            '2': voice_types[1], 
            '3': voice_types[2],
            '4': voice_types[3],
        }
        
        # 存储本次试验的映射关系，用于后续反馈
        thisExp.current_key_to_voice = key_to_voice
        
        # 为每个身份随机选择显示中文名还是英文名
        name_display = {}  # 存储本次试验中每个身份显示的名字
        language_choice = {}  # 存储语言选择（用于数据记录）
        
        for voice_type in voice_types:
            # 随机选择中文或英文
            chosen_language = random.choice(['chinese', 'english'])
            language_choice[voice_type] = chosen_language
            
            # 根据选择的语言获取对应的名字
            if chosen_language == 'chinese':
                name_display[voice_type] = thisExp.roleName[voice_type][0]  # 中文名
            else:
                name_display[voice_type] = thisExp.roleName[voice_type][1]  # 英文名
        
        # 设置三个选项文本组件的内容
        # 使用随机分配的映射关系
        option1_text.text = f"1. {name_display[key_to_voice['1']]}"    # 选项1
        option2_text.text = f"2. {name_display[key_to_voice['2']]}"    # 选项2
        option3_text.text = f"3. {name_display[key_to_voice['3']]}"    # 选项3
        option4_text.text = f"4. {name_display[key_to_voice['4']]}"    # 选项3
        
        # 记录语言选择信息和按键映射到实验数据
        thisExp.addData('familiar_name_language', language_choice.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_language', language_choice.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_language', language_choice.get('unfamiliar', '未出现'))
        thisExp.addData('lab_name_language', language_choice.get('lab', '未出现'))
        thisExp.addData('familiar_name_displayed', name_display.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_displayed', name_display.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_displayed', name_display.get('unfamiliar', '未出现'))
        thisExp.addData('lab_name_displayed', name_display.get('lab', '未出现'))
        
        # 记录本次试验的按键映射
        thisExp.addData('key1_mapping', key_to_voice['1'])
        thisExp.addData('key2_mapping', key_to_voice['2'])
        thisExp.addData('key3_mapping', key_to_voice['3'])
        thisExp.addData('key4_mapping', key_to_voice['4'])
        # create starting attributes for selection_key_resp
        selection_key_resp.keys = []
        selection_key_resp.rt = []
        _selection_key_resp_allKeys = []
        # store start times for whose_voice
        whose_voice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        whose_voice.tStart = globalClock.getTime(format='float')
        whose_voice.status = STARTED
        thisExp.addData('whose_voice.started', whose_voice.tStart)
        whose_voice.maxDuration = None
        # skip Routine whose_voice if its 'Skip if' condition is True
        whose_voice.skipped = continueRoutine and not (thisExp.show_attention_all[run_index] == False)
        continueRoutine = whose_voice.skipped
        # keep track of which components have finished
        whose_voiceComponents = whose_voice.components
        for thisComponent in whose_voice.components:
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
        
        # --- Run Routine "whose_voice" ---
        # if trial has changed, end Routine now
        if isinstance(formal1_1_loop_trials, data.TrialHandler2) and thisFormal1_1_loop_trial.thisN != formal1_1_loop_trials.thisTrial.thisN:
            continueRoutine = False
        whose_voice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *option1_text* updates
            
            # if option1_text is starting this frame...
            if option1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option1_text.frameNStart = frameN  # exact frame index
                option1_text.tStart = t  # local t and not account for scr refresh
                option1_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option1_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option1_text.started')
                # update status
                option1_text.status = STARTED
                option1_text.setAutoDraw(True)
            
            # if option1_text is active this frame...
            if option1_text.status == STARTED:
                # update params
                pass
            
            # if option1_text is stopping this frame...
            if option1_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option1_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option1_text.tStop = t  # not accounting for scr refresh
                    option1_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option1_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option1_text.stopped')
                    # update status
                    option1_text.status = FINISHED
                    option1_text.setAutoDraw(False)
            
            # *option2_text* updates
            
            # if option2_text is starting this frame...
            if option2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option2_text.frameNStart = frameN  # exact frame index
                option2_text.tStart = t  # local t and not account for scr refresh
                option2_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option2_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option2_text.started')
                # update status
                option2_text.status = STARTED
                option2_text.setAutoDraw(True)
            
            # if option2_text is active this frame...
            if option2_text.status == STARTED:
                # update params
                pass
            
            # if option2_text is stopping this frame...
            if option2_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option2_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option2_text.tStop = t  # not accounting for scr refresh
                    option2_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option2_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option2_text.stopped')
                    # update status
                    option2_text.status = FINISHED
                    option2_text.setAutoDraw(False)
            
            # *option3_text* updates
            
            # if option3_text is starting this frame...
            if option3_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option3_text.frameNStart = frameN  # exact frame index
                option3_text.tStart = t  # local t and not account for scr refresh
                option3_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option3_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option3_text.started')
                # update status
                option3_text.status = STARTED
                option3_text.setAutoDraw(True)
            
            # if option3_text is active this frame...
            if option3_text.status == STARTED:
                # update params
                pass
            
            # if option3_text is stopping this frame...
            if option3_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option3_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option3_text.tStop = t  # not accounting for scr refresh
                    option3_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option3_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option3_text.stopped')
                    # update status
                    option3_text.status = FINISHED
                    option3_text.setAutoDraw(False)
            
            # *option4_text* updates
            
            # if option4_text is starting this frame...
            if option4_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option4_text.frameNStart = frameN  # exact frame index
                option4_text.tStart = t  # local t and not account for scr refresh
                option4_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option4_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option4_text.started')
                # update status
                option4_text.status = STARTED
                option4_text.setAutoDraw(True)
            
            # if option4_text is active this frame...
            if option4_text.status == STARTED:
                # update params
                pass
            
            # if option4_text is stopping this frame...
            if option4_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option4_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option4_text.tStop = t  # not accounting for scr refresh
                    option4_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option4_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option4_text.stopped')
                    # update status
                    option4_text.status = FINISHED
                    option4_text.setAutoDraw(False)
            
            # *selection_key_resp* updates
            waitOnFlip = False
            
            # if selection_key_resp is starting this frame...
            if selection_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                selection_key_resp.frameNStart = frameN  # exact frame index
                selection_key_resp.tStart = t  # local t and not account for scr refresh
                selection_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(selection_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'selection_key_resp.started')
                # update status
                selection_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(selection_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(selection_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if selection_key_resp is stopping this frame...
            if selection_key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > selection_key_resp.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    selection_key_resp.tStop = t  # not accounting for scr refresh
                    selection_key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    selection_key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'selection_key_resp.stopped')
                    # update status
                    selection_key_resp.status = FINISHED
                    selection_key_resp.status = FINISHED
            if selection_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = selection_key_resp.getKeys(keyList=['1','2','3'], ignoreKeys=["escape"], waitRelease=False)
                _selection_key_resp_allKeys.extend(theseKeys)
                if len(_selection_key_resp_allKeys):
                    selection_key_resp.keys = _selection_key_resp_allKeys[-1].name  # just the last key pressed
                    selection_key_resp.rt = _selection_key_resp_allKeys[-1].rt
                    selection_key_resp.duration = _selection_key_resp_allKeys[-1].duration
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
                whose_voice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in whose_voice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "whose_voice" ---
        for thisComponent in whose_voice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for whose_voice
        whose_voice.tStop = globalClock.getTime(format='float')
        whose_voice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('whose_voice.stopped', whose_voice.tStop)
        # check responses
        if selection_key_resp.keys in ['', [], None]:  # No response was made
            selection_key_resp.keys = None
        formal1_1_loop_trials.addData('selection_key_resp.keys',selection_key_resp.keys)
        if selection_key_resp.keys != None:  # we had a response
            formal1_1_loop_trials.addData('selection_key_resp.rt', selection_key_resp.rt)
            formal1_1_loop_trials.addData('selection_key_resp.duration', selection_key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if whose_voice.maxDurationReached:
            routineTimer.addTime(-whose_voice.maxDuration)
        elif whose_voice.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[feedback_display],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from feedback_code
        # feedback routine的Begin Routine部分
        # 获取被试的反应
        if selection_key_resp.keys:
            # 处理反应键
            if type(selection_key_resp.keys) is list:
                participant_response_key = selection_key_resp.keys[0]
            else:
                participant_response_key = selection_key_resp.keys
            
            # 处理反应时
            if type(selection_key_resp.rt) is list:
                participant_rt = selection_key_resp.rt[0]
            else:
                participant_rt = selection_key_resp.rt
            
            # 使用本次试验的映射将按键转换为对应的身份
            participant_response = thisExp.current_key_to_voice.get(participant_response_key, '未知')
        else:
            participant_response = '无反应'
            participant_rt = -1
            participant_response_key = '无反应'
        
        # 判断回答是否正确
        is_correct = (participant_response == thisExp.correctAnswer)
        
        # 设置反馈文本
        if participant_response == '无反应':
            feedback_text = "未检测到反应，请尽快回答！"
            feedback_color = 'red'
        elif is_correct:
            feedback_text = "回答正确！"
            feedback_color = 'green'
        else:
            # 获取正确答案对应的中文名称
            correct_voice_type = thisExp.correctAnswer
            if correct_voice_type in thisExp.roleName:
                correct_name = thisExp.roleName[correct_voice_type][0]  # 中文名
                # 获取被试选择的身份名称
                if participant_response in thisExp.roleName:
                    chosen_name = thisExp.roleName[participant_response][0]  # 中文名
                    
                    # 找出正确答案对应的按键
                    correct_key = None
                    for key, voice in thisExp.current_key_to_voice.items():
                        if voice == correct_voice_type:
                            correct_key = key
                            break
                    
                    feedback_text = f"错误！您选择了{participant_response_key}({chosen_name})\n正确答案是{correct_key}({correct_name})"
                else:
                    feedback_text = f"错误！正确答案是{correct_name}"
            else:
                feedback_text = f"错误！正确答案是{thisExp.correctAnswer}"
            feedback_color = 'red'
        
        # 更新反馈文本组件
        feedback_display.text = feedback_text
        feedback_display.color = feedback_color
        
        # 记录反应数据
        thisExp.addData('response_key', participant_response_key)  # 记录的按键
        thisExp.addData('response_identity', participant_response)  # 记录的身份
        thisExp.addData('rt', participant_rt)
        thisExp.addData('correct', 1 if is_correct else 0)
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # skip Routine feedback if its 'Skip if' condition is True
        feedback.skipped = continueRoutine and not (thisExp.show_attention_all[run_index] == False)
        continueRoutine = feedback.skipped
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
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
        
        # --- Run Routine "feedback" ---
        # if trial has changed, end Routine now
        if isinstance(formal1_1_loop_trials, data.TrialHandler2) and thisFormal1_1_loop_trial.thisN != formal1_1_loop_trials.thisTrial.thisN:
            continueRoutine = False
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_display* updates
            
            # if feedback_display is starting this frame...
            if feedback_display.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_display.frameNStart = frameN  # exact frame index
                feedback_display.tStart = t  # local t and not account for scr refresh
                feedback_display.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_display, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_display.started')
                # update status
                feedback_display.status = STARTED
                feedback_display.setAutoDraw(True)
            
            # if feedback_display is active this frame...
            if feedback_display.status == STARTED:
                # update params
                pass
            
            # if feedback_display is stopping this frame...
            if feedback_display.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_display.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_display.tStop = t  # not accounting for scr refresh
                    feedback_display.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_display.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_display.stopped')
                    # update status
                    feedback_display.status = FINISHED
                    feedback_display.setAutoDraw(False)
            
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
                feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback.maxDurationReached:
            routineTimer.addTime(-feedback.maxDuration)
        elif feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "content" ---
        # create an object to store info about Routine content
        content = data.Routine(
            name='content',
            components=[content_text],
        )
        content.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for content
        content.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        content.tStart = globalClock.getTime(format='float')
        content.status = STARTED
        thisExp.addData('content.started', content.tStart)
        content.maxDuration = None
        # skip Routine content if its 'Skip if' condition is True
        content.skipped = continueRoutine and not (thisExp.show_attention_all[run_index] == False)
        continueRoutine = content.skipped
        # keep track of which components have finished
        contentComponents = content.components
        for thisComponent in content.components:
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
        
        # --- Run Routine "content" ---
        # if trial has changed, end Routine now
        if isinstance(formal1_1_loop_trials, data.TrialHandler2) and thisFormal1_1_loop_trial.thisN != formal1_1_loop_trials.thisTrial.thisN:
            continueRoutine = False
        content.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *content_text* updates
            
            # if content_text is starting this frame...
            if content_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                content_text.frameNStart = frameN  # exact frame index
                content_text.tStart = t  # local t and not account for scr refresh
                content_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(content_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'content_text.started')
                # update status
                content_text.status = STARTED
                content_text.setAutoDraw(True)
            
            # if content_text is active this frame...
            if content_text.status == STARTED:
                # update params
                pass
            
            # if content_text is stopping this frame...
            if content_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > content_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    content_text.tStop = t  # not accounting for scr refresh
                    content_text.tStopRefresh = tThisFlipGlobal  # on global time
                    content_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'content_text.stopped')
                    # update status
                    content_text.status = FINISHED
                    content_text.setAutoDraw(False)
            
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
                content.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in content.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "content" ---
        for thisComponent in content.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for content
        content.tStop = globalClock.getTime(format='float')
        content.tStopRefresh = tThisFlipGlobal
        thisExp.addData('content.stopped', content.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if content.maxDurationReached:
            routineTimer.addTime(-content.maxDuration)
        elif content.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
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
        import random
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
        if isinstance(formal1_1_loop_trials, data.TrialHandler2) and thisFormal1_1_loop_trial.thisN != formal1_1_loop_trials.thisTrial.thisN:
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
        # the Routine "rest_within" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 104.0 repeats of 'formal1_1_loop_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "rest" ---
    # create an object to store info about Routine rest
    rest = data.Routine(
        name='rest',
        components=[rest_image, rest_key_resp],
    )
    rest.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for rest_key_resp
    rest_key_resp.keys = []
    rest_key_resp.rt = []
    _rest_key_resp_allKeys = []
    # store start times for rest
    rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    rest.tStart = globalClock.getTime(format='float')
    rest.status = STARTED
    thisExp.addData('rest.started', rest.tStart)
    rest.maxDuration = None
    # keep track of which components have finished
    restComponents = rest.components
    for thisComponent in rest.components:
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
    
    # --- Run Routine "rest" ---
    rest.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rest_image* updates
        
        # if rest_image is starting this frame...
        if rest_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rest_image.frameNStart = frameN  # exact frame index
            rest_image.tStart = t  # local t and not account for scr refresh
            rest_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rest_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rest_image.started')
            # update status
            rest_image.status = STARTED
            rest_image.setAutoDraw(True)
        
        # if rest_image is active this frame...
        if rest_image.status == STARTED:
            # update params
            pass
        
        # *rest_key_resp* updates
        waitOnFlip = False
        
        # if rest_key_resp is starting this frame...
        if rest_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rest_key_resp.frameNStart = frameN  # exact frame index
            rest_key_resp.tStart = t  # local t and not account for scr refresh
            rest_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rest_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rest_key_resp.started')
            # update status
            rest_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(rest_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(rest_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if rest_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = rest_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _rest_key_resp_allKeys.extend(theseKeys)
            if len(_rest_key_resp_allKeys):
                rest_key_resp.keys = _rest_key_resp_allKeys[-1].name  # just the last key pressed
                rest_key_resp.rt = _rest_key_resp_allKeys[-1].rt
                rest_key_resp.duration = _rest_key_resp_allKeys[-1].duration
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
            rest.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in rest.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "rest" ---
    for thisComponent in rest.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for rest
    rest.tStop = globalClock.getTime(format='float')
    rest.tStopRefresh = tThisFlipGlobal
    thisExp.addData('rest.stopped', rest.tStop)
    # check responses
    if rest_key_resp.keys in ['', [], None]:  # No response was made
        rest_key_resp.keys = None
    thisExp.addData('rest_key_resp.keys',rest_key_resp.keys)
    if rest_key_resp.keys != None:  # we had a response
        thisExp.addData('rest_key_resp.rt', rest_key_resp.rt)
        thisExp.addData('rest_key_resp.duration', rest_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "rest" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "run2_code_routine" ---
    # create an object to store info about Routine run2_code_routine
    run2_code_routine = data.Routine(
        name='run2_code_routine',
        components=[],
    )
    run2_code_routine.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for run2_code_routine
    run2_code_routine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    run2_code_routine.tStart = globalClock.getTime(format='float')
    run2_code_routine.status = STARTED
    thisExp.addData('run2_code_routine.started', run2_code_routine.tStart)
    run2_code_routine.maxDuration = None
    # keep track of which components have finished
    run2_code_routineComponents = run2_code_routine.components
    for thisComponent in run2_code_routine.components:
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
    
    # --- Run Routine "run2_code_routine" ---
    run2_code_routine.forceEnded = routineForceEnded = not continueRoutine
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
            run2_code_routine.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in run2_code_routine.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "run2_code_routine" ---
    for thisComponent in run2_code_routine.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for run2_code_routine
    run2_code_routine.tStop = globalClock.getTime(format='float')
    run2_code_routine.tStopRefresh = tThisFlipGlobal
    thisExp.addData('run2_code_routine.stopped', run2_code_routine.tStop)
    # Run 'End Routine' code from run2_code
    run_index = 1
    thisExp.nextEntry()
    # the Routine "run2_code_routine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    formal1_2_loop_trials = data.TrialHandler2(
        name='formal1_2_loop_trials',
        nReps=104.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(formal1_2_loop_trials)  # add the loop to the experiment
    thisFormal1_2_loop_trial = formal1_2_loop_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisFormal1_2_loop_trial.rgb)
    if thisFormal1_2_loop_trial != None:
        for paramName in thisFormal1_2_loop_trial:
            globals()[paramName] = thisFormal1_2_loop_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisFormal1_2_loop_trial in formal1_2_loop_trials:
        currentLoop = formal1_2_loop_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisFormal1_2_loop_trial.rgb)
        if thisFormal1_2_loop_trial != None:
            for paramName in thisFormal1_2_loop_trial:
                globals()[paramName] = thisFormal1_2_loop_trial[paramName]
        
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
        if isinstance(formal1_2_loop_trials, data.TrialHandler2) and thisFormal1_2_loop_trial.thisN != formal1_2_loop_trials.thisTrial.thisN:
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
        
        # --- Prepare to start Routine "formal_run2_trial" ---
        # create an object to store info about Routine formal_run2_trial
        formal_run2_trial = data.Routine(
            name='formal_run2_trial',
            components=[formal1_run2_sound],
        )
        formal_run2_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from formal2_trial_code
        # 在每个run的正式trial的Begin Routine中
        run_index = 1  # 根据当前run设置
        trial_index = thisExp.trial_counter[run_index]
        
        # 检查是否需要显示attention check
        thisExp.show_attention_all[run_index] = False
        if (thisExp.current_attention_index_all[run_index] < len(thisExp.attention_trials_all[run_index]) and 
            trial_index == thisExp.attention_trials_all[run_index][thisExp.current_attention_index_all[run_index]]):
            thisExp.show_attention_all[run_index] = True
            thisExp.current_attention_index_all[run_index] += 1
        
        current_trial = thisExp.all_trials[run_index][trial_index]
        currentSentence = current_trial['sentence_list']
        currentSentenceID = current_trial['sentenceID']
        currentCategory = current_trial['category']
        currentMidCategory = current_trial['mid_category']
        currentSubCategory = current_trial['sub_category']
        currentcondition = current_trial['condition']
        currentfilepath = current_trial['Stimulus_path']
        currentStimulusID = current_trial['StimulusID']
        
        # 更新计数器
        thisExp.trial_counter[run_index] += 1
        thisExp.correctAnswer = correct_answers.get(currentcondition, '未知')
        
        thisExp.addData('sentence_list', currentSentence)
        thisExp.addData('sentenceID', currentSentenceID)
        thisExp.addData('category', currentCategory)
        thisExp.addData('mid_category', currentMidCategory)
        thisExp.addData('sub_category', currentSubCategory)
        thisExp.addData('condition', currentcondition)
        thisExp.addData('Stimulus_path', currentfilepath)
        thisExp.addData('StimulusID', currentStimulusID)
        thisExp.addData('filepath', currentfilepath)
        thisExp.addData('correct_answer', thisExp.correctAnswer)
        thisExp.addData('has_attention_check', thisExp.show_attention_all[run_index])  # 记录是否有attention check
        formal1_run2_sound.setSound(currentfilepath, secs=3, hamming=True)
        formal1_run2_sound.setVolume(1.0, log=False)
        formal1_run2_sound.seek(0)
        # store start times for formal_run2_trial
        formal_run2_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal_run2_trial.tStart = globalClock.getTime(format='float')
        formal_run2_trial.status = STARTED
        thisExp.addData('formal_run2_trial.started', formal_run2_trial.tStart)
        formal_run2_trial.maxDuration = None
        # keep track of which components have finished
        formal_run2_trialComponents = formal_run2_trial.components
        for thisComponent in formal_run2_trial.components:
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
        
        # --- Run Routine "formal_run2_trial" ---
        # if trial has changed, end Routine now
        if isinstance(formal1_2_loop_trials, data.TrialHandler2) and thisFormal1_2_loop_trial.thisN != formal1_2_loop_trials.thisTrial.thisN:
            continueRoutine = False
        formal_run2_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *formal1_run2_sound* updates
            
            # if formal1_run2_sound is starting this frame...
            if formal1_run2_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal1_run2_sound.frameNStart = frameN  # exact frame index
                formal1_run2_sound.tStart = t  # local t and not account for scr refresh
                formal1_run2_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('formal1_run2_sound.started', tThisFlipGlobal)
                # update status
                formal1_run2_sound.status = STARTED
                formal1_run2_sound.play(when=win)  # sync with win flip
            
            # if formal1_run2_sound is stopping this frame...
            if formal1_run2_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > formal1_run2_sound.tStartRefresh + 3-frameTolerance or formal1_run2_sound.isFinished:
                    # keep track of stop time/frame for later
                    formal1_run2_sound.tStop = t  # not accounting for scr refresh
                    formal1_run2_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    formal1_run2_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'formal1_run2_sound.stopped')
                    # update status
                    formal1_run2_sound.status = FINISHED
                    formal1_run2_sound.stop()
            
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
                    playbackComponents=[formal1_run2_sound]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                formal_run2_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal_run2_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal_run2_trial" ---
        for thisComponent in formal_run2_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal_run2_trial
        formal_run2_trial.tStop = globalClock.getTime(format='float')
        formal_run2_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal_run2_trial.stopped', formal_run2_trial.tStop)
        formal1_run2_sound.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if formal_run2_trial.maxDurationReached:
            routineTimer.addTime(-formal_run2_trial.maxDuration)
        elif formal_run2_trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "whose_voice" ---
        # create an object to store info about Routine whose_voice
        whose_voice = data.Routine(
            name='whose_voice',
            components=[option1_text, option2_text, option3_text, option4_text, selection_key_resp],
        )
        whose_voice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from selection_formal1_code
        # selection_formal1_2 routine的Begin Routine部分
        # 随机分配三个身份到三个按键位置
        voice_types = ['familiar', 'celebrity', 'unfamiliar', 'lab']
        random.shuffle(voice_types)  # 随机打乱顺序
        
        # 创建本次试验的按键与身份映射
        key_to_voice = {
            '1': voice_types[0],
            '2': voice_types[1], 
            '3': voice_types[2],
            '4': voice_types[3],
        }
        
        # 存储本次试验的映射关系，用于后续反馈
        thisExp.current_key_to_voice = key_to_voice
        
        # 为每个身份随机选择显示中文名还是英文名
        name_display = {}  # 存储本次试验中每个身份显示的名字
        language_choice = {}  # 存储语言选择（用于数据记录）
        
        for voice_type in voice_types:
            # 随机选择中文或英文
            chosen_language = random.choice(['chinese', 'english'])
            language_choice[voice_type] = chosen_language
            
            # 根据选择的语言获取对应的名字
            if chosen_language == 'chinese':
                name_display[voice_type] = thisExp.roleName[voice_type][0]  # 中文名
            else:
                name_display[voice_type] = thisExp.roleName[voice_type][1]  # 英文名
        
        # 设置三个选项文本组件的内容
        # 使用随机分配的映射关系
        option1_text.text = f"1. {name_display[key_to_voice['1']]}"    # 选项1
        option2_text.text = f"2. {name_display[key_to_voice['2']]}"    # 选项2
        option3_text.text = f"3. {name_display[key_to_voice['3']]}"    # 选项3
        option4_text.text = f"4. {name_display[key_to_voice['4']]}"    # 选项3
        
        # 记录语言选择信息和按键映射到实验数据
        thisExp.addData('familiar_name_language', language_choice.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_language', language_choice.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_language', language_choice.get('unfamiliar', '未出现'))
        thisExp.addData('lab_name_language', language_choice.get('lab', '未出现'))
        thisExp.addData('familiar_name_displayed', name_display.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_displayed', name_display.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_displayed', name_display.get('unfamiliar', '未出现'))
        thisExp.addData('lab_name_displayed', name_display.get('lab', '未出现'))
        
        # 记录本次试验的按键映射
        thisExp.addData('key1_mapping', key_to_voice['1'])
        thisExp.addData('key2_mapping', key_to_voice['2'])
        thisExp.addData('key3_mapping', key_to_voice['3'])
        thisExp.addData('key4_mapping', key_to_voice['4'])
        # create starting attributes for selection_key_resp
        selection_key_resp.keys = []
        selection_key_resp.rt = []
        _selection_key_resp_allKeys = []
        # store start times for whose_voice
        whose_voice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        whose_voice.tStart = globalClock.getTime(format='float')
        whose_voice.status = STARTED
        thisExp.addData('whose_voice.started', whose_voice.tStart)
        whose_voice.maxDuration = None
        # skip Routine whose_voice if its 'Skip if' condition is True
        whose_voice.skipped = continueRoutine and not (thisExp.show_attention_all[run_index] == False)
        continueRoutine = whose_voice.skipped
        # keep track of which components have finished
        whose_voiceComponents = whose_voice.components
        for thisComponent in whose_voice.components:
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
        
        # --- Run Routine "whose_voice" ---
        # if trial has changed, end Routine now
        if isinstance(formal1_2_loop_trials, data.TrialHandler2) and thisFormal1_2_loop_trial.thisN != formal1_2_loop_trials.thisTrial.thisN:
            continueRoutine = False
        whose_voice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *option1_text* updates
            
            # if option1_text is starting this frame...
            if option1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option1_text.frameNStart = frameN  # exact frame index
                option1_text.tStart = t  # local t and not account for scr refresh
                option1_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option1_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option1_text.started')
                # update status
                option1_text.status = STARTED
                option1_text.setAutoDraw(True)
            
            # if option1_text is active this frame...
            if option1_text.status == STARTED:
                # update params
                pass
            
            # if option1_text is stopping this frame...
            if option1_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option1_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option1_text.tStop = t  # not accounting for scr refresh
                    option1_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option1_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option1_text.stopped')
                    # update status
                    option1_text.status = FINISHED
                    option1_text.setAutoDraw(False)
            
            # *option2_text* updates
            
            # if option2_text is starting this frame...
            if option2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option2_text.frameNStart = frameN  # exact frame index
                option2_text.tStart = t  # local t and not account for scr refresh
                option2_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option2_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option2_text.started')
                # update status
                option2_text.status = STARTED
                option2_text.setAutoDraw(True)
            
            # if option2_text is active this frame...
            if option2_text.status == STARTED:
                # update params
                pass
            
            # if option2_text is stopping this frame...
            if option2_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option2_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option2_text.tStop = t  # not accounting for scr refresh
                    option2_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option2_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option2_text.stopped')
                    # update status
                    option2_text.status = FINISHED
                    option2_text.setAutoDraw(False)
            
            # *option3_text* updates
            
            # if option3_text is starting this frame...
            if option3_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option3_text.frameNStart = frameN  # exact frame index
                option3_text.tStart = t  # local t and not account for scr refresh
                option3_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option3_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option3_text.started')
                # update status
                option3_text.status = STARTED
                option3_text.setAutoDraw(True)
            
            # if option3_text is active this frame...
            if option3_text.status == STARTED:
                # update params
                pass
            
            # if option3_text is stopping this frame...
            if option3_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option3_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option3_text.tStop = t  # not accounting for scr refresh
                    option3_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option3_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option3_text.stopped')
                    # update status
                    option3_text.status = FINISHED
                    option3_text.setAutoDraw(False)
            
            # *option4_text* updates
            
            # if option4_text is starting this frame...
            if option4_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option4_text.frameNStart = frameN  # exact frame index
                option4_text.tStart = t  # local t and not account for scr refresh
                option4_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option4_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option4_text.started')
                # update status
                option4_text.status = STARTED
                option4_text.setAutoDraw(True)
            
            # if option4_text is active this frame...
            if option4_text.status == STARTED:
                # update params
                pass
            
            # if option4_text is stopping this frame...
            if option4_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option4_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option4_text.tStop = t  # not accounting for scr refresh
                    option4_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option4_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option4_text.stopped')
                    # update status
                    option4_text.status = FINISHED
                    option4_text.setAutoDraw(False)
            
            # *selection_key_resp* updates
            waitOnFlip = False
            
            # if selection_key_resp is starting this frame...
            if selection_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                selection_key_resp.frameNStart = frameN  # exact frame index
                selection_key_resp.tStart = t  # local t and not account for scr refresh
                selection_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(selection_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'selection_key_resp.started')
                # update status
                selection_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(selection_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(selection_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if selection_key_resp is stopping this frame...
            if selection_key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > selection_key_resp.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    selection_key_resp.tStop = t  # not accounting for scr refresh
                    selection_key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    selection_key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'selection_key_resp.stopped')
                    # update status
                    selection_key_resp.status = FINISHED
                    selection_key_resp.status = FINISHED
            if selection_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = selection_key_resp.getKeys(keyList=['1','2','3'], ignoreKeys=["escape"], waitRelease=False)
                _selection_key_resp_allKeys.extend(theseKeys)
                if len(_selection_key_resp_allKeys):
                    selection_key_resp.keys = _selection_key_resp_allKeys[-1].name  # just the last key pressed
                    selection_key_resp.rt = _selection_key_resp_allKeys[-1].rt
                    selection_key_resp.duration = _selection_key_resp_allKeys[-1].duration
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
                whose_voice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in whose_voice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "whose_voice" ---
        for thisComponent in whose_voice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for whose_voice
        whose_voice.tStop = globalClock.getTime(format='float')
        whose_voice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('whose_voice.stopped', whose_voice.tStop)
        # check responses
        if selection_key_resp.keys in ['', [], None]:  # No response was made
            selection_key_resp.keys = None
        formal1_2_loop_trials.addData('selection_key_resp.keys',selection_key_resp.keys)
        if selection_key_resp.keys != None:  # we had a response
            formal1_2_loop_trials.addData('selection_key_resp.rt', selection_key_resp.rt)
            formal1_2_loop_trials.addData('selection_key_resp.duration', selection_key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if whose_voice.maxDurationReached:
            routineTimer.addTime(-whose_voice.maxDuration)
        elif whose_voice.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[feedback_display],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from feedback_code
        # feedback routine的Begin Routine部分
        # 获取被试的反应
        if selection_key_resp.keys:
            # 处理反应键
            if type(selection_key_resp.keys) is list:
                participant_response_key = selection_key_resp.keys[0]
            else:
                participant_response_key = selection_key_resp.keys
            
            # 处理反应时
            if type(selection_key_resp.rt) is list:
                participant_rt = selection_key_resp.rt[0]
            else:
                participant_rt = selection_key_resp.rt
            
            # 使用本次试验的映射将按键转换为对应的身份
            participant_response = thisExp.current_key_to_voice.get(participant_response_key, '未知')
        else:
            participant_response = '无反应'
            participant_rt = -1
            participant_response_key = '无反应'
        
        # 判断回答是否正确
        is_correct = (participant_response == thisExp.correctAnswer)
        
        # 设置反馈文本
        if participant_response == '无反应':
            feedback_text = "未检测到反应，请尽快回答！"
            feedback_color = 'red'
        elif is_correct:
            feedback_text = "回答正确！"
            feedback_color = 'green'
        else:
            # 获取正确答案对应的中文名称
            correct_voice_type = thisExp.correctAnswer
            if correct_voice_type in thisExp.roleName:
                correct_name = thisExp.roleName[correct_voice_type][0]  # 中文名
                # 获取被试选择的身份名称
                if participant_response in thisExp.roleName:
                    chosen_name = thisExp.roleName[participant_response][0]  # 中文名
                    
                    # 找出正确答案对应的按键
                    correct_key = None
                    for key, voice in thisExp.current_key_to_voice.items():
                        if voice == correct_voice_type:
                            correct_key = key
                            break
                    
                    feedback_text = f"错误！您选择了{participant_response_key}({chosen_name})\n正确答案是{correct_key}({correct_name})"
                else:
                    feedback_text = f"错误！正确答案是{correct_name}"
            else:
                feedback_text = f"错误！正确答案是{thisExp.correctAnswer}"
            feedback_color = 'red'
        
        # 更新反馈文本组件
        feedback_display.text = feedback_text
        feedback_display.color = feedback_color
        
        # 记录反应数据
        thisExp.addData('response_key', participant_response_key)  # 记录的按键
        thisExp.addData('response_identity', participant_response)  # 记录的身份
        thisExp.addData('rt', participant_rt)
        thisExp.addData('correct', 1 if is_correct else 0)
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # skip Routine feedback if its 'Skip if' condition is True
        feedback.skipped = continueRoutine and not (thisExp.show_attention_all[run_index] == False)
        continueRoutine = feedback.skipped
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
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
        
        # --- Run Routine "feedback" ---
        # if trial has changed, end Routine now
        if isinstance(formal1_2_loop_trials, data.TrialHandler2) and thisFormal1_2_loop_trial.thisN != formal1_2_loop_trials.thisTrial.thisN:
            continueRoutine = False
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_display* updates
            
            # if feedback_display is starting this frame...
            if feedback_display.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_display.frameNStart = frameN  # exact frame index
                feedback_display.tStart = t  # local t and not account for scr refresh
                feedback_display.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_display, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_display.started')
                # update status
                feedback_display.status = STARTED
                feedback_display.setAutoDraw(True)
            
            # if feedback_display is active this frame...
            if feedback_display.status == STARTED:
                # update params
                pass
            
            # if feedback_display is stopping this frame...
            if feedback_display.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_display.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_display.tStop = t  # not accounting for scr refresh
                    feedback_display.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_display.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_display.stopped')
                    # update status
                    feedback_display.status = FINISHED
                    feedback_display.setAutoDraw(False)
            
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
                feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback.maxDurationReached:
            routineTimer.addTime(-feedback.maxDuration)
        elif feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "content" ---
        # create an object to store info about Routine content
        content = data.Routine(
            name='content',
            components=[content_text],
        )
        content.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for content
        content.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        content.tStart = globalClock.getTime(format='float')
        content.status = STARTED
        thisExp.addData('content.started', content.tStart)
        content.maxDuration = None
        # skip Routine content if its 'Skip if' condition is True
        content.skipped = continueRoutine and not (thisExp.show_attention_all[run_index] == False)
        continueRoutine = content.skipped
        # keep track of which components have finished
        contentComponents = content.components
        for thisComponent in content.components:
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
        
        # --- Run Routine "content" ---
        # if trial has changed, end Routine now
        if isinstance(formal1_2_loop_trials, data.TrialHandler2) and thisFormal1_2_loop_trial.thisN != formal1_2_loop_trials.thisTrial.thisN:
            continueRoutine = False
        content.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *content_text* updates
            
            # if content_text is starting this frame...
            if content_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                content_text.frameNStart = frameN  # exact frame index
                content_text.tStart = t  # local t and not account for scr refresh
                content_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(content_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'content_text.started')
                # update status
                content_text.status = STARTED
                content_text.setAutoDraw(True)
            
            # if content_text is active this frame...
            if content_text.status == STARTED:
                # update params
                pass
            
            # if content_text is stopping this frame...
            if content_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > content_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    content_text.tStop = t  # not accounting for scr refresh
                    content_text.tStopRefresh = tThisFlipGlobal  # on global time
                    content_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'content_text.stopped')
                    # update status
                    content_text.status = FINISHED
                    content_text.setAutoDraw(False)
            
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
                content.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in content.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "content" ---
        for thisComponent in content.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for content
        content.tStop = globalClock.getTime(format='float')
        content.tStopRefresh = tThisFlipGlobal
        thisExp.addData('content.stopped', content.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if content.maxDurationReached:
            routineTimer.addTime(-content.maxDuration)
        elif content.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
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
        import random
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
        if isinstance(formal1_2_loop_trials, data.TrialHandler2) and thisFormal1_2_loop_trial.thisN != formal1_2_loop_trials.thisTrial.thisN:
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
        # the Routine "rest_within" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 104.0 repeats of 'formal1_2_loop_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "rest" ---
    # create an object to store info about Routine rest
    rest = data.Routine(
        name='rest',
        components=[rest_image, rest_key_resp],
    )
    rest.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for rest_key_resp
    rest_key_resp.keys = []
    rest_key_resp.rt = []
    _rest_key_resp_allKeys = []
    # store start times for rest
    rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    rest.tStart = globalClock.getTime(format='float')
    rest.status = STARTED
    thisExp.addData('rest.started', rest.tStart)
    rest.maxDuration = None
    # keep track of which components have finished
    restComponents = rest.components
    for thisComponent in rest.components:
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
    
    # --- Run Routine "rest" ---
    rest.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rest_image* updates
        
        # if rest_image is starting this frame...
        if rest_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rest_image.frameNStart = frameN  # exact frame index
            rest_image.tStart = t  # local t and not account for scr refresh
            rest_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rest_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rest_image.started')
            # update status
            rest_image.status = STARTED
            rest_image.setAutoDraw(True)
        
        # if rest_image is active this frame...
        if rest_image.status == STARTED:
            # update params
            pass
        
        # *rest_key_resp* updates
        waitOnFlip = False
        
        # if rest_key_resp is starting this frame...
        if rest_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rest_key_resp.frameNStart = frameN  # exact frame index
            rest_key_resp.tStart = t  # local t and not account for scr refresh
            rest_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rest_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rest_key_resp.started')
            # update status
            rest_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(rest_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(rest_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if rest_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = rest_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _rest_key_resp_allKeys.extend(theseKeys)
            if len(_rest_key_resp_allKeys):
                rest_key_resp.keys = _rest_key_resp_allKeys[-1].name  # just the last key pressed
                rest_key_resp.rt = _rest_key_resp_allKeys[-1].rt
                rest_key_resp.duration = _rest_key_resp_allKeys[-1].duration
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
            rest.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in rest.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "rest" ---
    for thisComponent in rest.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for rest
    rest.tStop = globalClock.getTime(format='float')
    rest.tStopRefresh = tThisFlipGlobal
    thisExp.addData('rest.stopped', rest.tStop)
    # check responses
    if rest_key_resp.keys in ['', [], None]:  # No response was made
        rest_key_resp.keys = None
    thisExp.addData('rest_key_resp.keys',rest_key_resp.keys)
    if rest_key_resp.keys != None:  # we had a response
        thisExp.addData('rest_key_resp.rt', rest_key_resp.rt)
        thisExp.addData('rest_key_resp.duration', rest_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "rest" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "run3_code_routine" ---
    # create an object to store info about Routine run3_code_routine
    run3_code_routine = data.Routine(
        name='run3_code_routine',
        components=[],
    )
    run3_code_routine.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for run3_code_routine
    run3_code_routine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    run3_code_routine.tStart = globalClock.getTime(format='float')
    run3_code_routine.status = STARTED
    thisExp.addData('run3_code_routine.started', run3_code_routine.tStart)
    run3_code_routine.maxDuration = None
    # keep track of which components have finished
    run3_code_routineComponents = run3_code_routine.components
    for thisComponent in run3_code_routine.components:
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
    
    # --- Run Routine "run3_code_routine" ---
    run3_code_routine.forceEnded = routineForceEnded = not continueRoutine
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
            run3_code_routine.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in run3_code_routine.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "run3_code_routine" ---
    for thisComponent in run3_code_routine.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for run3_code_routine
    run3_code_routine.tStop = globalClock.getTime(format='float')
    run3_code_routine.tStopRefresh = tThisFlipGlobal
    thisExp.addData('run3_code_routine.stopped', run3_code_routine.tStop)
    # Run 'End Routine' code from run3_code
    run_index = 2
    thisExp.nextEntry()
    # the Routine "run3_code_routine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    formal2_3_loop_trials = data.TrialHandler2(
        name='formal2_3_loop_trials',
        nReps=104.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(formal2_3_loop_trials)  # add the loop to the experiment
    thisFormal2_3_loop_trial = formal2_3_loop_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisFormal2_3_loop_trial.rgb)
    if thisFormal2_3_loop_trial != None:
        for paramName in thisFormal2_3_loop_trial:
            globals()[paramName] = thisFormal2_3_loop_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisFormal2_3_loop_trial in formal2_3_loop_trials:
        currentLoop = formal2_3_loop_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisFormal2_3_loop_trial.rgb)
        if thisFormal2_3_loop_trial != None:
            for paramName in thisFormal2_3_loop_trial:
                globals()[paramName] = thisFormal2_3_loop_trial[paramName]
        
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
        if isinstance(formal2_3_loop_trials, data.TrialHandler2) and thisFormal2_3_loop_trial.thisN != formal2_3_loop_trials.thisTrial.thisN:
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
        
        # --- Prepare to start Routine "formal_run3_trial" ---
        # create an object to store info about Routine formal_run3_trial
        formal_run3_trial = data.Routine(
            name='formal_run3_trial',
            components=[formal3_sound],
        )
        formal_run3_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from formal2_run3_trial_code
        # 在每个run的正式trial的Begin Routine中
        run_index = 2  # 根据当前run设置
        trial_index = thisExp.trial_counter[run_index]
        
        # 检查是否需要显示attention check
        thisExp.show_attention_all[run_index] = False
        if (thisExp.current_attention_index_all[run_index] < len(thisExp.attention_trials_all[run_index]) and 
            trial_index == thisExp.attention_trials_all[run_index][thisExp.current_attention_index_all[run_index]]):
            thisExp.show_attention_all[run_index] = True
            thisExp.current_attention_index_all[run_index] += 1
        
        current_trial = thisExp.all_trials[run_index][trial_index]
        currentSentence = current_trial['sentence_list']
        currentSentenceID = current_trial['sentenceID']
        currentCategory = current_trial['category']
        currentMidCategory = current_trial['mid_category']
        currentSubCategory = current_trial['sub_category']
        currentcondition = current_trial['condition']
        currentfilepath = current_trial['Stimulus_path']
        currentStimulusID = current_trial['StimulusID']
        
        # 更新计数器
        thisExp.trial_counter[run_index] += 1
        thisExp.correctAnswer = correct_answers.get(currentcondition, '未知')
        
        thisExp.addData('sentence_list', currentSentence)
        thisExp.addData('sentenceID', currentSentenceID)
        thisExp.addData('category', currentCategory)
        thisExp.addData('mid_category', currentMidCategory)
        thisExp.addData('sub_category', currentSubCategory)
        thisExp.addData('condition', currentcondition)
        thisExp.addData('Stimulus_path', currentfilepath)
        thisExp.addData('StimulusID', currentStimulusID)
        thisExp.addData('filepath', currentfilepath)
        thisExp.addData('correct_answer', thisExp.correctAnswer)
        thisExp.addData('has_attention_check', thisExp.show_attention_all[run_index])  # 记录是否有attention check
        formal3_sound.setSound(currentfilepath, secs=3, hamming=True)
        formal3_sound.setVolume(1.0, log=False)
        formal3_sound.seek(0)
        # store start times for formal_run3_trial
        formal_run3_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal_run3_trial.tStart = globalClock.getTime(format='float')
        formal_run3_trial.status = STARTED
        thisExp.addData('formal_run3_trial.started', formal_run3_trial.tStart)
        formal_run3_trial.maxDuration = None
        # keep track of which components have finished
        formal_run3_trialComponents = formal_run3_trial.components
        for thisComponent in formal_run3_trial.components:
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
        
        # --- Run Routine "formal_run3_trial" ---
        # if trial has changed, end Routine now
        if isinstance(formal2_3_loop_trials, data.TrialHandler2) and thisFormal2_3_loop_trial.thisN != formal2_3_loop_trials.thisTrial.thisN:
            continueRoutine = False
        formal_run3_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *formal3_sound* updates
            
            # if formal3_sound is starting this frame...
            if formal3_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal3_sound.frameNStart = frameN  # exact frame index
                formal3_sound.tStart = t  # local t and not account for scr refresh
                formal3_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('formal3_sound.started', tThisFlipGlobal)
                # update status
                formal3_sound.status = STARTED
                formal3_sound.play(when=win)  # sync with win flip
            
            # if formal3_sound is stopping this frame...
            if formal3_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > formal3_sound.tStartRefresh + 3-frameTolerance or formal3_sound.isFinished:
                    # keep track of stop time/frame for later
                    formal3_sound.tStop = t  # not accounting for scr refresh
                    formal3_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    formal3_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'formal3_sound.stopped')
                    # update status
                    formal3_sound.status = FINISHED
                    formal3_sound.stop()
            
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
                    playbackComponents=[formal3_sound]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                formal_run3_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal_run3_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal_run3_trial" ---
        for thisComponent in formal_run3_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal_run3_trial
        formal_run3_trial.tStop = globalClock.getTime(format='float')
        formal_run3_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal_run3_trial.stopped', formal_run3_trial.tStop)
        formal3_sound.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if formal_run3_trial.maxDurationReached:
            routineTimer.addTime(-formal_run3_trial.maxDuration)
        elif formal_run3_trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "whose_voice" ---
        # create an object to store info about Routine whose_voice
        whose_voice = data.Routine(
            name='whose_voice',
            components=[option1_text, option2_text, option3_text, option4_text, selection_key_resp],
        )
        whose_voice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from selection_formal1_code
        # selection_formal1_2 routine的Begin Routine部分
        # 随机分配三个身份到三个按键位置
        voice_types = ['familiar', 'celebrity', 'unfamiliar', 'lab']
        random.shuffle(voice_types)  # 随机打乱顺序
        
        # 创建本次试验的按键与身份映射
        key_to_voice = {
            '1': voice_types[0],
            '2': voice_types[1], 
            '3': voice_types[2],
            '4': voice_types[3],
        }
        
        # 存储本次试验的映射关系，用于后续反馈
        thisExp.current_key_to_voice = key_to_voice
        
        # 为每个身份随机选择显示中文名还是英文名
        name_display = {}  # 存储本次试验中每个身份显示的名字
        language_choice = {}  # 存储语言选择（用于数据记录）
        
        for voice_type in voice_types:
            # 随机选择中文或英文
            chosen_language = random.choice(['chinese', 'english'])
            language_choice[voice_type] = chosen_language
            
            # 根据选择的语言获取对应的名字
            if chosen_language == 'chinese':
                name_display[voice_type] = thisExp.roleName[voice_type][0]  # 中文名
            else:
                name_display[voice_type] = thisExp.roleName[voice_type][1]  # 英文名
        
        # 设置三个选项文本组件的内容
        # 使用随机分配的映射关系
        option1_text.text = f"1. {name_display[key_to_voice['1']]}"    # 选项1
        option2_text.text = f"2. {name_display[key_to_voice['2']]}"    # 选项2
        option3_text.text = f"3. {name_display[key_to_voice['3']]}"    # 选项3
        option4_text.text = f"4. {name_display[key_to_voice['4']]}"    # 选项3
        
        # 记录语言选择信息和按键映射到实验数据
        thisExp.addData('familiar_name_language', language_choice.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_language', language_choice.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_language', language_choice.get('unfamiliar', '未出现'))
        thisExp.addData('lab_name_language', language_choice.get('lab', '未出现'))
        thisExp.addData('familiar_name_displayed', name_display.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_displayed', name_display.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_displayed', name_display.get('unfamiliar', '未出现'))
        thisExp.addData('lab_name_displayed', name_display.get('lab', '未出现'))
        
        # 记录本次试验的按键映射
        thisExp.addData('key1_mapping', key_to_voice['1'])
        thisExp.addData('key2_mapping', key_to_voice['2'])
        thisExp.addData('key3_mapping', key_to_voice['3'])
        thisExp.addData('key4_mapping', key_to_voice['4'])
        # create starting attributes for selection_key_resp
        selection_key_resp.keys = []
        selection_key_resp.rt = []
        _selection_key_resp_allKeys = []
        # store start times for whose_voice
        whose_voice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        whose_voice.tStart = globalClock.getTime(format='float')
        whose_voice.status = STARTED
        thisExp.addData('whose_voice.started', whose_voice.tStart)
        whose_voice.maxDuration = None
        # skip Routine whose_voice if its 'Skip if' condition is True
        whose_voice.skipped = continueRoutine and not (thisExp.show_attention_all[run_index] == False)
        continueRoutine = whose_voice.skipped
        # keep track of which components have finished
        whose_voiceComponents = whose_voice.components
        for thisComponent in whose_voice.components:
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
        
        # --- Run Routine "whose_voice" ---
        # if trial has changed, end Routine now
        if isinstance(formal2_3_loop_trials, data.TrialHandler2) and thisFormal2_3_loop_trial.thisN != formal2_3_loop_trials.thisTrial.thisN:
            continueRoutine = False
        whose_voice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *option1_text* updates
            
            # if option1_text is starting this frame...
            if option1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option1_text.frameNStart = frameN  # exact frame index
                option1_text.tStart = t  # local t and not account for scr refresh
                option1_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option1_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option1_text.started')
                # update status
                option1_text.status = STARTED
                option1_text.setAutoDraw(True)
            
            # if option1_text is active this frame...
            if option1_text.status == STARTED:
                # update params
                pass
            
            # if option1_text is stopping this frame...
            if option1_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option1_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option1_text.tStop = t  # not accounting for scr refresh
                    option1_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option1_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option1_text.stopped')
                    # update status
                    option1_text.status = FINISHED
                    option1_text.setAutoDraw(False)
            
            # *option2_text* updates
            
            # if option2_text is starting this frame...
            if option2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option2_text.frameNStart = frameN  # exact frame index
                option2_text.tStart = t  # local t and not account for scr refresh
                option2_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option2_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option2_text.started')
                # update status
                option2_text.status = STARTED
                option2_text.setAutoDraw(True)
            
            # if option2_text is active this frame...
            if option2_text.status == STARTED:
                # update params
                pass
            
            # if option2_text is stopping this frame...
            if option2_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option2_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option2_text.tStop = t  # not accounting for scr refresh
                    option2_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option2_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option2_text.stopped')
                    # update status
                    option2_text.status = FINISHED
                    option2_text.setAutoDraw(False)
            
            # *option3_text* updates
            
            # if option3_text is starting this frame...
            if option3_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option3_text.frameNStart = frameN  # exact frame index
                option3_text.tStart = t  # local t and not account for scr refresh
                option3_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option3_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option3_text.started')
                # update status
                option3_text.status = STARTED
                option3_text.setAutoDraw(True)
            
            # if option3_text is active this frame...
            if option3_text.status == STARTED:
                # update params
                pass
            
            # if option3_text is stopping this frame...
            if option3_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option3_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option3_text.tStop = t  # not accounting for scr refresh
                    option3_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option3_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option3_text.stopped')
                    # update status
                    option3_text.status = FINISHED
                    option3_text.setAutoDraw(False)
            
            # *option4_text* updates
            
            # if option4_text is starting this frame...
            if option4_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option4_text.frameNStart = frameN  # exact frame index
                option4_text.tStart = t  # local t and not account for scr refresh
                option4_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option4_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option4_text.started')
                # update status
                option4_text.status = STARTED
                option4_text.setAutoDraw(True)
            
            # if option4_text is active this frame...
            if option4_text.status == STARTED:
                # update params
                pass
            
            # if option4_text is stopping this frame...
            if option4_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option4_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option4_text.tStop = t  # not accounting for scr refresh
                    option4_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option4_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option4_text.stopped')
                    # update status
                    option4_text.status = FINISHED
                    option4_text.setAutoDraw(False)
            
            # *selection_key_resp* updates
            waitOnFlip = False
            
            # if selection_key_resp is starting this frame...
            if selection_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                selection_key_resp.frameNStart = frameN  # exact frame index
                selection_key_resp.tStart = t  # local t and not account for scr refresh
                selection_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(selection_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'selection_key_resp.started')
                # update status
                selection_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(selection_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(selection_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if selection_key_resp is stopping this frame...
            if selection_key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > selection_key_resp.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    selection_key_resp.tStop = t  # not accounting for scr refresh
                    selection_key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    selection_key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'selection_key_resp.stopped')
                    # update status
                    selection_key_resp.status = FINISHED
                    selection_key_resp.status = FINISHED
            if selection_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = selection_key_resp.getKeys(keyList=['1','2','3'], ignoreKeys=["escape"], waitRelease=False)
                _selection_key_resp_allKeys.extend(theseKeys)
                if len(_selection_key_resp_allKeys):
                    selection_key_resp.keys = _selection_key_resp_allKeys[-1].name  # just the last key pressed
                    selection_key_resp.rt = _selection_key_resp_allKeys[-1].rt
                    selection_key_resp.duration = _selection_key_resp_allKeys[-1].duration
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
                whose_voice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in whose_voice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "whose_voice" ---
        for thisComponent in whose_voice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for whose_voice
        whose_voice.tStop = globalClock.getTime(format='float')
        whose_voice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('whose_voice.stopped', whose_voice.tStop)
        # check responses
        if selection_key_resp.keys in ['', [], None]:  # No response was made
            selection_key_resp.keys = None
        formal2_3_loop_trials.addData('selection_key_resp.keys',selection_key_resp.keys)
        if selection_key_resp.keys != None:  # we had a response
            formal2_3_loop_trials.addData('selection_key_resp.rt', selection_key_resp.rt)
            formal2_3_loop_trials.addData('selection_key_resp.duration', selection_key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if whose_voice.maxDurationReached:
            routineTimer.addTime(-whose_voice.maxDuration)
        elif whose_voice.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[feedback_display],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from feedback_code
        # feedback routine的Begin Routine部分
        # 获取被试的反应
        if selection_key_resp.keys:
            # 处理反应键
            if type(selection_key_resp.keys) is list:
                participant_response_key = selection_key_resp.keys[0]
            else:
                participant_response_key = selection_key_resp.keys
            
            # 处理反应时
            if type(selection_key_resp.rt) is list:
                participant_rt = selection_key_resp.rt[0]
            else:
                participant_rt = selection_key_resp.rt
            
            # 使用本次试验的映射将按键转换为对应的身份
            participant_response = thisExp.current_key_to_voice.get(participant_response_key, '未知')
        else:
            participant_response = '无反应'
            participant_rt = -1
            participant_response_key = '无反应'
        
        # 判断回答是否正确
        is_correct = (participant_response == thisExp.correctAnswer)
        
        # 设置反馈文本
        if participant_response == '无反应':
            feedback_text = "未检测到反应，请尽快回答！"
            feedback_color = 'red'
        elif is_correct:
            feedback_text = "回答正确！"
            feedback_color = 'green'
        else:
            # 获取正确答案对应的中文名称
            correct_voice_type = thisExp.correctAnswer
            if correct_voice_type in thisExp.roleName:
                correct_name = thisExp.roleName[correct_voice_type][0]  # 中文名
                # 获取被试选择的身份名称
                if participant_response in thisExp.roleName:
                    chosen_name = thisExp.roleName[participant_response][0]  # 中文名
                    
                    # 找出正确答案对应的按键
                    correct_key = None
                    for key, voice in thisExp.current_key_to_voice.items():
                        if voice == correct_voice_type:
                            correct_key = key
                            break
                    
                    feedback_text = f"错误！您选择了{participant_response_key}({chosen_name})\n正确答案是{correct_key}({correct_name})"
                else:
                    feedback_text = f"错误！正确答案是{correct_name}"
            else:
                feedback_text = f"错误！正确答案是{thisExp.correctAnswer}"
            feedback_color = 'red'
        
        # 更新反馈文本组件
        feedback_display.text = feedback_text
        feedback_display.color = feedback_color
        
        # 记录反应数据
        thisExp.addData('response_key', participant_response_key)  # 记录的按键
        thisExp.addData('response_identity', participant_response)  # 记录的身份
        thisExp.addData('rt', participant_rt)
        thisExp.addData('correct', 1 if is_correct else 0)
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # skip Routine feedback if its 'Skip if' condition is True
        feedback.skipped = continueRoutine and not (thisExp.show_attention_all[run_index] == False)
        continueRoutine = feedback.skipped
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
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
        
        # --- Run Routine "feedback" ---
        # if trial has changed, end Routine now
        if isinstance(formal2_3_loop_trials, data.TrialHandler2) and thisFormal2_3_loop_trial.thisN != formal2_3_loop_trials.thisTrial.thisN:
            continueRoutine = False
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_display* updates
            
            # if feedback_display is starting this frame...
            if feedback_display.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_display.frameNStart = frameN  # exact frame index
                feedback_display.tStart = t  # local t and not account for scr refresh
                feedback_display.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_display, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_display.started')
                # update status
                feedback_display.status = STARTED
                feedback_display.setAutoDraw(True)
            
            # if feedback_display is active this frame...
            if feedback_display.status == STARTED:
                # update params
                pass
            
            # if feedback_display is stopping this frame...
            if feedback_display.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_display.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_display.tStop = t  # not accounting for scr refresh
                    feedback_display.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_display.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_display.stopped')
                    # update status
                    feedback_display.status = FINISHED
                    feedback_display.setAutoDraw(False)
            
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
                feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback.maxDurationReached:
            routineTimer.addTime(-feedback.maxDuration)
        elif feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "content" ---
        # create an object to store info about Routine content
        content = data.Routine(
            name='content',
            components=[content_text],
        )
        content.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for content
        content.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        content.tStart = globalClock.getTime(format='float')
        content.status = STARTED
        thisExp.addData('content.started', content.tStart)
        content.maxDuration = None
        # skip Routine content if its 'Skip if' condition is True
        content.skipped = continueRoutine and not (thisExp.show_attention_all[run_index] == False)
        continueRoutine = content.skipped
        # keep track of which components have finished
        contentComponents = content.components
        for thisComponent in content.components:
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
        
        # --- Run Routine "content" ---
        # if trial has changed, end Routine now
        if isinstance(formal2_3_loop_trials, data.TrialHandler2) and thisFormal2_3_loop_trial.thisN != formal2_3_loop_trials.thisTrial.thisN:
            continueRoutine = False
        content.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *content_text* updates
            
            # if content_text is starting this frame...
            if content_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                content_text.frameNStart = frameN  # exact frame index
                content_text.tStart = t  # local t and not account for scr refresh
                content_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(content_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'content_text.started')
                # update status
                content_text.status = STARTED
                content_text.setAutoDraw(True)
            
            # if content_text is active this frame...
            if content_text.status == STARTED:
                # update params
                pass
            
            # if content_text is stopping this frame...
            if content_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > content_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    content_text.tStop = t  # not accounting for scr refresh
                    content_text.tStopRefresh = tThisFlipGlobal  # on global time
                    content_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'content_text.stopped')
                    # update status
                    content_text.status = FINISHED
                    content_text.setAutoDraw(False)
            
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
                content.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in content.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "content" ---
        for thisComponent in content.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for content
        content.tStop = globalClock.getTime(format='float')
        content.tStopRefresh = tThisFlipGlobal
        thisExp.addData('content.stopped', content.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if content.maxDurationReached:
            routineTimer.addTime(-content.maxDuration)
        elif content.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
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
        import random
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
        if isinstance(formal2_3_loop_trials, data.TrialHandler2) and thisFormal2_3_loop_trial.thisN != formal2_3_loop_trials.thisTrial.thisN:
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
        # the Routine "rest_within" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 104.0 repeats of 'formal2_3_loop_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "rest" ---
    # create an object to store info about Routine rest
    rest = data.Routine(
        name='rest',
        components=[rest_image, rest_key_resp],
    )
    rest.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for rest_key_resp
    rest_key_resp.keys = []
    rest_key_resp.rt = []
    _rest_key_resp_allKeys = []
    # store start times for rest
    rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    rest.tStart = globalClock.getTime(format='float')
    rest.status = STARTED
    thisExp.addData('rest.started', rest.tStart)
    rest.maxDuration = None
    # keep track of which components have finished
    restComponents = rest.components
    for thisComponent in rest.components:
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
    
    # --- Run Routine "rest" ---
    rest.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rest_image* updates
        
        # if rest_image is starting this frame...
        if rest_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rest_image.frameNStart = frameN  # exact frame index
            rest_image.tStart = t  # local t and not account for scr refresh
            rest_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rest_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rest_image.started')
            # update status
            rest_image.status = STARTED
            rest_image.setAutoDraw(True)
        
        # if rest_image is active this frame...
        if rest_image.status == STARTED:
            # update params
            pass
        
        # *rest_key_resp* updates
        waitOnFlip = False
        
        # if rest_key_resp is starting this frame...
        if rest_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rest_key_resp.frameNStart = frameN  # exact frame index
            rest_key_resp.tStart = t  # local t and not account for scr refresh
            rest_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rest_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rest_key_resp.started')
            # update status
            rest_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(rest_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(rest_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if rest_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = rest_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _rest_key_resp_allKeys.extend(theseKeys)
            if len(_rest_key_resp_allKeys):
                rest_key_resp.keys = _rest_key_resp_allKeys[-1].name  # just the last key pressed
                rest_key_resp.rt = _rest_key_resp_allKeys[-1].rt
                rest_key_resp.duration = _rest_key_resp_allKeys[-1].duration
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
            rest.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in rest.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "rest" ---
    for thisComponent in rest.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for rest
    rest.tStop = globalClock.getTime(format='float')
    rest.tStopRefresh = tThisFlipGlobal
    thisExp.addData('rest.stopped', rest.tStop)
    # check responses
    if rest_key_resp.keys in ['', [], None]:  # No response was made
        rest_key_resp.keys = None
    thisExp.addData('rest_key_resp.keys',rest_key_resp.keys)
    if rest_key_resp.keys != None:  # we had a response
        thisExp.addData('rest_key_resp.rt', rest_key_resp.rt)
        thisExp.addData('rest_key_resp.duration', rest_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "rest" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "run4_code_routine" ---
    # create an object to store info about Routine run4_code_routine
    run4_code_routine = data.Routine(
        name='run4_code_routine',
        components=[],
    )
    run4_code_routine.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for run4_code_routine
    run4_code_routine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    run4_code_routine.tStart = globalClock.getTime(format='float')
    run4_code_routine.status = STARTED
    thisExp.addData('run4_code_routine.started', run4_code_routine.tStart)
    run4_code_routine.maxDuration = None
    # keep track of which components have finished
    run4_code_routineComponents = run4_code_routine.components
    for thisComponent in run4_code_routine.components:
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
    
    # --- Run Routine "run4_code_routine" ---
    run4_code_routine.forceEnded = routineForceEnded = not continueRoutine
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
            run4_code_routine.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in run4_code_routine.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "run4_code_routine" ---
    for thisComponent in run4_code_routine.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for run4_code_routine
    run4_code_routine.tStop = globalClock.getTime(format='float')
    run4_code_routine.tStopRefresh = tThisFlipGlobal
    thisExp.addData('run4_code_routine.stopped', run4_code_routine.tStop)
    # Run 'End Routine' code from run4_code
    run_index = 3
    thisExp.nextEntry()
    # the Routine "run4_code_routine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    formal2_4_loop_trials = data.TrialHandler2(
        name='formal2_4_loop_trials',
        nReps=104.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(formal2_4_loop_trials)  # add the loop to the experiment
    thisFormal2_4_loop_trial = formal2_4_loop_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisFormal2_4_loop_trial.rgb)
    if thisFormal2_4_loop_trial != None:
        for paramName in thisFormal2_4_loop_trial:
            globals()[paramName] = thisFormal2_4_loop_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisFormal2_4_loop_trial in formal2_4_loop_trials:
        currentLoop = formal2_4_loop_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisFormal2_4_loop_trial.rgb)
        if thisFormal2_4_loop_trial != None:
            for paramName in thisFormal2_4_loop_trial:
                globals()[paramName] = thisFormal2_4_loop_trial[paramName]
        
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
        if isinstance(formal2_4_loop_trials, data.TrialHandler2) and thisFormal2_4_loop_trial.thisN != formal2_4_loop_trials.thisTrial.thisN:
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
        
        # --- Prepare to start Routine "formal_run4_trial" ---
        # create an object to store info about Routine formal_run4_trial
        formal_run4_trial = data.Routine(
            name='formal_run4_trial',
            components=[formal4_sound],
        )
        formal_run4_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from formal2_run4_trial_code
        # 在每个run的正式trial的Begin Routine中
        run_index = 3  # 根据当前run设置
        trial_index = thisExp.trial_counter[run_index]
        
        # 检查是否需要显示attention check
        thisExp.show_attention_all[run_index] = False
        if (thisExp.current_attention_index_all[run_index] < len(thisExp.attention_trials_all[run_index]) and 
            trial_index == thisExp.attention_trials_all[run_index][thisExp.current_attention_index_all[run_index]]):
            thisExp.show_attention_all[run_index] = True
            thisExp.current_attention_index_all[run_index] += 1
        
        current_trial = thisExp.all_trials[run_index][trial_index]
        currentSentence = current_trial['sentence_list']
        currentSentenceID = current_trial['sentenceID']
        currentCategory = current_trial['category']
        currentMidCategory = current_trial['mid_category']
        currentSubCategory = current_trial['sub_category']
        currentcondition = current_trial['condition']
        currentfilepath = current_trial['Stimulus_path']
        currentStimulusID = current_trial['StimulusID']
        
        # 更新计数器
        thisExp.trial_counter[run_index] += 1
        thisExp.correctAnswer = correct_answers.get(currentcondition, '未知')
        
        thisExp.addData('sentence_list', currentSentence)
        thisExp.addData('sentenceID', currentSentenceID)
        thisExp.addData('category', currentCategory)
        thisExp.addData('mid_category', currentMidCategory)
        thisExp.addData('sub_category', currentSubCategory)
        thisExp.addData('condition', currentcondition)
        thisExp.addData('Stimulus_path', currentfilepath)
        thisExp.addData('StimulusID', currentStimulusID)
        thisExp.addData('filepath', currentfilepath)
        thisExp.addData('correct_answer', thisExp.correctAnswer)
        thisExp.addData('has_attention_check', thisExp.show_attention_all[run_index])  # 记录是否有attention check
        formal4_sound.setSound(currentfilepath, secs=3, hamming=True)
        formal4_sound.setVolume(1.0, log=False)
        formal4_sound.seek(0)
        # store start times for formal_run4_trial
        formal_run4_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal_run4_trial.tStart = globalClock.getTime(format='float')
        formal_run4_trial.status = STARTED
        thisExp.addData('formal_run4_trial.started', formal_run4_trial.tStart)
        formal_run4_trial.maxDuration = None
        # keep track of which components have finished
        formal_run4_trialComponents = formal_run4_trial.components
        for thisComponent in formal_run4_trial.components:
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
        
        # --- Run Routine "formal_run4_trial" ---
        # if trial has changed, end Routine now
        if isinstance(formal2_4_loop_trials, data.TrialHandler2) and thisFormal2_4_loop_trial.thisN != formal2_4_loop_trials.thisTrial.thisN:
            continueRoutine = False
        formal_run4_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *formal4_sound* updates
            
            # if formal4_sound is starting this frame...
            if formal4_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                formal4_sound.frameNStart = frameN  # exact frame index
                formal4_sound.tStart = t  # local t and not account for scr refresh
                formal4_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('formal4_sound.started', tThisFlipGlobal)
                # update status
                formal4_sound.status = STARTED
                formal4_sound.play(when=win)  # sync with win flip
            
            # if formal4_sound is stopping this frame...
            if formal4_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > formal4_sound.tStartRefresh + 3-frameTolerance or formal4_sound.isFinished:
                    # keep track of stop time/frame for later
                    formal4_sound.tStop = t  # not accounting for scr refresh
                    formal4_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    formal4_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'formal4_sound.stopped')
                    # update status
                    formal4_sound.status = FINISHED
                    formal4_sound.stop()
            
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
                    playbackComponents=[formal4_sound]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                formal_run4_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal_run4_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal_run4_trial" ---
        for thisComponent in formal_run4_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal_run4_trial
        formal_run4_trial.tStop = globalClock.getTime(format='float')
        formal_run4_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal_run4_trial.stopped', formal_run4_trial.tStop)
        formal4_sound.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if formal_run4_trial.maxDurationReached:
            routineTimer.addTime(-formal_run4_trial.maxDuration)
        elif formal_run4_trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "whose_voice" ---
        # create an object to store info about Routine whose_voice
        whose_voice = data.Routine(
            name='whose_voice',
            components=[option1_text, option2_text, option3_text, option4_text, selection_key_resp],
        )
        whose_voice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from selection_formal1_code
        # selection_formal1_2 routine的Begin Routine部分
        # 随机分配三个身份到三个按键位置
        voice_types = ['familiar', 'celebrity', 'unfamiliar', 'lab']
        random.shuffle(voice_types)  # 随机打乱顺序
        
        # 创建本次试验的按键与身份映射
        key_to_voice = {
            '1': voice_types[0],
            '2': voice_types[1], 
            '3': voice_types[2],
            '4': voice_types[3],
        }
        
        # 存储本次试验的映射关系，用于后续反馈
        thisExp.current_key_to_voice = key_to_voice
        
        # 为每个身份随机选择显示中文名还是英文名
        name_display = {}  # 存储本次试验中每个身份显示的名字
        language_choice = {}  # 存储语言选择（用于数据记录）
        
        for voice_type in voice_types:
            # 随机选择中文或英文
            chosen_language = random.choice(['chinese', 'english'])
            language_choice[voice_type] = chosen_language
            
            # 根据选择的语言获取对应的名字
            if chosen_language == 'chinese':
                name_display[voice_type] = thisExp.roleName[voice_type][0]  # 中文名
            else:
                name_display[voice_type] = thisExp.roleName[voice_type][1]  # 英文名
        
        # 设置三个选项文本组件的内容
        # 使用随机分配的映射关系
        option1_text.text = f"1. {name_display[key_to_voice['1']]}"    # 选项1
        option2_text.text = f"2. {name_display[key_to_voice['2']]}"    # 选项2
        option3_text.text = f"3. {name_display[key_to_voice['3']]}"    # 选项3
        option4_text.text = f"4. {name_display[key_to_voice['4']]}"    # 选项3
        
        # 记录语言选择信息和按键映射到实验数据
        thisExp.addData('familiar_name_language', language_choice.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_language', language_choice.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_language', language_choice.get('unfamiliar', '未出现'))
        thisExp.addData('lab_name_language', language_choice.get('lab', '未出现'))
        thisExp.addData('familiar_name_displayed', name_display.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_displayed', name_display.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_displayed', name_display.get('unfamiliar', '未出现'))
        thisExp.addData('lab_name_displayed', name_display.get('lab', '未出现'))
        
        # 记录本次试验的按键映射
        thisExp.addData('key1_mapping', key_to_voice['1'])
        thisExp.addData('key2_mapping', key_to_voice['2'])
        thisExp.addData('key3_mapping', key_to_voice['3'])
        thisExp.addData('key4_mapping', key_to_voice['4'])
        # create starting attributes for selection_key_resp
        selection_key_resp.keys = []
        selection_key_resp.rt = []
        _selection_key_resp_allKeys = []
        # store start times for whose_voice
        whose_voice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        whose_voice.tStart = globalClock.getTime(format='float')
        whose_voice.status = STARTED
        thisExp.addData('whose_voice.started', whose_voice.tStart)
        whose_voice.maxDuration = None
        # skip Routine whose_voice if its 'Skip if' condition is True
        whose_voice.skipped = continueRoutine and not (thisExp.show_attention_all[run_index] == False)
        continueRoutine = whose_voice.skipped
        # keep track of which components have finished
        whose_voiceComponents = whose_voice.components
        for thisComponent in whose_voice.components:
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
        
        # --- Run Routine "whose_voice" ---
        # if trial has changed, end Routine now
        if isinstance(formal2_4_loop_trials, data.TrialHandler2) and thisFormal2_4_loop_trial.thisN != formal2_4_loop_trials.thisTrial.thisN:
            continueRoutine = False
        whose_voice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *option1_text* updates
            
            # if option1_text is starting this frame...
            if option1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option1_text.frameNStart = frameN  # exact frame index
                option1_text.tStart = t  # local t and not account for scr refresh
                option1_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option1_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option1_text.started')
                # update status
                option1_text.status = STARTED
                option1_text.setAutoDraw(True)
            
            # if option1_text is active this frame...
            if option1_text.status == STARTED:
                # update params
                pass
            
            # if option1_text is stopping this frame...
            if option1_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option1_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option1_text.tStop = t  # not accounting for scr refresh
                    option1_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option1_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option1_text.stopped')
                    # update status
                    option1_text.status = FINISHED
                    option1_text.setAutoDraw(False)
            
            # *option2_text* updates
            
            # if option2_text is starting this frame...
            if option2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option2_text.frameNStart = frameN  # exact frame index
                option2_text.tStart = t  # local t and not account for scr refresh
                option2_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option2_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option2_text.started')
                # update status
                option2_text.status = STARTED
                option2_text.setAutoDraw(True)
            
            # if option2_text is active this frame...
            if option2_text.status == STARTED:
                # update params
                pass
            
            # if option2_text is stopping this frame...
            if option2_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option2_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option2_text.tStop = t  # not accounting for scr refresh
                    option2_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option2_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option2_text.stopped')
                    # update status
                    option2_text.status = FINISHED
                    option2_text.setAutoDraw(False)
            
            # *option3_text* updates
            
            # if option3_text is starting this frame...
            if option3_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option3_text.frameNStart = frameN  # exact frame index
                option3_text.tStart = t  # local t and not account for scr refresh
                option3_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option3_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option3_text.started')
                # update status
                option3_text.status = STARTED
                option3_text.setAutoDraw(True)
            
            # if option3_text is active this frame...
            if option3_text.status == STARTED:
                # update params
                pass
            
            # if option3_text is stopping this frame...
            if option3_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option3_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option3_text.tStop = t  # not accounting for scr refresh
                    option3_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option3_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option3_text.stopped')
                    # update status
                    option3_text.status = FINISHED
                    option3_text.setAutoDraw(False)
            
            # *option4_text* updates
            
            # if option4_text is starting this frame...
            if option4_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option4_text.frameNStart = frameN  # exact frame index
                option4_text.tStart = t  # local t and not account for scr refresh
                option4_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option4_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option4_text.started')
                # update status
                option4_text.status = STARTED
                option4_text.setAutoDraw(True)
            
            # if option4_text is active this frame...
            if option4_text.status == STARTED:
                # update params
                pass
            
            # if option4_text is stopping this frame...
            if option4_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option4_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option4_text.tStop = t  # not accounting for scr refresh
                    option4_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option4_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option4_text.stopped')
                    # update status
                    option4_text.status = FINISHED
                    option4_text.setAutoDraw(False)
            
            # *selection_key_resp* updates
            waitOnFlip = False
            
            # if selection_key_resp is starting this frame...
            if selection_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                selection_key_resp.frameNStart = frameN  # exact frame index
                selection_key_resp.tStart = t  # local t and not account for scr refresh
                selection_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(selection_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'selection_key_resp.started')
                # update status
                selection_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(selection_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(selection_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if selection_key_resp is stopping this frame...
            if selection_key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > selection_key_resp.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    selection_key_resp.tStop = t  # not accounting for scr refresh
                    selection_key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    selection_key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'selection_key_resp.stopped')
                    # update status
                    selection_key_resp.status = FINISHED
                    selection_key_resp.status = FINISHED
            if selection_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = selection_key_resp.getKeys(keyList=['1','2','3'], ignoreKeys=["escape"], waitRelease=False)
                _selection_key_resp_allKeys.extend(theseKeys)
                if len(_selection_key_resp_allKeys):
                    selection_key_resp.keys = _selection_key_resp_allKeys[-1].name  # just the last key pressed
                    selection_key_resp.rt = _selection_key_resp_allKeys[-1].rt
                    selection_key_resp.duration = _selection_key_resp_allKeys[-1].duration
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
                whose_voice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in whose_voice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "whose_voice" ---
        for thisComponent in whose_voice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for whose_voice
        whose_voice.tStop = globalClock.getTime(format='float')
        whose_voice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('whose_voice.stopped', whose_voice.tStop)
        # check responses
        if selection_key_resp.keys in ['', [], None]:  # No response was made
            selection_key_resp.keys = None
        formal2_4_loop_trials.addData('selection_key_resp.keys',selection_key_resp.keys)
        if selection_key_resp.keys != None:  # we had a response
            formal2_4_loop_trials.addData('selection_key_resp.rt', selection_key_resp.rt)
            formal2_4_loop_trials.addData('selection_key_resp.duration', selection_key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if whose_voice.maxDurationReached:
            routineTimer.addTime(-whose_voice.maxDuration)
        elif whose_voice.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[feedback_display],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from feedback_code
        # feedback routine的Begin Routine部分
        # 获取被试的反应
        if selection_key_resp.keys:
            # 处理反应键
            if type(selection_key_resp.keys) is list:
                participant_response_key = selection_key_resp.keys[0]
            else:
                participant_response_key = selection_key_resp.keys
            
            # 处理反应时
            if type(selection_key_resp.rt) is list:
                participant_rt = selection_key_resp.rt[0]
            else:
                participant_rt = selection_key_resp.rt
            
            # 使用本次试验的映射将按键转换为对应的身份
            participant_response = thisExp.current_key_to_voice.get(participant_response_key, '未知')
        else:
            participant_response = '无反应'
            participant_rt = -1
            participant_response_key = '无反应'
        
        # 判断回答是否正确
        is_correct = (participant_response == thisExp.correctAnswer)
        
        # 设置反馈文本
        if participant_response == '无反应':
            feedback_text = "未检测到反应，请尽快回答！"
            feedback_color = 'red'
        elif is_correct:
            feedback_text = "回答正确！"
            feedback_color = 'green'
        else:
            # 获取正确答案对应的中文名称
            correct_voice_type = thisExp.correctAnswer
            if correct_voice_type in thisExp.roleName:
                correct_name = thisExp.roleName[correct_voice_type][0]  # 中文名
                # 获取被试选择的身份名称
                if participant_response in thisExp.roleName:
                    chosen_name = thisExp.roleName[participant_response][0]  # 中文名
                    
                    # 找出正确答案对应的按键
                    correct_key = None
                    for key, voice in thisExp.current_key_to_voice.items():
                        if voice == correct_voice_type:
                            correct_key = key
                            break
                    
                    feedback_text = f"错误！您选择了{participant_response_key}({chosen_name})\n正确答案是{correct_key}({correct_name})"
                else:
                    feedback_text = f"错误！正确答案是{correct_name}"
            else:
                feedback_text = f"错误！正确答案是{thisExp.correctAnswer}"
            feedback_color = 'red'
        
        # 更新反馈文本组件
        feedback_display.text = feedback_text
        feedback_display.color = feedback_color
        
        # 记录反应数据
        thisExp.addData('response_key', participant_response_key)  # 记录的按键
        thisExp.addData('response_identity', participant_response)  # 记录的身份
        thisExp.addData('rt', participant_rt)
        thisExp.addData('correct', 1 if is_correct else 0)
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # skip Routine feedback if its 'Skip if' condition is True
        feedback.skipped = continueRoutine and not (thisExp.show_attention_all[run_index] == False)
        continueRoutine = feedback.skipped
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
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
        
        # --- Run Routine "feedback" ---
        # if trial has changed, end Routine now
        if isinstance(formal2_4_loop_trials, data.TrialHandler2) and thisFormal2_4_loop_trial.thisN != formal2_4_loop_trials.thisTrial.thisN:
            continueRoutine = False
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_display* updates
            
            # if feedback_display is starting this frame...
            if feedback_display.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_display.frameNStart = frameN  # exact frame index
                feedback_display.tStart = t  # local t and not account for scr refresh
                feedback_display.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_display, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_display.started')
                # update status
                feedback_display.status = STARTED
                feedback_display.setAutoDraw(True)
            
            # if feedback_display is active this frame...
            if feedback_display.status == STARTED:
                # update params
                pass
            
            # if feedback_display is stopping this frame...
            if feedback_display.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_display.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_display.tStop = t  # not accounting for scr refresh
                    feedback_display.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_display.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_display.stopped')
                    # update status
                    feedback_display.status = FINISHED
                    feedback_display.setAutoDraw(False)
            
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
                feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback.maxDurationReached:
            routineTimer.addTime(-feedback.maxDuration)
        elif feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "content" ---
        # create an object to store info about Routine content
        content = data.Routine(
            name='content',
            components=[content_text],
        )
        content.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for content
        content.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        content.tStart = globalClock.getTime(format='float')
        content.status = STARTED
        thisExp.addData('content.started', content.tStart)
        content.maxDuration = None
        # skip Routine content if its 'Skip if' condition is True
        content.skipped = continueRoutine and not (thisExp.show_attention_all[run_index] == False)
        continueRoutine = content.skipped
        # keep track of which components have finished
        contentComponents = content.components
        for thisComponent in content.components:
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
        
        # --- Run Routine "content" ---
        # if trial has changed, end Routine now
        if isinstance(formal2_4_loop_trials, data.TrialHandler2) and thisFormal2_4_loop_trial.thisN != formal2_4_loop_trials.thisTrial.thisN:
            continueRoutine = False
        content.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *content_text* updates
            
            # if content_text is starting this frame...
            if content_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                content_text.frameNStart = frameN  # exact frame index
                content_text.tStart = t  # local t and not account for scr refresh
                content_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(content_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'content_text.started')
                # update status
                content_text.status = STARTED
                content_text.setAutoDraw(True)
            
            # if content_text is active this frame...
            if content_text.status == STARTED:
                # update params
                pass
            
            # if content_text is stopping this frame...
            if content_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > content_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    content_text.tStop = t  # not accounting for scr refresh
                    content_text.tStopRefresh = tThisFlipGlobal  # on global time
                    content_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'content_text.stopped')
                    # update status
                    content_text.status = FINISHED
                    content_text.setAutoDraw(False)
            
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
                content.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in content.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "content" ---
        for thisComponent in content.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for content
        content.tStop = globalClock.getTime(format='float')
        content.tStopRefresh = tThisFlipGlobal
        thisExp.addData('content.stopped', content.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if content.maxDurationReached:
            routineTimer.addTime(-content.maxDuration)
        elif content.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
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
        import random
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
        if isinstance(formal2_4_loop_trials, data.TrialHandler2) and thisFormal2_4_loop_trial.thisN != formal2_4_loop_trials.thisTrial.thisN:
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
        # the Routine "rest_within" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 104.0 repeats of 'formal2_4_loop_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[end_image],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
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
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_image* updates
        
        # if end_image is starting this frame...
        if end_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_image.frameNStart = frameN  # exact frame index
            end_image.tStart = t  # local t and not account for scr refresh
            end_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_image.started')
            # update status
            end_image.status = STARTED
            end_image.setAutoDraw(True)
        
        # if end_image is active this frame...
        if end_image.status == STARTED:
            # update params
            pass
        
        # if end_image is stopping this frame...
        if end_image.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_image.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                end_image.tStop = t  # not accounting for scr refresh
                end_image.tStopRefresh = tThisFlipGlobal  # on global time
                end_image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_image.stopped')
                # update status
                end_image.status = FINISHED
                end_image.setAutoDraw(False)
        
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
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end.maxDurationReached:
        routineTimer.addTime(-end.maxDuration)
    elif end.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
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
