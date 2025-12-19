#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on 十一月 20, 2025, at 15:09
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
        originPath='E:\\lc\\Program_psychopy\\Voice_identity_categorization\\Sub02\\Voice_indentity_task_lastrun.py',
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
    if deviceManager.getDevice('instr_practice_key_resp') is None:
        # initialise instr_practice_key_resp
        instr_practice_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_practice_key_resp',
        )
    # create speaker 'lab_sound'
    deviceManager.addDevice(
        deviceName='lab_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_resp_lab_sound') is None:
        # initialise key_resp_lab_sound
        key_resp_lab_sound = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_lab_sound',
        )
    # create speaker 'sound_practice'
    deviceManager.addDevice(
        deviceName='sound_practice',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('selection_practice_key_resp') is None:
        # initialise selection_practice_key_resp
        selection_practice_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='selection_practice_key_resp',
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
    if deviceManager.getDevice('instr_formal2_key_resp') is None:
        # initialise instr_formal2_key_resp
        instr_formal2_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_formal2_key_resp',
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
    
    stimulus_file = os.path.join(os.getcwd(), 'stimulus', 'Stimulus.xlsx')
    roleName_file = os.path.join(os.getcwd(), 'stimulus', 'RoleName.xlsx')
    
    try:
        stimulus_df = pd.read_excel(stimulus_file)
        rolename_df = pd.read_excel(roleName_file)
        print(f"成功读取刺激材料文件，共{len(stimulus_df)}个刺激")
    except Exception as e:
        print(f"读取刺激材料文件失败: {e}")
        # 创建示例数据（仅用于测试，实际实验时应确保文件存在）
        stimulus_df = pd.DataFrame({
            'sentenceID': range(1, 121),
            'sentence_list': [f'句子_{i}' for i in range(1, 121)]
        })
        
    roleName = {row['Role']: [row['Name'], row['EnglishName']] for _, row in rolename_df.iterrows()}
    
    # 随机分配声音条件函数
    def assign_voice_conditions(df):
        """为刺激材料随机分配声音条件"""
        # 复制数据框
        df_copy = df.copy()
        
        # 确保sentenceID是整数
        df_copy['sentenceID'] = df_copy['sentenceID'].astype(int)
        
        # 随机打乱刺激顺序
        shuffled_indices = df_copy.index.tolist()
        random.shuffle(shuffled_indices)
        df_shuffled = df_copy.loc[shuffled_indices].reset_index(drop=True)
        
        # 分配声音条件（每个条件平均分配）
        n_stimuli = len(df_shuffled)
        n_per_condition = n_stimuli // 3
        
        # 计算每个条件的数量
        familiar_count = n_per_condition
        unfamiliar_count = n_per_condition
        celebrity_count = n_stimuli - (n_per_condition * 2)
        
        voice_conditions = (['familiar'] * familiar_count + 
                           ['unfamiliar'] * unfamiliar_count + 
                           ['celebrity'] * celebrity_count)
        
        df_shuffled['voice'] = voice_conditions
        
        # 重新按sentenceID排序（可选）
        df_assigned = df_shuffled.sort_values('sentenceID').reset_index(drop=True)
        
        # 添加文件路径
        df_assigned['filepath'] = df_assigned.apply(
            lambda row: f'stimulus/{row["voice"]}/Identity_Stimulus_{row["sentenceID"]:03d}.wav', 
            axis=1
        )
        
        return df_assigned
    
    # 将120个刺激随机分成两组
    all_stimuli = stimulus_df.copy()
    all_indices = list(all_stimuli.index)
    random.shuffle(all_indices)  # 随机打乱
    
    # 分成两组，每组60个
    group1_indices = all_indices[:30]
    group2_indices = all_indices[30:60]
    group3_indices = all_indices[60:90]
    group4_indices = all_indices[90:]
    
    group1_stimuli = all_stimuli.loc[group1_indices].reset_index(drop=True)
    group2_stimuli = all_stimuli.loc[group2_indices].reset_index(drop=True)
    group3_stimuli = all_stimuli.loc[group3_indices].reset_index(drop=True)
    group4_stimuli = all_stimuli.loc[group4_indices].reset_index(drop=True)
    
    print(f"第一组刺激数量: {len(group1_stimuli)}")
    print(f"第二组刺激数量: {len(group2_stimuli)}")
    print(f"第三组刺激数量: {len(group3_stimuli)}")
    print(f"第四组刺激数量: {len(group4_stimuli)}")
    
    # 为每组分配声音条件
    Assigned_Stimulus_Group1 = assign_voice_conditions(group1_stimuli)
    Assigned_Stimulus_Group2 = assign_voice_conditions(group2_stimuli)
    Assigned_Stimulus_Group3 = assign_voice_conditions(group3_stimuli)
    Assigned_Stimulus_Group4 = assign_voice_conditions(group4_stimuli)
    
    # 将分配结果转换为列表形式，便于PsychoPy使用
    trials_list_formal1 = []
    for idx, row in Assigned_Stimulus_Group1.iterrows():
        trial = {
            'sentenceID': row['sentenceID'],
            'sentence_list': row['sentence_list'],
            'voice': row['voice'],
            'filepath': row['filepath'],
            'group': 1  # 标记属于第一组
        }
        trials_list_formal1.append(trial)
    
    trials_list_formal2 = []
    for idx, row in Assigned_Stimulus_Group2.iterrows():
        trial = {
            'sentenceID': row['sentenceID'],
            'sentence_list': row['sentence_list'],
            'voice': row['voice'],
            'filepath': row['filepath'],
            'group': 2  # 标记属于第二组
        }
        trials_list_formal2.append(trial)
    
    trials_list_formal3 = []
    for idx, row in Assigned_Stimulus_Group3.iterrows():
        trial = {
            'sentenceID': row['sentenceID'],
            'sentence_list': row['sentence_list'],
            'voice': row['voice'],
            'filepath': row['filepath'],
            'group': 3  # 标记属于第二组
        }
        trials_list_formal3.append(trial)
    
    trials_list_formal4 = []
    for idx, row in Assigned_Stimulus_Group4.iterrows():
        trial = {
            'sentenceID': row['sentenceID'],
            'sentence_list': row['sentence_list'],
            'voice': row['voice'],
            'filepath': row['filepath'],
            'group': 4  # 标记属于第二组
        }
        trials_list_formal4.append(trial)
    # 随机化每组内的试验顺序
    random.shuffle(trials_list_formal1)
    random.shuffle(trials_list_formal2)
    random.shuffle(trials_list_formal3)
    random.shuffle(trials_list_formal4)
    
    print(f"第一组刺激随机化: {len(trials_list_formal1)}")
    print(trials_list_formal1)
    print(f"第二组刺激随机化: {len(trials_list_formal2)}")
    print(trials_list_formal2)
    thisExp.trials_list_formal1 = trials_list_formal1
    thisExp.trials_list_formal2 = trials_list_formal2
    thisExp.trials_list_formal3 = trials_list_formal3
    thisExp.trials_list_formal4 = trials_list_formal4
    
    print(f"身份姓名: {len(roleName)}")
    thisExp.roleName = roleName
    print(roleName)
    thisExp.trial_counter1 = 0
    thisExp.trial_counter2 = 0
    thisExp.trial_counter3 = 0
    thisExp.trial_counter4 = 0
    # Run 'Begin Experiment' code from practice_assign_code
    # 练习刺激的路径
    practice_path = os.path.join('stimulus', 'practice')
    
    # 获取练习文件夹中的所有音频文件
    practice_files = []
    for file in os.listdir(practice_path):
        if file.endswith(('.mp3', '.wav', '.m4a', '.ogg')):  # 支持多种音频格式
            practice_files.append(file)
    
    # 按文件名排序以确保顺序一致
    practice_files.sort()
    
    # 检查文件数量
    if len(practice_files) != 9:
        print(f"警告: 练习文件夹中应该有9个文件，但找到了{len(practice_files)}个文件")
    
    # 创建练习试验列表
    trials_list_practice = []
    
    # 分配声音条件
    for i, filename in enumerate(practice_files):
        # 根据索引分配声音条件
        if i < 3:  # 前三个文件是unfamiliar
            voice = 'unfamiliar'
        elif i < 6:  # 中间三个文件是celebrity
            voice = 'celebrity'
        else:  # 最后三个文件是familiar
            voice = 'familiar'
        
        # 创建完整的文件路径
        filepath = os.path.join(practice_path, filename)
        
        # 创建试验字典
        trial = {
            'voice': voice,
            'filepath': filepath
        }
        
        trials_list_practice.append(trial)
    
    # 将练习试验列表存储到实验变量中
    random.shuffle(trials_list_practice)
    thisExp.trials_list_practice = trials_list_practice
    thisExp.practicetrial_counter = 0
    # 打印调试信息
    print(f"成功创建练习试验列表，共{len(trials_list_practice)}个试验")
    for i, trial in enumerate(trials_list_practice):
        print(f"试验{i+1}: 声音条件={trial['voice']}, 文件={trial['filepath']}")
    
    # --- Initialize components for Routine "instruction" ---
    instruction_image = visual.ImageStim(
        win=win,
        name='instruction_image', 
        image='image/instr1.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.78, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    instruction_key_resp = keyboard.Keyboard(deviceName='instruction_key_resp')
    
    # --- Initialize components for Routine "instr_practice" ---
    instr_practice_image = visual.ImageStim(
        win=win,
        name='instr_practice_image', 
        image='image/instr2.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.78, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    instr_practice_key_resp = keyboard.Keyboard(deviceName='instr_practice_key_resp')
    
    # --- Initialize components for Routine "lab_voice" ---
    lab_voice_text = visual.TextStim(win=win, name='lab_voice_text',
        text='请先熟悉一段声音。\n吴婷婷\nWuTingTing',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    lab_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='lab_sound',    name='lab_sound'
    )
    lab_sound.setVolume(1.0)
    key_resp_lab_sound = keyboard.Keyboard(deviceName='key_resp_lab_sound')
    
    # --- Initialize components for Routine "fixation" ---
    text = visual.TextStim(win=win, name='text',
        text='＋',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "practice_trial" ---
    # Run 'Begin Experiment' code from practice_trial_code
    currentpracticefilepath = []
    sound_practice = sound.Sound(
        'A', 
        secs=2, 
        stereo=True, 
        hamming=True, 
        speaker='sound_practice',    name='sound_practice'
    )
    sound_practice.setVolume(1.0)
    
    # --- Initialize components for Routine "selection_practice" ---
    option1_practice_text = visual.TextStim(win=win, name='option1_practice_text',
        text=None,
        font='Arial',
        pos=(-0.65, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    option2_practice_text = visual.TextStim(win=win, name='option2_practice_text',
        text=None,
        font='Arial',
        pos=(-0.25, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    option3_practice_text = visual.TextStim(win=win, name='option3_practice_text',
        text=None,
        font='Arial',
        pos=(0.25, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    option4_practice_text = visual.TextStim(win=win, name='option4_practice_text',
        text=None,
        font='Arial',
        pos=(0.65, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    selection_practice_key_resp = keyboard.Keyboard(deviceName='selection_practice_key_resp')
    
    # --- Initialize components for Routine "feedback_practice" ---
    feedback_practice_display = visual.TextStim(win=win, name='feedback_practice_display',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "instr_formal1" ---
    instr_formal1_image = visual.ImageStim(
        win=win,
        name='instr_formal1_image', 
        image='image/instr_formal1.jpg', mask=None, anchor='center',
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
    
    # --- Initialize components for Routine "formal1_run1_trial" ---
    # Run 'Begin Experiment' code from formal_1_trial_code
    currentfilepath = []
    formal_1_sound = sound.Sound(
        'A', 
        secs=2, 
        stereo=True, 
        hamming=True, 
        speaker='formal_1_sound',    name='formal_1_sound'
    )
    formal_1_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "selection_formal" ---
    # Run 'Begin Experiment' code from selection_formal1_code
    # 在selection_formal1_2 routine的Begin Experiment标签中添加以下代码
    import random
    
    # 定义选项位置映射（固定位置）
    option_positions = {
        '1': (-0.3, 0),  # 选项1位置（左）
        '2': (0, 0),     # 选项2位置（中）
        '3': (0.3, 0),    # 选项3位置（右）
        '4': (0.3, 0)    # 选项3位置（右）
    }
    
    
    # 定义正确答案映射 - 对应角色名称
    correct_answers = {
        'familiar': 'familiar',
        'unfamiliar': 'unfamiliar', 
        'celebrity': 'celebrity',
        'someoneElse': 'someoneElse'
    }
    option1_text = visual.TextStim(win=win, name='option1_text',
        text=None,
        font='Arial',
        pos=(-0.65, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    option2_text = visual.TextStim(win=win, name='option2_text',
        text=None,
        font='Arial',
        pos=(-0.25, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    option3_text = visual.TextStim(win=win, name='option3_text',
        text=None,
        font='Arial',
        pos=(0.25, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    option4_text = visual.TextStim(win=win, name='option4_text',
        text=None,
        font='Arial',
        pos=(0.65, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
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
    
    # --- Initialize components for Routine "fixation" ---
    text = visual.TextStim(win=win, name='text',
        text='＋',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "formal1_run2_trial" ---
    formal1_run2_sound = sound.Sound(
        'A', 
        secs=2, 
        stereo=True, 
        hamming=True, 
        speaker='formal1_run2_sound',    name='formal1_run2_sound'
    )
    formal1_run2_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "selection_formal" ---
    # Run 'Begin Experiment' code from selection_formal1_code
    # 在selection_formal1_2 routine的Begin Experiment标签中添加以下代码
    import random
    
    # 定义选项位置映射（固定位置）
    option_positions = {
        '1': (-0.3, 0),  # 选项1位置（左）
        '2': (0, 0),     # 选项2位置（中）
        '3': (0.3, 0),    # 选项3位置（右）
        '4': (0.3, 0)    # 选项3位置（右）
    }
    
    
    # 定义正确答案映射 - 对应角色名称
    correct_answers = {
        'familiar': 'familiar',
        'unfamiliar': 'unfamiliar', 
        'celebrity': 'celebrity',
        'someoneElse': 'someoneElse'
    }
    option1_text = visual.TextStim(win=win, name='option1_text',
        text=None,
        font='Arial',
        pos=(-0.65, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    option2_text = visual.TextStim(win=win, name='option2_text',
        text=None,
        font='Arial',
        pos=(-0.25, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    option3_text = visual.TextStim(win=win, name='option3_text',
        text=None,
        font='Arial',
        pos=(0.25, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    option4_text = visual.TextStim(win=win, name='option4_text',
        text=None,
        font='Arial',
        pos=(0.65, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
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
    
    # --- Initialize components for Routine "instr_nofeedback" ---
    instr_no_feedback_image = visual.ImageStim(
        win=win,
        name='instr_no_feedback_image', 
        image='image/instr_formal2.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1.78, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    instr_formal2_key_resp = keyboard.Keyboard(deviceName='instr_formal2_key_resp')
    
    # --- Initialize components for Routine "fixation" ---
    text = visual.TextStim(win=win, name='text',
        text='＋',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "formal2_run3_trial" ---
    formal3_sound = sound.Sound(
        'A', 
        secs=2, 
        stereo=True, 
        hamming=True, 
        speaker='formal3_sound',    name='formal3_sound'
    )
    formal3_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "selection_formal" ---
    # Run 'Begin Experiment' code from selection_formal1_code
    # 在selection_formal1_2 routine的Begin Experiment标签中添加以下代码
    import random
    
    # 定义选项位置映射（固定位置）
    option_positions = {
        '1': (-0.3, 0),  # 选项1位置（左）
        '2': (0, 0),     # 选项2位置（中）
        '3': (0.3, 0),    # 选项3位置（右）
        '4': (0.3, 0)    # 选项3位置（右）
    }
    
    
    # 定义正确答案映射 - 对应角色名称
    correct_answers = {
        'familiar': 'familiar',
        'unfamiliar': 'unfamiliar', 
        'celebrity': 'celebrity',
        'someoneElse': 'someoneElse'
    }
    option1_text = visual.TextStim(win=win, name='option1_text',
        text=None,
        font='Arial',
        pos=(-0.65, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    option2_text = visual.TextStim(win=win, name='option2_text',
        text=None,
        font='Arial',
        pos=(-0.25, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    option3_text = visual.TextStim(win=win, name='option3_text',
        text=None,
        font='Arial',
        pos=(0.25, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    option4_text = visual.TextStim(win=win, name='option4_text',
        text=None,
        font='Arial',
        pos=(0.65, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    selection_key_resp = keyboard.Keyboard(deviceName='selection_key_resp')
    
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
    
    # --- Initialize components for Routine "fixation" ---
    text = visual.TextStim(win=win, name='text',
        text='＋',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "formal2_run4_trial" ---
    formal4_sound = sound.Sound(
        'A', 
        secs=2, 
        stereo=True, 
        hamming=True, 
        speaker='formal4_sound',    name='formal4_sound'
    )
    formal4_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "selection_formal" ---
    # Run 'Begin Experiment' code from selection_formal1_code
    # 在selection_formal1_2 routine的Begin Experiment标签中添加以下代码
    import random
    
    # 定义选项位置映射（固定位置）
    option_positions = {
        '1': (-0.3, 0),  # 选项1位置（左）
        '2': (0, 0),     # 选项2位置（中）
        '3': (0.3, 0),    # 选项3位置（右）
        '4': (0.3, 0)    # 选项3位置（右）
    }
    
    
    # 定义正确答案映射 - 对应角色名称
    correct_answers = {
        'familiar': 'familiar',
        'unfamiliar': 'unfamiliar', 
        'celebrity': 'celebrity',
        'someoneElse': 'someoneElse'
    }
    option1_text = visual.TextStim(win=win, name='option1_text',
        text=None,
        font='Arial',
        pos=(-0.65, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    option2_text = visual.TextStim(win=win, name='option2_text',
        text=None,
        font='Arial',
        pos=(-0.25, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    option3_text = visual.TextStim(win=win, name='option3_text',
        text=None,
        font='Arial',
        pos=(0.25, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    option4_text = visual.TextStim(win=win, name='option4_text',
        text=None,
        font='Arial',
        pos=(0.65, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    selection_key_resp = keyboard.Keyboard(deviceName='selection_key_resp')
    
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
    
    # --- Prepare to start Routine "instr_practice" ---
    # create an object to store info about Routine instr_practice
    instr_practice = data.Routine(
        name='instr_practice',
        components=[instr_practice_image, instr_practice_key_resp],
    )
    instr_practice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_practice_key_resp
    instr_practice_key_resp.keys = []
    instr_practice_key_resp.rt = []
    _instr_practice_key_resp_allKeys = []
    # store start times for instr_practice
    instr_practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr_practice.tStart = globalClock.getTime(format='float')
    instr_practice.status = STARTED
    thisExp.addData('instr_practice.started', instr_practice.tStart)
    instr_practice.maxDuration = None
    # keep track of which components have finished
    instr_practiceComponents = instr_practice.components
    for thisComponent in instr_practice.components:
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
    
    # --- Run Routine "instr_practice" ---
    instr_practice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr_practice_image* updates
        
        # if instr_practice_image is starting this frame...
        if instr_practice_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_practice_image.frameNStart = frameN  # exact frame index
            instr_practice_image.tStart = t  # local t and not account for scr refresh
            instr_practice_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_practice_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_practice_image.started')
            # update status
            instr_practice_image.status = STARTED
            instr_practice_image.setAutoDraw(True)
        
        # if instr_practice_image is active this frame...
        if instr_practice_image.status == STARTED:
            # update params
            pass
        
        # *instr_practice_key_resp* updates
        waitOnFlip = False
        
        # if instr_practice_key_resp is starting this frame...
        if instr_practice_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_practice_key_resp.frameNStart = frameN  # exact frame index
            instr_practice_key_resp.tStart = t  # local t and not account for scr refresh
            instr_practice_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_practice_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_practice_key_resp.started')
            # update status
            instr_practice_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_practice_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_practice_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_practice_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr_practice_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_practice_key_resp_allKeys.extend(theseKeys)
            if len(_instr_practice_key_resp_allKeys):
                instr_practice_key_resp.keys = _instr_practice_key_resp_allKeys[-1].name  # just the last key pressed
                instr_practice_key_resp.rt = _instr_practice_key_resp_allKeys[-1].rt
                instr_practice_key_resp.duration = _instr_practice_key_resp_allKeys[-1].duration
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
            instr_practice.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_practice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr_practice" ---
    for thisComponent in instr_practice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr_practice
    instr_practice.tStop = globalClock.getTime(format='float')
    instr_practice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr_practice.stopped', instr_practice.tStop)
    # check responses
    if instr_practice_key_resp.keys in ['', [], None]:  # No response was made
        instr_practice_key_resp.keys = None
    thisExp.addData('instr_practice_key_resp.keys',instr_practice_key_resp.keys)
    if instr_practice_key_resp.keys != None:  # we had a response
        thisExp.addData('instr_practice_key_resp.rt', instr_practice_key_resp.rt)
        thisExp.addData('instr_practice_key_resp.duration', instr_practice_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr_practice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "lab_voice" ---
    # create an object to store info about Routine lab_voice
    lab_voice = data.Routine(
        name='lab_voice',
        components=[lab_voice_text, lab_sound, key_resp_lab_sound],
    )
    lab_voice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    lab_sound.setSound('stimulus/voice_learned/吴婷婷_voice_learned.wav', hamming=True)
    lab_sound.setVolume(1.0, log=False)
    lab_sound.seek(0)
    # create starting attributes for key_resp_lab_sound
    key_resp_lab_sound.keys = []
    key_resp_lab_sound.rt = []
    _key_resp_lab_sound_allKeys = []
    # store start times for lab_voice
    lab_voice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    lab_voice.tStart = globalClock.getTime(format='float')
    lab_voice.status = STARTED
    thisExp.addData('lab_voice.started', lab_voice.tStart)
    lab_voice.maxDuration = None
    # keep track of which components have finished
    lab_voiceComponents = lab_voice.components
    for thisComponent in lab_voice.components:
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
    
    # --- Run Routine "lab_voice" ---
    lab_voice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *lab_voice_text* updates
        
        # if lab_voice_text is starting this frame...
        if lab_voice_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            lab_voice_text.frameNStart = frameN  # exact frame index
            lab_voice_text.tStart = t  # local t and not account for scr refresh
            lab_voice_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(lab_voice_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'lab_voice_text.started')
            # update status
            lab_voice_text.status = STARTED
            lab_voice_text.setAutoDraw(True)
        
        # if lab_voice_text is active this frame...
        if lab_voice_text.status == STARTED:
            # update params
            pass
        
        # *lab_sound* updates
        
        # if lab_sound is starting this frame...
        if lab_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            lab_sound.frameNStart = frameN  # exact frame index
            lab_sound.tStart = t  # local t and not account for scr refresh
            lab_sound.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('lab_sound.started', tThisFlipGlobal)
            # update status
            lab_sound.status = STARTED
            lab_sound.play(when=win)  # sync with win flip
        
        # if lab_sound is stopping this frame...
        if lab_sound.status == STARTED:
            if bool(False) or lab_sound.isFinished:
                # keep track of stop time/frame for later
                lab_sound.tStop = t  # not accounting for scr refresh
                lab_sound.tStopRefresh = tThisFlipGlobal  # on global time
                lab_sound.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lab_sound.stopped')
                # update status
                lab_sound.status = FINISHED
                lab_sound.stop()
        
        # *key_resp_lab_sound* updates
        waitOnFlip = False
        
        # if key_resp_lab_sound is starting this frame...
        if key_resp_lab_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_lab_sound.frameNStart = frameN  # exact frame index
            key_resp_lab_sound.tStart = t  # local t and not account for scr refresh
            key_resp_lab_sound.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_lab_sound, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_lab_sound.started')
            # update status
            key_resp_lab_sound.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_lab_sound.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_lab_sound.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_lab_sound.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_lab_sound.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_lab_sound_allKeys.extend(theseKeys)
            if len(_key_resp_lab_sound_allKeys):
                key_resp_lab_sound.keys = _key_resp_lab_sound_allKeys[-1].name  # just the last key pressed
                key_resp_lab_sound.rt = _key_resp_lab_sound_allKeys[-1].rt
                key_resp_lab_sound.duration = _key_resp_lab_sound_allKeys[-1].duration
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
                playbackComponents=[lab_sound]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            lab_voice.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in lab_voice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "lab_voice" ---
    for thisComponent in lab_voice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for lab_voice
    lab_voice.tStop = globalClock.getTime(format='float')
    lab_voice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('lab_voice.stopped', lab_voice.tStop)
    lab_sound.pause()  # ensure sound has stopped at end of Routine
    # check responses
    if key_resp_lab_sound.keys in ['', [], None]:  # No response was made
        key_resp_lab_sound.keys = None
    thisExp.addData('key_resp_lab_sound.keys',key_resp_lab_sound.keys)
    if key_resp_lab_sound.keys != None:  # we had a response
        thisExp.addData('key_resp_lab_sound.rt', key_resp_lab_sound.rt)
        thisExp.addData('key_resp_lab_sound.duration', key_resp_lab_sound.duration)
    thisExp.nextEntry()
    # the Routine "lab_voice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_loop_trials = data.TrialHandler2(
        name='practice_loop_trials',
        nReps=9.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(practice_loop_trials)  # add the loop to the experiment
    thisPractice_loop_trial = practice_loop_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop_trial.rgb)
    if thisPractice_loop_trial != None:
        for paramName in thisPractice_loop_trial:
            globals()[paramName] = thisPractice_loop_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_loop_trial in practice_loop_trials:
        currentLoop = practice_loop_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop_trial.rgb)
        if thisPractice_loop_trial != None:
            for paramName in thisPractice_loop_trial:
                globals()[paramName] = thisPractice_loop_trial[paramName]
        
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
        if isinstance(practice_loop_trials, data.TrialHandler2) and thisPractice_loop_trial.thisN != practice_loop_trials.thisTrial.thisN:
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
        
        # --- Prepare to start Routine "practice_trial" ---
        # create an object to store info about Routine practice_trial
        practice_trial = data.Routine(
            name='practice_trial',
            components=[sound_practice],
        )
        practice_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from practice_trial_code
        import pandas as pd
        
        # 获取当前trial索引
        trial_index = thisExp.practicetrial_counter
        
        currentpractice_trial = thisExp.trials_list_practice[trial_index]
        currentpracticeVoice = currentpractice_trial['voice']
        currentpracticefilepath = currentpractice_trial['filepath']
        
        print(f"试验{trial_index+1}: 声音条件={currentpracticeVoice}, 文件={currentpracticefilepath}")
        
        # 更新计数器
        thisExp.practicetrial_counter += 1
        thisExp.correctAnswer = correct_answers.get(currentpracticeVoice, '未知')
        
        thisExp.addData('voice', currentpracticeVoice)
        thisExp.addData('filepath', currentpracticefilepath)
        thisExp.addData('correct_answer', thisExp.correctAnswer)
        sound_practice.setSound(currentpracticefilepath, secs=2, hamming=True)
        sound_practice.setVolume(1.0, log=False)
        sound_practice.seek(0)
        # store start times for practice_trial
        practice_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_trial.tStart = globalClock.getTime(format='float')
        practice_trial.status = STARTED
        thisExp.addData('practice_trial.started', practice_trial.tStart)
        practice_trial.maxDuration = None
        # keep track of which components have finished
        practice_trialComponents = practice_trial.components
        for thisComponent in practice_trial.components:
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
        
        # --- Run Routine "practice_trial" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop_trials, data.TrialHandler2) and thisPractice_loop_trial.thisN != practice_loop_trials.thisTrial.thisN:
            continueRoutine = False
        practice_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *sound_practice* updates
            
            # if sound_practice is starting this frame...
            if sound_practice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_practice.frameNStart = frameN  # exact frame index
                sound_practice.tStart = t  # local t and not account for scr refresh
                sound_practice.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_practice.started', tThisFlipGlobal)
                # update status
                sound_practice.status = STARTED
                sound_practice.play(when=win)  # sync with win flip
            
            # if sound_practice is stopping this frame...
            if sound_practice.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_practice.tStartRefresh + 2-frameTolerance or sound_practice.isFinished:
                    # keep track of stop time/frame for later
                    sound_practice.tStop = t  # not accounting for scr refresh
                    sound_practice.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_practice.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_practice.stopped')
                    # update status
                    sound_practice.status = FINISHED
                    sound_practice.stop()
            
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
                    playbackComponents=[sound_practice]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                practice_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_trial" ---
        for thisComponent in practice_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_trial
        practice_trial.tStop = globalClock.getTime(format='float')
        practice_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_trial.stopped', practice_trial.tStop)
        sound_practice.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if practice_trial.maxDurationReached:
            routineTimer.addTime(-practice_trial.maxDuration)
        elif practice_trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "selection_practice" ---
        # create an object to store info about Routine selection_practice
        selection_practice = data.Routine(
            name='selection_practice',
            components=[option1_practice_text, option2_practice_text, option3_practice_text, option4_practice_text, selection_practice_key_resp],
        )
        selection_practice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from selection_practice_code
        # selection_formal1_2 routine的Begin Routine部分
        # 随机分配三个身份到三个按键位置
        voice_types = ['familiar', 'celebrity', 'unfamiliar', 'someoneElse']
        random.shuffle(voice_types)  # 随机打乱顺序
        
        # 创建本次试验的按键与身份映射
        key_to_voice = {
            '1': voice_types[0],
            '2': voice_types[1], 
            '3': voice_types[2],
            '4': voice_types[3]
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
        option1_practice_text.text = f"1. {name_display[key_to_voice['1']]}"    # 选项1
        option2_practice_text.text = f"2. {name_display[key_to_voice['2']]}"    # 选项2
        option3_practice_text.text = f"3. {name_display[key_to_voice['3']]}"    # 选项3
        option4_practice_text.text = f"4. {name_display[key_to_voice['4']]}"    # 选项4
        
        # 记录语言选择信息和按键映射到实验数据
        thisExp.addData('familiar_name_language', language_choice.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_language', language_choice.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_language', language_choice.get('unfamiliar', '未出现'))
        thisExp.addData('someoneElse_name_language', language_choice.get('someoneElse', '未出现'))
        thisExp.addData('familiar_name_displayed', name_display.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_displayed', name_display.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_displayed', name_display.get('unfamiliar', '未出现'))
        thisExp.addData('someoneElse_name_displayed', name_display.get('someoneElse', '未出现'))
        
        # 记录本次试验的按键映射
        thisExp.addData('key1_mapping', key_to_voice['1'])
        thisExp.addData('key2_mapping', key_to_voice['2'])
        thisExp.addData('key3_mapping', key_to_voice['3'])
        thisExp.addData('key4_mapping', key_to_voice['4'])
        # create starting attributes for selection_practice_key_resp
        selection_practice_key_resp.keys = []
        selection_practice_key_resp.rt = []
        _selection_practice_key_resp_allKeys = []
        # store start times for selection_practice
        selection_practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        selection_practice.tStart = globalClock.getTime(format='float')
        selection_practice.status = STARTED
        thisExp.addData('selection_practice.started', selection_practice.tStart)
        selection_practice.maxDuration = None
        # keep track of which components have finished
        selection_practiceComponents = selection_practice.components
        for thisComponent in selection_practice.components:
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
        
        # --- Run Routine "selection_practice" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop_trials, data.TrialHandler2) and thisPractice_loop_trial.thisN != practice_loop_trials.thisTrial.thisN:
            continueRoutine = False
        selection_practice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *option1_practice_text* updates
            
            # if option1_practice_text is starting this frame...
            if option1_practice_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option1_practice_text.frameNStart = frameN  # exact frame index
                option1_practice_text.tStart = t  # local t and not account for scr refresh
                option1_practice_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option1_practice_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option1_practice_text.started')
                # update status
                option1_practice_text.status = STARTED
                option1_practice_text.setAutoDraw(True)
            
            # if option1_practice_text is active this frame...
            if option1_practice_text.status == STARTED:
                # update params
                pass
            
            # if option1_practice_text is stopping this frame...
            if option1_practice_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option1_practice_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option1_practice_text.tStop = t  # not accounting for scr refresh
                    option1_practice_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option1_practice_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option1_practice_text.stopped')
                    # update status
                    option1_practice_text.status = FINISHED
                    option1_practice_text.setAutoDraw(False)
            
            # *option2_practice_text* updates
            
            # if option2_practice_text is starting this frame...
            if option2_practice_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option2_practice_text.frameNStart = frameN  # exact frame index
                option2_practice_text.tStart = t  # local t and not account for scr refresh
                option2_practice_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option2_practice_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option2_practice_text.started')
                # update status
                option2_practice_text.status = STARTED
                option2_practice_text.setAutoDraw(True)
            
            # if option2_practice_text is active this frame...
            if option2_practice_text.status == STARTED:
                # update params
                pass
            
            # if option2_practice_text is stopping this frame...
            if option2_practice_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option2_practice_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option2_practice_text.tStop = t  # not accounting for scr refresh
                    option2_practice_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option2_practice_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option2_practice_text.stopped')
                    # update status
                    option2_practice_text.status = FINISHED
                    option2_practice_text.setAutoDraw(False)
            
            # *option3_practice_text* updates
            
            # if option3_practice_text is starting this frame...
            if option3_practice_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option3_practice_text.frameNStart = frameN  # exact frame index
                option3_practice_text.tStart = t  # local t and not account for scr refresh
                option3_practice_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option3_practice_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option3_practice_text.started')
                # update status
                option3_practice_text.status = STARTED
                option3_practice_text.setAutoDraw(True)
            
            # if option3_practice_text is active this frame...
            if option3_practice_text.status == STARTED:
                # update params
                pass
            
            # if option3_practice_text is stopping this frame...
            if option3_practice_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option3_practice_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option3_practice_text.tStop = t  # not accounting for scr refresh
                    option3_practice_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option3_practice_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option3_practice_text.stopped')
                    # update status
                    option3_practice_text.status = FINISHED
                    option3_practice_text.setAutoDraw(False)
            
            # *option4_practice_text* updates
            
            # if option4_practice_text is starting this frame...
            if option4_practice_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                option4_practice_text.frameNStart = frameN  # exact frame index
                option4_practice_text.tStart = t  # local t and not account for scr refresh
                option4_practice_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(option4_practice_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'option4_practice_text.started')
                # update status
                option4_practice_text.status = STARTED
                option4_practice_text.setAutoDraw(True)
            
            # if option4_practice_text is active this frame...
            if option4_practice_text.status == STARTED:
                # update params
                pass
            
            # if option4_practice_text is stopping this frame...
            if option4_practice_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > option4_practice_text.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    option4_practice_text.tStop = t  # not accounting for scr refresh
                    option4_practice_text.tStopRefresh = tThisFlipGlobal  # on global time
                    option4_practice_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'option4_practice_text.stopped')
                    # update status
                    option4_practice_text.status = FINISHED
                    option4_practice_text.setAutoDraw(False)
            
            # *selection_practice_key_resp* updates
            waitOnFlip = False
            
            # if selection_practice_key_resp is starting this frame...
            if selection_practice_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                selection_practice_key_resp.frameNStart = frameN  # exact frame index
                selection_practice_key_resp.tStart = t  # local t and not account for scr refresh
                selection_practice_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(selection_practice_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'selection_practice_key_resp.started')
                # update status
                selection_practice_key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(selection_practice_key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(selection_practice_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if selection_practice_key_resp is stopping this frame...
            if selection_practice_key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > selection_practice_key_resp.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    selection_practice_key_resp.tStop = t  # not accounting for scr refresh
                    selection_practice_key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    selection_practice_key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'selection_practice_key_resp.stopped')
                    # update status
                    selection_practice_key_resp.status = FINISHED
                    selection_practice_key_resp.status = FINISHED
            if selection_practice_key_resp.status == STARTED and not waitOnFlip:
                theseKeys = selection_practice_key_resp.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                _selection_practice_key_resp_allKeys.extend(theseKeys)
                if len(_selection_practice_key_resp_allKeys):
                    selection_practice_key_resp.keys = _selection_practice_key_resp_allKeys[-1].name  # just the last key pressed
                    selection_practice_key_resp.rt = _selection_practice_key_resp_allKeys[-1].rt
                    selection_practice_key_resp.duration = _selection_practice_key_resp_allKeys[-1].duration
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
                selection_practice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in selection_practice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "selection_practice" ---
        for thisComponent in selection_practice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for selection_practice
        selection_practice.tStop = globalClock.getTime(format='float')
        selection_practice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('selection_practice.stopped', selection_practice.tStop)
        # check responses
        if selection_practice_key_resp.keys in ['', [], None]:  # No response was made
            selection_practice_key_resp.keys = None
        practice_loop_trials.addData('selection_practice_key_resp.keys',selection_practice_key_resp.keys)
        if selection_practice_key_resp.keys != None:  # we had a response
            practice_loop_trials.addData('selection_practice_key_resp.rt', selection_practice_key_resp.rt)
            practice_loop_trials.addData('selection_practice_key_resp.duration', selection_practice_key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if selection_practice.maxDurationReached:
            routineTimer.addTime(-selection_practice.maxDuration)
        elif selection_practice.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "feedback_practice" ---
        # create an object to store info about Routine feedback_practice
        feedback_practice = data.Routine(
            name='feedback_practice',
            components=[feedback_practice_display],
        )
        feedback_practice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from feedback_practice_code
        # feedback routine的Begin Routine部分
        # 获取被试的反应
        if selection_practice_key_resp.keys:
            # 处理反应键
            if type(selection_practice_key_resp.keys) is list:
                participant_response_key = selection_practice_key_resp.keys[0]
            else:
                participant_response_key = selection_practice_key_resp.keys
            
            # 处理反应时
            if type(selection_practice_key_resp.rt) is list:
                participant_rt = selection_practice_key_resp.rt[0]
            else:
                participant_rt = selection_practice_key_resp.rt
            
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
        feedback_practice_display.text = feedback_text
        feedback_practice_display.color = feedback_color
        
        # 记录反应数据
        thisExp.addData('response_key', participant_response_key)  # 记录的按键
        thisExp.addData('response_identity', participant_response)  # 记录的身份
        thisExp.addData('rt', participant_rt)
        thisExp.addData('correct', 1 if is_correct else 0)
        # store start times for feedback_practice
        feedback_practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback_practice.tStart = globalClock.getTime(format='float')
        feedback_practice.status = STARTED
        thisExp.addData('feedback_practice.started', feedback_practice.tStart)
        feedback_practice.maxDuration = None
        # keep track of which components have finished
        feedback_practiceComponents = feedback_practice.components
        for thisComponent in feedback_practice.components:
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
        
        # --- Run Routine "feedback_practice" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop_trials, data.TrialHandler2) and thisPractice_loop_trial.thisN != practice_loop_trials.thisTrial.thisN:
            continueRoutine = False
        feedback_practice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_practice_display* updates
            
            # if feedback_practice_display is starting this frame...
            if feedback_practice_display.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_practice_display.frameNStart = frameN  # exact frame index
                feedback_practice_display.tStart = t  # local t and not account for scr refresh
                feedback_practice_display.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_practice_display, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_practice_display.started')
                # update status
                feedback_practice_display.status = STARTED
                feedback_practice_display.setAutoDraw(True)
            
            # if feedback_practice_display is active this frame...
            if feedback_practice_display.status == STARTED:
                # update params
                pass
            
            # if feedback_practice_display is stopping this frame...
            if feedback_practice_display.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_practice_display.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_practice_display.tStop = t  # not accounting for scr refresh
                    feedback_practice_display.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_practice_display.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_practice_display.stopped')
                    # update status
                    feedback_practice_display.status = FINISHED
                    feedback_practice_display.setAutoDraw(False)
            
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
                feedback_practice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_practice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback_practice" ---
        for thisComponent in feedback_practice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback_practice
        feedback_practice.tStop = globalClock.getTime(format='float')
        feedback_practice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback_practice.stopped', feedback_practice.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback_practice.maxDurationReached:
            routineTimer.addTime(-feedback_practice.maxDuration)
        elif feedback_practice.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 9.0 repeats of 'practice_loop_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
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
    thisExp.nextEntry()
    # the Routine "instr_formal1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    formal1_1_loop_trials = data.TrialHandler2(
        name='formal1_1_loop_trials',
        nReps=30.0, 
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
        
        # --- Prepare to start Routine "formal1_run1_trial" ---
        # create an object to store info about Routine formal1_run1_trial
        formal1_run1_trial = data.Routine(
            name='formal1_run1_trial',
            components=[formal_1_sound],
        )
        formal1_run1_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from formal_1_trial_code
        # 获取当前trial索引
        trial_index = thisExp.trial_counter1
        
        current_trial = thisExp.trials_list_formal1[trial_index]
        currentSentence = current_trial['sentence_list']
        currentSentenceID = current_trial['sentenceID']
        currentVoice = current_trial['voice']
        currentfilepath = current_trial['filepath']
        
        # 更新计数器
        thisExp.trial_counter1 += 1
        thisExp.correctAnswer = correct_answers.get(currentVoice, '未知')
        
        thisExp.addData('sentence_list', currentSentence)
        thisExp.addData('sentenceID', currentSentenceID)
        thisExp.addData('voice', currentVoice)
        thisExp.addData('filepath', currentfilepath)
        thisExp.addData('correct_answer', thisExp.correctAnswer)
        formal_1_sound.setSound(currentfilepath, secs=2, hamming=True)
        formal_1_sound.setVolume(1.0, log=False)
        formal_1_sound.seek(0)
        # store start times for formal1_run1_trial
        formal1_run1_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal1_run1_trial.tStart = globalClock.getTime(format='float')
        formal1_run1_trial.status = STARTED
        thisExp.addData('formal1_run1_trial.started', formal1_run1_trial.tStart)
        formal1_run1_trial.maxDuration = None
        # keep track of which components have finished
        formal1_run1_trialComponents = formal1_run1_trial.components
        for thisComponent in formal1_run1_trial.components:
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
        
        # --- Run Routine "formal1_run1_trial" ---
        # if trial has changed, end Routine now
        if isinstance(formal1_1_loop_trials, data.TrialHandler2) and thisFormal1_1_loop_trial.thisN != formal1_1_loop_trials.thisTrial.thisN:
            continueRoutine = False
        formal1_run1_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
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
                if tThisFlipGlobal > formal_1_sound.tStartRefresh + 2-frameTolerance or formal_1_sound.isFinished:
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
                formal1_run1_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal1_run1_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal1_run1_trial" ---
        for thisComponent in formal1_run1_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal1_run1_trial
        formal1_run1_trial.tStop = globalClock.getTime(format='float')
        formal1_run1_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal1_run1_trial.stopped', formal1_run1_trial.tStop)
        formal_1_sound.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if formal1_run1_trial.maxDurationReached:
            routineTimer.addTime(-formal1_run1_trial.maxDuration)
        elif formal1_run1_trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "selection_formal" ---
        # create an object to store info about Routine selection_formal
        selection_formal = data.Routine(
            name='selection_formal',
            components=[option1_text, option2_text, option3_text, option4_text, selection_key_resp],
        )
        selection_formal.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from selection_formal1_code
        # selection_formal1_2 routine的Begin Routine部分
        # 随机分配三个身份到三个按键位置
        voice_types = ['familiar', 'celebrity', 'unfamiliar', 'someoneElse']
        random.shuffle(voice_types)  # 随机打乱顺序
        
        # 创建本次试验的按键与身份映射
        key_to_voice = {
            '1': voice_types[0],
            '2': voice_types[1], 
            '3': voice_types[2],
            '4': voice_types[3]
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
        option4_text.text = f"4. {name_display[key_to_voice['4']]}"    # 选项4
        # 记录语言选择信息和按键映射到实验数据
        thisExp.addData('familiar_name_language', language_choice.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_language', language_choice.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_language', language_choice.get('unfamiliar', '未出现'))
        thisExp.addData('someoneElse_name_language', language_choice.get('someoneElse', '未出现'))
        thisExp.addData('familiar_name_displayed', name_display.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_displayed', name_display.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_displayed', name_display.get('unfamiliar', '未出现'))
        thisExp.addData('someoneElse_name_displayed', name_display.get('someoneElse', '未出现'))
        
        # 记录本次试验的按键映射
        thisExp.addData('key1_mapping', key_to_voice['1'])
        thisExp.addData('key2_mapping', key_to_voice['2'])
        thisExp.addData('key3_mapping', key_to_voice['3'])
        thisExp.addData('key4_mapping', key_to_voice['4'])
        # create starting attributes for selection_key_resp
        selection_key_resp.keys = []
        selection_key_resp.rt = []
        _selection_key_resp_allKeys = []
        # store start times for selection_formal
        selection_formal.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        selection_formal.tStart = globalClock.getTime(format='float')
        selection_formal.status = STARTED
        thisExp.addData('selection_formal.started', selection_formal.tStart)
        selection_formal.maxDuration = None
        # keep track of which components have finished
        selection_formalComponents = selection_formal.components
        for thisComponent in selection_formal.components:
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
        
        # --- Run Routine "selection_formal" ---
        # if trial has changed, end Routine now
        if isinstance(formal1_1_loop_trials, data.TrialHandler2) and thisFormal1_1_loop_trial.thisN != formal1_1_loop_trials.thisTrial.thisN:
            continueRoutine = False
        selection_formal.forceEnded = routineForceEnded = not continueRoutine
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
                theseKeys = selection_key_resp.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
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
                selection_formal.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in selection_formal.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "selection_formal" ---
        for thisComponent in selection_formal.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for selection_formal
        selection_formal.tStop = globalClock.getTime(format='float')
        selection_formal.tStopRefresh = tThisFlipGlobal
        thisExp.addData('selection_formal.stopped', selection_formal.tStop)
        # check responses
        if selection_key_resp.keys in ['', [], None]:  # No response was made
            selection_key_resp.keys = None
        formal1_1_loop_trials.addData('selection_key_resp.keys',selection_key_resp.keys)
        if selection_key_resp.keys != None:  # we had a response
            formal1_1_loop_trials.addData('selection_key_resp.rt', selection_key_resp.rt)
            formal1_1_loop_trials.addData('selection_key_resp.duration', selection_key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if selection_formal.maxDurationReached:
            routineTimer.addTime(-selection_formal.maxDuration)
        elif selection_formal.forceEnded:
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
        
    # completed 30.0 repeats of 'formal1_1_loop_trials'
    
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
    
    # set up handler to look after randomisation of conditions etc
    formal1_2_loop_trials = data.TrialHandler2(
        name='formal1_2_loop_trials',
        nReps=30.0, 
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
        
        # --- Prepare to start Routine "formal1_run2_trial" ---
        # create an object to store info about Routine formal1_run2_trial
        formal1_run2_trial = data.Routine(
            name='formal1_run2_trial',
            components=[formal1_run2_sound],
        )
        formal1_run2_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from formal2_trial_code
        import pandas as pd
        
        # 获取当前trial索引
        trial_index = thisExp.trial_counter2
        
        current_trial = thisExp.trials_list_formal2[trial_index]
        currentSentence = current_trial['sentence_list']
        currentSentenceID = current_trial['sentenceID']
        currentVoice = current_trial['voice']
        currentfilepath = current_trial['filepath']
        
        # 更新计数器
        thisExp.trial_counter2 += 1
        thisExp.correctAnswer = correct_answers.get(currentVoice, '未知')
        
        thisExp.addData('sentence_list', currentSentence)
        thisExp.addData('sentenceID', currentSentenceID)
        thisExp.addData('voice', currentVoice)
        thisExp.addData('filepath', currentfilepath)
        thisExp.addData('correct_answer', thisExp.correctAnswer)
        formal1_run2_sound.setSound(currentfilepath, secs=2, hamming=True)
        formal1_run2_sound.setVolume(1.0, log=False)
        formal1_run2_sound.seek(0)
        # store start times for formal1_run2_trial
        formal1_run2_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal1_run2_trial.tStart = globalClock.getTime(format='float')
        formal1_run2_trial.status = STARTED
        thisExp.addData('formal1_run2_trial.started', formal1_run2_trial.tStart)
        formal1_run2_trial.maxDuration = None
        # keep track of which components have finished
        formal1_run2_trialComponents = formal1_run2_trial.components
        for thisComponent in formal1_run2_trial.components:
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
        
        # --- Run Routine "formal1_run2_trial" ---
        # if trial has changed, end Routine now
        if isinstance(formal1_2_loop_trials, data.TrialHandler2) and thisFormal1_2_loop_trial.thisN != formal1_2_loop_trials.thisTrial.thisN:
            continueRoutine = False
        formal1_run2_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
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
                if tThisFlipGlobal > formal1_run2_sound.tStartRefresh + 2-frameTolerance or formal1_run2_sound.isFinished:
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
                formal1_run2_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal1_run2_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal1_run2_trial" ---
        for thisComponent in formal1_run2_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal1_run2_trial
        formal1_run2_trial.tStop = globalClock.getTime(format='float')
        formal1_run2_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal1_run2_trial.stopped', formal1_run2_trial.tStop)
        formal1_run2_sound.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if formal1_run2_trial.maxDurationReached:
            routineTimer.addTime(-formal1_run2_trial.maxDuration)
        elif formal1_run2_trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "selection_formal" ---
        # create an object to store info about Routine selection_formal
        selection_formal = data.Routine(
            name='selection_formal',
            components=[option1_text, option2_text, option3_text, option4_text, selection_key_resp],
        )
        selection_formal.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from selection_formal1_code
        # selection_formal1_2 routine的Begin Routine部分
        # 随机分配三个身份到三个按键位置
        voice_types = ['familiar', 'celebrity', 'unfamiliar', 'someoneElse']
        random.shuffle(voice_types)  # 随机打乱顺序
        
        # 创建本次试验的按键与身份映射
        key_to_voice = {
            '1': voice_types[0],
            '2': voice_types[1], 
            '3': voice_types[2],
            '4': voice_types[3]
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
        option4_text.text = f"4. {name_display[key_to_voice['4']]}"    # 选项4
        # 记录语言选择信息和按键映射到实验数据
        thisExp.addData('familiar_name_language', language_choice.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_language', language_choice.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_language', language_choice.get('unfamiliar', '未出现'))
        thisExp.addData('someoneElse_name_language', language_choice.get('someoneElse', '未出现'))
        thisExp.addData('familiar_name_displayed', name_display.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_displayed', name_display.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_displayed', name_display.get('unfamiliar', '未出现'))
        thisExp.addData('someoneElse_name_displayed', name_display.get('someoneElse', '未出现'))
        
        # 记录本次试验的按键映射
        thisExp.addData('key1_mapping', key_to_voice['1'])
        thisExp.addData('key2_mapping', key_to_voice['2'])
        thisExp.addData('key3_mapping', key_to_voice['3'])
        thisExp.addData('key4_mapping', key_to_voice['4'])
        # create starting attributes for selection_key_resp
        selection_key_resp.keys = []
        selection_key_resp.rt = []
        _selection_key_resp_allKeys = []
        # store start times for selection_formal
        selection_formal.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        selection_formal.tStart = globalClock.getTime(format='float')
        selection_formal.status = STARTED
        thisExp.addData('selection_formal.started', selection_formal.tStart)
        selection_formal.maxDuration = None
        # keep track of which components have finished
        selection_formalComponents = selection_formal.components
        for thisComponent in selection_formal.components:
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
        
        # --- Run Routine "selection_formal" ---
        # if trial has changed, end Routine now
        if isinstance(formal1_2_loop_trials, data.TrialHandler2) and thisFormal1_2_loop_trial.thisN != formal1_2_loop_trials.thisTrial.thisN:
            continueRoutine = False
        selection_formal.forceEnded = routineForceEnded = not continueRoutine
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
                theseKeys = selection_key_resp.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
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
                selection_formal.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in selection_formal.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "selection_formal" ---
        for thisComponent in selection_formal.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for selection_formal
        selection_formal.tStop = globalClock.getTime(format='float')
        selection_formal.tStopRefresh = tThisFlipGlobal
        thisExp.addData('selection_formal.stopped', selection_formal.tStop)
        # check responses
        if selection_key_resp.keys in ['', [], None]:  # No response was made
            selection_key_resp.keys = None
        formal1_2_loop_trials.addData('selection_key_resp.keys',selection_key_resp.keys)
        if selection_key_resp.keys != None:  # we had a response
            formal1_2_loop_trials.addData('selection_key_resp.rt', selection_key_resp.rt)
            formal1_2_loop_trials.addData('selection_key_resp.duration', selection_key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if selection_formal.maxDurationReached:
            routineTimer.addTime(-selection_formal.maxDuration)
        elif selection_formal.forceEnded:
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
        
    # completed 30.0 repeats of 'formal1_2_loop_trials'
    
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
    
    # --- Prepare to start Routine "instr_nofeedback" ---
    # create an object to store info about Routine instr_nofeedback
    instr_nofeedback = data.Routine(
        name='instr_nofeedback',
        components=[instr_no_feedback_image, instr_formal2_key_resp],
    )
    instr_nofeedback.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_formal2_key_resp
    instr_formal2_key_resp.keys = []
    instr_formal2_key_resp.rt = []
    _instr_formal2_key_resp_allKeys = []
    # store start times for instr_nofeedback
    instr_nofeedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr_nofeedback.tStart = globalClock.getTime(format='float')
    instr_nofeedback.status = STARTED
    thisExp.addData('instr_nofeedback.started', instr_nofeedback.tStart)
    instr_nofeedback.maxDuration = None
    # keep track of which components have finished
    instr_nofeedbackComponents = instr_nofeedback.components
    for thisComponent in instr_nofeedback.components:
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
    
    # --- Run Routine "instr_nofeedback" ---
    instr_nofeedback.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr_no_feedback_image* updates
        
        # if instr_no_feedback_image is starting this frame...
        if instr_no_feedback_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_no_feedback_image.frameNStart = frameN  # exact frame index
            instr_no_feedback_image.tStart = t  # local t and not account for scr refresh
            instr_no_feedback_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_no_feedback_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_no_feedback_image.started')
            # update status
            instr_no_feedback_image.status = STARTED
            instr_no_feedback_image.setAutoDraw(True)
        
        # if instr_no_feedback_image is active this frame...
        if instr_no_feedback_image.status == STARTED:
            # update params
            pass
        
        # *instr_formal2_key_resp* updates
        waitOnFlip = False
        
        # if instr_formal2_key_resp is starting this frame...
        if instr_formal2_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_formal2_key_resp.frameNStart = frameN  # exact frame index
            instr_formal2_key_resp.tStart = t  # local t and not account for scr refresh
            instr_formal2_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_formal2_key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_formal2_key_resp.started')
            # update status
            instr_formal2_key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_formal2_key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_formal2_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_formal2_key_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr_formal2_key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_formal2_key_resp_allKeys.extend(theseKeys)
            if len(_instr_formal2_key_resp_allKeys):
                instr_formal2_key_resp.keys = _instr_formal2_key_resp_allKeys[-1].name  # just the last key pressed
                instr_formal2_key_resp.rt = _instr_formal2_key_resp_allKeys[-1].rt
                instr_formal2_key_resp.duration = _instr_formal2_key_resp_allKeys[-1].duration
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
            instr_nofeedback.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_nofeedback.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr_nofeedback" ---
    for thisComponent in instr_nofeedback.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr_nofeedback
    instr_nofeedback.tStop = globalClock.getTime(format='float')
    instr_nofeedback.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr_nofeedback.stopped', instr_nofeedback.tStop)
    # check responses
    if instr_formal2_key_resp.keys in ['', [], None]:  # No response was made
        instr_formal2_key_resp.keys = None
    thisExp.addData('instr_formal2_key_resp.keys',instr_formal2_key_resp.keys)
    if instr_formal2_key_resp.keys != None:  # we had a response
        thisExp.addData('instr_formal2_key_resp.rt', instr_formal2_key_resp.rt)
        thisExp.addData('instr_formal2_key_resp.duration', instr_formal2_key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instr_nofeedback" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    formal2_3_loop_trials = data.TrialHandler2(
        name='formal2_3_loop_trials',
        nReps=30.0, 
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
        
        # --- Prepare to start Routine "formal2_run3_trial" ---
        # create an object to store info about Routine formal2_run3_trial
        formal2_run3_trial = data.Routine(
            name='formal2_run3_trial',
            components=[formal3_sound],
        )
        formal2_run3_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from formal2_run3_trial_code
        import pandas as pd
        
        # 获取当前trial索引
        trial_index = thisExp.trial_counter3
        
        current_trial = thisExp.trials_list_formal3[trial_index]
        currentSentence = current_trial['sentence_list']
        currentSentenceID = current_trial['sentenceID']
        currentVoice = current_trial['voice']
        currentfilepath = current_trial['filepath']
        
        # 更新计数器
        thisExp.trial_counter3 += 1
        thisExp.correctAnswer = correct_answers.get(currentVoice, '未知')
        
        thisExp.addData('sentence_list', currentSentence)
        thisExp.addData('sentenceID', currentSentenceID)
        thisExp.addData('voice', currentVoice)
        thisExp.addData('filepath', currentfilepath)
        thisExp.addData('correct_answer', thisExp.correctAnswer)
        formal3_sound.setSound(currentfilepath, secs=2, hamming=True)
        formal3_sound.setVolume(1.0, log=False)
        formal3_sound.seek(0)
        # store start times for formal2_run3_trial
        formal2_run3_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal2_run3_trial.tStart = globalClock.getTime(format='float')
        formal2_run3_trial.status = STARTED
        thisExp.addData('formal2_run3_trial.started', formal2_run3_trial.tStart)
        formal2_run3_trial.maxDuration = None
        # keep track of which components have finished
        formal2_run3_trialComponents = formal2_run3_trial.components
        for thisComponent in formal2_run3_trial.components:
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
        
        # --- Run Routine "formal2_run3_trial" ---
        # if trial has changed, end Routine now
        if isinstance(formal2_3_loop_trials, data.TrialHandler2) and thisFormal2_3_loop_trial.thisN != formal2_3_loop_trials.thisTrial.thisN:
            continueRoutine = False
        formal2_run3_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
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
                if tThisFlipGlobal > formal3_sound.tStartRefresh + 2-frameTolerance or formal3_sound.isFinished:
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
                formal2_run3_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal2_run3_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal2_run3_trial" ---
        for thisComponent in formal2_run3_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal2_run3_trial
        formal2_run3_trial.tStop = globalClock.getTime(format='float')
        formal2_run3_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal2_run3_trial.stopped', formal2_run3_trial.tStop)
        formal3_sound.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if formal2_run3_trial.maxDurationReached:
            routineTimer.addTime(-formal2_run3_trial.maxDuration)
        elif formal2_run3_trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "selection_formal" ---
        # create an object to store info about Routine selection_formal
        selection_formal = data.Routine(
            name='selection_formal',
            components=[option1_text, option2_text, option3_text, option4_text, selection_key_resp],
        )
        selection_formal.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from selection_formal1_code
        # selection_formal1_2 routine的Begin Routine部分
        # 随机分配三个身份到三个按键位置
        voice_types = ['familiar', 'celebrity', 'unfamiliar', 'someoneElse']
        random.shuffle(voice_types)  # 随机打乱顺序
        
        # 创建本次试验的按键与身份映射
        key_to_voice = {
            '1': voice_types[0],
            '2': voice_types[1], 
            '3': voice_types[2],
            '4': voice_types[3]
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
        option4_text.text = f"4. {name_display[key_to_voice['4']]}"    # 选项4
        # 记录语言选择信息和按键映射到实验数据
        thisExp.addData('familiar_name_language', language_choice.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_language', language_choice.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_language', language_choice.get('unfamiliar', '未出现'))
        thisExp.addData('someoneElse_name_language', language_choice.get('someoneElse', '未出现'))
        thisExp.addData('familiar_name_displayed', name_display.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_displayed', name_display.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_displayed', name_display.get('unfamiliar', '未出现'))
        thisExp.addData('someoneElse_name_displayed', name_display.get('someoneElse', '未出现'))
        
        # 记录本次试验的按键映射
        thisExp.addData('key1_mapping', key_to_voice['1'])
        thisExp.addData('key2_mapping', key_to_voice['2'])
        thisExp.addData('key3_mapping', key_to_voice['3'])
        thisExp.addData('key4_mapping', key_to_voice['4'])
        # create starting attributes for selection_key_resp
        selection_key_resp.keys = []
        selection_key_resp.rt = []
        _selection_key_resp_allKeys = []
        # store start times for selection_formal
        selection_formal.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        selection_formal.tStart = globalClock.getTime(format='float')
        selection_formal.status = STARTED
        thisExp.addData('selection_formal.started', selection_formal.tStart)
        selection_formal.maxDuration = None
        # keep track of which components have finished
        selection_formalComponents = selection_formal.components
        for thisComponent in selection_formal.components:
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
        
        # --- Run Routine "selection_formal" ---
        # if trial has changed, end Routine now
        if isinstance(formal2_3_loop_trials, data.TrialHandler2) and thisFormal2_3_loop_trial.thisN != formal2_3_loop_trials.thisTrial.thisN:
            continueRoutine = False
        selection_formal.forceEnded = routineForceEnded = not continueRoutine
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
                theseKeys = selection_key_resp.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
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
                selection_formal.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in selection_formal.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "selection_formal" ---
        for thisComponent in selection_formal.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for selection_formal
        selection_formal.tStop = globalClock.getTime(format='float')
        selection_formal.tStopRefresh = tThisFlipGlobal
        thisExp.addData('selection_formal.stopped', selection_formal.tStop)
        # check responses
        if selection_key_resp.keys in ['', [], None]:  # No response was made
            selection_key_resp.keys = None
        formal2_3_loop_trials.addData('selection_key_resp.keys',selection_key_resp.keys)
        if selection_key_resp.keys != None:  # we had a response
            formal2_3_loop_trials.addData('selection_key_resp.rt', selection_key_resp.rt)
            formal2_3_loop_trials.addData('selection_key_resp.duration', selection_key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if selection_formal.maxDurationReached:
            routineTimer.addTime(-selection_formal.maxDuration)
        elif selection_formal.forceEnded:
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
        
    # completed 30.0 repeats of 'formal2_3_loop_trials'
    
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
    
    # set up handler to look after randomisation of conditions etc
    formal2_4_loop_trials = data.TrialHandler2(
        name='formal2_4_loop_trials',
        nReps=30.0, 
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
        
        # --- Prepare to start Routine "formal2_run4_trial" ---
        # create an object to store info about Routine formal2_run4_trial
        formal2_run4_trial = data.Routine(
            name='formal2_run4_trial',
            components=[formal4_sound],
        )
        formal2_run4_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from formal2_run4_trial_code
        import pandas as pd
        
        # 获取当前trial索引
        trial_index = thisExp.trial_counter4
        
        current_trial = thisExp.trials_list_formal4[trial_index]
        currentSentence = current_trial['sentence_list']
        currentSentenceID = current_trial['sentenceID']
        currentVoice = current_trial['voice']
        currentfilepath = current_trial['filepath']
        
        # 更新计数器
        thisExp.trial_counter4 += 1
        thisExp.correctAnswer = correct_answers.get(currentVoice, '未知')
        
        thisExp.addData('sentence_list', currentSentence)
        thisExp.addData('sentenceID', currentSentenceID)
        thisExp.addData('voice', currentVoice)
        thisExp.addData('filepath', currentfilepath)
        thisExp.addData('correct_answer', thisExp.correctAnswer)
        formal4_sound.setSound(currentfilepath, secs=2, hamming=True)
        formal4_sound.setVolume(1.0, log=False)
        formal4_sound.seek(0)
        # store start times for formal2_run4_trial
        formal2_run4_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        formal2_run4_trial.tStart = globalClock.getTime(format='float')
        formal2_run4_trial.status = STARTED
        thisExp.addData('formal2_run4_trial.started', formal2_run4_trial.tStart)
        formal2_run4_trial.maxDuration = None
        # keep track of which components have finished
        formal2_run4_trialComponents = formal2_run4_trial.components
        for thisComponent in formal2_run4_trial.components:
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
        
        # --- Run Routine "formal2_run4_trial" ---
        # if trial has changed, end Routine now
        if isinstance(formal2_4_loop_trials, data.TrialHandler2) and thisFormal2_4_loop_trial.thisN != formal2_4_loop_trials.thisTrial.thisN:
            continueRoutine = False
        formal2_run4_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
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
                if tThisFlipGlobal > formal4_sound.tStartRefresh + 2-frameTolerance or formal4_sound.isFinished:
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
                formal2_run4_trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in formal2_run4_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "formal2_run4_trial" ---
        for thisComponent in formal2_run4_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for formal2_run4_trial
        formal2_run4_trial.tStop = globalClock.getTime(format='float')
        formal2_run4_trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('formal2_run4_trial.stopped', formal2_run4_trial.tStop)
        formal4_sound.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if formal2_run4_trial.maxDurationReached:
            routineTimer.addTime(-formal2_run4_trial.maxDuration)
        elif formal2_run4_trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "selection_formal" ---
        # create an object to store info about Routine selection_formal
        selection_formal = data.Routine(
            name='selection_formal',
            components=[option1_text, option2_text, option3_text, option4_text, selection_key_resp],
        )
        selection_formal.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from selection_formal1_code
        # selection_formal1_2 routine的Begin Routine部分
        # 随机分配三个身份到三个按键位置
        voice_types = ['familiar', 'celebrity', 'unfamiliar', 'someoneElse']
        random.shuffle(voice_types)  # 随机打乱顺序
        
        # 创建本次试验的按键与身份映射
        key_to_voice = {
            '1': voice_types[0],
            '2': voice_types[1], 
            '3': voice_types[2],
            '4': voice_types[3]
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
        option4_text.text = f"4. {name_display[key_to_voice['4']]}"    # 选项4
        # 记录语言选择信息和按键映射到实验数据
        thisExp.addData('familiar_name_language', language_choice.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_language', language_choice.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_language', language_choice.get('unfamiliar', '未出现'))
        thisExp.addData('someoneElse_name_language', language_choice.get('someoneElse', '未出现'))
        thisExp.addData('familiar_name_displayed', name_display.get('familiar', '未出现'))
        thisExp.addData('celebrity_name_displayed', name_display.get('celebrity', '未出现'))
        thisExp.addData('unfamiliar_name_displayed', name_display.get('unfamiliar', '未出现'))
        thisExp.addData('someoneElse_name_displayed', name_display.get('someoneElse', '未出现'))
        
        # 记录本次试验的按键映射
        thisExp.addData('key1_mapping', key_to_voice['1'])
        thisExp.addData('key2_mapping', key_to_voice['2'])
        thisExp.addData('key3_mapping', key_to_voice['3'])
        thisExp.addData('key4_mapping', key_to_voice['4'])
        # create starting attributes for selection_key_resp
        selection_key_resp.keys = []
        selection_key_resp.rt = []
        _selection_key_resp_allKeys = []
        # store start times for selection_formal
        selection_formal.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        selection_formal.tStart = globalClock.getTime(format='float')
        selection_formal.status = STARTED
        thisExp.addData('selection_formal.started', selection_formal.tStart)
        selection_formal.maxDuration = None
        # keep track of which components have finished
        selection_formalComponents = selection_formal.components
        for thisComponent in selection_formal.components:
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
        
        # --- Run Routine "selection_formal" ---
        # if trial has changed, end Routine now
        if isinstance(formal2_4_loop_trials, data.TrialHandler2) and thisFormal2_4_loop_trial.thisN != formal2_4_loop_trials.thisTrial.thisN:
            continueRoutine = False
        selection_formal.forceEnded = routineForceEnded = not continueRoutine
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
                theseKeys = selection_key_resp.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
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
                selection_formal.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in selection_formal.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "selection_formal" ---
        for thisComponent in selection_formal.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for selection_formal
        selection_formal.tStop = globalClock.getTime(format='float')
        selection_formal.tStopRefresh = tThisFlipGlobal
        thisExp.addData('selection_formal.stopped', selection_formal.tStop)
        # check responses
        if selection_key_resp.keys in ['', [], None]:  # No response was made
            selection_key_resp.keys = None
        formal2_4_loop_trials.addData('selection_key_resp.keys',selection_key_resp.keys)
        if selection_key_resp.keys != None:  # we had a response
            formal2_4_loop_trials.addData('selection_key_resp.rt', selection_key_resp.rt)
            formal2_4_loop_trials.addData('selection_key_resp.duration', selection_key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if selection_formal.maxDurationReached:
            routineTimer.addTime(-selection_formal.maxDuration)
        elif selection_formal.forceEnded:
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
        
    # completed 30.0 repeats of 'formal2_4_loop_trials'
    
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
