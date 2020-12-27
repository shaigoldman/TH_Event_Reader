#!/usr/bin/env python
# coding: utf-8

# ## TH_EventReader
# 
# This code will load TH events using cmlreaders and then find the missing path data using the log files.

import os
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from cmlreaders import CMLReader, get_data_index


def get_cmlevents(subj, montage=None, session=None, exp='TH1'):
    """ Returns the reformatted events df for subj and mont.
        This events struct does not include pathInfo, since that isn't
        recorded in the system used by cmlreaders. To get pathInfo you
        need to use `read_path_log`.
    """
    #------Load data index for RAM
    df = get_data_index("r1")
    #------Specify the df for this subject and exp
    this_df =  df[(df['subject']==subj) & (df['experiment']==exp)]
    #------Find out the sessions, localization, and montage for this subject
    if session is None: # default to first sess
        session = this_df['session'].iloc[0]
    if montage is None: # default to first mont
        montage = this_df['montage'].iloc[0]
    #------Get more specific df
    this_specific_df = (this_df[(this_df['session'] == session)
                           & (this_df['montage'] == montage)])
    loc = int(this_specific_df.iloc()[0]['localization'])
    #-------Subjs with a montage above 0 have aliases used in log files
    subject_alias = this_specific_df['subject_alias'].iloc[0] 
                    # ^ use .iloc[0] bc this_specific_df has only one item
    #------For some subjs the sess ID system changed over time,
    #      and we need to know the original sess ID for certain log
    #      files access
    orig_sess_ID = this_specific_df['original_session'].iloc[0]
    if type(orig_sess_ID) == str:
        orig_sess_ID = np.float64(orig_sess_ID)
        # I do it as float first in case of NaN
        if orig_sess_ID == int(orig_sess_ID):
            orig_sess_ID = int(orig_sess_ID)
    if np.isnan(orig_sess_ID):
        orig_sess_ID = session
    #------Use CMLReader to read the events structure
    reader = CMLReader(subj, exp, session=session, 
                       montage=montage, localization=loc)
    events = reader.load('events')
    events['original_session_ID'] = orig_sess_ID
    events['subject_alias'] = subject_alias
    return events


def read_path_log(events):
    """ Reads the .par log file of navigation data and organizes it to
        fit with the rest of the events DatFrame.
        
        Args:
            events (pd.DataFrame): Events struct including 'subject'
                and 'original_session_ID', and event info.
        
        Returns
            pd.DataFrame with one field 'pathData', with the same index
                as events.
    """
    #-----Setup
    monts_and_sess = events[['subject_alias', 'original_session_ID']].drop_duplicates()
    #------This array will hold all the path data:
    all_pathInfo = []
    #------Iterate sessions bc each has its own par file
    for (index, (subj_str, sess)) in monts_and_sess.iterrows():
        #------Read the par log file
        par_file = f'/data10/RAM/subjects/{subj_str}/behavioral/TH1/session_{sess}/playerPaths.par'
        with open(par_file) as f:
            lines = f.read().split('\n')
        #------Init tracking vars
        current_event = events.index[0]
        current_event_start = 0
        event_pathInfo = []
        #------Go through the log file data
        for line_num, line in enumerate(lines):
            #------Collect the path data in this line of the log file
            # this dict will eventually have the keys: ['mstime', 'x', 'y', 'heading']
            path_data = {} 
            for index, token in enumerate(line.split('\t')): # see each datum in line
                if token == '':
                    # this is the empty last line of the log file
                    break 
                if index == 0: # event mstime
                    mstime = int(token)
                    path_data['mstime'] = mstime
                elif index == 1: # event start mstime
                    token_event_start = int(token)
                    #------Check if on a new event
                    if token_event_start > current_event_start:
                        if current_event_start > 0: # if its 0 that means we are on 1st line of log file
                            # add this event's pathInfo to all the events path info
                            all_pathInfo.append(event_pathInfo)
                            event_pathInfo = []
                            current_event += 1
                        current_event_start = token_event_start
                elif index == 5: # player x position
                    path_data['x'] = float(token)
                elif index == 6: # player y postion
                    path_data['y'] = float(token)
                elif index == 7: # player heading
                    path_data['heading'] = float(token)
            if path_data: # add this data to the event's pathInfo
                event_pathInfo.append(path_data)
        if event_pathInfo:
            all_pathInfo.append(event_pathInfo)
    #------Organize nav events by the overall events struct
    ordered_pathInfo = []
    event_num = 0
    for pathInfo in all_pathInfo:
        start_time = pathInfo[0]['mstime']
        # see which event is most assosiated in start-time with this nav vector:
        for index, mstime in events['mstime'][event_num:].iteritems():
            this_dist = np.abs(start_time-events['mstime'][index])
            try:
                next_dist = np.abs(start_time-events['mstime'][index+1])
            except KeyError: # no more events exist
                next_dist = np.inf
            if this_dist <= next_dist:
                # this event is most closely associated with this index
                ordered_pathInfo.append(pathInfo)
                event_num += 1
                break
            else:
                # this event is more closely associated with a future index
                # that means that there are some non-nav events in between
                # so we add those as empty nav vectors
                ordered_pathInfo.append([])
                event_num += 1
    # Add the non-nav events that occur after all nav events are over
    while len(ordered_pathInfo) < len(events):
        ordered_pathInfo.append([])
    #------Convert to pd.Series
    ordered_pathInfo = pd.DataFrame(data=pd.Series(ordered_pathInfo, index=events.index), columns=['pathInfo'])
    #------Done!
    return ordered_pathInfo


def get_savename(subj, montage, session, exp):
    if montage > 0:
        subj = f'{subj}_{montage}'
    main_dir = __file__.split('src')[0] + 'data/'
    if not os.path.exists(main_dir):
        # if installed with pip
        main_dir = __file__.split('TH_EventReader.py')[0] + 'data/'
    exp_dir = main_dir + exp + '/'
    if not os.path.exists(exp_dir):
        print(f'Creating {exp_dir}')
        os.mkdir(exp_dir) 
    subj_dir = exp_dir + subj + '/'
    if not os.path.exists(subj_dir):
        print(f'Creating {subj_dir}')
        os.mkdir(subj_dir)
    return subj_dir + f'session_{session}.pkl'


def save_events(events, subj, montage, session, exp):
    fname = get_savename(subj, montage, session, exp)
    events.to_pickle(fname)
    

def load_events(subj, montage, session, exp):
    return pd.read_pickle(get_savename(subj, montage, session, exp))


def get_events(subj, montage, session, exp, 
               recalc=False, save=True):
    """Returns the reformatted events df for subj and mont."""
    if not recalc:
        save_fname = get_savename(subj, montage, session, exp)
        if os.path.exists(save_fname):
            try:
                return load_events(subj, montage, session, exp)
            except:
                pass
        
    events = get_cmlevents(subj, montage, session, exp)
    events['pathInfo'] = read_path_log(events)
    
    if save:
        save_events(events, subj, montage, session, exp)
        
    return events


def get_monts_and_sess_pairs(subj, exp='TH1'):
    """ Returns a df of mont/sess pairs for this subj in this exp.
        iterate easily through this list in the following format:
            "for (index, (mont, sess)) in monts_and_sess_pairs(subj).iterrows():"
    """
    df = get_data_index("r1")
    subjexp_df = (df[(df['subject']==subj) & 
                     (df['experiment']==exp)]
                    )[['montage', 'session']]
    subjexp_df.index = range(len(subjexp_df))
    return subjexp_df


def exp_df(exp='TH1'):
    warnings.filterwarnings('ignore')
    df = get_data_index("r1")
    df = df[df['experiment'] == exp]
    df['subj'] = df.pop('subject')
    warnings.resetwarnings()
    return df[['subj', 'montage', 'session']]
