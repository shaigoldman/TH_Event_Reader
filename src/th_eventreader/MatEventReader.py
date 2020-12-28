""" Loads TH events from the old matlab events files. Not recommended. 
    Instead its a better idea to use TH_EventReader.
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib import pyplot as plt


def get_events_path(subj, montage=0, exp='TH1'):
    """ Returns path/filename for a subjects matlab events.
        
        Args:
            subj (str): Subject ID code. Sample: R1195E.
            montage (int): The number of montage changes that the
                subj had undergone at the time of this session.
            exp (str): Expirement ID (TH1, TH3, THR)
        
        Returns:
            filename (str)
    """
    root_dir = f'/data3/events/RAM_{exp}'
    mont_str = f'_{montage}' if montage else ''
    return f'{root_dir}/{subj}{mont_str}_events.mat'


def load_mat_events(subj, montage, exp='TH1'):
    """ Loads matlab events structure for the given subject using `get_events_path`."""
    path = get_events_path(subj, montage, exp)
    return pd.DataFrame(sio.loadmat(path, squeeze_me=True)["events"])


def mat_pathData_to_list(events):
    """ Does in-place conversion of the matlab loaded path data to
        an easy to work with formatting. The sio.loadmat format is
        terriblly confusing and difficult, using nested np.recarrays
        etc. This format is the same way that path data is stored in
        the python YC1 events: a list of dictionaries, where each dict
        looks like {'mstime': A, 'heading': B, 'x': C, 'y': D}.
        Unfortunately the naming convention is different than in YC1,
        but that is easy enough to deal with in higher level modules.
        Another note is that unlike YC1, these times are not relative
        to the onset of the event, but absolute mstimes. That will also
        need to be dealt with in higher level modules for conformity.
        
        Args:
            events (pd.DataFrame): an events structure loaded using
                sio.loadmat from the matlab events.
    """
    #------Iter though the raw path data
    path = events['pathInfo']
    for i, pathData in path.iteritems():
        #------Get the keys ('mstime', 'x', 'y', 'heading')
        keys = pathData.dtype.names
        #------Get the actual values out of these nested arrays
        #      and convert from the unworkable np.recarray to
        #      a simple dict
        squeezed_pathData = {}
        for key in keys:
            squeezed_pathData[key] = np.atleast_1d(pathData[key][()])
            length = len(squeezed_pathData[key])
        pathData = squeezed_pathData
        #------Convert to list of dictionaries
        list_pathData = [ { key: pathData[key][i]
                            for key in keys
                          }
                          for i in range(length)
                        ]
        #------Transform inplace
        events['pathInfo'].loc[i] = list_pathData


def rename_old_fieldnames(events, exp):
    """ Since I'm getting these events from the old matlab events
        structs, there's been some naming conventions since then that
        can cause problems for us, so here I can fix that in-place.
    """
    old_names = ['isStim', 'item', 'stimParams']
    new_names = ['is_stim', 'item_name', 'stim_params']
    for old_name, new_name in zip(old_names, new_names):
        if not new_name in events and old_name in events:
            events[new_name] = events.pop(old_name)
    #------Add missing experiment field
    events['experiment'] = [exp for i in range(len(events['mstime']))]
    #------Also add the montage field if its missing
    if not 'montage' in events:
        events['montage'] = np.zeros(len(events['experiment']))


def last_valid_event(subj):
    """ Some of these subjects can't load eeg from event X and on. 
        I'm not really sure what the deal is, but this tells you what
        event is the last valid event for loading eeg for that subject.
    """
    subj_event_pairs = (('R1154D', 780), ('R1167M', 260), ('R1180C', 522),
                        ('R1190P', 241), ('R1191J', 780), ('R1192C', 261),
                        ('R1195E', 780), ('R1024T', 520))
    subjs = [pair[0] for pair in subj_event_pairs]
    events = [pair[1] for pair in subj_event_pairs]
    if subj in subjs:
        return events[subjs.index(subj)]
    return None


def get_events_from_mat(subj, montage, exp):
    """Returns the reformatted events df for subj and mont."""
    #------Load events
    events = load_mat_events(subj, montage, exp)
    events['mstime'] = events['mstime'].astype(int)
    #------Adjust naming conventions
    rename_old_fieldnames(events, exp)
    #------Exclude events that don't have an eegfile
    #      (not really sure what's up with this, but some just have
    #      an empty list ([]) instead of a str file name)
    events = events[[type(i)==str for i in events['eegfile']]]
    #------Exclude events that don't work with loading eeg 
    if last_valid_event(subj) is not None:
        events = events[:last_valid_event(subj)-1]
    #------Convert the pathData into the list-dict format
    mat_pathData_to_list(events)
    #------Done!
    return events
