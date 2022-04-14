import numpy as np
import os

def convertPat(PAT):
    '''
    Convert Pattern PAT: "ABB..."
    --> "ABB" --> [True, False, False]
    '''
    return [ x == "A" for x in PAT ]

def interpolate_np(ARR:np.ndarray, MIN=0, MAX=1):
    return np.interp(ARR, (ARR.min(), ARR.max()), (MIN, MAX))

def hidePlotBounds(ax):
    '''
    Function: hidePlotBounds
    Summary: Hides the Matplotlib Plot Bounds
    Examples: hidePlotBounds(plt)
    Attributes:
        @param (ax): Matplotlib Axis Object
    Returns: None
    '''
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.set_xticks([])
    ax.set_yticks([])

def remove(path:str):
    if os.path.exists(path): os.remove(path)

def pathfix(path:str, safe:bool=False):
    if safe: remove(path)
    return os.path.join(
        os.path.dirname(__file__),
        *path.split("/")
    )