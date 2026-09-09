"""Explicit, model-declared elapsed-time input for finite maneuvers."""
import math
import numpy as np


def time_input_seconds(metadata):
    value=float(metadata.get("sim2sim_time_input_s",0.))
    if not math.isfinite(value) or value<0:raise ValueError("Invalid policy time-input duration")
    return value


def time_command(elapsed,seconds):
    if seconds<=0:raise ValueError("A time-input policy requires a positive duration")
    out=np.zeros(13,np.float32);out[0]=np.clip(elapsed/seconds,0.,1.)
    return out
