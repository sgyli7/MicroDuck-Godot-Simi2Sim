"""Explicit, opt-in motion control experiments shared by play and offline replay.

These change policy commands, not physics or model weights. Their configuration
and version must accompany every trace and deployment comparison.
"""
import math

import numpy as np

from sim2sim.coords import quat_wxyz_to_mat


class MotionControl:
    def __init__(self,settings=None):
        self.settings=settings or {};self.reset()

    def reset(self):
        self.started=False;self.target_yaw=0.;self.brake_elapsed=0.;self.was_braking=False
        self.brake_speed=0.
        self.brake_target_yaw=0.
        self.walk_has_moved=False
        self.walk_idle_elapsed=0.

    def command(self,command,state,skill,dt=.02):
        out=np.asarray(command,np.float32).copy()
        rotation=quat_wxyz_to_mat(state.base_quat_wxyz)
        yaw=math.atan2(rotation[1,0],rotation[0,0])
        if not self.started or skill not in ('walking','roller'):
            self.target_yaw=yaw;self.started=True
        if skill not in ('walking','roller'):
            self.brake_elapsed=0.;self.was_braking=False;self.walk_has_moved=False;return out
        if skill=='roller' and self.settings.get('heading_hold',False):
            self.target_yaw+=float(out[2])*dt
            difference=self.target_yaw-yaw
            out[2]=np.clip(math.atan2(math.sin(difference),math.cos(difference)),-1.,1.)
        elif skill=='walking' and self.settings.get('walk_heading_gain',0.)>0:
            idle_only=self.settings.get('walk_heading_scope','all')=='idle_after_motion'
            moving=math.sqrt(float(out[0])**2+float(out[1])**2)>.01 or abs(float(out[2]))>.05
            if moving:self.walk_has_moved=True;self.walk_idle_elapsed=0.
            if abs(float(out[2]))>.05 or (idle_only and moving):
                self.target_yaw=yaw # Requested turns retain their original velocity command.
            elif not idle_only or self.walk_has_moved:
                if idle_only and self.walk_idle_elapsed<float(self.settings.get('walk_idle_delay_s',0.))-1e-9:
                    self.target_yaw=yaw
                else:
                    difference=self.target_yaw-yaw
                    error=math.atan2(math.sin(difference),math.cos(difference))
                    out[2]=np.clip(float(self.settings['walk_heading_gain'])*error,-.3,.3)
                self.walk_idle_elapsed+=dt
        if skill=='walking':
            scale=float(self.settings.get('walk_translation_scale',1.))
            # Calibrate the actor's translational command; scoring retains the
            # user's requested velocity. Match GDScript double multiplication.
            out[0]=float(out[0])*scale
            out[1]=float(out[1])*scale
        braking=skill=='roller' and out[0]<0
        if braking:
            if not self.was_braking:self.brake_elapsed=0.
            heading_gain=float(self.settings.get('brake_heading_gain',0.))
            if heading_gain>0.:
                if not self.was_braking or abs(float(out[2]))>.05:self.brake_target_yaw=yaw
                if abs(float(out[2]))<=.05:
                    difference=self.brake_target_yaw-yaw
                    out[2]=np.clip(heading_gain*math.atan2(math.sin(difference),math.cos(difference)),-1.,1.)
            gain=float(self.settings.get('brake_velocity_gain',0.))
            if gain>0.:
                speed=math.cos(yaw)*float(state.base_linvel[0])+math.sin(yaw)*float(state.base_linvel[1])
                self.brake_speed=speed if not self.was_braking else .8*self.brake_speed+.2*speed
                out[0]=np.clip(-gain*self.brake_speed,-.5,.2)
            pulse=self.settings.get('brake_pulse_s')
            if pulse is not None and self.brake_elapsed>=float(pulse)-1e-9:out[0]=0.
            self.brake_elapsed+=dt
        else:self.brake_elapsed=0.
        self.was_braking=braking
        return out
