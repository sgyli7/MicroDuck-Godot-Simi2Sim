"""Actual ordinary/sprint policy prefixes for walking handoff training."""
import hashlib
import json
from pathlib import Path

import numpy as np

from sim2sim.play_input import PlayBrain, TwistLimits
from .models import NativeAnchor
from .tasks import DT


class SprintEntryBank:
    def __init__(self,path):
        config=json.loads(Path(path).read_text())
        if config['version']!='sprint_entry_v1':raise ValueError('Unknown sprint entry version')
        self.actors={k:NativeAnchor(config[k]) for k in ['walking','sprint']}
        if any(a.time_input_s or a.heading_input or a.yaw_memory_input or a.state_input or a.task_input
               for a in self.actors.values()):
            raise ValueError('Sprint prefixes require the ordinary 61D walking contract')
        self.hashes={k:a.sha256 for k,a in self.actors.items()}
        self.hashes['config']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        limits=TwistLimits(**config['twist_limits'])
        self.tapes={}
        programs={
            'ordinary_move':[(1,[]),(2,['fwd'])],
            'ordinary_left':[(1,[]),(2,['fwd','left'])],
            'ordinary_right':[(1,[]),(2,['fwd','right'])],
            'repeat_sprint':[(1,[]),(3,['fwd','sprint']),(2,['fwd'])],
            'restart_sprint':[(1,[]),(3,['fwd','sprint']),(3,[])],
        }
        for name,program in programs.items():
            brain=PlayBrain(has_standing=False,has_sprint=True,lim=limits);tape=[]
            for seconds,held in program:
                for _ in range(round(seconds/DT)):
                    out=brain.tick(set(held),[],DT,press_order=held)
                    tape.append(('sprint' if out.sprint else 'walking',out.command.copy()))
            self.tapes[name]=tape
        self.names=list(self.tapes)

    def send(self,w,name,step):
        if step==0:
            if w.motion is None:raise ValueError('Sprint entry requires deployment motion feedback')
            w._sprint_training_tape=w.command_tape.copy()
            w.command_tape=np.stack([command for _,command in self.tapes[name]])
            w._command_stamp=None
        actor,_=self.tapes[name][step]
        w.send(self.actors[actor](w.obs()[None])[0])

    def finish(self,w):
        # Keep actual pose, velocities, previous action and control memory.
        # Skip reset-only initial idle so learning begins at the handoff itself.
        tape=w._sprint_training_tape
        w.command_tape=np.concatenate([tape[50:],np.repeat(tape[-1:],50,axis=0)])
        w.finish_standing_entry(reset_motion=False)
