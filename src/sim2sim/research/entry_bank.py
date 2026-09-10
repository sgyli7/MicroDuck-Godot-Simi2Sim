"""Training-only native prefixes using the deployed controller and actor bank.

Every prefix ends before the maneuver clock starts. These are starting-state
distributions, not controller assistance inside an evaluated maneuver.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from sim2sim.play_input import PlayBrain
from sim2sim.play import kick_ball_position
from sim2sim.obs import build_obs
from sim2sim.policy import OnnxPolicy
from .models import NativeAnchor
from .tasks import DT


class EntryBank:
    def __init__(self,path):
        self.path=Path(path);paths=json.loads(self.path.read_text())
        walking=OnnxPolicy(Path(paths['walking']))
        if walking.has_standing_partner:raise ValueError('These prefixes require a walker that owns idle and braking')
        required={'walking','ground_pick','kick_left','kick_right'}
        if paths.get('sitstand'):required.add('sitstand')
        self.actors={k:NativeAnchor(paths[k]) for k in required}
        if any(p.time_input_s for p in self.actors.values()):raise ValueError('Prefix actors must retain ordinary commands')
        self.hashes={k:v.sha256 for k,v in self.actors.items()}
        sidecar=Path(paths['walking']).with_suffix('.manifest.json')
        if sidecar.exists():self.hashes['walking_manifest']=hashlib.sha256(sidecar.read_bytes()).hexdigest()
        idle=lambda s:(s,set(),None)
        programs={'idle':[idle(2.)],
            'walking':[(2.,{'fwd'},None),idle(2.)],
            'turning':[(1.5,{'left'},None),idle(2.)],
            'ground_pick':[(4.,set(),'pick'),idle(2.)],
            'kick_left':[(5.,set(),'kick_left'),idle(2.)],
            'kick_right':[(5.,set(),'kick_right'),idle(2.)],
            'game':[idle(1.),(2.,{'fwd'},None),idle(2.),(1.,{'left'},None),idle(1.),
                    (4.,set(),'pick'),idle(2.),(5.,set(),'kick_left'),idle(2.)]}
        if 'sitstand' in self.actors:
            programs['sit_rise']=[idle(1.),(6.,set(),'sit'),(6.,set(),'sit'),idle(1.)]
        self.tapes={}
        for name,program in programs.items():
            brain=PlayBrain(has_standing=False,lim=walking.twist_limits);tape=[]
            for seconds,held,tap in program:
                for k in range(round(seconds/DT)):
                    out=brain.tick(held,[tap] if tap and k==0 else [],DT)
                    tape.append((out.policy,out.command.copy(),out.started_skill))
            self.tapes[name]=tape
        self.names=list(self.tapes)

    def send(self,w,name,step):
        policy,cmd,started=self.tapes[name][step]
        if started in ('kick_left','kick_right'):w.pending_ball=kick_ball_position(w.state,started)
        obs=build_obs(w.state,w.last,cmd,w.home)
        w.send(self.actors[policy](obs[None])[0])
