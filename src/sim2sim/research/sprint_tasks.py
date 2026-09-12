"""Versioned walking speed curriculum, using the game's input ramp."""
import numpy as np
from sim2sim.play_input import PlayBrain, TwistLimits


def commands(condition, dt=.02, seconds=10., *, include_selection=False, ordinary_turn_rate=1.5):
    # Explicit cases keep one speed/turn cell auditable across training runs.
    _, speed_text, direction = condition.split('_')
    speed = int(speed_text)/100.
    if not .2 <= speed <= 1.:raise ValueError('Sprint speed outside experiment bounds')
    if direction not in ('straight','left','right','alternate','release','wfirst','turnrelease','repeat'):raise ValueError(direction)
    brain = PlayBrain(has_standing=False,has_sprint=True,
                      lim=TwistLimits(sprint_vmax_x=speed,sprint_vmax_ang=.8,vmax_ang=ordinary_turn_rate))
    tape=[];selection=[]
    for step in range(round(seconds/dt)):
        t=step*dt;held=set()
        if 1. <= t < seconds-2.:
            held={'sprint','fwd'}
            if direction=='left':held.add('left')
            elif direction=='right':held.add('right')
            elif direction=='alternate' and t>=3.:held.add('left' if t<5. else 'right')
            elif direction=='release' and t>=4.:held.discard('sprint')
            elif direction=='wfirst' and (t<3. or t>=6.):held.discard('sprint')
            elif direction=='turnrelease':
                if 2.<=t<6.:held.add('left')
                if t>=4.:held.discard('sprint')
            elif direction=='repeat':
                if 3.<=t<4.:held.discard('sprint')
                elif 6.<=t<7.:held=set()
        out=brain.tick(held,[],dt,press_order=sorted(held))
        tape.append(out.command);selection.append(out.sprint)
    result=np.stack(tape)
    return (result,np.array(selection,bool)) if include_selection else result
