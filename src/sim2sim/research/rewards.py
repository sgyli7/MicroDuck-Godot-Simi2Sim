"""Task-specific training objectives; never used by the physical evaluator."""
import math
import numpy as np
from sim2sim.train.rewards import sit_target_q
from .tasks import DT,CROUCH_STAND,CROUCH_DOWN,crouch_blend

EXTRA_DIM=26

class Objective:
    def __init__(self,w,weights=None,params=None):
        self.w=w;self.weights=weights or {};self.params=params or {};self.reset()
        if set(self.params)-{"velocity_variance","yaw_variance"}:raise ValueError("Unknown reward parameter")
        if any(float(v)<=0 for v in self.params.values()):raise ValueError("Reward variances must be positive")
        self.motion=None
        if any(k.startswith("motion_") for k in self.weights):
            if w.task.name!="roulade" or not getattr(w,"time_input_s",0.):raise ValueError("Motion-reference rewards require an explicit timed roll")
            from .roll_motion import reference
            self.motion=reference()

    def reset(self):
        w=self.w
        self.smooth=np.zeros(3);self.net=0.;self.frontier=0.;self.pivot=False
        self.inverted=False;self.touch=False;self.wrong=False;self.impact=False
        self.ball_best=0.;self.yaw0=w.features["yaw"];self.start_xy=w.features["xy"].copy()
        self.previous_score=0.
        self.was_idle=False
        if getattr(w,"roll_start",None) is not None:
            self.net,self.frontier,pivot,inverted=w.roll_start
            self.pivot=bool(pivot);self.inverted=bool(inverted)

    def extra(self):
        w=self.w;f=w.features;rot=f["rot"]
        relative=rot.T@(f["ball_pos"]-w.state.base_pos) if w.task.name.startswith("kick") else np.zeros(3)
        x=np.r_[f["vel"],f["contact"],f["foot_pos"][:,2]*10,f["z"]*10,
                np.nan_to_num([f["mouth_pos"][2]*10,f["mouth_down"]]),relative*10,
                rot.T@f["ball_vel"],w.t/w.task.seconds,self.net/6.283,self.pivot,self.inverted,
                self.touch,self.wrong,self.impact,self.smooth]
        assert x.shape==(EXTRA_DIM,),x.shape
        return x.astype(np.float32)

    def compute(self):
        w=self.w;f=w.features;t=w.t;task=w.task;name=task.name;cmd=w.executed_command
        self.smooth=.9*self.smooth+.1*np.r_[f["vel"][:2],f["gyro"][2]]
        up=float(np.exp(-((1-f["up"])/.06)**2)) # inverted receives zero
        rate=float(np.sum((w.last-w.old_last)**2))
        calm=float(np.exp(-np.sum(f["gyro"][:2]**2)/4))
        stand=float(np.exp(-((f["z"]-.115)/.03)**2))*up
        pose=float(np.exp(-np.mean((w.state.q-w.home)**2)/.12))
        terms={"action_rate":-.08*rate,"joint_speed":-.00002*float(np.sum(w.state.qd**2))}
        if name in ("standing","walking","roller"):
            target=cmd[:3] if name!="standing" else np.zeros(3)
            v_err=float(np.sum((self.smooth[:2]-target[:2])**2))
            yaw_err=float((self.smooth[2]-target[2])**2)
            terms.update(velocity=4*np.exp(-v_err/self.params.get("velocity_variance",.025)),
                         yaw=3*np.exp(-yaw_err/self.params.get("yaw_variance",.18)),upright=2*up,height=.5*stand)
            idle=np.linalg.norm(target)<.01
            if idle:
                if not self.was_idle:self.yaw0=f["yaw"];self.start_xy=f["xy"].copy()
                dy=math.atan2(math.sin(f["yaw"]-self.yaw0),math.cos(f["yaw"]-self.yaw0))
                terms.update(idle_position=2*np.exp(-float(np.sum((f["xy"]-self.start_xy)**2))/.0025),idle_heading=2*np.exp(-dy*dy/.03))
            self.was_idle=idle
            terms["pose"]=(.3 if not idle else 1.)*pose
            if name=="standing":terms["calm"]=calm
        elif name=="sitstand":
            sitting=cmd[0]>.5;target=sit_target_q(w.home) if sitting else w.home
            z=.06 if sitting else .115
            terms.update(posture=3*np.exp(-np.mean((w.state.q-target)**2)/.06),height=4*np.exp(-((f["z"]-z)/.02)**2),upright=2*up,calm=.5*calm)
        elif name=="ground_pick":
            phase=(t%task.period)/task.period
            if phase<.375:blend=phase/.375
            elif phase<.425:blend=1.
            elif phase<.8:blend=(.8-phase)/.375
            else:blend=0.
            # Source uses phase weights, not a prescribed tip trajectory.
            # Driving a linear target height would oppose its earlier reach.
            return_gate=float(np.clip((phase-.425)/.375,0,1))
            down=max(0.,f["mouth_down"])
            terms.update(tip_proximity=6*blend*np.exp(-(max(0.,f["mouth_pos"][2])/.06)**2),mouth_down=2*blend*down,return_stand=5*return_gate*stand,return_pose=3*return_gate*pose,feet_grounded=2*float(np.mean(f["contact"])))
            self.impact|=f["head_contact"]
            terms["head_impact"]=-15*float(f["head_contact"])
            terms["side_tilt"]=-float(f["rot"][2,1]**2)*3
        elif name.startswith("kick"):
            foot="ankle_left" if task.foot==0 else "ankle_right"
            other="ankle_right" if task.foot==0 else "ankle_left"
            newtouch=foot in f["kick_contacts"] and not self.touch
            self.touch|=foot in f["kick_contacts"];self.wrong|=other in f["kick_contacts"]
            ball_forward=float(f["ball_vel"][:2]@w.heading)
            ball_side=float(f["ball_vel"][:2]@np.array([-w.heading[1],w.heading[0]]))
            maxv=max(0.,min(ball_forward,1.));progress=max(0.,maxv-self.ball_best)
            self.ball_best=max(self.ball_best,maxv)
            terms.update(ball_progress=80*progress,ball_forward=8*maxv*float(self.touch),ball_side=-2*abs(ball_side),touch=15*float(newtouch),upright=2*up,stand=(4 if self.touch else 1)*stand,pose=.3*pose)
            terms["overspeed"]=-4*max(0.,ball_forward-1.)
            terms["wrong_foot"]=-8*float(other in f["kick_contacts"])
            terms["support"]=float(f["contact"][1-task.foot])*up
            if "heading" in self.weights:
                dy=math.atan2(math.sin(f["yaw"]-self.yaw0),math.cos(f["yaw"]-self.yaw0))
                terms["heading"]=-dy*dy
        elif name=="roller_crouch":
            blend=crouch_blend((t%task.period)/task.period)
            target=CROUCH_STAND*(1-blend)+CROUCH_DOWN*blend
            z=.12*(1-blend)+.065*blend
            terms.update(posture=5*np.exp(-np.mean((w.state.q-target)**2)/.15),height=3*np.exp(-((f["z"]-z)/.025)**2),upright=3*up,glide=.5*np.exp(-((f["vel"][0]-.2)/.3)**2))
        elif name=="roulade":
            self.net+=float(f["gyro"][1])*DT
            sagittal=float(np.clip((.866-abs(f["rot"][2,1]))/.366,0,1))
            headvalid=f["head_contact"] and f["head_up"]<-.3 and .35<self.net<2.97
            newpivot=headvalid and not self.pivot;self.pivot|=headvalid
            self.inverted|=self.pivot and f["up"]<-.7
            frontier=min(2*np.pi,max(self.frontier,self.net))
            newangle=max(0.,frontier-self.frontier);self.frontier=frontier
            supported=float(f["supported"])
            headgate=float(self.net<np.pi or self.pivot)
            terms.update(forward_progress=8*min(newangle/DT,5.)/5*sagittal*supported*headgate,pivot=10*float(newpivot),sagittal=1.5*sagittal,support=supported)
            landgate=float(self.inverted and self.net>4.5)
            terms["landing"]=15*landgate*stand*calm
            terms["land_pose"]=3*landgate*pose
            terms["land_bootstrap"]=4*landgate*(max(0.,f["up"])+np.exp(-((f["z"]-.115)/.05)**2))
            terms["stand_tax"]=-30*landgate*max(0.,.115-f["z"])
            terms["overspeed"]=-.1*max(0.,abs(f["gyro"][1])-7.)**2
            terms["stall"]=-2*float(t>2. and not (landgate and stand>.5))
            terms["reverse"]=-2*max(0.,-f["gyro"][1])*float(self.net<5.8)
            if "over_rotation" in self.weights:
                terms["over_rotation"]=-min(1.,(max(0.,self.net-2*np.pi)/np.pi)**2)
            if "land_leg_pose" in self.weights:
                legs=np.r_[0:5,9:14]
                terms["land_leg_pose"]=3*landgate*np.exp(-np.mean((w.state.q[legs]-w.home[legs])**2)/.12)
            if "land_heading" in self.weights:
                targetyaw=math.atan2(w.heading[1],w.heading[0])
                terms["land_heading"]=3*landgate*max(0.,f["up"])*math.cos(f["yaw"]-targetyaw)
            if self.motion is not None:
                ref=self.motion.sample(t+w.time_offset)
                terms["motion_pose"]=10*np.exp(-np.mean((w.state.q-ref["q"])**2)/.15)
                terms["motion_orientation"]=8*np.exp(-np.sum((f["rot"][2,:]-ref["gravity"])**2)/.25)
                terms["motion_height"]=3*np.exp(-((f["z"]-ref["z"])/.04)**2)
                terms["motion_rotation"]=4*np.exp(-((self.net-ref["net"])/.75)**2)
        else:raise ValueError(name)
        # Every trial records any adjusted term weights; the evaluator stays fixed.
        terms={k:float(v)*self.weights.get(k,1.) for k,v in terms.items()}
        reward=DT*sum(terms.values())
        fell=bool(f["z"]<.04 or f["tilt"]>75)
        terminate=fell and name in ("standing","walking","roller","kick_left","kick_right","ground_pick")
        if terminate:reward-=3.
        if not np.isfinite(reward):raise FloatingPointError("nonfinite task reward")
        return float(reward),terminate,terms
