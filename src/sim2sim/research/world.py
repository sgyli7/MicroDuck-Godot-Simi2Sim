"""Physical backends and task telemetry, independent of training rewards."""
import math,hashlib
from pathlib import Path
import mujoco
import numpy as np

from sim2sim.paths import load_robot_json,sim2sim_root
from sim2sim.obs import build_obs
from sim2sim.coords import quat_wxyz_to_mat, mat_to_quat_wxyz
from sim2sim.backends.godot_backend import GodotBackend, inertial_to_body
from sim2sim.backends.mujoco_backend import XL330_M6_KT
from sim2sim.train.reset_poses import HomePoseSampler
from sim2sim.train.rewards import sit_target_q
from .tasks import DT, command

class World:
    def __init__(self, task, backend="godot", headless=True):
        self.task, self.backend_name = task, backend
        self.cfg = load_robot_json(task.robot_path)
        files={"robot":task.robot_path,"mjcf":Path(self.cfg["mjcf"])}
        if backend=="godot":files.update(server=sim2sim_root()/"godot/physics_server.gd",spec=Path(self.cfg["godot_spec"]))
        self.physics={"backend":backend,"joint_limits":"signed_fresh_reset_v1","dt":DT,"current_limit_a":1.75,
                      "files":{k:{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in files.items()}}
        self.sampler = HomePoseSampler(self.cfg)
        self.mj = self.sampler.mj
        self.home = self.sampler.home
        self.meta = {mujoco.mj_id2name(self.mj.model,mujoco.mjtObj.mjOBJ_BODY,i): i for i in range(1,self.mj.model.nbody)}
        names = ["trunk_base", "jaw_soft", "ball", "ankle_left", "ankle_right", "ankle_l_v1", "ankle_r_v1"]
        names += [n for n in self.meta if n.startswith("tire")]
        self.report_names = [n for n in names if n in self.meta]
        if task.name == "roulade": self.report_names = list(self.meta)
        self.site_id = mujoco.mj_name2id(self.mj.model,mujoco.mjtObj.mjOBJ_SITE,"mouth_tip")
        if task.name == "ground_pick" and self.site_id < 0:
            raise RuntimeError("ground_pick requires the real mouth_tip site")
        self.backend = self.mj if backend == "mujoco" else GodotBackend(Path(self.cfg["godot_spec"]),headless=headless,current_limit_a=1.75,recv_timeout=15)
        if backend == "mujoco":
            limit = 1.75*XL330_M6_KT
            self.mj.model.actuator_forcerange[:] = [-limit,limit]
            self.mj.model.actuator_forcelimited[:] = 1
        self.state = None
        self.features = None
        self.last = np.zeros(14,np.float32)
        self.pending_ball=None

    def reset(self, seed, condition="default", randomize=True, entry_speed=None, phase_start=0., q_override=None):
        self.pending_ball=None
        self.rng = np.random.default_rng(seed)
        self.condition = condition
        self.t = float(phase_start)
        sitting = self.task.name == "sitstand" and condition in ("rise","sit_hold")
        q0 = sit_target_q(self.home) if sitting else q_override
        z = .06 if sitting else None
        yaw = float(self.rng.uniform(-math.pi,math.pi)) if randomize else 0.
        poses,_,_ = self.sampler.sample(self.rng,yaw_range=(yaw,yaw),joint_noise_rad=.015 if randomize else 0.,q_base=q0,z=z)
        d,m = self.mj.data,self.mj.model
        if entry_speed is None: entry_speed = .3 if self.task.name == "roller_crouch" else 0.
        self.heading = np.array([math.cos(yaw),math.sin(yaw)],np.float64)
        if entry_speed:
            d.qvel[self.mj.free_dofadr:self.mj.free_dofadr+2] = self.heading*entry_speed
        if "ball" in self.meta:
            bid=self.meta["ball"]; jid=int(m.body_jntadr[bid]); adr=int(m.jnt_qposadr[jid])
            off=np.array([.09,.042 if self.task.foot==0 else -.042])
            if randomize: off += self.rng.uniform(-.015,.015,2)
            rot=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
            d.qpos[adr:adr+3]=[*((rot@off)+d.qpos[self.mj.free_qposadr:self.mj.free_qposadr+2]),.036]
            d.qpos[adr+3:adr+7]=[1,0,0,0]
        mujoco.mj_forward(m,d)
        poses=self.mj.body_poses_mujoco()
        if self.backend_name == "mujoco":
            self.state=self.mj.reset(qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=self.home)
        else:
            self.state=self.backend.reset(ctrl=self.home,bodies=poses,report_bodies=self.report_names)
        self.last[:] = 0
        # Reset reports contain exact teleported poses; contact is not established yet.
        self.features=self.measure(reset=True)
        self.initial_xy=np.array(self.state.base_pos[:2],copy=True)
        self.initial_ball=self.features["ball_pos"].copy()
        return self.obs()

    def obs(self):
        return build_obs(self.state,self.last,command(self.task,self.t,self.condition),self.home)

    def send(self, action, capture_path=None):
        if not np.isfinite(action).all(): raise FloatingPointError("nonfinite policy action")
        self.executed_command=command(self.task,self.t,self.condition)
        self.old_last=self.last.copy()
        self.last=np.asarray(action,np.float32).copy()
        ctrl=self.home+self.last
        if self.backend_name == "godot": self.backend.send_step(ctrl,n_substeps=4,report="research",capture_path=capture_path,place_ball=self.pending_ball)
        else:
            if self.pending_ball is not None:
                m,d=self.mj.model,self.mj.data;bid=self.meta["ball"];jid=int(m.body_jntadr[bid]);adr=int(m.jnt_qposadr[jid]);vadr=int(m.jnt_dofadr[jid])
                d.qpos[adr:adr+7]=[*self.pending_ball,1,0,0,0];d.qvel[vadr:vadr+6]=0.;mujoco.mj_forward(m,d)
            self.contact_events={n:[] for n in self.report_names}
            for _ in range(4):
                self.state=self.mj.step(ctrl,n_substeps=1)
                for n,b in self._mj_bodies().items():self.contact_events[n].extend(b["contacts"])
        self.pending_ball=None

    def recv(self):
        if self.backend_name == "godot": self.state=self.backend.recv_step()
        self.t += DT
        self.features=self.measure()
        return self.obs()

    def step(self,action):
        self.send(action)
        return self.recv()

    def _mj_bodies(self):
        m,d=self.mj.model,self.mj.data
        reports={}
        for name in self.report_names:
            bid=self.meta[name];vel=np.zeros(6)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,bid,vel,0)
            reports[name]={"pos":d.xpos[bid].copy(),"rot":d.xmat[bid].reshape(3,3).copy(),"linvel":vel[3:].copy(),"ground_contact":False,"contacts":[]}
        for c in d.contact:
            b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            for a,b,geom in [(b1,b2,c.geom1),(b2,b1,c.geom2)]:
                name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,a)
                if name not in reports: continue
                other=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b) or "world"
                reports[name]["contacts"].append({"body":other,"ground":b==0,"shape":mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,geom) or f"geom_{geom}","impulse":0.})
                reports[name]["ground_contact"] |= b==0
        return reports

    def _godot_bodies(self,reset=False):
        raw=(self.state.extra or {}).get("raw",{})
        items=raw.get("body_states")
        if reset and items is None:items=raw.get("dump",[])
        if items is None: raise RuntimeError("research telemetry absent")
        out={}
        for b in items:
            name=b["name"]
            if name not in self.meta or "quat" not in b: continue
            bid=self.meta[name]
            pos,quat=inertial_to_body(np.asarray(b["pos"]),np.asarray(b["quat"]),self.mj.model.body_ipos[bid],self.mj.model.body_iquat[bid])
            out[name]={**b,"pos":pos,"rot":quat_wxyz_to_mat(quat),"linvel":np.asarray(b.get("linvel",[0,0,0])),"ground_contact":bool(b.get("ground_contact",False)),"contacts":b.get("contacts",[])}
        missing=set(self.report_names)-set(out)
        if missing: raise RuntimeError(f"missing task bodies: {missing}")
        return out

    def measure(self,reset=False):
        s=self.state
        b=self._mj_bodies() if self.backend_name=="mujoco" else self._godot_bodies(reset)
        for n,body in b.items():
            body["contact_events"] = [] if reset else (self.contact_events.get(n,[]) if self.backend_name=="mujoco" else body.get("contact_events",body["contacts"]))
        rot=quat_wxyz_to_mat(s.base_quat_wxyz)
        yaw=math.atan2(rot[1,0],rot[0,0]); cy,sy=math.cos(yaw),math.sin(yaw)
        # Evaluate the same trunk inertial-COM velocity in both simulators;
        # MuJoCo data.cvel is expressed at the subtree COM and is not this value.
        linear=np.asarray(s.base_linvel)
        if self.backend_name=="mujoco":
            bv=np.zeros(6);bid=self.mj.base_body_id
            mujoco.mj_objectVelocity(self.mj.model,self.mj.data,mujoco.mjtObj.mjOBJ_BODY,bid,bv,0)
            linear=bv[3:]+np.cross(bv[:3],self.mj.data.xipos[bid]-self.mj.data.xpos[bid])
        vel=np.array([[cy,sy,0],[-sy,cy,0],[0,0,1]])@linear
        supports = [["ankle_left"],["ankle_right"]]
        if self.task.robot == "microduck_roller":
            # Wheel groups resolved by their reset-side positions, not interleaved joint indices.
            supports=[[],[]]
            for name in b:
                if name.startswith("tire"):
                    local_y=float((rot.T@(b[name]["pos"]-s.base_pos))[1])
                    supports[0 if local_y>=0 else 1].append(name)
        contact=np.array([any(b[n]["ground_contact"] for n in group) for group in supports],np.float32)
        foot_pos=np.stack([np.mean([b[n]["pos"] for n in group],axis=0) if group else np.full(3,np.nan) for group in supports])
        foot_vel=np.stack([np.mean([b[n]["linvel"] for n in group],axis=0) if group else np.full(3,np.nan) for group in supports])
        jaw=b.get("jaw_soft")
        mouth_pos=np.full(3,np.nan);mouth_down=np.nan
        if self.site_id>=0:
            sid=self.site_id;bid=int(self.mj.model.site_bodyid[sid]);name=mujoco.mj_id2name(self.mj.model,mujoco.mjtObj.mjOBJ_BODY,bid)
            sb=b[name];mouth_pos=sb["pos"]+sb["rot"]@self.mj.model.site_pos[sid]
            sr=sb["rot"]@quat_wxyz_to_mat(self.mj.model.site_quat[sid]);mouth_down=float(-sr[2,0])
        ball=b.get("ball")
        kick_contacts=[] if ball is None else [c["body"] for c in ball["contact_events"] if not c["ground"]]
        f={"z":float(s.base_pos[2]),"xy":np.asarray(s.base_pos[:2]),"rot":rot,"yaw":yaw,
           "up":float(rot[2,2]),"tilt":float(np.degrees(np.arccos(np.clip(rot[2,2],-1,1)))),
           "vel":vel,"gyro":np.asarray(s.base_angvel_local),"contact":contact,"foot_pos":foot_pos,"foot_vel":foot_vel,
           "mouth_pos":mouth_pos,"mouth_down":mouth_down,"head_contact":False if jaw is None else any(c["ground"] for c in jaw["contact_events"]),
           "head_up":1. if jaw is None else float((jaw["rot"]@np.array([.882,0,.471]))[2]),
           "supported":any(v["ground_contact"] for k,v in b.items() if k!="ball"),
           "ball_pos":np.zeros(3) if ball is None else np.asarray(ball["pos"]),
           "ball_vel":np.zeros(3) if ball is None else np.asarray(ball["linvel"]),
           "kick_contacts":kick_contacts,"bodies":b}
        if not all(np.isfinite(x).all() for x in [s.q,s.qd,s.base_pos,s.base_quat_wxyz,vel,s.base_angvel_local]):
            raise FloatingPointError("nonfinite physical state")
        return f

    def nudge(self,velocity):
        if self.backend_name=="godot":self.backend.nudge(np.asarray(velocity))
        else:
            self.mj.data.qvel[self.mj.free_dofadr:self.mj.free_dofadr+3]+=velocity
            mujoco.mj_forward(self.mj.model,self.mj.data)

    def prepare_standing_entry(self):
        from .models import NativeAnchor
        from .tasks import TASKS
        teacher_name="roller" if self.task.robot=="microduck_roller" else "standing"
        seated=self.task.name=="sitstand" and self.condition in ("rise","sit_hold")
        if seated:teacher_name="sitstand"
        if not hasattr(self,"_entry_teachers"):self._entry_teachers={}
        if teacher_name not in self._entry_teachers:self._entry_teachers[teacher_name]=NativeAnchor(TASKS[teacher_name].source)
        teacher=self._entry_teachers[teacher_name]
        cmd=np.zeros(13,np.float32)
        if seated:cmd[0]=1.
        if self.task.robot=="microduck_roller" and self.task.name=="roller_crouch":cmd[0]=.3
        if "ball" in self.meta:self.pending_ball=[5.,5.,.035]
        return teacher,cmd

    def finish_standing_entry(self):
        self.t=0.
        self.initial_xy=self.features["xy"].copy()
        yaw=self.features["yaw"];self.heading=np.array([math.cos(yaw),math.sin(yaw)])
        if "ball" in self.meta:
            off=np.array([.09,.042 if self.task.foot==0 else -.042])+self.rng.uniform(-.015,.015,2)
            rotation=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
            self.pending_ball=[*(self.initial_xy+rotation@off),.035]
            self.initial_ball=np.asarray(self.pending_ball)
            self.features["ball_pos"]=self.initial_ball.copy();self.features["ball_vel"]=np.zeros(3)
        return self.obs()

    def enter_from_standing(self,seconds=1.):
        """A real policy handoff, retaining last_action as upstream play does."""
        teacher,cmd=self.prepare_standing_entry()
        for _ in range(round(seconds/DT)):
            obs=build_obs(self.state,self.last,cmd,self.home)
            self.step(teacher(obs[None])[0])
        return self.finish_standing_entry()

    def close(self):
        if self.backend_name=="godot":self.backend.close()
        self.sampler.close()
