"""Independent physical task evaluation. This module does not import rewards.

Protocol v1: task outcomes, trajectory diagnostics and source-simulator gaps.
Development seeds 100..; held-out seeds 1000.. . All rollouts run to their full
duration without automatic fall resets, including failed rolls and kicks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from .tasks import TASKS,SESSION,BASELINE,DT,conditions,command
from .world import World
from .models import NativeAnchor

PROTOCOL_VERSION="physical_tasks_v3"

def window_mean(x,n=50):
    x=np.asarray(x)
    if len(x)<n:return x.mean(axis=0,keepdims=True)
    cs=np.concatenate([np.zeros_like(x[:1]),np.cumsum(x,axis=0)],axis=0)
    return (cs[n:]-cs[:-n])/n

def summarize(task,rows,heading):
    a={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    final=slice(max(0,len(rows)-25),len(rows))
    tilt=a["tilt"];up=a["up"];z=a["z"];t=a["time"]
    grounded=a["contact"].sum(axis=1)>0
    stable=(tilt<15)&(z>.08)&grounded
    stand_final=bool(np.mean(stable[final])>=.9)
    failed=(z<.045)|(tilt>70)
    yaw=np.unwrap(a["yaw"])
    active=t>=1.
    err=window_mean(a["vel"][active,:2]-a["cmd"][active,:2])
    yawerr=window_mean((a["gyro"][active,2]-a["cmd"][active,2])[:,None])
    velocity_error=float(np.sqrt(np.mean(np.sum(err**2,axis=1))))
    yaw_error=float(np.sqrt(np.mean(yawerr**2)))
    out={"seconds":float(t[-1]),"fell":bool(failed.any()),"final_standing":stand_final,
         "final_z":float(z[-1]),"final_tilt":float(tilt[-1]),"max_tilt":float(tilt.max()),
         "vel_err_1s":velocity_error,"yaw_err_1s":yaw_error,
         "yaw_drift_deg":float(np.degrees(abs(yaw[-1]-yaw[0]))),
         "displacement":float(np.linalg.norm(a["xy"][-1]-a["xy"][0])),
         "mean_vx":float(a["vel"][active,0].mean()),"mean_wz":float(a["gyro"][active,2].mean()),
         "action_rate":float(np.mean(np.sum(np.diff(a["actions"],axis=0)**2,axis=1))),
         "head_contact_fraction":float(a["head_contact"].mean())}
    locomotion=float(np.exp(-velocity_error/.2-yaw_error/.5))*float((~failed).mean())
    if task.name in ("walking","roller"):
        out["success"]=bool(not failed.any() and velocity_error<.10 and yaw_error<.15)
        out["score"]=locomotion
    elif task.name=="standing":
        out["success"]=bool(not failed.any() and stand_final and out["displacement"]<.05 and out["yaw_drift_deg"]<5)
        out["score"]=float((~failed).mean())*float(np.exp(-out["displacement"]/.05-out["yaw_drift_deg"]/10))
    elif task.name=="sitstand":
        sit=a["cmd"][:,0]>.5
        target=np.where(sit,.06,.115)
        rest=((t>3)&(t<6))|(t>9)
        ok=(abs(z-target)<.015)&(tilt<15)
        fraction=float(ok[rest].mean())
        out.update(posture_hold_fraction=fraction,success=bool(fraction>=.9),score=fraction)
    elif task.name=="ground_pick":
        approach=t<task.period*.425
        ix=int(np.argmin(np.where(approach,a["mouth_z"],np.inf)))
        mz=float(a["mouth_z"][ix]);down=float(a["mouth_down"][ix])
        no_impact=not a["head_contact"].any()
        reach=float(np.exp(-max(mz,0)/.05))*max(0.,down)
        out.update(mouth_min_z=mz,mouth_down_at_min=down,head_impact=not no_impact,
                   success=bool(0<=mz<.03 and down>.7 and no_impact and stand_final),
                   score=reach*(.25+.75*float(stand_final))*float(no_impact))
    elif task.name.startswith("kick_"):
        disp=a["ball_pos"][-1,:2]-a["ball_pos"][0,:2]
        forward=float(disp@heading);side=float(disp@np.array([-heading[1],heading[0]]))
        peak=float(np.max(a["ball_vel"][:,:2]@heading))
        touch=bool(a["correct_kick"].any()); wrong=bool(a["wrong_kick"].any())
        directional=forward>abs(side)
        out.update(ball_forward=forward,ball_side=side,ball_peak_forward_speed=peak,
                   correct_foot_contact=touch,wrong_foot_contact=wrong,
                   success=bool(touch and not wrong and forward>.05 and peak>.15 and directional and stand_final),
                   score=float(touch and not wrong)*float(np.clip(forward/.25,0,1))*(.2+.8*float(stand_final))*float(directional))
    elif task.name=="roulade":
        # Ordered orientation and ground-contact events, independent of the
        # reward accumulator: head-top pivot -> inverted trunk -> feet again.
        head_top=a["head_contact"]&(a["head_up"]<-.3)
        sagittal=abs(a["lateral_z"])<.5
        pivot=np.flatnonzero(head_top&sagittal)
        inverted=np.flatnonzero((up<-.7)&sagittal)
        ordered=bool(len(pivot) and len(inverted) and inverted[-1]>=pivot[0])
        net=np.cumsum(a["gyro"][:,1]*DT)
        frontier=np.maximum.accumulate(np.maximum(net,0))
        new_forward=np.diff(np.r_[0.,frontier])
        fwd=float(np.sum(new_forward*a["supported"]*sagittal))
        continuous=float(np.mean(a["supported"]))
        out.update(head_top_pivot=bool(len(pivot)),inverted_trunk=bool(len(inverted)),
                   ordered_roll_events=ordered,supported_fraction=continuous,
                   supported_forward_rotation=fwd,
                   success=bool(ordered and fwd>4.5 and stand_final),
                   score=(.2*bool(len(pivot))+.2*ordered+.6*(ordered and stand_final))*min(1.,continuous/.8))
    elif task.name=="roller_crouch":
        down=(t>1)&(t<2.5); low=float(np.mean(z[down]))
        upright_down=float(np.mean(tilt[down]<45))
        out.update(crouch_z=low,glide_displacement=out["displacement"],
                   success=bool(low<.08 and upright_down>.9 and stand_final and out["displacement"]>.05),
                   score=float(np.clip((.12-low)/.06,0,1))*upright_down*(.2+.8*float(stand_final)))
    return out

def record(w,action):
    f=w.features; task=w.task
    foot="ankle_left" if task.foot==0 else "ankle_right"
    other="ankle_right" if task.foot==0 else "ankle_left"
    return {"time":w.t,"z":f["z"],"xy":f["xy"].copy(),"up":f["up"],"tilt":f["tilt"],
            "yaw":f["yaw"],"vel":f["vel"].copy(),"gyro":f["gyro"].copy(),"contact":f["contact"].copy(),
            "cmd":w.executed_command.copy(),"actions":np.array(action),"q":w.state.q.copy(),
            "mouth_z":float(f["mouth_pos"][2]),"mouth_down":f["mouth_down"],
            "head_contact":f["head_contact"],"head_up":f["head_up"],"supported":f["supported"],
            "lateral_z":f["rot"][2,1],"ball_pos":f["ball_pos"].copy(),"ball_vel":f["ball_vel"].copy(),
            "correct_kick":foot in f["kick_contacts"],"wrong_kick":other in f["kick_contacts"]}

def episode(skill,onnx,backend="godot",seed=100,condition="default",save_trace=None,noise_std=0.,headless=True):
    task=TASKS[skill];policy=NativeAnchor(onnx)
    w=World(task,backend,headless=headless)
    rows=[];observations=[];actions=[];rng=np.random.default_rng(seed+123456)
    start=time.monotonic()
    try:
        obs=w.reset(seed,condition)
        for k in range(round(task.seconds/DT)):
            action=policy(obs[None])[0]
            if noise_std:action=action+rng.normal(0,noise_std,14).astype(np.float32)
            observations.append(obs.copy());actions.append(action.copy())
            if condition=="walk_push" and k==200:w.nudge(np.array([.15,-.12,0]))
            obs=w.step(action);rows.append(record(w,action))
        result=summarize(task,rows,w.heading)
    finally:w.close()
    result.update(skill=skill,backend=backend,seed=int(seed),condition=condition,
                  policy=str(Path(onnx).resolve()),sha256=policy.sha256,protocol=PROTOCOL_VERSION,
                  physics=w.physics,
                  noise_std=noise_std,elapsed_s=time.monotonic()-start)
    if save_trace:
        p=Path(save_trace);p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,obs=np.stack(observations),actions=np.stack(actions),**{k:np.asarray([r[k] for r in rows]) for k in rows[0] if k!="actions"})
        result["trace"]=str(p)
    return result

def run_suite(skill,onnx,backend="godot",seeds=(100,101,102),workers=4,out=None,selected_conditions=None,noise_std=0.):
    task=TASKS[skill];conds=selected_conditions or conditions(task)
    jobs=[(c,s) for c in conds for s in seeds]
    results=[]
    if out:Path(out).mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(episode,skill,onnx,backend,s,c,
                            None if out is None else Path(out)/f"{c}_{s}.npz",noise_std):(c,s) for c,s in jobs}
        for f in as_completed(futures):
            c,s=futures[f]
            try:results.append(f.result())
            except Exception as e:results.append({"skill":skill,"condition":c,"seed":s,"backend":backend,"error":repr(e),"success":False,"score":0.})
    results.sort(key=lambda x:(x["condition"],x["seed"]))
    summary={"protocol":PROTOCOL_VERSION,"skill":skill,"onnx":str(onnx),"backend":backend,
             "success_rate":float(np.mean([x["success"] for x in results])),
             "score":float(np.mean([x["score"] for x in results])),"episodes":results,
             "errors":sum("error" in x for x in results)}
    if out:(Path(out)/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument("--skill",choices=list(TASKS));p.add_argument("--onnx",type=Path)
    p.add_argument("--backend",default="godot");p.add_argument("--baseline",action="store_true")
    p.add_argument("--seeds",type=int,default=3);p.add_argument("--seed-start",type=int,default=100)
    p.add_argument("--workers",type=int,default=4);p.add_argument("--out",type=Path);p.add_argument("--noise",type=float,default=0)
    args=p.parse_args()
    if args.baseline:
        for name in ([args.skill] if args.skill else TASKS):
            task=TASKS[name]
            for label,source,backend in [("factory_mujoco",task.source,"mujoco"),("factory_godot",task.source,"godot"),("previous_godot",BASELINE/task.previous,"godot")]:
                out=SESSION/"evaluation_v3"/name/label
                if (out/"summary.json").exists():continue
                r=run_suite(name,source,backend,range(args.seed_start,args.seed_start+args.seeds),args.workers,out)
                print(name,label,'success',r["success_rate"],'score',round(r["score"],4),'errors',r["errors"],flush=True)
    else:
        r=run_suite(args.skill,args.onnx,args.backend,range(args.seed_start,args.seed_start+args.seeds),args.workers,args.out,noise_std=args.noise)
        print(json.dumps({k:v for k,v in r.items() if k!="episodes"},indent=2))

if __name__=="__main__":main()
