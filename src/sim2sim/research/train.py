"""Deadline-aware PPO experiments with exact native anchors and honest exports."""
import argparse,copy,hashlib,json,math,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from .tasks import TASKS,SESSION,conditions,DT
from .world import World
from .models import Policy,Critic,export_policy,parity
from .rewards import Objective,EXTRA_DIM
from .evaluate import run_suite,PROTOCOL_VERSION
from .mirror import OBS_PERM,OBS_SIGN,JOINT_PERM,JOINT_SIGN
from sim2sim.obs import build_obs

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def reference_observations(task):
    """Native MuJoCo states, with failed full trajectories excluded.

    Locomotion references retain stable moving trajectories and idle explicitly;
    no claim that their command tracking satisfies the stricter new protocol.
    """
    root=SESSION/"evaluation_v4"/task.name/"factory_mujoco"
    summary=json.loads((root/"summary.json").read_text())
    handoff=SESSION/"evaluation_v4_standing"/task.name/"factory_mujoco/summary.json"
    if not handoff.exists():handoff=SESSION/"handoff_probe"/task.name/"factory_mujoco/summary.json"
    if handoff.exists():summary["episodes"].extend(json.loads(handoff.read_text())["episodes"])
    selected=[]
    for e in summary["episodes"]:
        ok=e["success"]
        if task.name in ("standing","walking","roller"):
            ok=not e["fell"] and (e["condition"] in ("default","idle","game_seq") or abs(e["mean_vx"])>.05 or abs(e["mean_wz"])>.1)
        if ok:selected.append(np.load(e["trace"])["obs"])
    if not selected:raise RuntimeError(f"No valid source reference trajectories for {task.name}")
    return torch.from_numpy(np.concatenate(selected))

class Vector:
    def __init__(self,task,num_envs,seed,weights=None,training_conditions=None,entry="reset",roll_starts=0.,reward_params=None,random_commands=0.,time_input_s=0.,heading_input=False):
        self.task=task;self.worlds=[];self.objectives=[];self.seed=seed
        self.rng=np.random.default_rng(seed);self.count=0
        self.conditions=training_conditions or conditions(task)
        self.entry=entry;self.entry_counts={"reset":0,"standing":0}
        self.random_commands=random_commands
        if random_commands and task.name not in ("walking","roller"):raise ValueError("Random commands require locomotion")
        self.roll_starts_fraction=roll_starts;self.roll_library=None
        if roll_starts:
            if task.name!="roulade":raise ValueError("--roll-starts requires roulade")
            from .curriculum import RollStarts
            self.roll_library=RollStarts();self.entry_counts["midroll"]=0
        try:
            for i in range(num_envs):
                self.worlds.append(World(task,time_input_s=time_input_s,heading_input=heading_input))
            self.reset_worlds(range(num_envs))
            self.objectives=[Objective(w,weights,reward_params) for w in self.worlds]
        except BaseException:
            self.close();raise

    def next_seed(self):
        self.count+=1;return self.seed*100000+self.count

    def reset_worlds(self,indices):
        warm=[]
        for i in indices:
            w=self.worlds[i];w.reset(self.next_seed(),self.next_condition(i))
            if self.roll_library is not None and self.rng.random()<self.roll_starts_fraction:
                self.roll_library.reset(w,self.rng);self.entry_counts["midroll"]+=1
                continue
            standing=self.entry=="standing" or (self.entry=="mixed" and self.rng.random()<.5)
            if standing:
                teacher,cmd=w.prepare_standing_entry();warm.append((w,teacher,cmd))
            self.entry_counts["standing" if standing else "reset"]+=1
        # Warmups still use real dynamics and source actions, but independent
        # workers advance together instead of serializing fifty round trips.
        for _ in range(50 if warm else 0):
            for w,teacher,cmd in warm:
                obs=build_obs(w.state,w.last,cmd,w.home);w.send(teacher(obs[None])[0])
            for w,_,_ in warm:w.recv()
        for w,_,_ in warm:w.finish_standing_entry()

    def next_condition(self,i=0):
        if self.random_commands and self.rng.random()<self.random_commands:return "random_seq"
        return self.conditions[int(self.rng.integers(len(self.conditions)))]

    def observations(self):
        actor=np.stack([w.obs() for w in self.worlds])
        critic=np.concatenate([actor,np.stack([r.extra() for r in self.objectives])],axis=1)
        return torch.from_numpy(actor),torch.from_numpy(critic)

    def step(self,actions):
        for w,a in zip(self.worlds,actions):w.send(a)
        reward=[];done=[];timeouts=[];terms=[]
        for w,obj in zip(self.worlds,self.objectives):
            w.recv();r,terminal,detail=obj.compute()
            timeout=w.t>=self.task.seconds-1e-6 and not terminal
            reward.append(r);done.append(terminal or timeout)
            # One-shot maneuvers end naturally at their task deadline. Only
            # continuous control uses artificial time-limit bootstrapping.
            timeouts.append(timeout and self.task.name in ("standing","walking","roller","sitstand"));terms.append(detail)
        terminal_obs=self.observations()[1]
        finished=[i for i,d in enumerate(done) if d]
        self.reset_worlds(finished)
        for i in finished:self.objectives[i].reset()
        return torch.tensor(reward),torch.tensor(done),torch.tensor(timeouts),terminal_obs,terms

    def close(self):
        for w in self.worlds:
            try:w.close()
            except Exception:pass

def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}

def save_checkpoint(path,policy,critic,ao,co,iteration,config):
    torch.save({"policy":policy.state_dict(),"critic":critic.state_dict(),"actor_optimizer":ao.state_dict(),"critic_optimizer":co.state_dict(),"iteration":iteration,"config":config,"rng":rng_state(),"factory_sha256":policy.anchor.sha256,"protocol":PROTOCOL_VERSION},path)

def run(args):
    torch.set_num_threads(args.threads);seed_all(args.seed)
    resume_checkpoint=torch.load(args.resume,weights_only=False) if args.resume else None
    if resume_checkpoint is not None:
        recorded_gate=resume_checkpoint['config'].get('time_gate','')
        if not args.time_gate:args.time_gate=recorded_gate
        if args.time_gate!=recorded_gate:raise ValueError('Resume cannot change the actor time gate')
    task=TASKS[args.skill];session=json.loads((SESSION/"session.json").read_text())
    hard_deadline=float(session["deadline_unix"])-60
    start=time.time();deadline=min(hard_deadline,start+args.minutes*60)
    if start>=deadline:raise RuntimeError("Eight-hour experiment window has ended")
    out=SESSION/"runs"/args.name;out.mkdir(parents=True,exist_ok=False)
    source=Path(args.source) if args.source else task.source
    config=vars(args).copy();config.update(start_unix=start,deadline_unix=deadline,source=str(source),protocol=PROTOCOL_VERSION)
    config["code_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    (out/"config.json").write_text(json.dumps(config,indent=2))
    template=Path(args.template) if args.template else source
    config["increment_template_sha256"]=hashlib.sha256(template.read_bytes()).hexdigest()
    time_gate=None if not args.time_gate else tuple(float(x) for x in args.time_gate.split(','))
    policy=Policy(source,args.variant,args.std,args.bound,template=template,time_gate=time_gate)
    policy.task_name=task.name
    critic=Critic(template,EXTRA_DIM,time_input_s=policy.anchor.time_input_s,heading_input=policy.anchor.heading_input)
    obs_sign=OBS_SIGN.copy()
    if policy.anchor.heading_input:obs_sign[50]=1. # Relative-heading cosine is even under reflection.
    actor_parameters=list(policy.delta.net.parameters())+[policy.log_std]
    ao=torch.optim.Adam(actor_parameters,lr=args.actor_lr);co=torch.optim.Adam(critic.parameters(),lr=args.critic_lr)
    initial_iteration=0
    if args.resume:
        ck=resume_checkpoint
        if ck["factory_sha256"]!=policy.anchor.sha256:raise RuntimeError("Resume source mismatch")
        policy.load_state_dict(ck["policy"]);critic.load_state_dict(ck["critic"])
        ao.load_state_dict(ck["actor_optimizer"]);co.load_state_dict(ck["critic_optimizer"])
        initial_iteration=ck["iteration"]
        random.setstate(ck["rng"]["python"]);np.random.set_state(ck["rng"]["numpy"]);torch.set_rng_state(ck["rng"]["torch"])
    references=reference_observations(task) if args.variant=="anchor" else None
    initial_export=export_policy(policy,out/"initial.onnx")
    initial_parity=parity(policy,initial_export,n=1000)
    (out/"initial_parity.json").write_text(json.dumps(initial_parity,indent=2))
    if not initial_parity["passed"]:raise RuntimeError("Initial actor export parity failed")
    weights=json.loads(args.weights)
    training_conditions=None if not args.conditions else args.conditions.split(",")
    env=Vector(task,args.envs,args.seed,weights,training_conditions,args.entry,args.roll_starts,json.loads(args.reward_params),args.random_commands,policy.anchor.time_input_s,policy.anchor.heading_input)
    config["time_input_s"]=policy.anchor.time_input_s
    config["heading_input"]=policy.anchor.heading_input
    eval_entry=args.eval_entry or ("both" if args.entry=="mixed" else args.entry)
    config["physics"]=env.worlds[0].physics
    if env.roll_library is not None:config["roll_starts_sha256"]=env.roll_library.sha256
    if env.objectives[0].motion is not None:config["roll_motion_sha256"]=env.objectives[0].motion.sha256
    if any(w.physics!=config["physics"] for w in env.worlds):raise RuntimeError("Worker physics fingerprints differ")
    config["resume_starts_new_physical_episodes"]=bool(args.resume)
    (out/"config.json").write_text(json.dumps(config,indent=2))
    log=(out/"metrics.jsonl").open("a",buffering=1)
    best_score=-math.inf;best_success=-1.;iteration=initial_iteration;last_eval=time.time();total_samples=0;status="time_limit"
    evaluation_records=[]
    try:
        while time.time()<deadline and (not args.iterations or iteration<initial_iteration+args.iterations):
            if (out/"STOP").exists():status="stopped_for_review";break
            iteration+=1;iteration_start=time.time()
            buffers={k:[] for k in ["obs","critic_obs","anchor","action","logprob","mean","std","value","reward","physical_reward","done"]}
            faults=0;all_terms={};term_count=0
            policy.train()
            with torch.no_grad():
                for step in range(args.steps):
                    if step%25==0 and time.time()>deadline:break
                    obs,cobs=env.observations();anchor=policy.anchor_values(obs)
                    dist=policy.distribution(obs,anchor);action=dist.sample();value=critic(cobs)
                    reward,done,timeouts,terminal_cobs,terms=env.step(action.numpy())
                    physical_reward=reward.clone()
                    # Truncated time limits bootstrap terminal state, never the reset state.
                    reward=reward+args.gamma*critic(terminal_cobs)*timeouts
                    for k,v in [("obs",obs),("critic_obs",cobs),("anchor",anchor),("action",action),("logprob",dist.log_prob(action).sum(-1)),("mean",dist.mean),("std",dist.stddev),("value",value),("reward",reward),("physical_reward",physical_reward),("done",done)]:buffers[k].append(v)
                    for entry in terms:
                        for k,v in entry.items():all_terms[k]=all_terms.get(k,0.)+v
                        term_count+=1
                if not buffers["obs"]:break
                b={k:torch.stack(v) for k,v in buffers.items()}
                T,N=b["value"].shape;total_samples+=T*N
                advantage=torch.zeros_like(b["reward"]);gae=torch.zeros(N)
                next_value=critic(env.observations()[1])
                for s in reversed(range(T)):
                    active=(~b["done"][s]).float()
                    delta=b["reward"][s]+args.gamma*next_value*active-b["value"][s]
                    gae=delta+args.gamma*args.lam*active*gae;advantage[s]=gae;next_value=b["value"][s]
                returns=advantage+b["value"]
                advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
                flat={k:v.flatten(0,1) for k,v in b.items()};advantage=advantage.flatten();returns=returns.flatten()
                if args.symmetry_weight:
                    flat["mirrored_obs"]=flat["obs"][:,OBS_PERM]*torch.from_numpy(obs_sign)
                    flat["mirrored_anchor"]=policy.anchor_values(flat["mirrored_obs"])
            before=copy.deepcopy(policy.state_dict());optbefore=copy.deepcopy(ao.state_dict())
            losses=[];kls=[];stop_actor=False;update_count=0
            for epoch in range(args.epochs):
                for ix in torch.randperm(T*N).split(args.minibatch):
                    pred=critic(flat["critic_obs"][ix]);closs=(pred-returns[ix]).square().mean()
                    if not torch.isfinite(closs):raise FloatingPointError("nonfinite critic loss")
                    co.zero_grad();closs.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);co.step()
                    if stop_actor or iteration-initial_iteration<=args.critic_warmup:continue
                    dist=policy.distribution(flat["obs"][ix],flat["anchor"][ix])
                    with torch.no_grad():
                        old=torch.distributions.Normal(flat["mean"][ix],flat["std"][ix])
                        kl=torch.distributions.kl_divergence(old,dist).sum(-1).mean()
                    kls.append(float(kl))
                    if float(kl)>2*args.target_kl:stop_actor=True;continue
                    logprob=dist.log_prob(flat["action"][ix]).sum(-1)
                    ratio=(logprob-flat["logprob"][ix]).clamp(-20,20).exp()
                    loss=-torch.minimum(ratio*advantage[ix],ratio.clamp(.8,1.2)*advantage[ix]).mean()
                    if references is not None:
                        ref=references[torch.randint(len(references),(min(args.minibatch,len(references)),))]
                        loss=loss+args.anchor_weight*policy.delta(ref).square().mean()
                    if args.variant=="residual":loss=loss+args.residual_weight*policy.delta(flat["obs"][ix]).square().mean()
                    if args.symmetry_weight:
                        mirrored=policy(flat["mirrored_obs"][ix],flat["mirrored_anchor"][ix])
                        reflected=dist.mean[:,JOINT_PERM]*torch.from_numpy(JOINT_SIGN)
                        loss=loss+args.symmetry_weight*(mirrored-reflected).square().mean()
                    if not torch.isfinite(loss):raise FloatingPointError("nonfinite actor loss")
                    ao.zero_grad();loss.backward()
                    if iteration<=args.freeze_std:policy.log_std.grad=None
                    torch.nn.utils.clip_grad_norm_(actor_parameters,.5);ao.step()
                    with torch.no_grad():policy.log_std.clamp_(np.log(.005),np.log(.3))
                    losses.append(float(loss.detach()));update_count+=1
            with torch.no_grad():
                ix=torch.arange(0,T*N,max(1,T*N//2048))
                dist=policy.distribution(flat["obs"][ix],flat["anchor"][ix])
                old=torch.distributions.Normal(flat["mean"][ix],flat["std"][ix])
                actual_kl=float(torch.distributions.kl_divergence(old,dist).sum(-1).mean())
                prediction=critic(flat["critic_obs"][ix]);truth=returns[ix]
                value_loss=float((prediction-truth).square().mean())
                explained_variance=float(1-(truth-prediction).var()/(truth.var()+1e-8))
                delta_rms=float(policy.delta(flat["obs"][ix]).square().mean().sqrt())
            rejected=not math.isfinite(actual_kl) or actual_kl>.15
            if rejected:
                policy.load_state_dict(before);ao.load_state_dict(optbefore)
                for g in ao.param_groups:g["lr"]*=.5
            elif actual_kl>args.target_kl*1.5:
                for g in ao.param_groups:g["lr"]=max(1e-6,g["lr"]*.8)
            entry={"iteration":iteration,"elapsed":time.time()-start,"samples":total_samples,"reward":float(b["physical_reward"].mean()),"bootstrapped_reward":float(b["reward"].mean()),"reward_logging":"physical_v2","value_loss":value_loss,"explained_variance":explained_variance,"delta_rms":delta_rms,"kl":actual_kl,"actor_lr":ao.param_groups[0]["lr"],"std":float(policy.log_std.detach().exp().mean()),"rejected":rejected,"actor_updates":update_count,"fps":T*N/(time.time()-iteration_start),"done_fraction":float(b["done"].float().mean()),"terms":{k:v/term_count for k,v in all_terms.items()}}
            log.write(json.dumps(entry)+"\n")
            print(json.dumps({k:v for k,v in entry.items() if k!="terms"}),flush=True)
            if iteration%5==0:save_checkpoint(out/"latest.pt",policy,critic,ao,co,iteration,config)
            evaluate_now=time.time()-last_eval>args.eval_seconds or (args.iterations and iteration>=initial_iteration+args.iterations)
            if evaluate_now and time.time()<hard_deadline-30:
                export=export_policy(policy,out/f"iteration_{iteration:05d}.onnx")
                save_checkpoint(out/f"iteration_{iteration:05d}.pt",policy,critic,ao,co,iteration,config)
                # Report export roundoff independently of physical outcomes.
                report=parity(policy,export,n=200)
                selected=None if not args.eval_conditions else args.eval_conditions.split(",")
                result=run_suite(task.name,export,seeds=(100,101,102),workers=4,out=out/f"eval_{iteration:05d}",selected_conditions=selected,entry=eval_entry)
                result_short={k:v for k,v in result.items() if k!="episodes"};result_short.update(iteration=iteration,parity=report)
                evaluation_records.append(result_short)
                (out/"evaluations.json").write_text(json.dumps(evaluation_records,indent=2))
                print("EVALUATION "+json.dumps(result_short),flush=True)
                if report["passed"] and not result["errors"] and (result["success_rate"],result["score"])>(best_success,best_score):
                    best_score=result["score"]
                    best_success=result["success_rate"]
                    save_checkpoint(out/"best.pt",policy,critic,ao,co,iteration,config)
                    (out/"best.onnx").write_bytes(export.read_bytes())
                last_eval=time.time()
            if rejected and ao.param_groups[0]["lr"]<1e-6:status="repeated_kl_rejection";break
        save_checkpoint(out/"latest.pt",policy,critic,ao,co,iteration,config)
        export=export_policy(policy,out/"final.onnx")
        report=parity(policy,export,n=1000)
        (out/"final_parity.json").write_text(json.dumps(report,indent=2))
        if time.time()<hard_deadline-30:
            selected=None if not args.eval_conditions else args.eval_conditions.split(",")
            result=run_suite(task.name,export,seeds=(100,101,102),workers=4,out=out/"eval_final",selected_conditions=selected,entry=eval_entry)
            if report["passed"] and not result["errors"] and (result["success_rate"],result["score"])>(best_success,best_score):
                (out/"best.onnx").write_bytes(export.read_bytes());save_checkpoint(out/"best.pt",policy,critic,ao,co,iteration,config);best_score=result["score"];best_success=result["success_rate"]
        (out/"completed.json").write_text(json.dumps({"status":status,"iterations":iteration,"samples":total_samples,"elapsed":time.time()-start,"best_dev_score":best_score if math.isfinite(best_score) else None,"best_dev_success":best_success,"final_parity":report},indent=2))
    except BaseException as e:
        (out/"error.txt").write_text(traceback.format_exc())
        save_checkpoint(out/"failed.pt",policy,critic,ao,co,iteration,config)
        raise
    finally:env.close();log.close()
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--skill",required=True,choices=list(TASKS));p.add_argument("--name",required=True)
    p.add_argument("--variant",choices=["plain","anchor","residual"],default="residual")
    p.add_argument("--minutes",type=float,default=15);p.add_argument("--iterations",type=int,default=0)
    p.add_argument("--envs",type=int,default=16);p.add_argument("--steps",type=int,default=512)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--threads",type=int,default=2)
    p.add_argument("--epochs",type=int,default=2);p.add_argument("--minibatch",type=int,default=2048)
    p.add_argument("--actor-lr",type=float,default=3e-5);p.add_argument("--critic-lr",type=float,default=3e-4)
    p.add_argument("--std",type=float,default=.03);p.add_argument("--bound",type=float,default=.2)
    p.add_argument("--freeze-std",type=int,default=20);p.add_argument("--critic-warmup",type=int,default=2)
    p.add_argument("--gamma",type=float,default=.99);p.add_argument("--lam",type=float,default=.95)
    p.add_argument("--target-kl",type=float,default=.015);p.add_argument("--anchor-weight",type=float,default=10)
    p.add_argument("--residual-weight",type=float,default=1);p.add_argument("--weights",default="{}")
    p.add_argument("--conditions");p.add_argument("--eval-conditions");p.add_argument("--eval-seconds",type=float,default=300)
    p.add_argument("--source");p.add_argument("--resume");p.add_argument("--template")
    p.add_argument("--entry",choices=["reset","standing","mixed"],default="reset")
    p.add_argument("--eval-entry",choices=["reset","standing","both"])
    p.add_argument("--roll-starts",type=float,default=0.,help="Training-only fraction of source mid-roll resets")
    p.add_argument("--symmetry-weight",type=float,default=0.,help="Bilateral actor consistency loss using the upstream observation/action transform")
    p.add_argument("--reward-params",default="{}",help="Explicit tracking-kernel variances")
    p.add_argument("--random-commands",type=float,default=0.,help="Fraction of training episodes with varied interactive command tapes")
    p.add_argument("--time-gate",default="",help="Optional start,end seconds for a learned increment on a declared time-input actor")
    args=p.parse_args()
    if not 0<=args.roll_starts<=1:p.error("--roll-starts must be in [0,1]")
    if not 0<=args.random_commands<=1:p.error("--random-commands must be in [0,1]")
    out=SESSION/"runs"/args.name;existed_before=out.exists()
    try:print(run(args),flush=True)
    except BaseException:
        if not existed_before and out.is_dir() and not (out/"error.txt").exists():
            (out/"error.txt").write_text(traceback.format_exc())
        raise

if __name__=="__main__":main()
