"""Exact frozen ONNX anchor plus a differentiable policy increment.

For full finetuning the increment is MLP(theta)-MLP(theta_initial). This retains
the original runtime's finite-precision forward at initialization, while giving
ordinary full-MLP gradients. It avoids changing the normalization epsilon or
loosening parity to conceal cross-library GEMM roundoff. Export retains the
original ONNX graph and adds the learned increment; no runtime Python is needed.
"""
import copy
import hashlib
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from sim2sim.train.onnx_import import parse_mlp_onnx, _sample_realistic_obs61

class NativeAnchor:
    def __init__(self, path):
        self.path=Path(path)
        self.raw=self.path.read_bytes()
        so=ort.SessionOptions();so.intra_op_num_threads=1;so.inter_op_num_threads=1
        self.session=ort.InferenceSession(self.raw,sess_options=so,providers=["CPUExecutionProvider"])
        self.input=self.session.get_inputs()[0].name
        self.sha256=hashlib.sha256(self.raw).hexdigest()

    def __call__(self, obs):
        obs=np.asarray(obs,np.float32).reshape(-1,61)
        return np.concatenate([self.session.run(None,{self.input:o[None]})[0] for o in obs],axis=0)

class Increment(nn.Module):
    def __init__(self, source, variant="anchor", bound=.2):
        super().__init__()
        rec=parse_mlp_onnx(source)
        self.variant=variant
        self.bound=float(bound)
        self.register_buffer("mean",torch.from_numpy(rec.mean.copy()))
        self.register_buffer("denominator",torch.from_numpy(rec.std.copy()))
        if variant=="residual":
            self.net=nn.Sequential(nn.Linear(61,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,14))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
            self.initial=None
        else:
            layers=[]
            for i,(w,b) in enumerate(rec.layers):
                layer=nn.Linear(w.shape[1],w.shape[0])
                with torch.no_grad():layer.weight.copy_(torch.from_numpy(w.copy()));layer.bias.copy_(torch.from_numpy(b.copy()))
                layers.append(layer)
                if i+1<len(rec.layers):layers.append(nn.ELU())
            self.net=nn.Sequential(*layers)
            self.initial=copy.deepcopy(self.net).requires_grad_(False)
            # Subtract nearby full-network outputs in float64. Otherwise two
            # independent float32 GEMMs can swamp a small learned increment.
            self.double()

    def forward(self, obs):
        x=(obs-self.mean)/self.denominator
        if self.variant=="residual":return self.bound*torch.tanh(self.net(x.clamp(-20,20)))
        return (self.net(x)-self.initial(x)).to(obs.dtype)

class Policy(nn.Module):
    def __init__(self, source, variant="anchor", std=.03, bound=.2, template=None):
        super().__init__()
        self.anchor=NativeAnchor(source)
        self.delta=Increment(template or source,variant,bound)
        self.log_std=nn.Parameter(torch.full((14,),float(np.log(std))))
        self.variant=variant

    def anchor_values(self, obs):
        if isinstance(obs,torch.Tensor):obs=obs.detach().cpu().numpy()
        return torch.from_numpy(self.anchor(obs))

    def forward(self,obs,anchor=None):
        if anchor is None:anchor=self.anchor_values(obs).to(obs.device)
        return anchor+self.delta(obs)

    def distribution(self,obs,anchor=None):
        mean=self(obs,anchor)
        return torch.distributions.Normal(mean,self.log_std.clamp(np.log(.005),np.log(.3)).exp())

    @torch.no_grad()
    def predict(self,obs):
        x=torch.as_tensor(np.asarray(obs,np.float32).reshape(-1,61))
        return self(x).numpy()

class Critic(nn.Module):
    def __init__(self,source,extra_dim):
        super().__init__()
        rec=parse_mlp_onnx(source)
        self.register_buffer("mean",torch.from_numpy(rec.mean.copy()))
        self.register_buffer("denominator",torch.from_numpy(rec.std.copy()))
        self.net=nn.Sequential(nn.Linear(61+extra_dim,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1))

    def forward(self,obs):
        x=torch.cat([((obs[:,:61]-self.mean)/self.denominator).clamp(-20,20),obs[:,61:].clamp(-20,20)],dim=-1)
        return self.net(x).squeeze(-1)

def export_policy(policy,path):
    import onnx
    from onnx import helper,compose
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    delta_path=path.with_suffix(".delta.onnx")
    torch.onnx.export(policy.delta.eval(),(torch.zeros(1,61),),str(delta_path),input_names=["obs"],output_names=["increment"],opset_version=18,dynamo=False)
    original=compose.add_prefix(onnx.load_model_from_string(policy.anchor.raw),"factory/")
    delta=compose.add_prefix(onnx.load(str(delta_path)),"adapt/")
    for model in (original,delta):
        old=model.graph.input[0].name
        for node in model.graph.node:
            for i,name in enumerate(node.input):
                if name==old:node.input[i]="obs"
    nodes=list(original.graph.node)+list(delta.graph.node)
    nodes.append(helper.make_node("Add",[original.graph.output[0].name,delta.graph.output[0].name],["actions"]))
    graph=helper.make_graph(nodes,"factory_plus_adaptation",[helper.make_tensor_value_info("obs",onnx.TensorProto.FLOAT,[1,61])],[helper.make_tensor_value_info("actions",onnx.TensorProto.FLOAT,[1,14])],initializer=list(original.graph.initializer)+list(delta.graph.initializer))
    result=helper.make_model(graph,opset_imports=[helper.make_opsetid("",18)])
    result.ir_version=min(original.ir_version,10)
    for prop in original.metadata_props:
        result.metadata_props.add(key=prop.key,value=prop.value)
    result.metadata_props.add(key="sim2sim_factory_sha256",value=policy.anchor.sha256)
    result.metadata_props.add(key="sim2sim_adaptation",value=policy.variant)
    if getattr(policy,"task_name",None):
        from .tasks import TASKS
        task=TASKS[policy.task_name]
        result.metadata_props.add(key="sim2sim_task",value=task.name)
        result.metadata_props.add(key="sim2sim_command_mode",value=task.mode)
        result.metadata_props.add(key="sim2sim_period_s",value=str(task.period if task.mode=="phase" else 0))
    onnx.checker.check_model(result)
    onnx.save(result,str(path));delta_path.unlink()
    return path

def parity(policy,exported=None,n=10000):
    rng=np.random.default_rng(0)
    reference=policy.anchor if exported is None else NativeAnchor(exported)
    result={}
    for label,obs in [("random",rng.standard_normal((n,61),dtype=np.float32)),("realistic",_sample_realistic_obs61(n,rng))]:
        with torch.no_grad():
            # The deployment contract is batch=1. Batched torch training can
            # have different summation roundoff; report it separately.
            batch=policy.predict(obs)
            single=np.concatenate([policy.predict(x[None]) for x in obs])
        ref=reference(obs)
        result[label+"_max_abs"]=float(np.max(np.abs(single-ref)))
        result[label+"_batch_max_abs"]=float(np.max(np.abs(batch-ref)))
    result["threshold"]=1e-5
    result["passed"]=all(result[k]<1e-5 for k in ["random_max_abs","realistic_max_abs"])
    return result
