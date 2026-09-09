"""Exact bilateral policy transform, using the upstream 61D symmetry contract."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,compose

JOINT_PERM=np.array([9,10,11,12,13,5,6,7,8,0,1,2,3,4],np.int64)
JOINT_SIGN=np.array([-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,-1],np.float32)
OBS_PERM=np.r_[np.arange(6),6+JOINT_PERM,20+JOINT_PERM,34+JOINT_PERM,np.arange(48,61)].astype(np.int64)
OBS_SIGN=np.r_[[-1,1,-1],[1,-1,1],JOINT_SIGN,JOINT_SIGN,JOINT_SIGN,[1,-1,-1],[1,1,-1,-1],[1,-1,1,-1,1,-1]].astype(np.float32)

def reflect_obs(x):return np.asarray(x)[...,OBS_PERM]*OBS_SIGN
def reflect_action(x):return np.asarray(x)[...,JOINT_PERM]*JOINT_SIGN

def mirror(source,dest,task="kick_right"):
    original=onnx.load(source);m=compose.add_prefix(original,"mirror_source/")
    input_name=m.graph.input[0].name
    arrays={"obs_perm":OBS_PERM,"obs_sign":OBS_SIGN,"joint_perm":JOINT_PERM,"joint_sign":JOINT_SIGN}
    nodes=[helper.make_node("Gather",["obs","obs_perm"],["permuted_obs"],axis=1),helper.make_node("Mul",["permuted_obs","obs_sign"],[input_name])]
    nodes+=list(m.graph.node)
    nodes+=[helper.make_node("Gather",[m.graph.output[0].name,"joint_perm"],["permuted_action"],axis=1),helper.make_node("Mul",["permuted_action","joint_sign"],["actions"])]
    graph=helper.make_graph(nodes,"bilateral_policy_transform",[helper.make_tensor_value_info("obs",onnx.TensorProto.FLOAT,[1,61])],[helper.make_tensor_value_info("actions",onnx.TensorProto.FLOAT,[1,14])],initializer=list(m.graph.initializer)+[numpy_helper.from_array(v,k) for k,v in arrays.items()])
    result=helper.make_model(graph,opset_imports=list(original.opset_import));result.ir_version=original.ir_version
    for p in original.metadata_props:
        if p.key not in ("sim2sim_transform","sim2sim_task","sim2sim_source_sha256"):result.metadata_props.add(key=p.key,value=p.value)
    result.metadata_props.add(key="sim2sim_transform",value="bilateral_reflection_61d")
    result.metadata_props.add(key="sim2sim_task",value=task)
    result.metadata_props.add(key="sim2sim_source_sha256",value=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    onnx.checker.check_model(result);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);onnx.save(result,str(dest));return dest

def main():
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("dest",type=Path);p.add_argument("--task",default="kick_right")
    a=p.parse_args();print(mirror(a.source,a.dest,a.task))

if __name__=="__main__":main()
