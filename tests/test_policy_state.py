import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from sim2sim.policy_state import BRAKE_STATE_V1, STATE_KEY, inject_state, state_input, wrap_anchor
from sim2sim.research.models import NativeAnchor


class ResidualStateTests(unittest.TestCase):
    def test_world_velocity_is_projected_into_heading_and_old_contract_is_unchanged(self):
        obs = np.arange(61, dtype=np.float32)
        state = SimpleNamespace(base_quat_wxyz=np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)]),
                                base_linvel=np.array([.2, .6, 9.]), base_pos=np.array([2., 3., .115]))
        actual = inject_state(obs, state, BRAKE_STATE_V1)
        np.testing.assert_array_equal(actual[:58], obs[:58])
        np.testing.assert_allclose(actual[58:], [.6, -.2, .115], atol=1e-7)
        self.assertIs(inject_state(obs, state, ''), obs)
        with self.assertRaises(ValueError): state_input({STATE_KEY: 'unknown'})
        state.base_linvel[0] = np.nan
        with self.assertRaises(ValueError): inject_state(obs, state, BRAKE_STATE_V1)

    def test_anchor_nodes_and_double_tensors_are_identical_and_state_is_masked(self):
        rng = np.random.default_rng(915002)
        nodes = [helper.make_node('Cast', ['obs'], ['precise'], to=TensorProto.DOUBLE),
                 helper.make_node('MatMul', ['precise', 'weights'], ['product']),
                 helper.make_node('Cast', ['product'], ['actions'], to=TensorProto.FLOAT)]
        tensor = numpy_helper.from_array(rng.normal(0, .1, (61, 14)), 'weights')
        graph = helper.make_graph(nodes, 'double_actor',
            [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, 61])],
            [helper.make_tensor_value_info('actions', TensorProto.FLOAT, [1, 14])], [tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'old.onnx', Path(directory)/'new.onnx'
            onnx.save(model, source); wrap_anchor(source, target)
            original, wrapped = NativeAnchor(source), NativeAnchor(target)
            obs = rng.normal(size=(64, 61)).astype(np.float32)
            masked = obs.copy(); masked[:, 58:61] = 0
            np.testing.assert_array_equal(wrapped(obs), original(masked))
            self.assertEqual(wrapped.state_input, BRAKE_STATE_V1)
            emitted = onnx.load(target)
            self.assertEqual([n.SerializeToString() for n in emitted.graph.node[1:]],
                             [n.SerializeToString() for n in model.graph.node])
            self.assertEqual(emitted.graph.initializer[0].SerializeToString(), tensor.SerializeToString())
            with self.assertRaises(ValueError): wrap_anchor(target, source)


if __name__ == '__main__': unittest.main()
