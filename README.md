# MicroDuck Sim2Sim

将 MuJoCo 训练的机器人控制策略迁移到 **Godot / Jolt**，比较两种物理引擎中的运动表现。以 MicroDuck 为例，提供场景转换、策略运行、物理校准与训练评估工具。

![MicroDuck：场景行走、刚体交互、前滚与轮足动作](docs/media/microduck-sim2sim.gif)

## 功能

- 从编译后的 MuJoCo 模型生成 Godot 刚体、碰撞和关节。
- 由 Python 运行 ONNX 策略，同步推进 MuJoCo 与 Jolt，采集并对比轨迹。
- 支持站立、行走、踢球、翻滚和轮足等动作，以及键鼠交互。
- 支持 Godot / Jolt 环境中的 PPO 微调与策略评估。

<details>
<summary>完整技能与场景演示</summary>

**技能演示**

![MicroDuck 在 Godot / Jolt 中运行 ONNX 控制策略](docs/media/microduck-service-bay-15s.gif)

**行走与跟随视角**

![MicroDuck 行走与环视](docs/media/distant-scenery-preview.gif)

**轻质刚体交互**

![踢动箱子、小瓶和小球](docs/media/loose-props.gif)

**Godot 场景**

![维修站场景](docs/media/yard-atmosphere.gif)

</details>

## 快速开始

需要 Python 3.12、[uv](https://docs.astral.sh/uv/)、Godot 4.7.2，以及 [microduck_rl](https://github.com/pollen-robotics/microduck_rl) 的机器人资源和 ONNX 权重。

```bash
git clone https://github.com/sgyli7/MicroDuck-Godot-Simi2Sim.git
cd MicroDuck-Godot-Simi2Sim
uv sync

export MICRODUCK_RL=/path/to/microduck_rl
export MICRODUCK_POLICIES=/path/to/onnx_models
export GODOT=/path/to/godot

uv run --no-sync python -m sim2sim.research.setup scenes
uv run --no-sync sim2sim-play
```

机器人网格与模型权重需单独准备。资源配置和模型加载见[运行说明](REPRODUCING.md)。

## 文档

- [物理映射、校准与双后端对比](SIM2SIM.md)
- [环境配置、模型加载与复现](REPRODUCING.md)
- [实验结果](RESEARCH_RESULT_20260910.md)
- [Godot 交互场景](docs/showcase.md)

## 许可

原创代码采用 [Apache-2.0](LICENSE)。第三方软件、机器人资源及其许可见 [NOTICE](NOTICE)。
