# MicroDuck-Godot-Simi2Sim

MuJoCo 训练出的 ONNX policy，在 **Godot 4.7 + 内置 Jolt** 上当第二个物理后端重放，并和 MuJoCo lockstep 对比。

Python 是唯一控制器。编译后的 `MjModel` 是模型真源；ONNX 只在 Python 里跑。这不是「Godot 里嵌 MuJoCo」也不是 Hakoniwa 那种 viewer，而是 **sim2sim**：两端各自步进，能量化 walk / run 是否还站得住。

详细映射、已知不映射项和门禁数字见 [SIM2SIM.md](SIM2SIM.md)。

## 需要

- Godot **4.7.2**（`godot` 在 `PATH`，或 `export GODOT=...`）
- Python 3.12 + [uv](https://docs.astral.sh/uv/)
- [microduck_rl](https://github.com/pollen-robotics/microduck_rl) 的 MJCF（本仓不发布 NC 网格）
- ONNX：官方 `alpha_walking.onnx` 或你自己训的权重（本仓不提交 `.onnx`）

```bash
export MICRODUCK_RL=$HOME/Projects/microduck_rl
export MICRODUCK_POLICIES=$HOME/Projects/MicroDuck/policies   # 或任意放 onnx 的目录
export PATH="$HOME/.local/bin:$PATH"
export DISPLAY="${DISPLAY:-:1}"

cd MicroDuck-Godot-Simi2Sim
uv sync
./run.sh                 # convert → spikes → calib → dual rollout → compare
uv run sim2sim-play      # 窗口；按住 W/↑ 才走
uv run sim2sim-play --local-ppo
uv run sim2sim-play --scene res://scenes/rough_forest_play.tscn
```

崎岖森林地形要先跑 `godot/scripts/setup_forest_vendor.sh`（克隆 [godot-forest-demo](https://github.com/GamesNotDeveloped/godot-forest-demo)，CC BY 4.0，需署名）。默认平地 `main.tscn` 不依赖 vendor。

## 环境变量

| 变量 | 含义 |
|---|---|
| `SIM2SIM_ROOT` | 本仓库根。一般不用设 |
| `MICRODUCK_RL` | `microduck_rl` 检出路径 |
| `MICRODUCK_POLICIES` | ONNX 目录 |
| `GODOT` | Godot 可执行文件，默认 `~/.local/bin/godot` |

`robots/*.json` 里的路径用 `${MICRODUCK_RL}` / `${MICRODUCK_POLICIES}` / `${SIM2SIM_ROOT}`，不要写死本机绝对路径。

## 布局

```
src/mjcf2godot/    MJCF → Godot 场景
src/sim2sim/       双后端、obs、policy、校准、compare、play
godot/             Godot 4.7.2 + Jolt 工程
robots/            机器人 JSON
```

协议是 TCP 行分隔 JSON（MuJoCo 坐标系）。`hello` / `reset` / `step` / `pin` / `set_tau_limit` / `close`。

## 许可

- 本仓库原创代码：**Apache-2.0**（[LICENSE](LICENSE)）
- 第三方与可选森林资源：[NOTICE](NOTICE)
