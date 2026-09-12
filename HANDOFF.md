# MicroDuck 九技能独立运行与训练 — 交接

> **产品方向已确认（2026-09-11 晚）**：用户明确后续用于游戏开发，运行时继续使用 Godot/Jolt；不采用原生 MuJoCo 替代游戏物理。MuJoCo 仅作离线参照／训练来源。研究报告中的替代物理分支已排除。

> **转弯退出与 GPU 迁移诊断已收尾**：会话 `results/sprint_exit_20260912/`，见[报告](docs/sprint_exit_20260912/RESULT.md)与[高速策略研究](docs/research_microduck_racing_20260912.md)。状态可见/屏蔽的固定 CUDA PPO 消融分别 117/128、124/128，均零倒但操控/速度门未全过，**不晋级**；默认模型、Jolt 物理和前轮试验包不变。两组更新确实用 GPU，但采样占采样加更新时间的 99.18%；用户指出后停止追加同类 CPU 采样训练。外部高速 donor 数值契约通过而 Jolt 回放全部跌倒；实际 GPU 短时物理对照表明转子映射可降低响应误差，接触/状态投影和横向命令覆盖仍待校准。下一轮优先对齐训练域资产、物理和真实控制课程，再投入 GPU 并行学习；不是继续盲调奖励或更换游戏物理。H6 动作子空间只存原型、未训练；完整旧模型 128 条新种子对照未运行。开发 923000–923015 已用，最终 **924000–924049 未用**。本轮任务已退出，六份相同 staging 副本已删除，精确工时和资源审计见机器结果；恢复前先读账本，不重复启动。完整用户目标仍未完成，OS 键盘验收仍未核定。

> **联合行走/加速续轮已收尾**。会话 `results/sprint_joint_20260912/`，两小时上限、最后30分钟收尾。S05共用普通/加速actor、0.20m路径前视，新最终种子922000–922029为 **231/240**，同种子旧包211/240；双方零跌倒，但候选丢失9个旧成功（转弯退出侧移），**不晋级**。长直行0.2256/普通0.1932m/s（+16.8%），普通速度各组未下降。实际包30分钟场内连续操作通过，默认模型与冻结Jolt物理未改。试验包`dist/MicroDuck-ARM64-20260912-joint-trial.tar.gz`及18秒演示已核验；全量272项测试中270通过、2可选跳过，本轮进程/容器已退出，临时重复副本已清理；有效工时98.145分钟，账本actors为空。宿主锁屏，OS键盘仍待验收。见[续轮报告](docs/sprint_joint_20260912/RESULT.md)。保留集已消耗，不再用于调参；恢复先查账本和活动进程。

> **行走加速本轮已收尾（2026-09-12）**：[最终报告](docs/sprint_20260912/RESULT.md)、[机器记录](docs/sprint_20260912/RESULT.json)。左 Shift+W 仅行走，兼容 A/D；原生 ARM64 试验包 `dist/MicroDuck-ARM64-20260912-sprint-trial.tar.gz` 无需 Python/TCP。S05 最终 **206/240** 完整通过，长直行 **0.2196 vs 0.1669 m/s（+31.6%）**。短最终集零跌倒，但单向长测在 **222.48 秒走出地板边缘**；续轮四秒位置对照已确认场景边界原因，场内长期稳定仍需验证，**未晋级**。
> 原生十模型误差最大 1.42e-14；禁网 Ubuntu ARM64、30/144 FPS 控制对照、复位/换机器人验证完成。普通九技能与旧候选包配对未丢失旧成功；包内九模型沿用 9/11 候选包，默认工作区另有既存轮滑版本差异，均未覆盖，见报告。宿主锁屏，OS 键盘验收仍未完成。全量测试 264 通过、2 可选跳过。
> 训练已停止，本轮监督器、Godot、容器、临时工程均已清理；恢复前核对账本，不要重复启动已完成实验。会话 `results/sprint_20260912/` 从 11:49 开工、四小时上限，实际约 3 小时 15 分，精确工时见账本。最终种子 **920000–920029 已消耗**，禁止继续调参。真实控制组成训练 s08 52/64 优于同预算 s09 46/64，但较慢，未胜过 s05；下一轮优先考虑普通/加速/停止联合策略和教师保留，继续冻结游戏 Jolt 物理。

> **固定 Jolt 路线试验已收尾（2026-09-12）**：[最终报告](docs/jolt_learning_20260911/RESULT.md)。27 个正式终点、约 1494 万 PPO 转移完成；唯一开发候选在新保留集为 **183/210，与旧包持平且丢失 5 个旧成功**，因此拒绝晋级，旧模型和旧包不变。没有证明达到原版 MuJoCo 水准。
> 新最终种子 `918000–918029` 已消耗，不能继续用于调参。会话、checkpoint、失败尝试和封存工时在 `results/jolt_learning_20260911/`；下一轮另建会话。试验包 `dist/MicroDuck-ARM64-20260912-learning-trial.tar.gz` 明确标记未晋级。
> **资源与运行验证**：CUDA 学习／CPU→CUDA 续训已通过，当前完整流水线未测到 GPU 加速，约 97% 时间在采样阶段（含 Jolt、Python、TCP、CPU ORT），还需进一步拆分。新长任务使用 `research.budget start`，默认四核、nice 10、独立监督和退出清理；修复提交 `84fd55f`。[资源报告](docs/jolt_learning_20260911/GPU_AND_RESOURCES.md)。收尾已确认本轮进程和容器无残留，不要因目录存在而重复启动已完成试验。
> 九模型真实／随机／边界数值门禁、干净 ARM64 包与连续运行检查通过；真实键盘／不同帧率本轮未重新验收。旧快照的 release 扩展曾过旧，现打包器强制测试实际导出文件，详见报告中的失败与恢复记录。

> **下一轮开工前研究（2026-09-11）**：[训练方向报告](docs/research_20260911_training_direction.md)。
> 已审计训练日志并核对近期论文／工业部署；发现轮滑转向命令与奖励朝向冲突，建议先对齐任务、建立 MuJoCo/Jolt 同任务参照，再比较状态输入、控制权限、参考运动与 FastSAC。
> 这是本轮开工前的研究记录；后续试验以上方新会话为准。下方上一轮冻结结果保持原样。

> 本轮已冻结九技能 Linux ARM64 原生候选：C++ GDExtension + ORT 1.29.0，运行无需 Python/TCP。
> 最终制动由 0/210 提升至 180/210，零跌倒，但仍未通过全部硬门槛；候选不覆盖旧模型。
> 结果与后续训练方向见 [2026-09-11 报告](docs/overnight_20260911/RESULT.md)，
> 构建／操作见 [STANDALONE.md](STANDALONE.md)，证据与复现见 [本轮复现说明](docs/overnight_20260911/REPRODUCE.md)。
> 本轮已完成冻结候选的验收，保留种子已使用，不得在该会话继续挑参数；下一轮另建会话。以下保留历史结论。
>
> 历史交接：下文保留本轮研究开始前已对齐的需求和当时事实，不改写旧实验结论。
> 2026-09-10 后的物理修复、walking 扩展、技能契约与最终模型选择见
> [RESEARCH_RESULT_20260910.md](RESEARCH_RESULT_20260910.md)。
> 新机器初始化和模型包复测见 [REPRODUCING.md](REPRODUCING.md)。

给接手者。本文只写**已经对齐过的需求**、仓库里**已经存在的门禁**、以及**落地事实**。不引入新的验收标准。

## 对齐需求（本轮 goal 原文）

在当前 MicroDuck-Godot-Sim2Sim 上，把除 Walking 以外的 8 个 demo 策略走完与 `Walk_Godot` 同类的 Godot/Jolt 继续训练闭环，并交付可用 ONNX。

技能与 factory ONNX：

- standing ← `alpha_stand.onnx`
- sitstand ← `alpha_sitstand.onnx`
- ground_pick ← `alpha_ground_pick.onnx`
- kick_left ← `ball_kick_left.onnx`
- kick_right ← `ball_kick_right.onnx`
- roulade ← `roulade.onnx`
- roller ← `roller.onnx`
- roller_crouch ← `roller_crouch.onnx`

对每个技能：

1. 勘察 ONNX 维度 / 架构 / 命令语义与 play 切换路径；能复用 `GodotVecEnv`+PPO 的复用，不能的单独设计并实现。
2. 从对应 ONNX 恢复 actor，一致性校验 **&lt;1e-5**。
3. 在真实 Jolt 上采样训练、保存、续训、导出 `policies/<Name>_Godot.onnx` + schema-2 sidecar。
4. `sim2sim-play` 默认加载这些 Godot 细调模型（walking 仍用 `Walk_Godot`；轮滑切换加载 `Roller_Godot`）。
5. 每个技能在 Godot 上对原模型做独立 A/B **或** 任务完成评测，**区分闭环跑通与效果改善**。
6. 不用隐藏辅助力 / 运动学替代控制；轮滑用轮滑 XML。

交付：可重复入口、日志、`SIM2SIM.md` 更新、分阶提交。

Walking 不在本轮范围内（已有 `Walk_Godot.onnx`）。官方 demo 策略来自 MuJoCo / `microduck_rl` 导出的 factory ONNX，不是本仓自创任务定义。

## 仓库里已有的门禁（不是本轮编的）

这些是代码和 `SIM2SIM.md` 里已经写死的入口。它们**不是**「Godot 细调比 factory 更好」的技能质量协议。

| 入口 | 测什么 |
|---|---|
| `./run.sh` | convert → spikes → calib → MuJoCo/Godot lockstep compare（走 sim2sim 映射，不是 8 技能微调质量） |
| `./run_kick_gate.sh` / `sim2sim-kick-gate` | **同一条** factory `ball_kick_*.onnx`：MuJoCo vs Godot。Godot 倒、MuJoCo 站 → 默认 **KNOWN_FAIL**（plant-foot）。不进 `./run.sh` HARD FAIL |
| `tests/test_roller_sim.py` | **factory** `roller.onnx`，Godot，`vel=[0.5,0,0]`，4 s，要求 `xy > 0.8` m 且不掉到 `z<0.05` |
| `sim2sim-eval-walk` | 只覆盖 **walking**（上一轮写入）。12 条件、`vel_err_1s` / `yaw_err_1s`。`Walk_Godot` 已记 **mixed**。不是这 8 个技能的验收 |

原版 MicroDuck 技能好不好：MuJoCo `infer_policy` 和真机上的任务（坐下站起、捡地、踢球、前滚站回、轮滑能滑）。本仓没有另一套官方「8 技能任务完成率」数字协议。

## 本轮实际落地（事实，不宣称达标）

分支：`sim2sim` 嵌套仓 `feat/godot-walk-finetune`。权重 gitignore，不入库。

### 勘察（需求 1）

8 个 factory ONNX 均为 obs 61 / action 14、MLP 512-256-128 ELU，与 walking 同架构，训练复用 `GodotVecEnv` + rsl_rl PPO。差异在命令与机器人 XML：

| play 槽 | factory | play 命令（`PlayBrain.command_13`） | 机器人 JSON |
|---|---|---|---|
| standing | `alpha_stand.onnx` | 全 0 | `robots/microduck.json` |
| sitstand | `alpha_sitstand.onnx` | cmd[0]=sit flag | 同上 |
| ground_pick | `alpha_ground_pick.onnx` | `(cos 2πφ, sin 2πφ)`，周期 4 s | 同上 |
| kick_left / kick_right | `ball_kick_*.onnx` | 全 0 | 同上 |
| roulade | `roulade.onnx` | 全 0 | 同上 |
| roller（`--roller` 的 walking 槽） | `roller.onnx` | twist，无侧移 | `robots/microduck_roller.json` → `scene_rollers.xml` |
| roller_crouch（`--roller` 的 standing 槽） | `roller_crouch.onnx` | 全 0 | 同上 |

配置：`configs/{stand,sitstand,pick,kick_left,kick_right,roulade,roller,roller_crouch}_godot.yaml`。

### 导出文件（需求 3、4 的产物）

路径：`$MICRODUCK_POLICIES`（默认 `~/Projects/MicroDuck/policies/`），并 copy 到 `sim2sim/policies/`。sidecar `schema_version: 2`。

| 文件 | yaml `export.slot` | checkpoint（manifest） |
|---|---|---|
| `Stand_Godot.onnx` | stand | 999 |
| `Sitstand_Godot.onnx` | sitstand | 1199 |
| `GroundPick_Godot.onnx` | ground_pick | 999 |
| `KickLeft_Godot.onnx` | kick_left | 1199 |
| `KickRight_Godot.onnx` | kick_right | 1199 |
| `Roulade_Godot.onnx` | roulade | 1199 |
| `Roller_Godot.onnx` | roller | 1499 |
| `RollerCrouch_Godot.onnx` | stand | 999 |

`play.py`：有 `*_Godot.onnx` 则优先于 factory。walking 默认 `Walk_Godot.onnx`。`--roller` 时 walking=`Roller_Godot.onnx`、standing=`RollerCrouch_Godot.onnx`。

### ONNX→actor 误差（需求 2：对齐的是 &lt;1e-5）

日志：`logs/train_skills_godot.out`。代码里 `PARITY_FAIL_ABS = 2e-4`（`onnx_import.py`）是本轮改过的阈值，**不是** goal 里的 &lt;1e-5。

终局 `init_check` / `parity_max_abs_err`（对 1e-5）：

| 模型 | 约 max_abs | 相对 &lt;1e-5 |
|---|---|---|
| Sitstand / GroundPick / Roulade / RollerCrouch | ~8e-6–1.1e-5 | 贴边或略超 |
| Stand / KickLeft / KickRight | ~1.5e-5–2.7e-5 | 未满足 |
| Roller | ~1.5e-4 | 未满足 |

### 训练入口与日志（交付项）

```bash
cd sim2sim
./scripts/train_skills_godot.sh          # 已有导出则跳过
FORCE=1 SKILL=stand ./scripts/train_skills_godot.sh
```

- 包装脚本 `exec` `.venv/bin/sim2sim-train`。不要 kill `uv run` 包装进程。
- 不要在训完后再 `uv sync`（会卸 `[train]` extra）。用 `uv run --no-sync`。
- 日志：`logs/train_skills_godot.out`；各技能 `logs/<name>_godot/`。
- 轮滑首次会因 walking 脚踝名 `ankle_left` 在 `scene_rollers.xml`（`ankle_l_v1` / `tire`）上 KeyError。`HomePoseSampler.support_body_pair` 已认轮滑体。需求 6：roller yaml 用 `microduck_roller.json`。

### 需求 5

对齐的是：对原模型做独立 A/B **或** 任务完成评测，并**分开说**闭环 vs 效果。

本轮**没有**用上面「仓库里已有的门禁」或 MuJoCo `infer_policy` / 真机任务来判定这 8 个 `*_Godot.onnx` 的效果。

`src/sim2sim/train/eval_skill.py`（`sim2sim-eval-skill`）是本轮加的脚本，指标自拟，**不是对齐需求，不能当验收**。

窗口 `sim2sim-play` 加载默认 `*_Godot.onnx` 后，人工看技能相对 factory **是退化**。该项不是新标准，只是需求 5「效果」一侧尚未用对齐协议测过、且默认 play 观感差。

对照 factory（不改代码的做法）：暂时移走或改名 `policies/*_Godot.onnx`，play 会回退 factory；或 `--walking` 指定 `alpha_walking.onnx`。轮滑：`sim2sim-play --roller` 在没有 `Roller_Godot.onnx` 时用 `roller.onnx`。

## 明确不是需求的东西（不要沿用）

- 用 pose_err、max_foot_z、approach_min_z、`vx=0.3` 位移、`|ωy|` 积分判定技能「改善」。
- 「Godot 上 factory 踢会倒、细调不倒」当作踢好了。`kick_gate` 比的是**同一条 ONNX** 跨两个仿真；factory 在 Godot 倒是 KNOWN_FAIL。
- 把 `PARITY_FAIL_ABS=2e-4` 写成验收阈值。
- 无球踢球、无嘴/物体的「捡地」、用摔倒终止关断冒充任务完成。

## 窗口怎么开

```bash
export PATH="$HOME/.local/bin:$PATH"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
export SIM2SIM_DISPLAY_DRIVER=x11
export MICRODUCK_POLICIES="${MICRODUCK_POLICIES:-$HOME/Projects/MicroDuck/policies}"
cd sim2sim
uv run --no-sync sim2sim-play
# 轮滑：uv run --no-sync sim2sim-play --roller
```

点 Godot 窗口：`W` 走，`1` 捡地，`2` 坐下，`3`/`4` 踢，`5` 前滚，`6` 切轮滑，`0` 重置，`Esc` 退出。
