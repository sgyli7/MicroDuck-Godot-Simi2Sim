# Shift 加速：来源、训练路线与验收依据

研究日期：2026-09-12。项目审计起点为 `73b9f95`；源训练仓库为 `/home/ethan/Projects/microduck_rl` 的 `5946fd9cdbc58956424420153e51975af3b30d77`。本文只做源码、模型文件与一手文献研究，没有运行训练／仿真、安装依赖或修改模型。源仓库已有 `scripts/infer_policy.py` 工作区改动，未触碰。

**建议把 sprint 作为单独训练、显式选择的运动能力，保留普通模式；先查清已有速度能力，再用“速度×转弯”的成功边界课程学习，并用成功行为回放约束退化。** 加大 W 命令可以作为对照，不能冒称新训练成果。游戏继续使用冻结 Godot/Jolt，最终所有速度和稳定性结论来自实际独立程序。

## 已核实的起点

上一轮 27 个开发终点、约 1494 万个 PPO 转移没有产生可替换模型。成功教师回放比无约束 PPO 更能保留行为，但唯一晋级候选在保留集仍为 183/210，并丢失 5 个旧成功。约 97% 整轮时间在采样链路，CUDA 学习跑通没有带来整轮加速。因此本轮不能把“更多相同 PPO”“GPU 利用率”或单一奖励上升当成进步。[上一轮实测](jolt_learning_20260911/RESULT.md)、[原研究及其条件](research_20260911_training_direction.md)

| 来源／接口 | 核实内容 | 对 sprint 的含义 |
|---|---|---|
| 上游 walking | `vx∈[-0.4,0.4] m/s`，`vy∈[-0.3,0.3] m/s`，`yaw_rate∈[-1,1] rad/s`；固定范围，15% 原地转向桶。源码记录更激进角速课程曾伴随退化 | 超过 0.4 的速度属于训练外探索；不能凭文件名含 run 假定学过高速 |
| 本地默认游戏 walking | W 为 0.3，A/D 为 ±1.5；命令有斜向归一化和加减速斜坡 | 新源 actor 的训练 yaw 范围和默认 A/D 不同；改转向限幅须单独记控制版本 |
| 上游 roller | `cmd_x∈[-0.5,0.6]` 是推进强度，零是滑行、负是刹车；`cmd[2]` 是相对朝向误差 | 正油门 0.6 不是 0.6 m/s；轮滑不能直接复用 walking 的速度损失 |
| 当前 roller 源配方 | heading error 范围 `(0,0)`，训练重点为直线 heading hold | 快速 A/D 转向没有在这份源配方得到训练覆盖，必须独立验证和学习 |

来源：[walking 配方](https://github.com/pollen-robotics/microduck_rl/blob/5946fd9cdbc58956424420153e51975af3b30d77/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)、[roller 配方](https://github.com/pollen-robotics/microduck_rl/blob/5946fd9cdbc58956424420153e51975af3b30d77/src/mjlab_microduck/tasks/microduck_velocity_rollers_env_cfg.py)、[本地键盘输入](../src/sim2sim/play_input.py)、[命令控制](../src/sim2sim/motion_control.py)。表中上游历史失败是源码作者记录，本文未复现其历史运行。

轮滑主奖励实际是 `max(cmd_x,0) * tanh(max(mean_wheel_omega,0)/(vel_scale/wheel_radius))`，当前配方 `vel_scale=0.3`、轮半径默认 0.0175；这是轮速奖励的饱和尺度，既不是目标身体速度，也不是最大车速。源码注释称某个已测 checkpoint 在最大推进时约 0.33 m/s，这不能当成本地模型测量。`RelativeHeadingVelocityCommand` 的旧 docstring 与实现符号不一致：实际代码为 `target_yaw-current_yaw`，正号为左转，应以代码和实际符号回放为准。[MDP 实现](https://github.com/pollen-robotics/microduck_rl/blob/5946fd9cdbc58956424420153e51975af3b30d77/src/mjlab_microduck/tasks/mdp.py)

### 可用的本地 walking 教师候选

目录为 `/home/ethan/Projects/microduck_rl/logs/rsl_rl/local_ppo_velocity/2026-09-02_01-47-57_local_walk_run_idle/`。

| 文件 | SHA256 |
|---|---|
| `2026-09-02_01-47-57_local_walk_run_idle.onnx` | `8e0727f9cf3367adc34ce934ef94a323959dbbf8c1164ca0e3e8972579d78509` |
| `model_2999.pt` | `c148a9e3bbde1c1c981a9576258f1535cd7f5afe4e0e9141f1529c84e792365d` |

保存的 [agent.yaml](/home/ethan/Projects/microduck_rl/logs/rsl_rl/local_ppo_velocity/2026-09-02_01-47-57_local_walk_run_idle/params/agent.yaml) 指定 seed 42、3000 次迭代、每环境 24 步、actor/critic 512/256/128 ELU、PPO、观测归一化。保存的 [env.yaml](/home/ethan/Projects/microduck_rl/logs/rsl_rl/local_ppo_velocity/2026-09-02_01-47-57_local_walk_run_idle/params/env.yaml) 指定 4096 环境、0.005 s 物理、decimation 4；命令范围与上表一致，3–8 秒重采样。这些是配置证据，不单凭配置断言完成样本数或模型质量。

该模型训练来源是 MuJoCo Warp 上的 BAM XL330/m6，`kp_fw=200`、电压随机范围 6.5–8.2 V、动作延迟 3–6 个物理步（15–30 ms）。动作是 `HOME+action`，比例 1.0；14 个主动关节顺序与本项目一致。actor 61D：角速 3、重力 3、关节位置 14、关节速度 14、上一动作 14、命令 13；线速度只给 critic。源训练有噪声、延迟和动力学随机化，不能假定同一 XML 的普通 MuJoCo 路径已经精确复现 BAM 训练环境。现有 [World](../src/sim2sim/research/world.py) 的 `backend=mujoco` 主要复用 XML 和限矩设置，作为“源模型在 XML 下的对照”有用，但不是完整源训练域。

静态读取 ONNX 证实 `obs[1,61]→actions[1,14]`，Sub/Div 归一化、4 个 Gemm、3 个 ELU；十个 initializer 原本都是 FLOAT32。这是不同来源自身的精度，不能写成对既有 DOUBLE 模型降精度。若采用此来源，须如实记录来源变更，并继续保持原生／Python动作误差 `<1e-5`。其 metadata 的 HOME 数字只打印到三位小数，不能拿它覆盖精确机器人资源 HOME。规范导出器把归一化烘焙进图。[源导出实现](https://github.com/pollen-robotics/microduck_rl/blob/5946fd9cdbc58956424420153e51975af3b30d77/scripts/export.py)

本地已有 `.venv`，文件版本为 torch 2.9.1+cu129、mjlab 1.3.0、Warp 1.12.0、MuJoCo Warp 3.8.1、MuJoCo 3.10.0、rsl_rl 5.0.1、BAM 1.0.1。无需先升级框架；是否适合继续预训练，仍须先做受限 GPU smoke、真实吞吐和目标域可迁移性检查。[源依赖锁定规则](https://github.com/pollen-robotics/microduck_rl/blob/5946fd9cdbc58956424420153e51975af3b30d77/pyproject.toml)

## 能直接借鉴的研究机制

**能力驱动的二维命令课程。** Rapid Locomotion 在真机 Mini Cheetah 展示高速直行和转向，官方实现将线速与角速奖励同时达到阈值的格子及相邻格子提高采样权重。本项目可借鉴该扩展规则，但四足的速度数值、Isaac Gym 栈和系统辨识模块不能直接移植到双足 Jolt。把“直线快”和“快速转弯”分别训练再期望自动组合，证据不足。[作者项目](https://agility.csail.mit.edu/)、[官方 curriculum.py](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/main/mini_gym/envs/base/curriculum.py)

**保留访问不到的成功状态。** Fine-tuning RL Models 的研究区分了在固定旧状态上做教师 BC/KL 与只在当前状态上做 kickstarting；前者能覆盖刚开始微调还访问不到的旧成功区域。其主要任务并非本机器人，不能把结论当效果保证；本项目上一轮回放 KL 相对更少退化，是本地支持证据。PPO 应将旧数据用于独立的保留损失，不能将旧动作当作新的 on-policy rollout。显式 sprint 按键还允许冻结普通策略、只训练新增分支，但仍须训练边界进入／退出。[论文](https://arxiv.org/html/2402.02868v2)、[本地结果](jolt_learning_20260911/RESULT.md)

**相位和动作过渡。** Disney BDX 使用速度相关的参考步态和相位，在退出 walking 时等到双支撑，上一动作帮助过渡连续。这为“已能走但加速切换容易倒”提供明确结构，代价是可靠参考和相位契约。应先证实 sprint 的主要失败确实集中在过渡，再增加相位；没有来源运动时，不把整段轨迹硬拟合当捷径。[BDX 论文 V、VII 节](https://arxiv.org/html/2501.05204v1)

**离策略算法与更快采样是第二条路线。** Amazon FAR FastSAC 使用大批量并行仿真和 replay，并给出实物人形部署；低维任务／硬件不同，其训练分钟数不能套给 Jolt。它同时强调简单速度／角速目标、存活、足部和动作平滑约束，罚项随能力增强。本轮若只有数小时，先复用已经存在的 mjlab/PPO 来源或目标域教师保留，只有当相同预算下的路线不学且值函数诊断明确时，再做有界 FastSAC 对照。[论文](https://arxiv.org/html/2512.01996v1)、[官方 Holosoma](https://github.com/amazon-far/holosoma)

2026 年 LIFT 用 SAC 预训练、物理先验世界模型和确定性真实采集做微调；其普通神经世界模型消融会给出不合理身体高度并导致 critic 发散。这支持继续研究数据重用与更新方向，却不支持临时搭建一个未经验证的 Jolt 替身再宣称进步。该论文使用大规模预训练和额外模型，当前预算应优先实现任务能力。[LIFT 论文](https://arxiv.org/html/2601.21363v1)

## 本轮建议的实验顺序

以下是待实验的工程建议，不是已经证明的最优配置。

1. **先有限测来源能力。** 原 walking、当前保留 walking、上列 2999 来源分别在冻结 Jolt 做普通 W、较高 W 命令、W+A、W+D 的相同短／长完整回放，记录实际前速、侧向速度、路径偏离、姿态和跌倒。轮滑以推进强度扫描建立实际速度曲线。来源模型与控制修改分开记账，不扫大量参数。
2. **建立明确 sprint 策略分支。** Shift+W 请求新增能力，普通策略及其他技能先保持冻结；不能通过加快游戏时间、修改执行器、施加外力或直接写速度实现。分支可复用原 actor 初始化，但需要真实训练产物。若靠命令幅度区分 sprint，须确认普通／冲刺命令不会别名、斜坡期间策略选择明确；显式模式不能占用仍有头部／姿态语义的命令槽而不版本化。
3. **收集分桶教师和进入分布。** 对直行、左转、右转、普通→冲刺、冲刺→普通、松 W、转弯中松 Shift 分桶，只将满足各桶标准的来源行为作为教师。完整轨迹保留上一动作和真实可达入口；不把旧失败转向当正确示范。成功数据供 BC/KL 保留与 critic 的实际回报校准使用。当前 [TeacherRetention](../src/sim2sim/research/teacher_retention.py) 硬编码 68D、负油门及三个刹车时间桶，不能直接用它训练 sprint。
4. **按速度与转弯能力扩展。** 从已测稳速起步，速度先以小档位上探；左右转向与过渡从第一阶段就有固定覆盖比例。只在当前格子的开发回放无跌倒、方向合格、速度确实跟上时扩相邻格；留下一定旧格子经验防遗忘。不能因到了第 N 分钟无条件同时提升前速与角速。转弯时的最大可持续前速单独报告，不以直线峰速冒称全操作速度。
5. **奖励与约束对齐。** walking 主项为实际前速和所需角速，直行阶段补累积朝向／侧向偏差约束，转弯时参考方向随意图转动；roller 正油门训练改用可解释的身体前进与可控性目标，避免仅让轮子空转。无跌倒是独立晋级门，不能靠大速度奖励抵消一次跌倒。动作平滑不能强到使“不动”胜出，也不应限制到几个关节再次复制上一轮失败。
6. **短实验先检验更新方向。** 初期固定一组完整成功／失败前缀，测更新前后的真实 Jolt 回报与任务指标；若 critic 优势预测反复把成功行为推坏，先处理值函数／截断／覆盖问题。停止无教师约束的同配方加时，不以降低学习率无限续跑。只有确有增长趋势的路线再投入剩余小时并做独立训练种子复核。

若 Jolt 实际新样本吞吐仍限制能力，而本地 BAM/mjlab 模型在 Jolt 有可接受起点，可以隔离源训练配置，GPU 预训练扩速度，然后每个阶段回到冻结 Jolt 验收与适配。源域收益必须和目标域收益分别报告，不能切换运行时物理。准备和编译时间计入总成本。

## 独立验收和 Auto-Research 记录

最终优先级应是：**所有规定操作无跌倒 → 方向／转弯合格 → 在此可行集合中最大化可持续速度**。用户没给数值容差，应在首次选模前冻结合理的直行路径／朝向、转向响应和加速收益门槛；不能看候选表现后放宽。有限回放零跌倒只能证明该测试覆盖，不能写成所有输入下绝不会倒。

验收至少包含：Shift 单独不前移；仅左 Shift+W 加速；先 Shift 后 W、先 W 后 Shift；A/D 同时转向；短点按／长按、交替左右、转弯松键；冲刺进入和退出；普通九技能回归、复位／机器人切换后模式清理。直行测世界路径相对起点方向的偏差与持续前速，转弯测实际期望符号、延迟、转角和切线方向速度，避免把转弯位移误判成走歪。不能只评第一次加速的峰值，应测较长稳定段和持续转弯。

设置新的训练／开发／最终保留种子和命令时序族，已有 `918000–918029` 已使用。冻结候选后一次性最终验证；同批模型在无 Python/TCP、禁网的独立导出包重放，并核验真实左 Shift 输入、数值误差 `<1e-5`、200/50 Hz、模型加载与资源释放。普通／候选比较采用同一控制契约；不同契约单列。

autoresearch 可借鉴的是：固定评测、有限预算、每次可审核的代码／假设改动、保留／拒绝／崩溃记录和 Git 版本；不是机器人算法，也无需运行其语言模型代码。每轮记录模型 SHA、控制／物理／观测／奖励版本、速度×转弯覆盖、教师数据 SHA、实际样本与工时、旧成功损失、新成功、去留原因。评分先过上述硬门再比较速度，不能压成可被速度抵消跌倒的加权总分。[官方 README](https://github.com/karpathy/autoresearch)、[实验约定](https://github.com/karpathy/autoresearch/blob/master/program.md)

运行资源继续用现有独立监督、四核默认和任务级进程组清理。最终报告须列实际速度提升、转弯覆盖和失败边界，以及本轮训练／仿真进程已经退出的核查结果。若仍有硬门未通过，应保留实验成果和原因，不把只接通 Shift 按键当成完成。

## 后备实验补充：仅 walking，GPU 训练代理的执行器对照

用户随后明确仅训练行走，轮滑不进入本轮实施。主线程新 Jolt 基线表明：保留 walking 在 W 命令 0.3／0.4／0.5 时实际约 0.17 m/s，零跌但约 31° 偏航；factory 在 0.4 时约 0.22 m/s，但累计绕转约 264°，0.5 会跌；2999 来源在 0.3 时约 0.23 m/s，0.4／0.5 和右转容易跌。这些为主线程提供的当轮初步测量，最终数字应引用其原始轨迹。它们支持“源策略有较快动作，但当前迁移不能安全使用”，不支持继续单纯放大命令。

### 源随机化能否覆盖当前 Jolt

**现有随机化不足以证明覆盖。** 只调延迟或扩大摩擦范围没有消除下面的结构差异。

| 项目 | source 2999 的 GPU 训练环境 | 当前游戏 Jolt | 结论 |
|---|---|---|---|
| 动作延迟 | BAM 配置 3–6 个 5 ms 物理步 | `_ctrl` 经固定采样后直接用于各物理步，没有这段额外延迟缓存 | 0–6 步随机化可包含零额外延迟，但不能修复其他动力学差异 |
| 扭矩 | BAM 电压控制、反电动势、负载／Stribeck 摩擦；电压 6.5–8.2 V | `clip(.55*(target-q), ±.6405236) -.053*qd -.0048*tanh(qd/.05)`；先限驱动扭矩，再加被动阻尼摩擦 | 当前 BAM 电压／摩擦标量随机化不能自动变成该公式；不能把固件 kp=200 与位置增益 .55 混比 |
| 惯量 | 关节空间 armature=.0018（并有约 ±10% DR） | armature 投影到子刚体对角惯量，之后以最大主惯量的 1/10 设下限 | 两者不是相同参数表示；质量 ±5% 和 armature ±10% 不保证覆盖 |
| 接触 | MuJoCo Warp 接触、足面 μ∈[.7,1.3]；该范围只是摩擦系数 | Jolt 接触及当前冻结碰撞几何 | μ DR 不包含 solver、接触刚度和接触点集差异，不能宣布等价 |

限矩数字为 `1.75 A × 0.36601349688984386 Nm/A`。依据：[Jolt 扭矩与惯量实现](../godot/physics_server.gd)、[生成的机器人参数](../godot/generated/microduck/robot_spec.json)、[电流限矩入口](../src/sim2sim/backends/godot_backend.py)、[本地 BAM 执行器](</home/ethan/Projects/microduck_rl/.venv/lib/python3.12/site-packages/bam/mjlab.py>)、[源机器人与 DR 配方](https://github.com/pollen-robotics/microduck_rl/blob/5946fd9cdbc58956424420153e51975af3b30d77/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)。本轮不修改这些游戏参数。

### 一小时内值得验证的一项假设

**H-GPU：在 GPU 中使用目标域扭矩规律的训练代理，比继续在原 BAM 域微调更容易把已有较快步态迁到 Jolt。** 两臂都从相同 2999 checkpoint 出发，固定相同命令、种子、网络、预算；一臂保持 BAM，另一臂使用训练专用显式扭矩执行器。修改的是离线训练域；游戏仍运行当前 Jolt。

一小时只做路线筛查：约 10 分钟配置／两臂 64 环境×5 次 smoke，约 12 分钟／臂正式采样，余下至少 20 分钟做固定 Jolt 完整命令回放与清理。按 smoke 的实际耗时选择 256 或 512 环境及迭代上限；并行仿真和学习都在单块 GPU，同一时刻仅一个进程。若编译加 smoke 已超过 10 分钟、代理初始 actor 立即失稳、或推理／导出不一致，记录路线未就绪，不侵占验收时间。此预算不是承诺一小时学会 sprint。

隔离入口放在 **sim2sim 的新 results 会话**，通过已有源 `.venv/bin/python` 运行；不要编辑源仓库、重装依赖或把 registry 全局对象直接改掉。[`load_env_cfg()`](</home/ethan/Projects/microduck_rl/.venv/lib/python3.12/site-packages/mjlab/tasks/registry.py>) 返回深拷贝。实现入口可以叫 `results/<会话>/gpu_proxy/train_proxy.py`，接受 `--kind bam|jolt_torque --envs --iterations --output`。下面是已经核对本机 API 的核心实现片段，**尚未保存为运行脚本，也未通过 smoke**；正式入口还需统一监督、日志和错误退出。

```python
# 文件属于 sim2sim/results/<会话>/gpu_proxy/，不修改 microduck_rl。
from dataclasses import dataclass, asdict
import torch
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata, attach_metadata_to_onnx

class JoltTorqueProxy(IdealPdActuator):
    def compute(self, cmd):
        drive = (.55 * (cmd.position_target - cmd.pos)).clamp(
            -1.75 * .36601349688984386, 1.75 * .36601349688984386
        )
        return drive - .053 * cmd.vel - .0048 * torch.tanh(cmd.vel / .05)

@dataclass(kw_only=True)
class JoltTorqueProxyCfg(IdealPdActuatorCfg):
    def build(self, entity, target_ids, target_names):
        return JoltTorqueProxy(self, entity, target_ids, target_names)

task = "Mjlab-Velocity-Flat-MicroDuck"
ec, ac = load_env_cfg(task), load_rl_cfg(task)
ec.scene.num_envs = args.envs
ec.seed = ac.seed = 920101  # 新训练种子；开发/最终种子另列。
ec.sim.mujoco.timestep = .005
ec.decimation = 4
ec.commands["twist"].ranges.lin_vel_x = (.20, .45)
ec.commands["twist"].ranges.lin_vel_y = (0., 0.)
ec.commands["twist"].ranges.ang_vel_z = (-.8, .8)
ec.commands["twist"].rel_standing_envs = .10
ec.commands["twist"].rel_forward_envs = .40
ec.commands["twist"].rel_turn_in_place_envs = .10
# 固定成熟阶段，防止恢复权重却从第0步重新启动训练课程。
ec.curriculum.clear()
ec.rewards["action_rate_l2"].weight = -1.0
if args.kind == "jolt_torque":
    ec.scene.entities["robot"].articulation.actuators = (
        JoltTorqueProxyCfg(
            target_names_expr=(r"^(?!passive_).*",),
            stiffness=.55, damping=0., effort_limit=float("inf"),
            frictionloss=0., viscous_damping=0., armature=.0018,
            delay_min_lag=0, delay_max_lag=0,
        ),
    )
    ec.events.pop("expand_bam_friction_fields", None)
    ec.events.pop("randomize_joint_friction", None)  # BAM专属回调不能作用于proxy。
    ec.events.pop("randomize_motor_gains", None)
ac.logger = "tensorboard"
ac.upload_model = False
ac.save_interval = 25
ac.algorithm.learning_rate = 3e-5
ac.algorithm.desired_kl = .005
ac.algorithm.entropy_coef = .005
env = None
try:
    env = ManagerBasedRlEnv(cfg=ec, device="cuda:0")
    vec = RslRlVecEnvWrapper(env, clip_actions=ac.clip_actions)
    runner = load_runner_cls(task)(vec, asdict(ac), str(args.output), "cuda:0")
    runner.load(str(checkpoint_2999), load_cfg={
        "actor": True, "critic": True, "optimizer": False,
        "iteration": False, "rnd": False,
    }, map_location="cuda:0")
    runner.learn(num_learning_iterations=args.iterations,
                 init_at_random_ep_len=True)
    runner.save(str(args.output / "final.pt"))
    runner.export_policy_to_onnx(str(args.output), "final.onnx")
    attach_metadata_to_onnx(str(args.output / "final.onnx"),
                           get_base_metadata(vec.unwrapped, "local-sprint-proxy"))
finally:
    if env is not None:
        env.close()
```

**代理边界**：这里的 `effort_limit=inf` 只避免 MuJoCo 再对“驱动限矩之后的总被动扭矩”二次截断；显式公式仍按游戏驱动限矩夹紧，不能误改为无限驱动力。原生关节摩擦／阻尼必须置零，避免重复施力。片段暂保留真正关节空间 armature，未复制 Jolt 对角惯量映射，接触也仍属 MuJoCo，故只有扭矩规律对齐，不是完整物理等价。两臂现有质量／接触 DR 可以保持；不要同时新增宽随机化和多种奖励，模糊这一对照。动作归一化、HOME、14关节、61D、原始 FP32 来源保持一致。源 normalizer 属于 actor state，应一起加载和导出。

监督调用方式（保存完整入口并通过参数检查后）：

```bash
# 从 sim2sim 目录运行；RUN 替换为当轮已有的新会话绝对路径。
.venv/bin/python -m sim2sim.research.budget --directory "$RUN" start \
  --timeout 600 --reserve 1200 --cpus 4 --nice 10 --label gpu-proxy-smoke \
  -- /home/ethan/Projects/microduck_rl/.venv/bin/python \
  "$RUN/gpu_proxy/train_proxy.py" --kind jolt_torque --envs 64 \
  --iterations 5 --output "$RUN/gpu_proxy/smoke_jolt_torque"

# 两臂必须串行且均通过smoke。200只是上限，依smoke测时在启动前冻结。
.venv/bin/python -m sim2sim.research.budget --directory "$RUN" start \
  --timeout 720 --reserve 1200 --cpus 4 --nice 10 --label gpu-proxy-jolt-torque \
  -- /home/ethan/Projects/microduck_rl/.venv/bin/python \
  "$RUN/gpu_proxy/train_proxy.py" --kind jolt_torque --envs 512 \
  --iterations 200 --output "$RUN/gpu_proxy/jolt_torque_s920101"
```

第二臂把 `--kind`、label 和输出目录改为 `bam`，不复用同一输出目录。`runner.learn()` 的迭代参数是**新增迭代数**；不把 2999 误算成还须从零走到 2999。源 `VelocityOnPolicyRunner.save()` 内置导出失败会仅警告并继续，因此上面的最终显式导出和后续 `<1e-5` 门禁不可省略。旧优化器不加载是为了避免其动量和学习率覆盖新实验设置；这不是完整续训复现，应明确记为权重初始化微调。[本机 runner API](</home/ethan/Projects/microduck_rl/.venv/lib/python3.12/site-packages/rsl_rl/runners/on_policy_runner.py>)、[源导出行为](</home/ethan/Projects/microduck_rl/.venv/lib/python3.12/site-packages/mjlab/tasks/velocity/rl/runner.py>)

资源快照（本次只读查询）：GB10 单卡、GPU 约 7%、47°C；统一内存 121 GiB 总量，约 29 GiB free、104 GiB available，nvidia-smi 显存数值 N/A。它只是当时余量，不能保证后续分配。先 64 环境编译，再选 256／512；设置 PyTorch／OMP 线程不超过四核，禁视频和网络 logger。与主线程训练串行；正式 Jolt 性能测量时必须停掉 GPU 代理。不要清全机 cache，也不要动其他项目进程。

成功标准不是代理 reward：候选回到冻结 Jolt，配对原 2999、保留 walking，在相同 W／左Shift+W／A/D、起步／退出和长直行新开发回放中，必须无跌倒、方向合格且比保留模型实际更快。若 BAM 与代理在源域都变快、目标域仍跌或走歪，则不再扩大这个分支；它只能说明当前代理未解决迁移。若代理明显改善，才值得进入多种子与更完整目标域适配，仍不能直接发布。该筛查本身不能证明解决了 PPO 遗忘，普通策略继续隔离，任何旧成功损失都须列出。
