# Flow-Research 第四批完整交付说明

日期：2026-10-02。新项目继续位于独立的 `Flow-Research` 目录，分支为 `codex/research-v3`。本批首先按要求将前三批代码、实验和三份完整中文说明上传到 [GitHub 研究分支](https://github.com/roy-pyke/Flow/tree/codex/research-v3)，随后完成时空面风、源项、指定入流和完整质量收支。原 Flow 目录保持 V2 基线。

首次远端核验提交为 `6f5390ed9b904f180665bc5799488339d5b82f37`；对应 [GitHub Actions 检查](https://github.com/roy-pyke/Flow/actions/runs/36981213458) 已成功，包含 Python、原生 C++ 和前端构建/lint。本批科学源码固定在 `1fd4e7af81851aeb66fbb130f0a51adba2b86492`；正式实验在该提交的干净工作区生成，另有结果与说明提交。没有将研究分支合并进 main。

**本批验收：全套 609 项测试通过；正式研究的 108 个原始数组从归档源码逐位重放一致；普通配置示例的 3 个场数组与 8 个驱动数组重放一致。现在可以配置持续／脉冲源和变风，在独立参考下比较数值方法，并核对每一项质量收支。**

完整路线图由 44/186 增至 **52/186 项必做任务已验收**，剩余 134 项必做和 12 项可选。任务数量不是工时百分比；第四批完成不代表整个深化项目已经完成。

## 1. 本批解决的实际问题

此前研究者可以改变材料扩散系数和路径出发时刻，但生产求解器的风仍只能是全域恒定向量，也没有可配置体源。用一个初始 Gaussian 代替持续排放，或者在每一步随意添加数值，会改变正在比较的物理问题。带入流的问题如果只检查总质量不变，也会错误地把正确结果判成失败。

本批提供一个统一的实验入口：

    版本化 JSON 或完整 NPZ
      → 明确网格、面方向、单位、时间节点和输入预算
      → 不可变 PrescribedFields
      → 全时间范围 CFL 上界、按输出及驱动节点拆步
      → 保守面输运 + 扩散 + 与方法一致的源积分
      → 分开记录注入、流入、流出、场质量
      → 独立连续解／固定空间时间参考／解析积分
      → 数组、源码、图表、报告和清单封存
      → 验证、重放与包外重绘

所有浓度依旧是合成相对浓度。这里解决的是科研模型与算法比较，不是已经校准的真实空气质量预测。前端地图仍沿用 V2 冻结场界面；这些丰富输入通过 CLI/Python 使用。

## 2. 方程、数组和输入含义

方程为：

    ∂c/∂t + div(u c) = div(κ grad c) + S.

长度 m、时间 s、速度 m/s、κ 为 m²/s、源项为相对浓度/s。源是与当前 c 无关的外加生成率，可以有符号以构造数学参考；没有加入依赖 c 的反应、化学机制或观测同化。

| 数组 | 形状 | 含义 |
| --- | --- | --- |
| times_s | (nt,) | 至少两个严格递增节点，必须从 0 开始 |
| velocity_x | (nt,ny,nx+1) | x 面平均速度，正方向始终为 +x |
| velocity_y | (nt,ny+1,nx) | y 面平均速度，正方向始终为 +y |
| source | (nt,ny,nx) | 单元平均体源率 |
| inflow_left/right | (nt,ny) | 左／右边界面入流浓度 |
| inflow_bottom/top | (nt,nx) | 下／上边界面入流浓度 |

所有数组采用 float64，y 行从南向北。Python 输入的实数数组被复制到不可变 bytes 所有者上；不能通过修改原数组或重新打开 writeable 改变驱动。复数、布尔、对象、非有限值和错形状明确拒绝。Python 中遗漏的数组／边界侧扩展为零；已提供数组不自动广播、翻轴或重采样。

共享节点之间全部使用分段线性时间重构，运行必须完整落在其覆盖区间内；没有末帧冻结或时间外推。字段内容身份包含几何、边界、时间规则、单位及完整数组字节。不同驱动只改变右端和输运，不错误复用其场值。

解析 generator_version=1 支持：

- 常向量风；其节点倍率可以实现时间反转。
- 线性剪切风：x 方向速度跨 y 改变，或 y 方向速度跨 x 改变。
- 绕指定中心的线性旋转风。矩形域的周期版本是法向面通量的周期延拓，切向分量可能在接缝跳变；不把它称为无界平面的刚体轨迹实验。
- 常量源与 Gaussian 空间源。Gaussian 通过 erf 差计算真实单元平均，不用中心值冒充平均。
- 四侧非负入流浓度及随节点变化的倍率。

source_scale 的 `[0,1,0]` 表示有限时段内的连续三角脉冲；不是矩形突变或 Dirac 瞬时源。输入可以完整通过 NPZ 定义八个数组，并声明 hash、网格、节点、单位、轴序与 south_to_north。不会执行输入代码或 pickle。

## 3. 保守输运、边界和非零散度

内部每个面使用面速度与上游单元浓度的乘积作为通量。同一物理面的贡献对相邻两个单元符号相反。因此实现的是 `-div(u c)`，而不是漏掉散度项的 `-u·grad(c)`：

    -div(u c) = -u·grad(c) - c div(u).

压缩流能够增加局部峰值和方差，同时守住闭／周期域的总质量。本批对此提供直接算子测试和解析特征线反例，诊断不会无条件要求常数场不变、最大值不增或能量下降。

| 边界 | 唯一的外部数值通量规则 |
| --- | --- |
| zero_flux | 外边界法向风速必须为零，总外面通量为零；允许域内部的非零风 |
| periodic | 两份接缝面速度必须精确一致，双方共用一个接缝通量；两单元轴仍保留两张物理面 |
| open 入流 | 用外法向速度乘指定入流浓度作为总入流通量 |
| open 出流 | 用外法向速度乘内部上游浓度作为出流通量 |

开放边界不额外加入外部扩散面贡献，也不把入流浓度同时当作另一套独立扩散 Dirichlet 条件。内部扩散仍采用此前的保守算子。源、面流和单元质量共享同一面积／面长单位。

有限体积的通用推导可见 [NIST FiPy 官方文档](https://pages.nist.gov/fipy/en/stable/numerical/discret.html)；本项目的算子和独立参考是自己的实现，没有引入 FiPy 依赖或声称等价于该库。

## 4. 时间推进与稳定步长

| 方法 | 源项取值 | 风与入流取值 | 本批后端 |
| --- | --- | --- | --- |
| FE diffusion | 步左端 | 必须零风 | NumPy |
| BE diffusion | 步右端 | 必须零风 | SciPy |
| CN diffusion | 左右端梯形平均 | 必须零风 | SciPy |
| 显式 upwind advection–diffusion | 步左端 | 步左端 | NumPy |
| IMEX Euler | 步左端 | 步左端 | NumPy + SciPy |

Rannacher 第一个 CN 宏步拆成两个 BE 半步；第二个半步使用其自己的右端时刻。每个方法都会在所有驱动节点处拆步，即使用户没有要求在那里输出。每个区间从固定时间原点构造时钟，避免反复累加 dt 的漂移；无法在浮点时钟上前进的步长明确报错。

设单元外向流率为：

    r = max(u_right,0)/dx + max(-u_left,0)/dx
      + max(v_top,0)/dy + max(-v_bottom,0)/dy.

逐单元、逐保存节点取最大值 r_max。每个速度分量在节点间线性变化，正部和及其单元最大值是凸函数，因此节点最大值约束整个中间时段。初始风为零也不能忽略之后变大的风。

显式输运使用 `dt ≤ 0.9 / [r_max + 2κ(dx^-2+dy^-2)]`；IMEX 使用 `dt ≤ 0.9/r_max`，纯扩散沿用其已有稳定界。这是全时段充分界，尚不是自适应最优步长。非负初值／源／入流在相应 FE CFL 或 BE/IMEX 前提下具有正性；CN 不承诺无条件正性。

纯扩散加源支持标量或正单元材料 κ、闭／周期边界、FE/BE/CN。变风输运本批仍要求标量 κ，支持三种边界和两个一阶输运方法。指定 cpp 会明确拒绝；有 prescribed fields 时 auto FE 选择 NumPy。没有伪造一个支持新驱动的 C++ 内核。

## 5. 质量账本到底检验什么

每个真实内部步累计与更新公式完全相同的数值源积分及边界通量：

    R(t) = M(t)-M(0) + 累计流出 - 累计流入 - 累计注入.

每步检查 R，输出时保存各项历史。字段包括 cumulative_source_mass、cumulative_boundary_outward_mass、cumulative_boundary_inward_mass、mass_balance_history 和 max_internal_mass_balance_error。为了兼容旧诊断，cumulative_outward_flux 仍表示“流出减流入”的净值，而不是只表示流出。

源积分按实际时间方法的求积计算，不称为精确物理积分。若场本身有符号，几何出流项也可能有符号；不会裁剪。原始质量漂移、最小／最大浓度与能量仍保留。有源问题质量变化可以很大，正确条件是完整收支闭合。

**R 很小只能说明账本与更新自洽，不能证明场准确，也不能证明注入／入流／出流各自积分准确。** 独立制造解专门让这两件事同时可见。能量、极值和场诊断在初始及输出帧取样；质量账本另在所有内部步取样。

## 6. 配置身份、预算与归档

ProblemSpec 新增可选 forcing。没有该字段或它为 None 时，physical_descriptor 主动移除它，保持旧 physical/configuration ID。旧没有驱动的计算仍进入原来的 solve 分支，减少无关数值算术变化。

有驱动时，旧 velocity 必须是零向量，避免两套风同时生效。解析驱动记录构造版本；NPZ 的本地路径不参与物理身份，文件内容 hash 与网格／时间／单位参与身份。capabilities 单独列出 28 个合法的 prescribed 方法、后端、边界和启动方式组合，实际风的法向约束继续检查输入数组。

预算分为 max_output_bytes、max_forcing_bytes 和 max_cell_updates。读输入或分配解析数组之前先核对输出和驱动的 float64 字节预算。NPZ 完整文件及解压成员总大小允许固定 1,000,000 bytes 头部／容器余量；八个成员名、唯一性、各 NPY header 的形状／dtype／payload 大小在 NumPy 分配前检查。工作估计包含驱动节点、输出节点和 Rannacher 额外拆步。这些都是准入估计，不是峰值 RSS 或运行时间硬上限。

普通研究包新增 forcing.npz 保存实际使用的八组数组；原始 NPZ 另复制为 forcing_input.npz。旧 fields.npz 的布局保持原样。验证只检查已经保存的驱动，不重新求解析采样公式；原始 NPZ 与转换后的数组也会逐项对照。

普通 replay 使用当前代码重新生成解析驱动，或读取保存的原始 NPZ，并比较所有驱动与输出数组。改变解析生成器会反映为驱动数组差异，不会静默称为相同重放。正式独立研究另外保存原生成器和完整 Python 源码，从而可以明确重放当时的实现。

发布仍使用“暂存 → 完整验证 → 原子改名到新目录”。已存在输出不覆盖，失败会清理暂存。哈希是完整性检查，不是真实性认证或数学正确性证明。

## 7. 三族独立参考及正式结果

正式结果见 [英文完整报告](../../reports/research/forced_transport_baseline/REPORT.md)、[机器报告](../../reports/research/forced_transport_baseline/report.json) 和 [原始数组](../../reports/research/forced_transport_baseline/raw_fields.npz)。

### 7.1 周期压缩与反转：空间误差和时间误差分开

取 `u_x=a(t)sin(2πx/L)`、u_y=0，初值常数 c0。a 是五节点分段线性反转信号；最终有符号时间积分为零。保守方程的精确连续解为：

    c = c0 / [cosh(s) + sinh(s) cos(2πx/L)],
    s = (2π/L) ∫ a(t) dt.

独立逆特征线在每个单元两端的角度差给出真实单元平均；另用积分求积核验。连续解最终回到均匀场，但一阶上风的数值扩散不会因风向反转而消失。

时间参考由独立逐面组装的一维上风矩阵产生。在风保持同号的区间，矩阵只乘一个时间标量，因此可以用指数精确推进；遇到零点／节点必须按时间顺序相乘，不把正负风的矩阵错误交换。参考又由独立自适应 ODE 积分核验。

| nx | 空间 RMS：半离散参考对连续解 | 时间 RMS：数值对半离散参考 | 观察空间阶 |
| ---: | ---: | ---: | ---: |
| 16 | 0.016301097 | 0.00018281575 | — |
| 32 | 0.0091514019 | 0.00020484148 | 0.832904 |
| 64 | 0.0048738846 | 0.00021799307 | 0.908921 |
| 128 | 0.0025180817 | 0.00022515220 | 0.952747 |

这验证 x 方向、y 不变问题的一阶空间趋势，不是一般二维二阶结论。固定空间的变风 FE 最后时间阶为 1.004567。最细网格的时间／空间 RMS 比约 8.94%，两者已经分别计算而非混合拟合。在压缩输出时刻，最细精确单元平均最大值为 1.368867；错误的常数保持模型 Linf 误差为 0.368867。

### 7.2 非齐次扩散：源时间求积和整体时间阶

闭边界的源包含随时间仿射变化的均值和一个 cosine 单元平均模态。独立逐面矩阵增加“时钟”和“常数”两个状态，使用增广矩阵指数直接积分；再用独立离散模态 Duhamel 公式核对。连续 cosine 参考单独记录空间误差。

| 方法 | 最细时间 RMS | 最后时间阶 | 注入质量误差 |
| --- | ---: | ---: | ---: |
| FE | 0.00077127339 | 1.000088 | -0.00075 |
| BE | 0.00077117974 | 0.999913 | +0.00075 |
| CN | 5.5106646e-8 | 2.000002 | -6.66e-16 |

非零扩散使 CN 的整体时间误差确实非零，避免用 κ=0、线性源时梯形求积恰好精确的退化例来冒称二阶。该固定网格的独立空间 RMS 为 0.0002660938；精确累计源质量为 0.656。额外求解器测试覆盖正变材料 κ 与非均匀时变源组合，闭／周期三种扩散方法都对照独立增广系统。

### 7.3 开放边界：正确场、闭合账本和非零积分误差

取 `u_x=v0+a x`、`c=c0+b t`、`S=b+a c`，左侧入流为 c，右侧出流。两个方法都用相同标量 κ=0.02；扩散作用于这个空间常数场为零。内部有不与输出中点对齐的 0.37T 驱动节点。

连续积分可手算：初始质量 0.5，末质量 0.62，源注入 0.2992，入流 0.2688，出流 0.448。对于每个仿射时间被积函数，左矩形法误差为 `-斜率/2 × Σ实际dt²`；报告从真实步长直方图独立计算此误差，没有重复调用生产通量。

| 最细方法 | 场 Linf | 源积分误差 | 入流积分误差 | 出流积分误差 | 质量残差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 显式输运 | 4.66e-15 | -0.00014946 | -0.00022419 | -0.00037365 | 2.78e-15 |
| IMEX Euler | 1.18e-14 | -0.00014946 | -0.00022419 | -0.00037365 | 4.22e-15 |

这个例子清楚地区分舍入量级的闭合残差与一阶积分误差。报告与图同时保存二者，没有只挑看起来更好的数字。

## 8. 验证过程、审查修复和性能范围

- 全套 `FLOW_REQUIRE_NATIVE=1 .venv/bin/python -m pytest -q`：**609 passed, 2 warnings, 14.27s**。两条为已有 Starlette/httpx 与 anyio 弃用提示；本批没有改变依赖。
- 四个新增测试文件合计 150 项：forcing 33，forced_solver 32，forcing_research 38，forced_transport_study 47。覆盖独立面对参考、正性条件、压缩流、时钟、源权重、CFL、身份、预算、畸形 NPZ、失败发布和重绘。
- 图表 QA 后只修改横轴刻度显示，相关原子发布／渲染失败／重绘测试再跑 **3 passed**。正式图已目视核验，标签清晰。
- 普通配置示例成功运行、验证、重放，3 个初值／时间／场数组和 8 个驱动数组全部一致，Linf=0。
- 正式包 31 个 manifest 列表文件，另含 manifest 本身，共 32 个路径；其中 25 个源码／锁文件。108 个数组共 642,136 bytes，配置 ID 为 `3cccffe91568ac30eb99b62d99a4258f555022006b23d087d1b88224b9dc1fcc`。
- 正式包从已提交源码的干净工作区生成到本地输出，再原样移动到公开报告目录。归档源码重放：108 数组最大差值 0；报告除 provenance、solve_ms、timings_ms 之外完全一致；重绘 PNG 字节一致。

本批审查中修复了两组真实一致性缺口。第一，无 forcing 配置原可混入 manifest 登记过的驱动文件，导致 verify 通过、replay 失败；现在保留文件／配置／结果必须一致，重放在创建输出前拒绝。第二，研究包原可在重新计算 hash 后删掉绘图时间轴而通过校验；现在检查固定输出／驱动时钟、必要源码包初始化文件，以及重绘需要的数值字段、长度和取值范围。

性能上保留“驱动一次预处理、之后复用”，并沿用已有稀疏 LU 缓存。缓存键继续包含网格、κ、边界、方法和真实 dt；风／源在右端变化不要求重建扩散 LU。驱动保留字节和单个快照字节有显式元数据，但不可变插值快照和逐步校验仍有开销。本批没有测量新的端到端速度收益，不把测试耗时或单次 study 运行时间包装为性能基准。

## 9. 如何运行与复现

所有命令从 `Flow-Research` 根目录执行，输出目录必须尚不存在：

```bash
.venv/bin/python -m scripts.run_research capabilities
.venv/bin/python -m scripts.run_research run configs/research/prescribed_transport.json --output data/experiments/research/my-prescribed
.venv/bin/python -m scripts.run_research verify data/experiments/research/my-prescribed
.venv/bin/python -m scripts.run_research replay data/experiments/research/my-prescribed --output data/experiments/research/my-prescribed-replay

.venv/bin/python -m scripts.run_forced_transport_experiments --config configs/research/forced_transport_study.json --output reports/research/my-forced-study
.venv/bin/python -m scripts.run_forced_transport_experiments --verify reports/research/forced_transport_baseline
.venv/bin/python -m scripts.run_forced_transport_experiments --replot reports/research/forced_transport_baseline
```

replot 会写入新的同级 PNG，不修改封存目录；同名重绘文件已存在则拒绝。要重放归档代码，进入 `reports/research/forced_transport_baseline/source_snapshot/` 后执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /path/to/Flow-Research/.venv/bin/python -m scripts.run_forced_transport_experiments --config ../config.json --output /absolute/fresh/replay-directory
```

禁止写 bytecode 可以避免污染封存目录。重放使用所选解释器安装的依赖，不自动重建操作系统；同环境的逐位一致不等于跨机器保证。

## 10. 完成范围与下一层工作

本批验收 M02-06 至 M02-12，以及 M02-14：时空面风、非零散度、可控体源、源时间离散、边界通量、完整诊断、独立参考族、模型适用说明。第三批的变系数参考与本批的风／源参考共同支撑参考族验收。

M02-01 仍缺统一研究 HTTP/API 接入；M02-02 的多峰和紧支撑初值还未齐全，因此没有把整条 M02 直接宣称完成。高阶 MUSCL/SSP 输运、可扩展迭代隐式求解、伴随与目标误差控制、自适应、不确定性和实际用户任务验证仍在后续路线图。运输 κ 仍为标量，没有张量／时变扩散、非线性反应或不连续源信号。

先前的 sealed baseline 没有重写。第一至第三批说明继续固定各自历史提交；本文件是第四批的完整记录。[全部批次入口](README.md) 汇总公开版本。

## 11. 全部文件设计与变更清单

下表逐路径覆盖本批相对于前三批上传提交的变更，包括正式包中的源码副本。快照文件不是另一套实现：它们保留生成这次报告的精确源字节，独立于未来主目录修改。

本批共 **50 个新增／修改路径**。

| 状态 | 路径 | 设计及变化 |
| --- | --- | --- |
| 修改 | [README.md](../../README.md) | 公开项目入口；接入驱动功能与新指南，明确旧地图方程的历史范围。 |
| 修改 | [backend/app/numerics/__init__.py](../../backend/app/numerics/__init__.py) | 导出 PrescribedFields 和 face_advection_rate。 |
| 新增 | [backend/app/numerics/forced_solver.py](../../backend/app/numerics/forced_solver.py) | 节点拆步、FE/BE/CN/IMEX 源时间规则、真实半步时钟、复用 LU 与完整质量账本。 |
| 新增 | [backend/app/numerics/forcing.py](../../backend/app/numerics/forcing.py) | 不可变输入、严格时间与边界契约、插值、全局出流率上界、保守面通量及分项流量。 |
| 修改 | [backend/app/numerics/solver.py](../../backend/app/numerics/solver.py) | 兼容增加 forcing 参数；旧输入保留原分支，新驱动分派到独立实现。 |
| 修改 | [backend/app/research/experiments.py](../../backend/app/research/experiments.py) | 实际驱动／原始输入归档、保留文件一致性、历史数组验证及全数组重放比较。 |
| 修改 | [backend/app/research/problems.py](../../backend/app/research/problems.py) | 新增解析／NPZ 驱动 schema、单位与方向、旧身份兼容、输入与工作预算、能力组合。 |
| 新增 | [configs/research/forced_transport_study.json](../../configs/research/forced_transport_study.json) | 固定三族独立研究的几何、物理参数、细化网格和预算。 |
| 新增 | [configs/research/prescribed_transport.json](../../configs/research/prescribed_transport.json) | 可运行的风向反转、Gaussian 三角源、两侧指定入流示例。 |
| 新增 | [docs/milestones/PHASE4_COMPLETE.zh-CN.md](../../docs/milestones/PHASE4_COMPLETE.zh-CN.md) | 本文件：工作、设计、改动、数学、证据、修复及全部路径。 |
| 修改 | [docs/milestones/README.md](../../docs/milestones/README.md) | 加入第四批说明的公开导航。 |
| 新增 | [docs/prescribed_transport.md](../../docs/prescribed_transport.md) | 英文完整数学、数组、配置、能力、运行、归档和局限。 |
| 修改 | [docs/research.md](../../docs/research.md) | 研究入口与当前实测证据更新，保留旧批次历史。 |
| 新增 | [reports/research/forced_transport_baseline/REPORT.md](../../reports/research/forced_transport_baseline/REPORT.md) | 英文科学报告与结果解释。 |
| 新增 | [reports/research/forced_transport_baseline/config.json](../../reports/research/forced_transport_baseline/config.json) | 规范化完整研究配置。 |
| 新增 | [reports/research/forced_transport_baseline/convergence.csv](../../reports/research/forced_transport_baseline/convergence.csv) | 可直接分析的空间／时间误差表。 |
| 新增 | [reports/research/forced_transport_baseline/forced_evidence.png](../../reports/research/forced_transport_baseline/forced_evidence.png) | 经过目视核验的四面板科学图。 |
| 新增 | [reports/research/forced_transport_baseline/manifest.json](../../reports/research/forced_transport_baseline/manifest.json) | 其余 31 文件的哈希与大小。 |
| 新增 | [reports/research/forced_transport_baseline/raw_fields.npz](../../reports/research/forced_transport_baseline/raw_fields.npz) | 108 个实际输入、数值解、参考解和独立参考矩阵。 |
| 新增 | [reports/research/forced_transport_baseline/report.json](../../reports/research/forced_transport_baseline/report.json) | 全指标、诊断、身份、环境和源码哈希。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/__init__.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/__init__.py) | 不可变源码／锁文件副本：`backend/__init__.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/__init__.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/__init__.py) | 不可变源码／锁文件副本：`backend/app/__init__.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/diffusion.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/diffusion.py) | 不可变源码／锁文件副本：`backend/app/diffusion.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/main.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/main.py) | 不可变源码／锁文件副本：`backend/app/main.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/__init__.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/__init__.py) | 不可变源码／锁文件副本：`backend/app/numerics/__init__.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/coefficients.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/coefficients.py) | 不可变源码／锁文件副本：`backend/app/numerics/coefficients.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/forced_solver.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/forced_solver.py) | 不可变源码／锁文件副本：`backend/app/numerics/forced_solver.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/forcing.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/forcing.py) | 不可变源码／锁文件副本：`backend/app/numerics/forcing.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/native.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/native.py) | 不可变源码／锁文件副本：`backend/app/numerics/native.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/operators.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/operators.py) | 不可变源码／锁文件副本：`backend/app/numerics/operators.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/solver.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/solver.py) | 不可变源码／锁文件副本：`backend/app/numerics/solver.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/validation.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/numerics/validation.py) | 不可变源码／锁文件副本：`backend/app/numerics/validation.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/observations.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/observations.py) | 不可变源码／锁文件副本：`backend/app/observations.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/research/__init__.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/research/__init__.py) | 不可变源码／锁文件副本：`backend/app/research/__init__.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/research/decision.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/research/decision.py) | 不可变源码／锁文件副本：`backend/app/research/decision.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/research/dynamic_routing.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/research/dynamic_routing.py) | 不可变源码／锁文件副本：`backend/app/research/dynamic_routing.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/research/experiments.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/research/experiments.py) | 不可变源码／锁文件副本：`backend/app/research/experiments.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/research/problems.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/research/problems.py) | 不可变源码／锁文件副本：`backend/app/research/problems.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/research/travel.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/research/travel.py) | 不可变源码／锁文件副本：`backend/app/research/travel.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/routing.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/routing.py) | 不可变源码／锁文件副本：`backend/app/routing.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/temporal_observations.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/temporal_observations.py) | 不可变源码／锁文件副本：`backend/app/temporal_observations.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/backend/app/validation.py](../../reports/research/forced_transport_baseline/source_snapshot/backend/app/validation.py) | 不可变源码／锁文件副本：`backend/app/validation.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/requirements.lock.txt](../../reports/research/forced_transport_baseline/source_snapshot/requirements.lock.txt) | 不可变源码／锁文件副本：`requirements.lock.txt`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/scripts/__init__.py](../../reports/research/forced_transport_baseline/source_snapshot/scripts/__init__.py) | 不可变源码／锁文件副本：`scripts/__init__.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [reports/research/forced_transport_baseline/source_snapshot/scripts/run_forced_transport_experiments.py](../../reports/research/forced_transport_baseline/source_snapshot/scripts/run_forced_transport_experiments.py) | 不可变源码／锁文件副本：`scripts/run_forced_transport_experiments.py`；来自科学源码提交，重放使用本副本。 |
| 新增 | [scripts/run_forced_transport_experiments.py](../../scripts/run_forced_transport_experiments.py) | 特征线／独立指数／解析积分参考，指标、CSV、图、原子封存、源码清单、verify 与包外 replot。 |
| 新增 | [tests/test_forced_solver.py](../../tests/test_forced_solver.py) | 32 项源权重、半步时钟、旧行为退化、完整账本及正变材料增广参考专项。 |
| 新增 | [tests/test_forced_transport_study.py](../../tests/test_forced_transport_study.py) | 47 项特征线求积、ODE/Duhamel 核对、收敛、积分账本、协议预算及封存／重绘专项。 |
| 新增 | [tests/test_forcing.py](../../tests/test_forcing.py) | 33 项独立算子、边界、CFL、不可变输入和时间覆盖专项。 |
| 新增 | [tests/test_forcing_research.py](../../tests/test_forcing_research.py) | 38 项配置、身份、预算、输入 header、归档篡改、旧包兼容与完整重放专项。 |
