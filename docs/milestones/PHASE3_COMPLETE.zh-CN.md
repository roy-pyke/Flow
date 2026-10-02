# Flow-Research 第三批完整交付说明

> GitHub 公开副本：按用户要求于 2026-10-02 发布。正文记录该批交付时的状态；“未推送”等历史表述不代表当前分支状态。代码链接已改为仓库相对路径；本地验收日志、内部规划记录仍以文件名标识。

本批从第二批提交 `001a4b85118a641934b67d19eac096d4c08942e8` 继续，在独立目录 `/path/to/Flow-Research` 实现并验收正变系数扩散。最终本地提交为 `b318246e9be2baea25f4d798e9e52c620ec9ba0c`，验收记录见 IMPLEMENTATION_PHASE3.json（本地记录 `planning/IMPLEMENTATION_PHASE3.json`）。原 Flow 没有编辑，仍处于 V2 基线提交，120 个跟踪文件、Git 工作区干净。

应本次要求，先完成了两份各自独立的历史档案：[第一批完整说明](PHASE1_COMPLETE.zh-CN.md) 共 713 行，覆盖全部 57 个历史变更路径；[第二批完整说明](PHASE2_COMPLETE.zh-CN.md) 共 461 行，覆盖全部 41 个历史变更路径。两份文件都固定历史提交，集中保存做过的事情、文件设计、代码改动、数学、审查修复、测试、运行方法和限制。本批不追写前两批的封存实验数据。

**第三批已交付一个可直接研究材料非均匀性的完整入口：定义正扩散系数场 → 计算 → 对照独立参考 → 检查界面通量与收敛阶 → 保存原始数组和源码 → 重放与重绘。全套 459 项测试通过，正式实验的 66 个数组从归档源码逐位重放一致。**

这是一批完成的数值研究功能；整个深化项目仍未全部完成。当前清单累计 44/186 项必做验收，剩余 142 项必做与 12 项可选。任务数量不代表工时百分比。

## 1. 本批实际解决的问题

此前生产求解器只能给全域一个标量 κ。研究者即使怀疑不同位置具有不同扩散能力，也不能在同一套求解、验证和归档协议里表达它。随意把 κ 乘在旧 Laplacian 外面还会算成 `κ Δc`，并不是本批要解的 `div(κ grad c)`；材料跳变处尤其容易得到错误通量。

现在支持静态、各向同性、每单元一个正材料值的 κ(x,y)。浓度仍是合成的相对浓度，长度 m、时间 s、κ 单位 m²/s。输入是材料单元值，不是浓度单元平均，不含时间变化、张量扩散、体积源、反应或真实观测校准。

| 部分 | 本批之前 | 本批之后 |
| --- | --- | --- |
| 扩散系数 | 非负常标量 | 保留标量；新增严格正的 `(ny,nx)` 场与不可变预处理对象 |
| 方法 | 原有五种时间推进 | 变系数先接 FE/BE/CN；恒定风输运保持原范围 |
| 边界 | 常系数既有边界 | 变系数支持 zero_flux 与 periodic |
| 后端 | 常系数 NumPy/C++ FE、SciPy 隐式 | 变系数 FE 用 NumPy，BE/CN 用 SciPy；显式 cpp 请求报错 |
| 面系数 | 全域同一 κ | 两个半格串联阻力产生调和面系数 |
| 显式稳定界 | 常系数历史公式 | 变量场按实际最大单元总出流率计算；标量路径保留 |
| 隐式缓存 | 标量 κ 身份 | 加入系数内容与面策略哈希，仍含几何、边界、方法、真实 dt |
| 研究输入 | 常数 κ | JSON 分层材料或 hash 校验 NPZ，支持 Python 原始数组 |
| 证据 | 常系数参考 | 分层解析通量、变系数精确 PDE、独立小矩阵时间参考 |

## 2. 调和面通量为什么正确

方程为 `c_t = div(κ grad c)`。相邻单元 P、N 的中心距为 h，界面在公共面上，每边材料在自己的半格内为常数。无界面源或额外接触阻力、法向通量连续时：

    q = -(c_N-c_P) / [h/(2κ_P) + h/(2κ_N)]
      = -κ_f (c_N-c_P)/h
    κ_f = 2κ_Pκ_N/(κ_P+κ_N)

所以每个面的速率 `a_f=κ_f/h²` 对 P 加 `a_f(c_N-c_P)`，对 N 加相反数。这保持离散收支。调和值的实际计算写成 `min*(2/(1+min/max))`，避免先乘 κ_Pκ_N 或相加大数导致溢出；面速率连续除两次间距，避免直接构造 h² 的溢出/下溢。不可表示的面速率或总出流率明确拒绝。

这些假设限定在正交等体积网格、标量材料、对齐界面。没有宣称支持未对齐界面、非正交网格或各向异性张量。平滑材料采用同一两点通量，其一致性由独立空间实验检验。

闭边界没有外部面贡献。周期边界加入首尾接缝；某轴只有两个单元时，同一对单元之间确实有两张物理面，不能去重掉一张。

对生产离散矩阵 L：

    L 1 = 0,    1^T L = 0,    L = L^T
    z^T L z = -Σ_faces a_f (z_N-z_P)² ≤ 0

因此 L 是对称负半定，具有常数零空间；不是 SPD。对正 dt 和 θ，`I-θ dt L` 才是 SPD。此处单元体积相同，体积加权内积是欧氏内积的正常数倍。已有非零风 upwind 算子一般不对称，不把这一结论推广给它。

FE 的单调性界为 `dt≤1/max_P Σ_faces(P) a_f`，自动步长使用其 90%。BE 对这些闭/周期扩散问题保持正性与最大值原理；CN 对二次能量稳定，却不保证任意步长下正性。负值保留供诊断，没有裁剪或重新缩放质量。Rannacher 仅将第一个 CN 宏步替换为两个 BE 半步，不能当成以后所有大步长的正性保证。

## 3. 文件设计与调用链

    layered JSON / NPZ / Python array
      → 严格问题与几何校验、输出预算
      → diffusivity_field 解析实际材料值
      → prepare_diffusivity 生成不可变面速率与内容身份
      → 用实际出流率估计工作预算
      → solve(FE/BE/CN)
          FE: 无矩阵面通量散度
          BE/CN: 同一面速率组装 CSC → 缓存 LU → 求解与残差
      → 浓度/时间/初值/材料数组及原始输入一起归档
      → schema/hash 验证 → 原子发布新目录
      → 重放所有数组、独立参考评价

| 新增/修改的核心文件 | 设计与具体变化 |
| --- | --- |
| `backend/app/numerics/coefficients.py` | 新增 `DiffusionCoefficients`、`prepare_diffusivity`、稳定调和均值。保存单元值、x/y 内面速率、周期面速率、逐单元出流率、最大出流率、系数哈希和内存元数据。所有数组使用不可变 bytes 作为底层存储，不能重新打开 writeable 标记。预处理对象只能复用于完全相同的网格/边界。 |
| `backend/app/numerics/operators.py` | 增加变量场/预处理对象分派，保留旧标量代码。CSC 与 matrix-free 都消费相同的面速率。matrix-free 每次只保留一个方向的通量临时数组；复数/对象/布尔场不再被默默转成实数。 |
| `backend/app/numerics/solver.py` | 接受 float/ndarray/prepared；一次预处理后复用。变量 FE 按能力选择 NumPy，显式 cpp 拒绝；CFL 用真实出流率；缓存键用内容哈希。增加系数诊断、backend_selection。另修复初值在验证前转换导致复数虚部被丢弃的问题。 |
| `backend/app/numerics/__init__.py` | 导出新的公共系数接口；原有数值入口继续兼容。 |
| `backend/app/research/problems.py` | κ 保留 float，新增 `LayeredDiffusivity` 与 `ArrayDiffusivity`。增加材料加载、分配前 NPY header 验证、先输出后计算的预算检查、真实 CFL 预算和完整机读组合。旧常系数配置身份不变。 |
| `backend/app/research/experiments.py` | 归档实际 `diffusivity`，NPZ 原样保留为 `kappa_input.npz`；增加稳定归档数组身份；校验不重算解析材料公式；重放比较全部数组。 |
| `configs/research/layered_diffusion.json` | 可立即运行的 40×20 分层 CN 示例，κ 左 0.05、右 0.5，界面 x=1，闭边界。 |
| `scripts/run_variable_diffusion_experiments.py` | 三种独立参考、原始数组库存、数值/参考预算、CSV/图/报告、完整源码归档、verify/replot CLI。 |
| `configs/research/variable_diffusion_study.json` | 固定正式研究协议：四档空间网格、两种时间边界、三方法与四档步数、一个分层反例。 |
| `scripts/benchmark_diffusion_operator.py` | 先检查等价输出，再交错计时 matrix-free/CSC；分别保存预处理/组装/应用原始重复、明确数组字节、tracemalloc 分配、环境和测量源码。 |
| `tests/test_variable_diffusion.py` | 57 个新增用例：独立逐面对参考、矩形/两单元周期、守恒/能量/正性、极端系数、内容哈希、缓存、标量退化、后端拒绝和初值类型保护。 |
| `tests/test_variable_research.py` | 48 个新增用例：配置身份、分层与大坐标对齐、NPZ dtype/header/hash、真实 CFL 预算、便携归档、重放与篡改后重新 hash 的拒绝。 |
| `tests/test_variable_study.py` | 36 个新增用例：独立投影积分、PDE 恒等式、空间/时间阶、所有界面通量、聚合预算、源码/数组库存、头部伪造、发布失败、重绘等。 |
| `docs/variable_diffusion.md` | 英文完整数学、适用条件、输入格式、示例、验证与性能结果。 |
| `docs/research.md`、`README.md` | 接入新入口并修正能力边界；明确地图仍冻结场，变量材料目前是 CLI/Python 能力。 |

## 4. 配置、身份与持久化细节

一个对齐材料界面可以写为：

    "kappa": {"kind":"layered", "axis":"x", "interface_m":1,
              "left":0.05, "right":0.5}

接口也允许 y 方向；此时 left/right 指坐标较低/较高一侧。界面必须严格在域内并落在网格面上。最终实现先重建最近物理面，在物理坐标上比较两个 ULP 的容差，避免投影坐标 500000 附近的相减误差。间距不足 32 个坐标 ULP 时明确拒绝，以免容差跨越邻面。周期层同时会在域首尾再次相接。

NPZ 必须只有一个名为 values 的实数正数组，并声明 SHA-256、bounds、nx/ny、y,x 轴序、south_to_north、cell_material、m2/s。不会自动翻转或重采样。文件 hash、ZIP 展开大小和 NPY header 都在分配数组前检查；header 的形状、dtype、payload 字节必须一致，禁止 object/pickle、重复成员、伪造巨大 shape。输出预算在读取输入前执行。

两种身份承担不同职责：

- 求解器的 `coefficient_sha256` 覆盖材料字节及形状/单位/解释/面策略，用于缓存；完整缓存键另含几何、边界、方法和每个实际 dt 的十六进制值。
- 归档的 `values_sha256` 只按已声明归档 schema 对实际 float64 little-endian 材料值取 hash，用于历史数组完整性。验证时不需要按新版本重建面速率或重新评价解析材料函数。

旧 float κ 的 JSON 和三种问题身份保持原值；NPZ 的本地路径不参与物理身份，内容与网格声明参与。导入的离散系数网格本身是物理输入，改变该网格/数值会改变身份；解析分层界面在合法网格细化时则保持物理身份。

新变量包中 fields.npz 保存 initial、times_s、frames、diffusivity。若材料来自 NPZ，原文件还按字节保存在 manifest 列出的 kappa_input.npz；验证导入值必须与实际归档材料一致。旧三个数组的常系数包继续有效。完整暂存目录通过校验后才发布，已存在目的地拒绝覆盖。hash 是完整性证据，不是认证签名或科学真理证明。

## 5. 三套数值参考与正式结果

正式包：[variable_diffusion_baseline](../../reports/research/variable_diffusion_baseline/REPORT.md)。共有 66 组原始数组、514,368 字节数组数据、29 个 manifest 产物，含 23 份源码/依赖快照。

### 5.1 真正的无源变系数精确 PDE

令 θ=πx/L、r=1+2cos(2θ)，取

    κ(x)=D [1+(b/3)r] / [1+3br],    0<|b|<1/9
    c(x,t)=mean+A exp(-Dπ²t/L²)[cosθ+b cos3θ]

则 κc_x 的括号正好变成 `sinθ+(b/3)sin3θ`，再微分得到 c_t；两个端点通量为零。这是人为构造的验证问题，不是测得的环境扩散率。初值和解析比较值都使用 exact sinc 单元平均，投影没有数值求积误差。

空间实验固定 ny=4，仅加密 x；解和材料均与 y 无关，不宣称一般二维空间收敛已经完全验证。生产 CN 使用 dt=0.0005、T=0.2，另用固定生产空间矩阵的指数解分开测时间误差。矩阵指数不是独立空间离散参考；独立连续解析解才负责检验空间正确性。

| nx | 空间 RMS | CN 时间 RMS | 空间观测阶 |
| ---: | ---: | ---: | ---: |
| 16 | 2.8921424e-4 | 2.0987165e-9 | — |
| 32 | 7.2639725e-5 | 2.1270833e-9 | 1.993308 |
| 64 | 1.8181012e-5 | 2.1343515e-9 | 1.998326 |
| 128 | 4.5465716e-6 | 2.1361696e-9 | 1.999582 |

最细网格时间/空间 RMS 比例约 0.00046984，即 0.047%，不会掩盖本曲线的空间阶。空间指数参考另受 `T*maxOutgoing≤10000` 限制，默认最大约 1192.79；不能只凭隐式步数少就允许无界代价的指数参考。该限制仍不是 wall-time 上限。

### 5.2 独立小矩阵时间参考

固定 10×6 矩形网格，用另一套 Python 逐面对循环组装 dense L，再用对称特征分解计算 exp(TL)c0；组装没有调用生产 diffusion_matrix。系数为一个正二维平滑函数，分别检查闭边界和周期边界。FE/BE/CN 的步数为 8/16/32/64。

| 边界 | FE 最后观测阶 | BE 最后观测阶 | CN 最后观测阶 |
| --- | ---: | ---: | ---: |
| zero_flux | 1.005887 | 0.994244 | 2.000185 |
| periodic | 1.008847 | 0.991358 | 2.000130 |

这里衡量固定空间离散下的时间误差，不把矩阵指数称为连续 PDE 真值。浮点特征分解/指数也不是向外舍入的严格区间证明。误差低于可表示精度时，报告写 unresolved，不强行输出收敛阶。

### 5.3 分层稳态界面通量

单位长度，左右各半，κ_left=1、κ_right=20，左端浓度 1、右端 0。独立解析串联阻力给出 q=1/(0.5/1+0.5/20)=1.90476190476。解是分段线性；界面对齐后中心值等于精确单元平均。

验证器在生产闭边界矩阵的副本上加入左右半格 Dirichlet 项，顶部/底部保持零通量。**这些边界项只用于该验证问题，生产求解器没有新增 Dirichlet 边界能力。**

| 方案 | 浓度 Linf 误差 | 所有 x 面最大通量误差 | 平均 x 通量 |
| --- | ---: | ---: | ---: |
| 生产调和面 | 5.55e-16 | 1.33e-14 | 1.90476190476 |
| 算术均值对照 | 0.02422118 | 0.05000502 | 1.95476692604 |

算术方案自身也能得到近乎常通量，却比正确通量偏约 2.63%。所以只检查“左右通量连续”还不够，必须同时对照正确串联阻力。两套方案的所有 x/y 面通量、矩阵和 RHS 均完整保存，调和解最大 y 通量约 6.66e-15。

## 6. 性能与分配证据：不预设 matrix-free 更快

[正式原始测量](../../reports/research/variable_operator_baseline/benchmark.json) 使用周期 256×160 网格、固定随机种子 73、同一输入数组、21 次重复。测量之前检查等价输出；应用阶段交错次序，分别记录材料预处理、CSC 组装和重复应用。另保留测量源码与依赖锁，共 24 项 manifest 产物。

| 项目 | 预处理后 matrix-free | CSC |
| --- | ---: | ---: |
| 常驻系数数组 / 单独矩阵数组 | 1,310,720 B | 2,621,444 B |
| 重复应用中位数 | 0.144250 ms | 0.107083 ms |
| 单次 warm call 的 tracemalloc 峰值 | 786,184 B | 327,923 B |
| 结果数组 | 327,680 B | 327,680 B |

系数预处理中位数 0.620625 ms，CSC 从 prepared 组装另花 2.475833 ms。输出差 Linf=3.64e-12，相对输出最大尺度约 10,577。matrix-free 的结果加最大单方向通量缓冲算术估算为 654,080 B，实际跟踪分配更高，因此文档没有把该估算当成进程峰值上限。

本例 matrix-free 系数表示约是单独 CSC 数组的一半，但重复应用 CSC 更快。两列没有比较整个 PDE 进程内存；当前隐式求解器同时保留材料、CSC 和 LU。计时不包括 PDE 时间推进、LU、道路、路由、HTTP 或前端，不能据此声称完整系统加速。正式性能测量在其他 agent 数值工作停止后独立执行。

## 7. 测试、审查修复与验收

完整命令：`FLOW_REQUIRE_NATIVE=1 .venv/bin/python -m pytest -q`。结果 **459 passed, 2 warnings, 9.52 s**；相较第二批新增 141 项。两条 warning 仍来自 Starlette/httpx 与 anyio 的已有弃用提醒。随后只调整了图的横轴标签以消除重叠，相关归档/重绘 2 项检查再次通过。前端与 C++ 源码没有变化，本批没有重复前端构建或声称新 Linux CI 结果。

本批修复与防护包括：

1. 变量系数不能使用标量真假判断；分派使用明确的 `diffusion_active` 与 coefficient identity。
2. 改动原始数组不能悄悄改变面速率或污染 LU 缓存；新快照使用不可逆只读 bytes backing。
3. 极大/极小材料对比度不能先做 κ_Pκ_N 或 h² 导致伪溢出；不可表示速率明确失败。
4. 两单元周期域保留两张真实面；独立 oracle 有专门用例。
5. 复数初值及算子场不能先转换后丢虚部；明确拒绝复杂/对象/布尔/文本初值，允许正常实数整数输入。
6. 大坐标平移不能误拒绝同一分层界面；改用物理 ULP，并拒绝网格精度不足。
7. NPZ ZIP 大小通过不代表 header shape 安全；头部先检查，拒绝伪造、重复及多余成员。
8. 隐式步数预算不足以约束指数参考；添加独立 stiffness 工作量上限。
9. 零可辨误差产生 None 比例时，报告标记 unresolved，不再发生字符串格式化异常。
10. 图表视觉检查发现 log 横轴次刻度标签拥挤；改成离散实验刻度。调整前的自建试包移到忽略的 intermediate 目录，没有改写前两批或覆盖已封存包。

正式研究包验证 29 项产物；从 source_snapshot 重放后 66/66 数组逐位一致；排除 provenance/solve_ms/timings_ms 后科学记录完全一致；单独重绘 PNG 与原图逐字节相同。两个新包的源码 hash 都与最终实现一致。第一批 scalar 包、第二批 temporal 包仍验证通过。分层通用 CLI 示例四个数组也重放一致，质量漂移约 1.33e-15、最大相对线性残差约 3.52e-16。

勾选任务：M02-04（正变系数）、M02-05（材料面系数与通量）、M04-01（当前算子的真实数学性质）、M04-02（无矩阵应用、矩阵等价与内存记录）。M02-11/12/14 只新增了变量扩散范围的证据；变量风、源项、完整模型目录等缺口仍在，未一并勾选。

## 8. 运行与重放

在 `/path/to/Flow-Research` 执行：

```bash
.venv/bin/python -m scripts.run_research capabilities
.venv/bin/python -m scripts.run_research run configs/research/layered_diffusion.json --output data/experiments/research/my-layered
.venv/bin/python -m scripts.run_research verify data/experiments/research/my-layered
.venv/bin/python -m scripts.run_research replay data/experiments/research/my-layered --output data/experiments/research/my-layered-replay
.venv/bin/python -m scripts.run_variable_diffusion_experiments --config configs/research/variable_diffusion_study.json --output reports/research/my-variable-study
.venv/bin/python -m scripts.run_variable_diffusion_experiments --verify reports/research/my-variable-study
.venv/bin/python -m scripts.run_variable_diffusion_experiments --replot reports/research/my-variable-study
.venv/bin/python -m scripts.benchmark_diffusion_operator --output reports/research/my-variable-benchmark
```

目的目录必须是新的；已有产物不覆盖。正式数值包源码重放：

```bash
cd reports/research/variable_diffusion_baseline/source_snapshot
PYTHONDONTWRITEBYTECODE=1 /path/to/Flow-Research/.venv/bin/python -m scripts.run_variable_diffusion_experiments --config ../config.json --output /absolute/new/output
```

禁用 bytecode 写入可保持源码包封存状态。重放使用当前已安装的解释器/依赖，依赖锁仅记录版本，不会自动重建操作系统。数组逐位一致只描述本机本次记录，跨平台可以有合理浮点差别。

## 9. 下一批与当前限制

下一个相邻阶段优先补空间/时间变化的风与源项，并统一边界总通量和质量收支，提供各自独立制造解。随后再深化迭代隐式解法、预条件与性能路线。当前材料场仍静态、正标量、均匀笛卡尔网格；开放变量扩散、任意 Dirichlet、非对齐界面、张量材料、观测校准尚未实现。

已有时变观察器可消费这些求解得到的场数组，但本批不新增城市级动态路线/UI 工作流。地图继续使用冻结场和 V2 控件，研究入口是 CLI/Python。原资料和旧报告保留，新成果集中在 Flow-Research；未推送远端。

## 10. 本地材料与完整文件清单

本文件、两份历史完整说明、START_HERE、主待办清单、派生 BACKLOG、IMPLEMENTATION_PHASE3 和 validation 日志属于本地研究规划；按既有约定通过 `.git/info/exclude` 保留，未混入公开源码提交。首次图表 QA 之前的临时包及重放/重绘放在忽略的 output/research-validation；环境 `.venv` 和生成缓存也不属于源码变更。

下面的完整 Git 路径清单按本批最终工作树生成；measurement_source / source_snapshot 每条均为封存副本，不表示对同名旧项目又做了一次修改。

<!-- PHASE3_FILE_INVENTORY -->

| 序号 | 状态 | 文件 | 作用 |
| ---: | --- | --- | --- |
| 1 | M | `README.md` | 公开使用、数学与能力范围说明。 |
| 2 | M | `backend/app/numerics/__init__.py` | 生产实现或实验/测量入口；设计与具体变化见第 3 节。 |
| 3 | A | `backend/app/numerics/coefficients.py` | 生产实现或实验/测量入口；设计与具体变化见第 3 节。 |
| 4 | M | `backend/app/numerics/operators.py` | 生产实现或实验/测量入口；设计与具体变化见第 3 节。 |
| 5 | M | `backend/app/numerics/solver.py` | 生产实现或实验/测量入口；设计与具体变化见第 3 节。 |
| 6 | M | `backend/app/research/experiments.py` | 生产实现或实验/测量入口；设计与具体变化见第 3 节。 |
| 7 | M | `backend/app/research/problems.py` | 生产实现或实验/测量入口；设计与具体变化见第 3 节。 |
| 8 | A | `configs/research/layered_diffusion.json` | 可重现问题/研究配置；见第 3–5 节。 |
| 9 | A | `configs/research/variable_diffusion_study.json` | 可重现问题/研究配置；见第 3–5 节。 |
| 10 | M | `docs/research.md` | 公开使用、数学与能力范围说明。 |
| 11 | A | `docs/variable_diffusion.md` | 公开使用、数学与能力范围说明。 |
| 12 | A | `reports/research/variable_diffusion_baseline/REPORT.md` | 正式结果、原始数据、配置、图或完整性清单；见第 5–6 节。 |
| 13 | A | `reports/research/variable_diffusion_baseline/config.json` | 正式结果、原始数据、配置、图或完整性清单；见第 5–6 节。 |
| 14 | A | `reports/research/variable_diffusion_baseline/convergence.csv` | 正式结果、原始数据、配置、图或完整性清单；见第 5–6 节。 |
| 15 | A | `reports/research/variable_diffusion_baseline/manifest.json` | 正式结果、原始数据、配置、图或完整性清单；见第 5–6 节。 |
| 16 | A | `reports/research/variable_diffusion_baseline/raw_fields.npz` | 正式结果、原始数据、配置、图或完整性清单；见第 5–6 节。 |
| 17 | A | `reports/research/variable_diffusion_baseline/report.json` | 正式结果、原始数据、配置、图或完整性清单；见第 5–6 节。 |
| 18 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/__init__.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 19 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/__init__.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 20 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/diffusion.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 21 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/main.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 22 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/numerics/__init__.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 23 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/numerics/coefficients.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 24 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/numerics/native.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 25 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/numerics/operators.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 26 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/numerics/solver.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 27 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/numerics/validation.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 28 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/observations.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 29 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/research/__init__.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 30 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/research/decision.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 31 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/research/dynamic_routing.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 32 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/research/experiments.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 33 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/research/problems.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 34 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/research/travel.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 35 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/routing.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 36 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/temporal_observations.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 37 | A | `reports/research/variable_diffusion_baseline/source_snapshot/backend/app/validation.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 38 | A | `reports/research/variable_diffusion_baseline/source_snapshot/requirements.lock.txt` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 39 | A | `reports/research/variable_diffusion_baseline/source_snapshot/scripts/__init__.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 40 | A | `reports/research/variable_diffusion_baseline/source_snapshot/scripts/run_variable_diffusion_experiments.py` | 数值研究封存副本；原模块职责见第 3 节及历史文档。 |
| 41 | A | `reports/research/variable_diffusion_baseline/variable_evidence.png` | 正式结果、原始数据、配置、图或完整性清单；见第 5–6 节。 |
| 42 | A | `reports/research/variable_operator_baseline/benchmark.json` | 正式结果、原始数据、配置、图或完整性清单；见第 5–6 节。 |
| 43 | A | `reports/research/variable_operator_baseline/manifest.json` | 正式结果、原始数据、配置、图或完整性清单；见第 5–6 节。 |
| 44 | A | `reports/research/variable_operator_baseline/measurement_source/backend/__init__.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 45 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/__init__.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 46 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/diffusion.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 47 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/main.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 48 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/numerics/__init__.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 49 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/numerics/coefficients.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 50 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/numerics/native.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 51 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/numerics/operators.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 52 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/numerics/solver.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 53 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/numerics/validation.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 54 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/observations.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 55 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/research/__init__.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 56 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/research/decision.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 57 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/research/dynamic_routing.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 58 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/research/experiments.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 59 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/research/problems.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 60 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/research/travel.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 61 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/routing.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 62 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/temporal_observations.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 63 | A | `reports/research/variable_operator_baseline/measurement_source/backend/app/validation.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 64 | A | `reports/research/variable_operator_baseline/measurement_source/requirements.lock.txt` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 65 | A | `reports/research/variable_operator_baseline/measurement_source/scripts/__init__.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 66 | A | `reports/research/variable_operator_baseline/measurement_source/scripts/benchmark_diffusion_operator.py` | 性能测量封存副本；保留实际测量时的依赖与实现。 |
| 67 | A | `scripts/benchmark_diffusion_operator.py` | 生产实现或实验/测量入口；设计与具体变化见第 3 节。 |
| 68 | A | `scripts/run_variable_diffusion_experiments.py` | 生产实现或实验/测量入口；设计与具体变化见第 3 节。 |
| 69 | A | `tests/test_variable_diffusion.py` | 本批回归与独立参考验收；见第 3、7 节。 |
| 70 | A | `tests/test_variable_research.py` | 本批回归与独立参考验收；见第 3、7 节。 |
| 71 | A | `tests/test_variable_study.py` | 本批回归与独立参考验收；见第 3、7 节。 |

本批跟踪文件变更合计 71 个路径；本地中文说明和验证日志另列，不计入源码提交文件数。
