# Flow-Research 第二批完整交付说明

> GitHub 公开副本：按用户要求于 2026-10-02 发布。正文记录该批交付时的状态；“未推送”等历史表述不代表当前分支状态。代码链接已改为仓库相对路径；本地验收日志、内部规划记录仍以文件名标识。

本文把第二批的目标、代码设计、数学定义、实验、审查修复、测试、归档、运行方式和全部文件变化集中保存在一个文件中。阅读本文不需要先读另一份进度报告。

**本批完成了真实物理时间上的完整旅程积分，以及有明确等待规则的有限时间展开图搜索。** 研究者现在可以区分：场在变、旅客在移动、帧之间需要插值、搜索时钟需要离散，究竟是哪一项改变了路线决策。

本文是历史交付记录，范围固定为第一批提交 **420ac772dccbd3c3f047303e70f835f954bcecc8** 到第二批提交 **001a4b85118a641934b67d19eac096d4c08942e8**。代码说明依据后者的 Git 对象读取，实验数字依据该提交中的 temporal_baseline 报告；第三批及以后修改不属于本文。以下本地文件链接方便打开文件，但工作树随后可能演进；要还原本批代码，应以这个完整提交号为准。

第二批交付日期为 2026-10-01（America/Los_Angeles）；归档与核验的 UTC 时间跨到了 2026-10-02。本批在独立目录 /path/to/Flow-Research、分支 codex/research-v3 中完成，本地提交后未推送。原目录 /path/to/Flow 的 120 个跟踪文件经过逐字节核对，没有变化，原仓库保持干净。

## 1. 这一批解决的问题，以及仍然保留的研究边界

第一批能够计算扩散场，并把某一时刻冻结的场映射到道路上。在冻结假设下，每条边只有一个固定暴露成本，可以调用静态最短路径。这个假设无法回答：旅客走到下一条边时，污染场已经发生变化，会不会应该选另一条路？等待能否降低暴露？为什么某个更晚到达中间节点的状态不能直接丢掉？

第二批把路径评价改为整个旅程上的函数：

    E(γ) = ∫[t_depart,t_arrive] c̃(γ(t), t) dt
    J(γ) = (t_arrive - t_depart) + λ E(γ),   λ ≥ 0

这里 γ(t) 同时包含行走和原地等待；c̃ 是由离散网格帧定义的明确重构，浓度单位仍是相对浓度。J 的数值单位按本实验约定记为目标成本秒，不是健康损失，也不等于真正节省的通勤时间。

本批交付包括：

- 非均匀保存帧上的严格时空采样；整段轨迹必须被帧覆盖。
- 包含折线弯点、重复边、起点/途中/终点等待的轨迹表示。
- 对声明的时空重构精确至浮点误差的分段两点 Gauss 积分，以及可分别细化空间和时间的梯形积分。
- 一个保留节点与时间状态的有限图参考搜索器，显式记录主动等待及时间取整等待。
- 三个实际调用生产 CN 求解器的 PDE 场景、独立解析参考、时间离散对照、动态标签反例。
- 同时封存配置、图、数组、源码、CSV、图和报告的完整实验包，以及结构与哈希验证、归档源码重放和无损重绘。

研究入口是 Python/CLI。网页地图和原有 API 路线比较仍使用冻结场，没有把新时变能力自动接进地图，也没有增加本批性能提升声明。上述能力建立了可审查的小规模参考，不是已经完成全城市连续时间动态优化。

## 2. 整体结构与调用顺序

代码按“场、轨迹、积分、搜索、证据”分工。通用线性观察器允许带符号的数学场；需要非负成本的路径研究适配器负责额外限制。

    temporal_study.json → Study 校验与总数组/工作量预检查
      → corridor_network + enumerate_simple_paths 构造完整的两路径候选集
      → Grid + 精确单元平均初值
      → numerics.solve(CN, scipy) 生成带真实保存时刻的帧
      → TimeDependentExposure 拥有不可写的帧/时刻副本
          → build_path_trajectory 构造每个候选的真实轨迹
          → build_trajectory_observer 构造稀疏积分行
          → 与独立连续解析积分比较，得到冻结/时变选择及候选 regret
      → solve_time_expanded 调用 edge_exposure 与 waiting_exposure
          → 返回 travel/wait 动作、每条边时间与 E/J
          → trajectory_from_schedule 重建整个旅程
          → 再积分整个旅程，并用解析解重新评分全部动作
      → reconstruction_study + label_counterexample
      → write_bundle 暂存所有文件 → verify_bundle → 发布新目录

在这条链上，PDE 内部步长、输出保存时刻、沿轨迹求积规则、搜索时间格点是四个不同参数。代码和报告分别记录它们，防止把提高某一层分辨率误说成所有误差都减小了。

## 3. 时空场到底如何定义

### 3.1 保存数组和真实时间插值

场数组的逻辑顺序是 (frame, y, x)，底层按 C 顺序展平；空间位置来自单元中心。帧时刻至少两个，必须有限、严格递增，而且相邻时间差必须有限。帧不必等间隔。

给定 τₙ ≤ t ≤ τₙ₊₁，令 α=(t−τₙ)/(τₙ₊₁−τₙ)。空间双线性重构记作 Bₙ(x,y)，则：

    c̃(x,y,t) = (1−α) Bₙ(x,y) + α Bₙ₊₁(x,y)

这按实际秒数插值，不按帧下标插值。恰好等于末帧时刻时，使用最后一个区间的 α=1；早于首帧、晚于末帧的查询一律拒绝。即使只晚一个可表示浮点数，也不能静默把末帧当成无限延续。

空间端沿用共享 bilinear_weights 的契约：zero_flux/open 在物理矩形内部的外侧半格采用已声明的常值延拓，periodic 在首末单元中心之间接续周期重构；这不是允许查询跑出物理矩形后再默认 clip 回来。轨迹端点也会检查，不能因为 Gauss 采样点没有碰到越界位置就漏掉短暂越界。

### 3.2 为什么必须分段，为什么两点 Gauss 足够

在一个固定空间重构单元内，每一帧可写成 a+b x+c y+d xy。相邻帧之间做线性时间插值后，各系数是 t 的一次式。沿直线匀速运动，x(t)、y(t) 都是一次式，因此被积函数最高是三次多项式。

两点 Gauss-Legendre 对次数不超过三次的多项式精确。代码先在以下位置分段，保证每个小区间确实使用同一个多项式表达式：

1. 每个道路原始折线弯点，以及每个行走/等待分界。
2. 穿过 x/y 单元中心重构节点的时刻，包括边界半格与周期接缝相关节点。
3. 穿过保存帧时刻的时刻。

每段 [a,b] 使用中点 m=(a+b)/2、半宽 h=(b−a)/2，采样 m±h/√3，两点权重均为 h。原地等待时位置固定，帧间函数最多是时间一次式，同一规则依然正确。

“精确”限定为**所声明的离散重构积分**。原始 PDE 的离散误差、有限体积平均值转为点重构的误差、帧间线性插值误差以及模型误差都没有被 Gauss 积分消除。

### 3.3 稀疏观察矩阵与两种求积规则

每个求积点最多连接相邻两帧、每帧四个空间邻点，因此可以把一次完整旅程的积分写成：

    E = Hγ · vec(frames)

Hγ 是 CSR 格式的一行，列顺序为 frame/y/x。重复贡献会合并，零项移除，索引排序；矩阵数据和索引缓冲区设为不可写。对有效插值和正求积权重，观察权重非负；作用于常数 1 时给出整段旅程时长，包含等待。

默认 gauss2_split 使用上述强制分段，不接受空间/时间步长参数，以免调用者误以为这些参数正在起作用。trapezoid 同样保留强制分段，再根据 spatial_step_m 和 time_step_s 两个独立上限加密。省略某个上限只表示不增加该类细化，并不取消强制分段。

max_samples 默认 1,000,000。在分配采样点、求积数组和稀疏矩阵之前累计预计求值数量；极小步长、极大网格穿越或过多分段会直接拒绝。这个限制约束求积样本数，不是整个进程的峰值内存保证。

## 4. 逐文件实现说明

### 4.1 backend/app/observations.py：轨迹契约、等待及统一时钟

第一批已有稀疏空间道路观察器。本批保留这部分职责，主要扩展 PathTrajectory 和轨迹构造。

**PathTrajectory(vertex_xy, vertex_times_s, edge_indices, metadata)** 现在即使直接调用数据类构造，也会验证并复制数组：顶点为有限 (n,2)，n≥1；时间数量匹配，严格递增且时间差、总时长有限；边编号为非负整数；metadata 是字典。两组数组均由对象拥有且不可写。只有一个顶点表示合法的零时长轨迹；相邻坐标相同而时间增加表示等待。positions(times) 严格限制在轨迹时间范围内。

**build_stationary_trajectory(point, start, end)** 是新增的原地等待入口。正时长生成同坐标的两顶点轨迹；start=end 生成单顶点轨迹；时间倒序或非有限输入拒绝。

**build_path_trajectory(network, edge_indices, speed_mps=1.4, departure_time_s=0, *, waits_s=None)** 新增 waits_s。对 N 次边经过，需要 N+1 个非负等待时长：每次边经过之前一个，最终到达后一个。重复边按每次经过分别保留。例如边序列 [0,1,0] 和 waits_s=[.3,.2,.4,.5] 包含三次行走与四次等待，不会按边编号去重。

边之间必须按方向连通，几何端点必须精确衔接。函数不会偷偷补一个未计费的短连接段。零长度折线子段跳过，真实弯点全部保留。元数据记录边身份、总长度、等待清单和每次边经过的出发/到达时间。

本批还统一了逐边的权威时钟：

    edge_end = edge_start + edge.length_m / speed

内部弯点使用相对于该边起点的累计弧长；最后一个顶点明确锚定到 edge_end。下条边及等待都从这个已确定的时间继续。这样与搜索器采用完全一致的终点公式，避免“逐段除以速度再求和”与“先求总长度再除以速度”相差 1 ULP，导致恰好到达 horizon 的路径被误拒绝。

### 4.2 backend/app/temporal_observations.py：通用时空采样和积分

本批新增此文件，负责数学观察算子，不负责路线优化。

| API/内部部分 | 职责与契约 |
| --- | --- |
| sample_time_series(grid, times_s, frames, points_xy, query_times_s, boundary) | 对应位置与时刻逐点采样；位置形状为查询时间形状加末尾维度 2，输出保持查询时间形状；支持标量和非均匀帧。 |
| TrajectoryObserver | 保存 grid、只读 times_s、CSR matrix 和元数据；apply(frames) 返回完整时间积分，检查形状、有限性及计算溢出。 |
| build_trajectory_observer(...) | 根据验证后的 PathTrajectory 构造观察行；默认 gauss2_split，另有 trapezoid 与空间/时间细化参数。 |
| _frame_times / _time_weights | 检查帧和查询覆盖；末帧定位到最后区间，绝不外推。 |
| _centre_crossings / _segment_breaks | 先确定节点范围和预算，再生成空间/帧分割；避免为巨大网格盲目生成全部坐标。 |
| 元数据 | 保存网格、帧时刻、轨迹顶点和时间、重构版本、边界、求积规则、控制步长、operator_sha256、nnz、CSR 字节数、样本数、构造耗时及解释范围。 |

算子接受带符号但有限的场，便于线性代数检查。非负路线成本要求由下一层实施。直接输入越界时间、非法 shape、未知规则、不匹配的积分参数或产生非有限结果都会明确失败。

### 4.3 backend/app/research/travel.py：非负场适配与动作复核

本批新增 TimeDependentExposure，把观察器连接到路径和图搜索。初始化输入 network、grid、times_s、frames，以及速度、边界、缓存/数组/样本限额。

它检查：帧为实数有限非负 (time,ny,nx) 数组；时间与相邻间隔有限且递增；复制后的 float64 数据不超过默认 128 MiB；速度正；所有道路端点精确等于对应图节点；全部几何位于有效空间域。帧和时刻被复制为自己拥有的只读数组。网络仍按实例使用期间不可修改的契约持有，并非额外深拷贝整个路网。

| API | 作用 |
| --- | --- |
| trajectory_exposure(trajectory) | 检查完整时间覆盖，构建时空观察器并积分。 |
| edge_exposure(edge_index, departure_s) | 用该边的完整折线和实际出发时间构造轨迹，返回整次经过的积分。 |
| waiting_exposure(node_id, start_s, end_s) | 在该图节点原地积分；合法零时长直接返回 0。 |
| metadata() | 给出场/图身份、覆盖范围、数据字节数、速度、缓存使用、求积规则和模型限制。 |
| trajectory_from_schedule(network, result) | 从最优搜索结果的显式动作恢复单条完整轨迹，供独立审计。 |

默认标量缓存最多保留 2,048 个结果，使用 LRU；键包括边与实际出发时刻，或等待节点和完整时间区间。不同时间经过同一边不会误命中同一值。max_cache_entries=0 可禁用缓存；该限制不是总进程内存上限。场/图身份包含帧内容 SHA-256、网格、时刻、速度、边界、节点与完整道路几何。

trajectory_from_schedule 不修补错误动作：相邻动作的首尾时刻必须完全接续；等待节点必须是当前节点；行走边必须按方向连通且几何精确相接；行走结束时刻必须等于 start+length/speed；最终节点、时刻和有序边列表必须与结果一致。它保留每个内部弯点和显式等待，再让观察器对完整轨迹积分，用于发现漏算等待、时间空档或未计费连接段。

### 4.4 backend/app/research/dynamic_routing.py：有限时间展开参考搜索

本批新增 solve_time_expanded，主要参数为 network/start/end、departure_time_s、horizon_time_s、time_step_s、speed_mps、lambda_weight、edge_exposure、waiting_exposure、allow_wait、max_states、max_transitions。

全局格点为 k·Δt，原点固定为 0。若出发时刻不在格点上，它是一个单独的初始状态，而不是先强制等到下个格点。状态标签为“节点、时间格点”，不能只按空间节点存一个累计最低成本。

对每次非终点行走：先走完完整边，实际到达 t_actual=t_start+length/speed；再选满足 t_grid≥t_actual 的最小格点。差值形成一条真实的 rounding wait 动作，在到达节点支付等待时间和等待积分。关闭 allow_wait 仅关闭主动等待，不免除这段时间取整等待。主动等待则把当前节点推进到下个格点，同样收费。

若道路到达目的地，直接按 t_actual 终止，不再增加没有必要的终点取整等待。所有真实到达和后续等待都受 horizon 严格约束。ceil_index 用浮点大小关系校正除法舍入，没有 isclose 宽容区间；一个 nextafter-later 到达绝不能被向下当作已到达前一个格点。过小的时钟步长、无法分辨的正旅行时长或超过支持的 binary64 索引分辨率会被拒绝。

虽然空间图可以有回路，每个动作都使时间严格增加，所以展开图是 DAG。实现按时间顺序处理节点/时间标签，在同一节点同一时间仅保留严格更小的目标值；不同时间状态保留。终点是吸收状态。平局按时间、节点标识、有类型的边身份顺序确定，保持确定性；平行边的整数 key=1 和字符串 key="1" 不会合并。

默认预算为 100,000 个状态、1,000,000 次可行转移。开始前按非终点节点数×格点数加离格初态保守检查状态容量；执行时累计已求值转移数，超限直接抛错，不能把中途找到的路线当成最优。起终点相同返回零时长空路线，不因无须展开的大格点表耗尽状态预算。不可达返回 complete=True、status=unreachable、成本为空，这表示完整搜索证明此有限模型下不可达。

结果包括 status/complete/scope、动作、逐边进入/真实到达/取整时刻、取整和主动等待、边序列和有类型边 ID、节点序列、到达时间、travel/wait/elapsed、E/exposure、J/objective、状态和转移计数、格点/预算/等待策略。回调的有限非负假设、普通浮点比较限制也写入结果。

最优性只针对**声明的有限取整图、输入成本与等待策略**。它不是连续时间最优性证书，不自动给出空间或时间离散误差上界。

### 4.5 scripts/run_temporal_experiments.py：实验协议、独立答案及发布

本批新增脚本，可作为模块或直接脚本运行。Case/Study 使用继承的严格 Pydantic 配置：未知字段和非有限数拒绝，schema_version 只能为 1。Study 限制网格每方向 8..128，case 数 1..12 且 ID 不重复，κ 在 0..100；mean≥amplitude≥0；出发早于 horizon；1..8 个正搜索时间步。

协议先检查帧/内部步长比率有限，至多 1,025 帧，估计总 cell updates 不超过 100,000,000。总归档数组预算覆盖**每个 case 的数值帧、连续节点参考、精确单元平均参考三组数组**，再加公共初值、时刻和制造案例预留量，合计不超过 128 MiB。它没有宣称限制稀疏分解、临时复制或 Python 进程的峰值 RSS。

| 函数 | 实际工作 |
| --- | --- |
| analytic_cosine_exposure | 连续余弦扩散解沿每个实际移动/等待线段的独立闭式积分；复指数与 expm1 减少短区间相消误差。 |
| polynomial_exposure | 用独立 Polynomial 多项式积分计算制造场的完整轨迹值。 |
| reconstruction_study | 非均匀帧、重复边和等待；分别做空间求积、时间求积与保存帧插值细化。 |
| label_counterexample | 保存单空间标签错误的手算小图、显式积分规则与 4 对 102 的答案。 |
| run_study | 生产 CN 求解、冻结/时变全路径评价、解析比较、有限图搜索、完整动作重新积分和解析重新评分。 |
| plot_report | 从报告数据画三联图：冻结选择失误、求积误差、独立时间插值误差。 |
| verify_bundle | 检查完整产物哈希、配置/候选身份、源码记录和原始数组结构。 |
| write_bundle | 新目录专用的暂存、封装、自验证、重命名发布与失败清理。 |
| main | --output、--verify、--replot 互斥入口；--config 指定协议。 |

所有无等待候选都必须完整落在保存场覆盖内。冻结对照在**实际出发时刻**插值得到冻结场；出发时刻位于两帧之间也不会偷换成第 0 帧。路径的最终参考按完整行程的连续解析场计算。

时间展开结果除了逐动作回调累计，还从动作恢复完整轨迹，独立做一次重构积分；随后既逐动作做解析积分，也对完整轨迹做解析积分，检查两套记账一致。将返回日程与两条连续无等待候选最优值比较时，字段明确叫 difference_from_continuous_no_wait_candidate_optimum_s，且说明它是不同策略的比较，不是连续时间 regret 证书。

### 4.6 configs/research/temporal_study.json：受版本控制的研究协议

默认协议固定：矩形 [0,0,10,10] m；40×40 网格；速度 1.4 m/s；出发 0 s；horizon 12 s；保存间隔 0.125 s；内部 CN 步长 0.005 s；λ=1；mean=1、amplitude=0.95；三个 κ 场景为 0、0.01、5；搜索步长为 1、0.5、0.25、0.125 s。

两条走廊来自既有 corridor_network：起点/终点横坐标为域宽的 0.1/0.9，高度为 0.5；上走廊高度 0.8，下走廊高度 0.35。它是明确制造的合成图，不冒充 OSM 真实街区。完整简单路径只有两条，编号 0 和 1。

### 4.7 三份新增测试文件

**tests/test_temporal_observations.py** 覆盖观察层：非负权重与常数积分；三次复合多项式；非均匀帧端点与标量采样；原地等待和时间仿射函数；冻结帧与旧空间积分一致；zero_flux/open/periodic 一致语义；独立自适应积分对照；重复边/四段等待；非法轨迹和只读副本；独立空间/时间细化；严格覆盖和短暂空间越界；样本分配前预算；多段折线在不同出发/等待条件下与搜索时钟逐位一致。

**tests/test_dynamic_routing.py** 覆盖搜索层：用完全枚举动作树的独立 oracle 对照多个 λ、离格出发和等待策略；显式取整等待及目的地不取整；主动等待有利/昂贵两种情况；单空间标签反例；nextafter 严格向上取整；horizon 恰好可达/刚好不可达；不连通、空路线、平行边类型和确定性平局；负数/NaN/Inf 积分；非法配置、重复边身份、预算耗尽和累计溢出。该文件独立运行时有 37 项通过记录。

**tests/test_temporal_research.py** 覆盖整条链：余弦闭式参考对照 SciPy 自适应积分；适配器拥有数组、缓存限制和非法场；精确几何及覆盖；特定 1 ULP 折线反例；动作重建与多项式整程参考；生产 PDE 三场景；分别细化求积和保存帧；与波速同行的周期平移波；配置版本、总数组预算；新包不覆盖；哈希重新计算之后仍能检出错 shape、NaN、schema、源码记录及额外文件；绘图失败时不发布半成品。

三个新增文件与既有测试一起使总数从 217 增至 318，即新增 101 项。它们测试的是科学不变量、独立答案和失败行为，并不是只复制实现内部公式。

### 4.8 README.md 与 docs/research.md：公开入口准确反映能力

README 更新研究扩展介绍，明确时变旅程和有限图能力，保留地图/API 冻结场的事实，链接新增时变报告，并在目录表加入 temporal_observations.py。

docs/research.md 从“第一批静态里程碑”改为静态/时变研究工作流，新增实际 CLI、归档重放、字节码关闭、等待/时钟/覆盖、CSR 语义、适配器、动态标签与范围限制；记录 318 项测试和 15 数组重放。原来的第一批能力、原始 V2 历史证据以及观测基准仍保留，没有把旧速度测量改说成这批动态搜索性能。

## 5. 实验的数学设置与实际结果

### 5.1 生产 PDE 和独立参考

主实验使用：

    ∂c/∂t = κ Δc，矩形零通量边界
    c(y,t) = 1 + 0.95 exp(−κ (π/10)² t) cos(πy/10)

初始数组不是简单中心点值，而是余弦模式的精确有限体积平均：振幅乘以 sinc(k Δy/2)，实现用 NumPy 的归一化 sinc 参数 kΔy/(2π)。数值场与精确单元平均参考比较；道路观察则使用声明的中心双线性重构。连续节点参考另外保存，避免把平均值误差和点值误差混淆。

上/下走廊旅程长度分别 14 m 和 11 m，速度相同，对应时长 10 s 和约 7.857143 s。每次选择都在共同的两条完整无主动等待候选上评价。

| 场景 ID | κ | 冻结选择 | 时变选择 | 冻结选择的连续参考 regret | 时变选择的连续参考 regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| stationary_equivalent | 0 | 0 | 0 | 0 | 0 |
| slow_evolution | 0.01 | 0 | 0 | 0 | 0 |
| frozen_selects_wrong_route | 5 | 0 | 1 | 2.645895542563977 s | 0 |

第一行说明场不变时，时变积分退化为冻结积分。第二行给出冻结选择仍有效的缓慢变化情景。第三行证明快速变化时，仅按出发场选路会在此候选集内选错。这个 deliberate synthetic 案例不说明现实路线多大比例会选错，也没有实测污染校准。

### 5.2 时间展开参考细化与完整动作复核

以下只对最后一个 κ=5 场景执行，关闭主动等待，保留并收费取整等待：

| Δt_search (s) | 取整等待 (s) | 数值 J | 所选完整动作的连续解析 J | 解析 J 与连续无等待候选最优值之差 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.2142857143 | 18.8119040178 | 18.8119376176 | 2.4392667306 |
| 0.5 | 0.7142857143 | 17.8082523732 | 17.8082907192 | 1.4356198322 |
| 0.25 | 0.2142857143 | 16.8035794209 | 16.8036232492 | 0.4309523622 |
| 0.125 | 0.0892857143 | 16.5522197324 | 16.5522650517 | 0.1795941647 |

完整轨迹重构积分与动作积分和的绝对差分别约 1.07e−14、1.78e−15、1.42e−14、1.78e−15，验证等待没有从账目里消失。

本例步长减小时目标下降，但不同时间网格产生不同等待日程。尤其关闭主动等待后，更细图不一定能复制粗图的所有等待安排，因此不能据此证明一般目标单调收敛；表中最后一列也不是连续时间误差界或最优 regret 保证。

### 5.3 为什么 FIFO 不允许只存一个节点标签

独立手算图有 s、a、b、t。速度为 1；s→a 和 s→b 长度为 1，b→a 长度为 √2，a→t 长度为 1。旅行时间恒定，所以满足通常的旅行时间 FIFO。除 a→t 外积分为 0；a→t 在出发早于 3 时暴露 100，否则 0；等待暴露为 0。使用 Δt=1，关闭主动等待。

- s→a 在 t=1 到达 a，累计成本 1，之后到 t 的目标为 102。
- s→b→a 在真实时刻 1+√2 到达 a，向上取整等待至 t=3；累计成本 3，之后目标为 4。

只保留 a 节点最低累计成本会删掉第二种状态，返回 102；保存节点/时间标签得到 4。这里的时变成本通过显式 callback 指定，并未声称它来自同一套 PDE 污染场。反例证明的是状态删减逻辑错误，旅行 FIFO 不能替代暴露目标的动态状态分析。

### 5.4 分离求积误差与保存帧插值误差

制造场为 1+0.2x+0.3y+0.1t+0.02xyt。8×8 网格，非均匀帧 [0,0.7,2.1,4.8,8,12]，折线边经过顺序 [0,1,0]，速度 4，出发 0.4，等待 [.3,.2,.4,.5]。独立解析积分为 45.19066468575263；Gauss 得到 45.190664685752644；绝对差 1.4210854715202004e−14。

梯形求积分两组实验，把 spatial_step_m 或 time_step_s 依次取 .8、.4、.2、.1、.05，另一上限设为 100，使其不主导细化。两组均保持同一重构与强制分段，从而单独检查积分近似。

保存帧插值另用空间常数场 1+exp(−0.3t)，同一完整旅程，独立精确时间积分。非均匀帧为 12·linspace(0,1,N+1)^1.2：

| 帧区间数 | 最大帧间隔 (s) | 暴露绝对误差 |
| ---: | ---: | ---: |
| 4 | 3.5032123983 | 0.1534200354 |
| 8 | 1.7767046035 | 0.0373634688 |
| 16 | 0.8942785172 | 0.0093398028 |
| 32 | 0.4485818635 | 0.0023497903 |
| 64 | 0.2246469623 | 0.0005870733 |

这个实验没有混入 PDE 数值推进或空间重构改变。另有测试检查周期平移波 c=2+0.3 cos(k(x−2t))，旅客按相同速度移动时的独立常暴露参考；这是制造解观察测试，不是本批新增了输运求解器。

## 6. 实验包设计与验证边界

归档目录为 /path/to/Flow-Research/reports/research/temporal_baseline。

| 顶层产物 | 内容 |
| --- | --- |
| config.json | 规范化后的 schema_version=1 完整协议。 |
| network.json | 合成走廊图，完整节点、边、坐标与来源说明。 |
| report.json | 主结果、求解诊断、候选身份、逐路径结果、重构实验、动态反例、搜索动作与解析复核、源码和环境 provenance。 |
| paths.csv | 三个 case × 两条候选，共 6 行数据；长度、时刻、积分、目标及误差。 |
| raw_fields.npz | 15 个原始数组，禁止 pickle 读取。 |
| temporal_evidence.png | 三联证据图。 |
| REPORT.md | 包内英文结果说明、范围与重放命令。 |
| source_snapshot/ | 22 个源码/包入口/依赖锁文件，保存本次实验实际使用的实现。 |
| manifest.json | schema_version、completed 状态、其余 29 个文件的大小和 SHA-256；不对自己递归计算哈希。 |

15 数组分别是 times_s、initial_cell_averages；每个 case 的 __fields、__continuous_nodes、__exact_cell_averages 共 9 个；以及 polynomial_frames、polynomial_times_s、trajectory_xy、trajectory_times_s 共 4 个。主帧形状 (97,40,40)，初值 (40,40)，多项式帧 (6,8,8)。

write_bundle 必须写入新目录，已有输出拒绝覆盖。全部内容先写入同父目录的临时目录，最后生成 manifest 并自验证，成功后通过目录 rename 发布；失败时清理暂存目录，不留下可被误认为完成的目标包。此流程实现完整目录的发布；不把它额外宣传为经过故障注入验证的断电持久性协议。

verify_bundle 实际执行的检查包括：

1. manifest 版本与完成状态、必要顶层文件。
2. 文件路径不允许绝对路径、父目录穿越和反斜线；目标必须是安全的常规文件，不是符号链接，解析后仍位于包内。
3. 每个文件的大小与 SHA-256；实际文件集合等于 manifest 集合加 manifest 自身，拒绝未列出的文件。
4. 使用 Study 验证配置；报告版本、configuration_id、报告内重复 config 和 case ID 顺序必须一致。
5. 必需源码必须存在；报告的 source_snapshot_sha256 与 manifest 的所有源码列表及哈希完全对应。
6. 从 network.json 重建 RoadNetwork，重新枚举完整简单候选，比较报告中的图/候选身份。
7. 在读取 NPZ 数组前检查 ZIP 的未压缩总大小；限制为 128 MiB 加头部预留；allow_pickle=False。
8. 数组键集合必须符合协议；shape、实数 dtype、有限性正确；输出时刻与配置重建结果一致；制造帧/轨迹记录与原始数组一致；轨迹再通过 PathTrajectory 验证。

这比“文件没有传坏”更严格，但仍不是科学正确性证明。验证器不会重新求解每个 PDE、重新证明每个搜索最优值或对报告每个数字做完整推导。哈希也不是身份认证签名。CLI 的 scope 明确限制为产物哈希、配置身份、候选及原始数组 schema。

实验运行时 HEAD 仍为 420ac77，工作树已有本批改动，因此包内 provenance 保留 git_dirty=true 和当时 HEAD，而不事后改成 001a4b8 假装实验已经在该提交后运行。精确源码哈希随后与最终本地提交源码核对一致。记录的实验环境为 Python 3.12.6、macOS 26.5.2 arm64、NumPy 2.5.3、SciPy 1.18.1、Pydantic 2.13.5。

## 7. 本批审查发现并修复的细节

| 问题 | 最终处理 | 核验方式 |
| --- | --- | --- |
| 多段折线逐段计时与搜索整边计时相差 1 ULP | 每条边的最终时刻统一为 edge_start+stored_length/speed，保留内部弯点；不放宽 horizon。 | 特定四点折线、不同出发/等待/重复经过及 exact horizon 回归。 |
| 时刻在格点之后一个浮点数，被近似相等错误归回前一格 | 使用严格大小比较校正 ceil，禁止 isclose 可行性容差。 | nextafter 到达必须向上取整。 |
| 只验证 manifest 哈希，错误 shape/NaN/schema 重新哈希后仍可能通过 | 增加配置、case、源码、候选、数组结构和实际文件集合验证。 | 破坏内容后重新计算哈希，验证仍拒绝。 |
| 数组预算只检查单 case，三类数组乘多个 case 后膨胀 | 总量预算覆盖所有 case 的三组主数组及共同数组。 | 128² 网格、8 case 等曾可达到约 1.9 GiB 的参数现在预检查拒绝。 |
| 时间端点有限但差值溢出 | 轨迹、帧和适配器同时验证相邻间隔/总时长有限。 | 极端有限端点与无限差值输入拒绝。 |
| 路径连通但几何端点略错，可能瞬移到等待节点 | 研究适配器要求精确节点/几何连接，动作恢复不自动补连接。 | 0.01 m 端点偏移与断裂动作拒绝。 |
| 逐动作积分类似正确，但可能漏算整程等待 | 从动作恢复一个完整轨迹，再做重构积分与独立解析积分。 | 四档搜索结果的差异仅约 1e−14。 |
| 开始就等于目的地，却按全时间表扣状态预算 | 直接返回零成本空路线，状态容量为 1。 | max_states=1 的空路线测试。 |
| 单次回调值有限，累计积分/目标却溢出 | 求和和 objective 检查，溢出抛 ValueError。 | 巨大有限暴露和 λ 回归。 |
| 归档源码重放写入 __pycache__，污染封存包 | 重放命令显式设置 PYTHONDONTWRITEBYTECODE=1。 | 重新核验源包 29 项哈希/文件集合。 |
| 重绘或失败发布改变原产物 | 重绘输出同级新 PNG；新包暂存验证后发布，失败清理。 | 字节一致重绘及绘图失败回归。 |
| 文档把新能力误说成网页已接入或连续时间最优 | README/指南/报告统一标记 CLI/Python、有限取整图及冻结地图边界。 | 交付文档与实现核对。 |

最初的 1 ULP 复现折线为 [(0.1,0.1), (0.31529526772962263,0.7067003751060119), (0.5493078171597499,0.5681219753894384), (0.9,0.9)]，速度 1.4：整边公式得到 0.998977204912758，旧逐段累加得到 0.9989772049127581。这个差别虽然数值极小，却会改变严格时间边界下是否允许整段观测，因而专门修复并保留回归测试。

## 8. 本批验收证据与数量

最终全量命令为：

    FLOW_REQUIRE_NATIVE=1 .venv/bin/python -m pytest -q

结果：**318 passed，2 warnings，7.17 s**。环境变量要求原生模块真正可用，避免以跳过 C++ 测试冒充通过。两个提醒来自既有依赖：Starlette TestClient 的 httpx 使用弃用提醒，以及 anyio BlockingPortal 别名弃用提醒。它们不是本批数值测试失败。

其他已完成核验：

- 从归档 source_snapshot 重放，15 个 raw 数组逐位相同。
- 科学记录在去掉 provenance、build_ms、solve_ms、timings_ms 后一致；这几个字段涉及运行环境或耗时，不强求重放逐字相等。
- 29 个 manifest 产物的磁盘与 Git 索引哈希核对通过；来源快照与最终提交源码匹配。
- 从保存报告单独重绘 PNG，与原图逐字节相同，源实验包未改写；图做过视觉检查。
- Git diff 格式检查通过；原项目 120 文件不变。
- 本批前端没有改动，因此没有重复构建或 lint；第一批通过记录保留。没有新增 Linux CI 运行声明。

测试时间与逐位一致只描述记录时的本地环境，不承诺在不同机器、BLAS/编译器或依赖版本上完全逐位相同。本文编写期间未重新跑基准或新增数值实验，避免把新的工作树状态混入历史第二批证据。

本批新增验收 12 项，第一批 28 项，累计 **40/186 项必做**；余下 146 项必做和 12 项可选。条目数量不是工程工时百分比。

| 新完成 ID | 条目与对应证据 |
| --- | --- |
| M01-02 | 显式传递重构与边界规则：空间、时空观察共享 bilinear_weights 契约，三类边界测试。 |
| M05-07 | 时空插值：真实物理秒数与非均匀帧。 |
| M05-08 | 时间覆盖：完整旅程、等待和严格末帧边界。 |
| M05-09 | 时变解析案例：常数、时间仿射、多项式、平移波、冻结极限。 |
| M05-10 | 空间/时间采样独立控制：内部 dt、输出帧、求积、搜索时间分别记录。 |
| M05-11 | 等待与重复路段：同一边不同到达时刻分别积分。 |
| M06-03 | 手算小图：平行边、不可达、空路线、回路、换路、不同到达时间和独立动作枚举。 |
| M06-07 | 时间展开参考图：真实行走、收费 ceil wait、主动等待、horizon 与终点语义。 |
| M06-08 | 动态标签：4 对 102 反例与节点/时间保留。 |
| M06-09 | FIFO 与暴露目标区分：恒定旅行时间仍不能按空间节点删状态。 |
| M06-10 | 时间离散误差跟踪：四档步长、完整动作解析重新评分，明确连续精度未被证明。 |
| M06-14 | 冻结与时变系统比较：不变、缓变、明显失效的相同模型/出发条件对照。 |

## 9. 复现、验证和阅读命令

运行需要 Flow-Research 已准备好的独立 .venv。每次实验输出目录必须尚不存在；以下 my-* 名字如果已经使用，应换一个新名字。命令不会覆盖 temporal_baseline。

    cd /path/to/Flow-Research
    .venv/bin/python -m scripts.run_temporal_experiments --config configs/research/temporal_study.json --output reports/research/my-temporal-study
    .venv/bin/python -m scripts.run_temporal_experiments --verify reports/research/my-temporal-study
    .venv/bin/python -m scripts.run_temporal_experiments --replot reports/research/my-temporal-study

--replot 不重求 PDE，也不重跑速度测量。它输出与包同级的 my-temporal-study-replot.png；若已存在则拒绝覆盖。

从本批封存源码重放，即使工作树以后已进入第三批，仍可以使用：

    cd /path/to/Flow-Research/reports/research/temporal_baseline/source_snapshot
    PYTHONDONTWRITEBYTECODE=1 /path/to/Flow-Research/.venv/bin/python -m scripts.run_temporal_experiments --config ../config.json --output /path/to/Flow-Research/data/experiments/research/my-temporal-replay
    PYTHONDONTWRITEBYTECODE=1 /path/to/Flow-Research/.venv/bin/python -m scripts.run_temporal_experiments --verify /path/to/Flow-Research/data/experiments/research/my-temporal-replay

禁用字节码写入是为了不向封存源目录添加 __pycache__。这里仍使用已有解释器和安装依赖；依赖锁记录了版本，但不会自动重建整个操作系统。严格复现时需同时保持相容依赖环境。

查看本批提交和清单，而不切换当前工作树：

    git show 001a4b85118a641934b67d19eac096d4c08942e8:backend/app/research/dynamic_routing.py
    git diff --name-status 420ac77..001a4b85118a641934b67d19eac096d4c08942e8

下面两个主产物可以直接打开：[第二批机器报告](../../reports/research/temporal_baseline/report.json)、[第二批图](../../reports/research/temporal_baseline/temporal_evidence.png)。其内容和数值已在本文解释，而不是要求读者转去别处才能理解交付。

## 10. 本批没有完成的部分与第三批衔接

本批没有实现一般变系数/变风/源项、高阶输运、PCG/多重网格、伴随、自适应网格、可靠误差估计、传感器噪声建模或完整研究 UI。没有用真实观测校准空气质量，也没有给动态积分或搜索做新的可信速度提升测量。

有限图参考的预算限制使它适合可穷举/可独立核对的小图。它不自动扩展成大规模城市动态导航，也不解决连续出发时刻、无限等待策略和连续控制的全局最优问题。观察采样预算、归档数组预算和缓存保留预算分别限制各自资源，都不是进程峰值内存上限。

第三批将从 M02 的正标量变扩散系数、面通量和制造解验证推进，再支撑 M04 的 matrix-free 迭代和预条件比较。这个顺序让后续性能研究比较的是相同模型与精度。第三批实际代码和验收另行记录，不能追写为第二批已完成。

## 11. 完整 Git 文件清单：全部 41 个路径

以下清单由历史区间 git diff --name-status 420ac77..001a4b8 提取；M 表示修改，A 表示新增。共 41 个路径，历史统计为 11,695 行增加、31 行删除，其中大量行是报告 JSON 和已用源码快照，并不等于新增算法代码行数。每个 source_snapshot 文件都是实验封存副本；即使副本为 A，也不表示原模块在第二批重新实现。

| 序号 | 状态 | 路径 | 职责/本批变化 |
| ---: | --- | --- | --- |
| 1 | M | [README.md](../../README.md) | 更新时变能力概览、地图/API 冻结边界、研究报告链接和目录导航。 |
| 2 | M | [backend/app/observations.py](../../backend/app/observations.py) | 严格 PathTrajectory，新增 stationary trajectory、waits_s 与逐边权威时钟。 |
| 3 | A | [backend/app/research/dynamic_routing.py](../../backend/app/research/dynamic_routing.py) | 节点/时间 DAG 参考搜索、收费取整等待、严格 horizon 与预算/身份契约。 |
| 4 | A | [backend/app/research/travel.py](../../backend/app/research/travel.py) | 非负时变场适配器、有界标量缓存、完整动作轨迹恢复。 |
| 5 | A | [backend/app/temporal_observations.py](../../backend/app/temporal_observations.py) | 真实时刻采样、CSR 时空积分、Gauss/梯形分段与样本预算。 |
| 6 | A | [configs/research/temporal_study.json](../../configs/research/temporal_study.json) | 三场景生产 PDE 和四档搜索时间步的版本化默认协议。 |
| 7 | M | [docs/research.md](../research.md) | 研究使用指南；时空/等待/动态搜索语义、复现命令、证据与限制。 |
| 8 | A | [reports/research/temporal_baseline/REPORT.md](../../reports/research/temporal_baseline/REPORT.md) | 包内英文研究结果、解释范围和归档重放命令。 |
| 9 | A | [reports/research/temporal_baseline/config.json](../../reports/research/temporal_baseline/config.json) | 本次运行规范化配置；与报告 configuration_id 交叉验证。 |
| 10 | A | [reports/research/temporal_baseline/manifest.json](../../reports/research/temporal_baseline/manifest.json) | 完成标记及其余 29 文件大小/SHA-256 清单。 |
| 11 | A | [reports/research/temporal_baseline/network.json](../../reports/research/temporal_baseline/network.json) | 完整合成走廊图及来源元数据。 |
| 12 | A | [reports/research/temporal_baseline/paths.csv](../../reports/research/temporal_baseline/paths.csv) | 三个场景的六条候选记录，方便外部表格或脚本分析。 |
| 13 | A | [reports/research/temporal_baseline/raw_fields.npz](../../reports/research/temporal_baseline/raw_fields.npz) | 15 个数值/解析/轨迹原始数组，二进制归档。 |
| 14 | A | [reports/research/temporal_baseline/report.json](../../reports/research/temporal_baseline/report.json) | 完整机器结果、动作、对照、诊断、控制参数、来源和限制。 |
| 15 | A | [reports/research/temporal_baseline/source_snapshot/backend/__init__.py](../../reports/research/temporal_baseline/source_snapshot/backend/__init__.py) | 封存副本；backend 包入口，保证离开工作树仍可导入。 |
| 16 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/__init__.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/__init__.py) | 封存副本；app 包入口。 |
| 17 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/diffusion.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/diffusion.py) | 封存副本；实验采用的 Grid、基础场和共享空间双线性重构。 |
| 18 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/main.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/main.py) | 封存副本；归档当时的 API 服务实现；本批未把时变控制接进 API。 |
| 19 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/numerics/__init__.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/numerics/__init__.py) | 封存副本；数值求解统一导出入口。 |
| 20 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/numerics/native.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/numerics/native.py) | 封存副本；已有原生后端加载、校验和分派逻辑。 |
| 21 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/numerics/operators.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/numerics/operators.py) | 封存副本；本批实际使用的常系数空间算子实现。 |
| 22 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/numerics/solver.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/numerics/solver.py) | 封存副本；生产时间推进/CN 求解与诊断，作为实际实验依赖封存。 |
| 23 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/numerics/validation.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/numerics/validation.py) | 封存副本；已有数值验证框架。 |
| 24 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/observations.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/observations.py) | 封存副本；本批空间观察、等待轨迹与权威时钟的实际实验副本。 |
| 25 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/research/__init__.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/research/__init__.py) | 封存副本；research 包入口。 |
| 26 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/research/decision.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/research/decision.py) | 封存副本；完整简单路径枚举、走廊图、候选成本和已有条件误差界。 |
| 27 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/research/dynamic_routing.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/research/dynamic_routing.py) | 封存副本；本次运行的有限时间展开参考搜索副本。 |
| 28 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/research/experiments.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/research/experiments.py) | 封存副本；已有实验框架和运行环境/源码 provenance 功能。 |
| 29 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/research/problems.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/research/problems.py) | 封存副本；严格配置基类、规范化内容身份及已有 PDE 协议。 |
| 30 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/research/travel.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/research/travel.py) | 封存副本；本次运行的帧适配、缓存及动作恢复副本。 |
| 31 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/routing.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/routing.py) | 封存副本；有向多重图、完整折线长度和既有静态路线支持。 |
| 32 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/temporal_observations.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/temporal_observations.py) | 封存副本；本次运行的时空插值与稀疏观察副本。 |
| 33 | A | [reports/research/temporal_baseline/source_snapshot/backend/app/validation.py](../../reports/research/temporal_baseline/source_snapshot/backend/app/validation.py) | 封存副本；既有应用验证工具；随完整 backend 源树归档。 |
| 34 | A | [reports/research/temporal_baseline/source_snapshot/requirements.lock.txt](../../reports/research/temporal_baseline/source_snapshot/requirements.lock.txt) | 封存副本；记录 Python 依赖版本，便于解释和重建相容运行环境。 |
| 35 | A | [reports/research/temporal_baseline/source_snapshot/scripts/__init__.py](../../reports/research/temporal_baseline/source_snapshot/scripts/__init__.py) | 封存副本；允许 python -m scripts.run_temporal_experiments 的包入口。 |
| 36 | A | [reports/research/temporal_baseline/source_snapshot/scripts/run_temporal_experiments.py](../../reports/research/temporal_baseline/source_snapshot/scripts/run_temporal_experiments.py) | 封存副本；可直接重放本次研究的实验脚本快照。 |
| 37 | A | [reports/research/temporal_baseline/temporal_evidence.png](../../reports/research/temporal_baseline/temporal_evidence.png) | 已视觉核查的三联图，支持不重算直接重绘。 |
| 38 | A | [scripts/run_temporal_experiments.py](../../scripts/run_temporal_experiments.py) | 研究协议、闭式/多项式参考、实验运行、绘图、归档验证与 CLI。 |
| 39 | A | [tests/test_dynamic_routing.py](../../tests/test_dynamic_routing.py) | 独立动作树 oracle、反例、等待、严格边界、预算和失效测试。 |
| 40 | A | [tests/test_temporal_observations.py](../../tests/test_temporal_observations.py) | 插值/积分不变量、制造参考、覆盖、轨迹及权威时钟回归。 |
| 41 | A | [tests/test_temporal_research.py](../../tests/test_temporal_research.py) | 生产 PDE、适配器、动作审计、解析积分、预算和归档整体验证。 |

## 12. Git 清单之外的本地规划、运行与核验文件

本项目沿用本地研究规划不随源码提交的约定：独立仓库的 .git/info/exclude 排除 START_HERE.zh-CN.md 和 planning/。因此下列本地进度文件不是上述 41 个跟踪路径，不能把它们当成缺失的提交产物。它们补充管理和本地验收信息。

| 本地文件/目录 | 作用与本批状态 |
| --- | --- |
| START_HERE.zh-CN.md（本地记录 `START_HERE.zh-CN.md`） | 本地中文入口，说明当前能力、里程碑和下一阶段。 |
| planning/03_MASTER_TODO.zh-CN.md（本地记录 `planning/03_MASTER_TODO.zh-CN.md`） | 完成状态的主清单；第二批勾选上述 12 项，累计 40 项。 |
| planning/BACKLOG.json（本地记录 `planning/BACKLOG.json`） | 从主清单生成的机器索引，避免维护两套互相矛盾的完成状态。 |
| planning/06_TEMPORAL_PROGRESS.zh-CN.md（本地记录 `planning/06_TEMPORAL_PROGRESS.zh-CN.md`） | 第二批精简进度版；本文件在其基础上加入完整设计、数学和文件说明。 |
| planning/IMPLEMENTATION_PHASE2.json（本地记录 `planning/IMPLEMENTATION_PHASE2.json`） | 本地验收：318 测试、15 数组、29 哈希、原项目不变、12 新完成 ID、环境与范围记录。 |
| planning/validation/phase2-pytest.txt（本地记录 `planning/validation/phase2-pytest.txt`） | 保存最终全量测试输出及两条依赖弃用提醒。 |
| planning/refresh_backlog.py（本地记录 `planning/refresh_backlog.py`） | 从主清单刷新 BACKLOG.json 的既有工具，本批复用。 |
| 本文件 planning/PHASE2_COMPLETE.zh-CN.md | 应用户后续要求汇总的单文件完整第二批说明；不追写进历史提交，不更改封存实验包。 |

.venv、已有原生构建、缓存、忽略的 data/experiments 运行输出及临时重放/重绘副本属于运行环境或核验辅助，不是可交付源码清单的替代品。可复现的主证据已经完整保存在受跟踪的 temporal_baseline，具体机器上的临时目录名不影响本批实验定义。

本文读取历史 Git 源码、历史报告和本地验收记录后编写；没有新增第二批测试、改动实验、重跑基准或修改原项目。41 项历史路径及本文本地链接均做存在性核对。本文件负责把已经完成的第二批讲清楚，第三批以新的实现与证据继续推进。
