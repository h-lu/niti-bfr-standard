# Braided Device Video-Only Formal Method

更新时间: 2026-04-20

本文档用于把当前仓库对 `braided device` 的方法学定位正式落下来，供主线程后续指挥实现与验收。

本文只讨论一个明确前提:

- 输入只有视频
- 不额外假设 `3DRA`、`CBCT`、血管 `centerline`
- 不额外假设器械的完整几何参数、编织参数或部署边界条件

本文是对现有调研与 README 的收敛版方法定义，相关上下文见:

- [README.md](/Users/wangxq/Documents/niti-bfr-standard-2/README.md)
- [braided-device-survey.md](/Users/wangxq/Documents/niti-bfr-standard-2/docs/braided-device-survey.md)
- [research-notes.md](/Users/wangxq/Documents/niti-bfr-standard-2/docs/research-notes.md)

本文同时补充一个新的收敛要求:

- 对 braided 对象, 当前仓库也需要提供一条 `YY/T 1771` 对齐的 `formal Af` 工作流
- 但该“对齐”只指对齐 `恢复量 vs 温度 -> Af-95 / Af-tan` 的测试与计算框架
- 不应被改写成“braided finished product 已经被原始 `wire / tube / strip` 标准范围逐字覆盖”

为便于后续像 `wire-like` 一样简洁比较三条 braided 主量, 本文同时约定:

- `A = length_axis`
- `B = diameter_max`
- `C = area_proj`

其中 `B` 的长期方法学定位是 braided 的宽度/直径视角; 当前实现里这一路线由 strict body-only 最大正交宽度
`diameter_max = max_s w_orth(s)` 承载, 不应用 `diameter_mid_p90`、厚度 proxy 或 Feret proxy 混写成 route B 本身。

其中:

- `A / B / C` 代表 braided 的三条并行分析路线
- 每一次 braided 分析都应为 `A / B / C` 三条路线分别输出自己的 route-level result
- route-level result 至少表示该路线“算出了什么、值列/恢复列是什么、状态如何、为什么可能不能 formal”，不等于该路线已经 `formal passed`
- 当前对象级默认 formal 主量仍是 `A`
- 当前代码实现里 `B` 已进入 braided object-level formal candidate
- `C` 当前稳定输出自己的 route-level result, 并带独立 `reportability_status / gate_reason / warning_codes / route_qc_summary`, 但仍不是默认 object-level formal candidate
- 对外表述可写成 `A:length_axis`、`B:diameter_max`、`C:area_proj`

这里要明确区分:

- “当前实现默认 formal 放行 `A`” 属于实现状态
- “当前代码里 `B` 已进入对象级 formal candidate, `C` 稳定输出 route-level result 但仍未进入默认对象级 formal candidate” 也属于实现状态
- “A / B / C 长期并行保留并比较” 属于项目的方法学目标

进一步地, 当前文档默认采用下面的语义:

- `route-level result` = 该路线自己的量、值列/恢复列、Af 估计、QC 与状态
- `object-level formal result` = 当前 run 最终允许对外作为正式结论的主路线结果
- `object-level selection` = 当前 run 在 route-level result 之上做出的 `primary / formal / recommended` 选择

route-level 的稳定字段口径也应固定为:

- `alias`
- `metric_key`
- `display_label`
- `value_series_col`
- `recovery_series_col`
- `af95_c`
- `aftan_c`
- `fit_rmse`
- `monotonic_violation_fraction`
- `dynamic_range`
- `reportability_status`
- `gate_reason`
- `warning_codes`
- `formal_candidate`
- `accepted_as_formal_candidate`
- `selected_as_primary`
- `selected_as_formal`
- `formal_role`
- `auxiliary_metric_keys`
- `route_qc_summary`

对象级字段则应作为 route-level 的平行层读取:

- 主读路径先用 canonical `object_*` contract
- `object_reportability_status`
- `object_formal_metric_key` / `object_formal_route_alias`
- `object_provisional_metric_key` / `object_provisional_route_alias`
- `object_recommended_metric_key` / `object_recommended_route_alias`
- `object_formal_gate_reason`
- `object_formal_af95_c` / `object_formal_aftan_c`
- `object_provisional_af95_c` / `object_provisional_aftan_c`
- `formal_metric_label`、`provisional_metric_label`、`primary_metric_label` 继续保留为兼容现有 summary/UI 的展示字段, 但不再作为主 contract

同时要明确:

- route-level `formal_passed` 表示该路线自己的 gate 已放行
- `selected_as_primary` / `selected_as_formal` 表示该路线是否被对象级流程选中
- 对象级 canonical `object_*` 字段表示当前 run 最终推荐展示的是哪一路; `formal_metric_label` / `provisional_metric_label` 仅保留为兼容展示层
- 二者相关, 但不是同一个层次的判断

因此, 同一个 braided run 里可以同时出现:

- `A / B / C` 三条路线都有结果
- 其中只有一条是当前对象级 formal 主量
- 或三条路线都只有 `quicklook / provisional / formal_blocked` 状态, 没有任何一路被放行为正式结论

route-level QC 摘要的读取顺序也应固定:

1. 先读 `reportability_status`、`gate_reason`、`warning_codes`
2. 再读跨路线可比的 `fit_rmse`、`monotonic_violation_fraction`、`dynamic_range`
3. 再读该路线自己的 `route_qc_summary`, 并固定按统一骨架理解:
   `valid_points`、`quality_median`、`monotonic_violation_fraction`、`tail_recovery_median`、`stability_metric`
4. 对 braided 再补读 area/axis/body-only 相关 QC:
   `formal_qc`、`centerline_disagreement_median`、`body_mask_attachment_leak_fraction_median`、`area_proj_definition_gap_px2`

其中 braided 三条路线当前应按下面的 route-specific QC 语义理解:

- `A:length_axis` 重点看 `route_qc_summary.stability_metric.key = centerline_disagreement_median`
- `B:diameter_max` 重点看 `route_qc_summary.stability_metric.key = axis_peak_position_stability_p95`
- `C:area_proj` 重点看 `route_qc_summary.stability_metric.key = area_definition_gap_fraction_p95`

换句话说, braided 的 route-level QC 摘要并不意味着“所有 QC 都塞回 route entry 一个字段”; 更准确的定义是:

- route entry 提供稳定的路线级结果骨架
- 对象级 summary 和 benchmark summary 提供对象/场景共享的关键 QC 量
- 页面、JSON、benchmark 汇总都应按这套口径组织阅读

## 1. 方法边界

### 1.1 只有视频时，可正式采用的方法

在只有视频输入的条件下，当前仓库可以正式采用的是一类:

- `2D video-based geometric measurement method`
- 或更具体地说:
- `video-only 2D axis-and-diameter-profile method for braided device geometry`

它的核心不是 patient-specific deployment 模拟，而是从视频中稳定提取二维投影几何量，并围绕这些量组织正式指标。

这类方法可以正式围绕以下对象表示展开:

- `axis / projected centerline`
  在二维投影下定义器械主轴或投影中心线
- `D(x)`
  沿主轴位置 `x` 的局部宽度或投影直径分布
- `zone segmentation`
  基于 `D(x)` 定义 `landing zone / transition zone / compaction zone`
- `foreshortening`
  跟踪长度变化及其相对缩短比

### 1.2 只有视频时，不应越界声称的方法

在只有视频输入时，不应把当前仓库的方法包装成下面这些更高级方法:

- `patient-specific virtual deployment`
- `clinical-grade local porosity quantification`
- `device sizing and landing prediction` for patient-specific anatomy
- `wall apposition` or `stent-wall apposition`
- 基于真实血管与器械参数的部署预测

这些主张通常需要至少一部分额外输入:

- `3D/CBCT/Dyna-CT/3DRA`
- 血管 `centerline` 与局部半径
- 器械结构参数或编织参数
- 部署边界条件
- 更高阶的几何模型、虚拟部署模型或 FE / reduced-order model

因此，只有视频时，当前仓库的正式目标应是:

- 稳定地量测二维投影几何
- 给出分区化、可解释的几何主量
- 为后续更高阶方法保留接口

而不是提前声称已经实现商业规划软件那一档能力。

## 2. 术语对齐

为便于与论文和商业软件对话，当前仓库后续文档与实现建议统一采用下面的术语。

### 2.1 `centerline / axis`

在只有视频输入时，当前仓库可正式提取的是:

- 二维投影下的 `axis`
- 或 `projected centerline`

这里不应把它写成真实三维血管中心线，也不应把它等同于 patient-specific vascular `centerline`。

### 2.2 `D(x)`

`D(x)` 指沿 `axis` 或 `projected centerline` 的局部投影宽度分布。

在当前问题里，它是第一层正式几何对象，因为它直接承载:

- 最大直径位置
- 两端收口
- 中段鼓包
- 分区边界
- 长度与缩短行为

### 2.3 `foreshortening`

`foreshortening` 是 braided device 文献和商业软件都重视的主量。

在只有视频时，当前仓库可以正式主张的是:

- 二维投影意义下的 `foreshortening`
- 即基于视频中投影长度变化得到的轴向缩短比

不应主张的是:

- 在真实三维解剖中的部署缩短预测

### 2.4 `landing zone / transition zone / compaction zone`

当前仓库后续应把左右收口和中段鼓包，正式改写为分区语义:

- `landing zone`
  近端或远端的着陆/收口区
- `transition zone`
  从端部向中段过渡的区域
- `compaction zone`
  中段更致密、更鼓起的主区

在只有视频时，这些分区应基于 `D(x)` 的几何分段来定义，而不是基于 3D 部署后丝线接触状态来定义。

### 2.5 `porosity`

`porosity` 是文献与商业软件里的高价值术语，但在只有视频时需要谨慎。

可以正式写的是:

- `porosity proxy`
- `mesh-density proxy`

不建议直接写成:

- 真实 `local porosity`
- clinical-grade porosity quantification

原因是二维图像中的网纹、照明、焦点、缩放和 aliasing 都会显著影响估计。

### 2.6 `virtual deployment`

`virtual deployment` 在文献和商业软件里通常意味着:

- 已有解剖几何
- 已有器械模型
- 通过几何规则、仿真或力学模型预测部署结果

当前仓库在只有视频输入时，不应把现有 braided 路线称为 `virtual deployment`。

### 2.7 `apposition`

`apposition` 或 `wall apposition` 指器械与血管壁的贴附情况。

这类结论需要血管壁信息、部署几何或 3D 成像支持。只有视频输入时，当前仓库不应声称已经量测或预测 `apposition`。

## 3. 论文/商业软件主流方法 vs 当前仓库 braided 方法

下表用于明确当前仓库方法在行业坐标里的位置。

| 方法类型 | 典型输入 | 几何表示 | 主要输出量 | 可以可信主张的内容 | 不能主张的内容 |
| --- | --- | --- | --- | --- | --- |
| 增强可视化类方法 | 视频、透视序列、ROI | 增强后的可视对象 | visibility enhancement、overlay、ROI visualization | 把器械看得更清楚，辅助人工观察 | 不能代替正式几何量测、部署预测、porosity/apposition 结论 |
| 文献中的视频/图像量测法 | 2D 图像或视频 | `axis/projected centerline + D(x)` | `L`、`Dmax`、`D(x)`、`foreshortening`、zone-based geometry | 二维投影几何量测、分区化描述、时序变化追踪 | 不能直接替代 3D patient-specific virtual deployment |
| 商业/学术虚拟部署方法 | `3DRA/CBCT`、血管 `centerline`、器械参数、部署边界条件 | vessel centerline + envelope + device model + wire model | deployment result、landing、device sizing、local porosity、apposition | patient-specific planning、部署后几何预测、落点和贴壁分析 | 不能由单纯视频 quicklook 等价获得 |
| 高保真仿真/FE/ROM 方法 | 3D 几何、器械参数、材料与接触模型 | wire-level geometry / shell / FE model | deployment mechanics、stress、apposition、porosity、final geometry | 更高保真地研究部署行为与结构响应 | 不能由当前仓库视频输入直接落地 |
| 当前仓库 braided 方法 | 视频 | 外轮廓 + 主轴/宽度/面积三路线量测 | 主轴长度、最大直径、投影面积、左右收口定位、route-level status/gate reason、QC 叠加帧 | 可作为 `video-only 2D geometric measurement and zone analysis`，适合做路线级比较、QC 与方法学验证 | 不应声称 patient-specific virtual deployment、clinical-grade porosity、apposition |

当前仓库与主流方法的关系应理解为:

- 它已经进入了 `2D 图像几何量测法` 这一大类
- 当前实现已经稳定输出 `A / B / C` 三条路线的 route-level result, 不再只是单一路径的 quicklook 对照
- 当前 route B 的稳定口径是 braided 的宽度/直径路线, 当前实现键为 strict body-only `diameter_max`, 不是中段 proxy 的统称
- 与文献和商业软件的共同语言已经出现，但尚未达到它们的高级主张条件

因此在对外展示层, 更合适的读取顺序是:

1. 先看 `A / B / C` 三条路线各自的 route-level result 与 QC 摘要
2. 再看对象级 canonical `object_*` 字段, 尤其是 `object_reportability_status`、`object_formal_metric_key`、`object_provisional_metric_key`、`object_recommended_metric_key`、`object_formal_gate_reason`
3. 如需兼容旧 summary/UI，再回看 `formal_metric_label`、`provisional_metric_label`、`primary_metric_label`
4. 最后才下结论“本次 braided run 的对象级正式推荐是谁”

## 4. 当前仓库 braided 路线的建议定位

当前仓库不建议继续把 braided 路线仅仅描述为:

- `quicklook`

更适合的正式说法是:

- `video-only 2D geometric measurement workflow for braided device`
- 或
- `axis-and-diameter-profile based geometric analysis for braided device videos`

如果需要更偏论文语言的简洁表述，建议优先采用:

- `video-only 2D geometric measurement and zone analysis`

这样既保留了视频输入的现实边界，也把方法的正式对象从“单一长度快看”提升为“几何量测 + 分区分析”。

## 5. 建议主量

当前仓库 braided 路线后续建议围绕下面这些主量组织实现、QC 与验收。

### 5.1 第一层主量

- `L_axis`
  基于 `axis` 或 `projected centerline` 的功能长度
- `L_env`
  基于包络边界定义的投影长度
- `Dmax`
  最大投影直径
- `D(x)`
  沿轴向的局部投影宽度分布
- `x_peak`
  最大直径所在的归一化轴向位置
- `FS`
  二维投影意义下的 `foreshortening`

### 5.2 第二层分区量

- `landing_zone_left/right`
  左右端着陆/收口区长度
- `transition_zone_left/right`
  左右过渡区长度
- `compaction_zone_length`
  中段鼓包或更致密主区长度
- `zone symmetry`
  左右分区对称性

### 5.3 第三层增强量

- `mesh-density proxy(x)`
  局部网纹密度代理量
- `porosity proxy(x)`
  局部孔隙率代理量

这些增强量可以作为解释层，但不建议在当前阶段取代第一层主量。

## 5A. `YY/T 1771` 对齐的 braided formal Af 定义

如果项目目标是把 braided 也纳入与 `wire-like` 类似的
`YY/T 1771-aligned` Af 工作流, 当前实现里最稳妥的正式定义应为:

- 当前默认 formal 主量为 `length_axis(T)`
- `A / B / C` 三条路线都应输出各自的 route-level result
- route-level result 可以包含各自的 `Af-95` / `Af-tan` 估计、QC、status 与 gate reason
- 但 route-level result 本身不等于对象级 formal passed
- 当前对象级默认 formal 主量仍为 `length_axis(T)`
- 当前代码实现已允许 `diameter_max(T)` 进入对象级 formal candidate; 这里的 `diameter_max(T)` 固定指 strict body-only `max_s w_orth(s)`
- `area_proj(T)` 当前稳定输出自己的 route-level result, 并带 area-specific gate、status 与 gate reason, 但仍不作为默认对象级 formal candidate
- 该量表示二维投影下 braided 试样的主轴功能长度
- 升温恢复时, 该量对当前对象应单调减小
- 正式恢复率定义为:
  `R_axis(T) = (L_M - L_axis(T)) / (L_M - L_A)`

其中:

- `L_M`: 低温起始状态下的主轴长度参考值
- `L_A`: 高温完全恢复平台下的主轴长度参考值

据此:

- `Af-95`: `R_axis(T) = 0.95` 时对应的温度
- `Af-tan`: 对 `R_axis(T)` 的主转变区作切线，并与高温平台线求交得到的温度

当前代码侧的字段约定也应保持一致:

- `length_axis_px`: formal `pruned centerline arc length`
- `length_axis_body_bins_px`: 旧 body-bin centerline 弧长, 仅作对照
- `length_axis_alt_px`: 第二条 centerline 估计, 用于 disagreement/QC
- `diameter_max_px`: strict body-only `max_s w_orth(s)`; 与 `diameter_max_orth_px` 同义
- `diameter_mid_p90_px`: 中段窗口 proxy, 只作稳健性/解释层参考
- `area_proj_px2`: formal `integral_body(w_orth(s) ds)`
- `area_proj_contour_width_integral_px2`: 旧 contour-width integral, 仅作对照
- `area_proj_contour_px2`: body-only contour area, 用于面积定义差检查

route-level / summary 命名也应按下面的层级理解:

- route-level 入口: `route_results`、`route_results_by_alias`、`route_alias_order`
- route-level 状态字段: `reportability_status`、`gate_reason`、`warning_codes`
- route-level 选择字段: `formal_candidate`、`accepted_as_formal_candidate`、`selected_as_primary`、`selected_as_formal`、`formal_role`
- route-level QC 字段: `fit_rmse`、`monotonic_violation_fraction`、`dynamic_range`、`route_qc_summary`
- `route_qc_summary` 标准骨架: `valid_points`、`quality_median`、`monotonic_violation_fraction`、`tail_recovery_median`、`stability_metric`
- canonical 对象级字段: `object_reportability_status`、`object_formal_metric_key`、`object_provisional_metric_key`、`object_recommended_metric_key`
- 兼容展示字段: `formal_metric_label`、`provisional_metric_label`、`primary_metric_label`
- benchmark/summary 里的 QC 补充字段: `formal_qc`、`quality_median`、`centerline_disagreement_median`、`body_mask_attachment_leak_fraction`、`axis_peak_position_stability_p95`

这四层不能混写成“一个字段同时表达路线结果和对象级推荐”。

路线级对照与交叉验证量允许保留:

- `length_env(T)`
- `diameter_max(T)`
- `foreshortening_axis(T)`
- `foreshortening_env(T)`

这些量在当前实现中的角色应区分为:

- `diameter_max(T)`:
  braided 的 route B 当前固定由 strict body-only `max_s w_orth(s)` 承载; 当前代码里它已进入对象级 formal candidate, 但不是默认 formal 主量
- `length_env(T)`、`foreshortening_axis(T)`、`foreshortening_env(T)`:
  当前主要用于 route-level 对照、交叉检查与解释层
- `area_proj(T)`:
  当前应稳定输出 route-level result, 并带独立 gate / status / gate_reason, 但仍主要是对照量与方法学验证量, 不是默认对象级 formal candidate

除已进入对象级 candidate 的 `diameter_max(T)` 外, 其他量当前主要用于:

- 交叉检查
- QC
- 方法学验证

不直接参与对象级正式主量竞争。

这样做的原因是:

- `length_axis(T)` 最接近 `YY/T 1771 / ASTM F2082` 中“恢复位移/形变量随温度变化”的主框架
- 它比局部网纹、局部孔隙率代理量更稳健
- 它比 `diameter_max(T)` 更接近“轴向恢复行为”的主运动方向
- 它仍然保持在“二维投影几何量测”的真实边界内

## 6. 当前仓库与正式方法之间的升级方向

主线程后续如果要把 braided 路线从当前状态升级为正式方法，建议按下面的顺序推进。

1. 保留现有外轮廓与 QC 能力，继续作为回退基线。
2. 把当前“主轴长度 + 最大直径 + 左右 taper”统一改写为 `axis + D(x) + zone segmentation`。
3. 明确 `L_env` 与 `L_axis` 的定义差异，避免不同口径混写。
4. 把 `foreshortening` 升级为主量，而不是只做单帧长度展示。
5. 把左右收口正式升级为 `landing zone / transition zone` 语义。
6. 将 `compaction zone` 作为中段核心分区，而不是仅以 `x_peak` 代替。
7. 如需继续扩展，再把 `mesh-density proxy / porosity proxy` 作为增强层接入。

在 `formal Af` 这一条线上, 当前升级顺序建议固定为:

1. 固定 `A / B / C` 三条路线都输出 route-level result, 不因对象级 formal 只放行一条路线就隐藏其他路线结果。
2. 对象级默认 formal 主量仍保持 `length_axis(T)`。
3. 在当前实现状态下, 允许 `diameter_max(T)` 作为对象级 formal candidate 参与比较, 但不要把这件事误写成 “braided 已经三路等权 formal 化”。
4. `area_proj(T)` 先保持 route-level 输出、对照与方法学验证角色; 它应稳定带自己的 gate / status / gate_reason, 但不提前包装成已放行的对象级 formal 主量。
5. 只在温度同步可用、恢复覆盖到高温平台、相应主量曲线单调性可接受时给对象级正式 `Af`。
6. 需要更高层解释时，再把分区量和密度代理量作为结果解释层叠加。
7. benchmark 页面或汇总文件应优先支持六路线横向比较, 但在当前实现状态下先以 braided benchmark suite 打底, 不把“目标中的六路线页面”误写成“今天已经完全对称可用”。

## 6A. Benchmark 页面 / 汇总文件应如何比较路线

当前 braided benchmark 的稳定读取入口是:

- 单场景目录下的 `analysis_metrics.json`
- 单场景目录下的 `analysis.csv`
- suite 目录下的 `benchmark_summary.json`
- suite 目录下的 `benchmark_summary.csv`

对 braided A/B/C 三条路线, 当前推荐的横向比较顺序是:

1. 先看 `af_comparison` 或 `af_comparison_by_alias` 里的 `af95_error_c` / `aftan_error_c`
2. 再看 `route_results` / `route_results_by_alias` 里的 `reportability_status`、`gate_reason`、`accepted_as_formal_candidate`
3. 再看各路线自己的 `selected_as_primary`、`selected_as_formal` 与标准化 `route_qc_summary`
4. 再看 benchmark 共享 QC:
   `quality_median`、`body_mask_attachment_leak_fraction`、`centerline_disagreement_median`、`endpoint_jump_p95_px`、`axis_peak_position_stability_p95`
5. 如需解释面积路线, 额外回看 `area_proj_definition_gap_px2` 或 `area_definition_gap` 相关字段

项目长期上, benchmark 页面或汇总文件应能横向比较六条路线:

- wire `A:x_route_a`
- wire `B:kappa_fit`
- wire `C:kappa_route_c`
- braided `A:length_axis`
- braided `B:diameter_max`
- braided `C:area_proj`

但在当前实现状态下:

- braided 与 wire 都已经有 suite 级 `benchmark_summary.json/csv` 汇总入口
- 两侧单场景文件形态仍不完全相同, 但 summary 阅读体验应统一成“先路线级比较, 再 canonical 对象级选择, 最后回到单场景 QC”
- 因此文档里应把“六路线横向比较”写成当前可落地的 summary 体验与后续页面层继续对齐的方向, 而不是说仓库今天已经在所有页面上完全对称

## 7. 应如何对外表述

后续文档、汇报或实现说明里，建议采用下面这类表述:

- 当前 braided 路线是一种 `video-only 2D geometric measurement and zone analysis` 方法。
- 该方法从视频中提取 `axis / projected centerline`、`D(x)`、`foreshortening` 以及 `landing / transition / compaction` 分区。
- 该方法适合做二维投影几何量测、QC、时序跟踪和方法学验证。
- 对同一个 braided run, `A / B / C` 三条路线都应输出自己的 route-level result。
- route-level result 不等于 formal passed; 它也可以是 `quicklook`、`provisional`、`formal_blocked` 或 `formal_passed`。
- route B 在长期方法学上表示 braided 的宽度/直径路线; 当前实现固定用 strict body-only `diameter_max` 承载, 不与 `diameter_mid_p90` 等 proxy 混写。
- canonical `object_*` 字段是在 route-level result 之上的第二层选择语义, 不能反过来覆盖掉 A/B/C 三条路线自身的输出; `formal_metric_label` / `provisional_metric_label` 继续保留为兼容展示层。
- 当实验布置满足温度同步与完整恢复条件时, 当前仓库还可提供一条 `YY/T 1771-aligned` 的 braided `formal Af` 工作流。
- 这条 `formal Af` 工作流当前默认以 `length_axis(T)` 为对象级 formal 主量; 当前代码实现里 `diameter_max(T)` 已进入对象级 formal candidate, `area_proj(T)` 则稳定输出带 gate / status / gate_reason 的 route-level result, 但仍不是默认对象级 formal candidate。
- benchmark 汇总应优先用于横向比较各路线的可用性、QC 稳定性与 Af 偏差, 不应只盯对象级最后选中的一路。

同时，建议明确写出下面这些限制:

- 当前方法不是 `patient-specific virtual deployment`。
- 当前方法不提供 clinical-grade `porosity` 定量。
- 当前方法不提供 `apposition` 结论。
- 当前方法不能替代需要 `3D/CBCT/centerline/device parameters` 的商业规划工作流。
- 当前方法中的 `YY/T 1771-aligned` 表述, 只表示与标准的 BFR 计算框架对齐, 不表示 braided finished product 已被原始标准范围逐字覆盖。

## 8. 一句话结论

在只有视频输入的条件下，当前仓库 braided 路线最合适的正式定位不是“部署仿真”，而是:

- 以 `axis / projected centerline` 与 `D(x)` 为核心对象的
- `video-only 2D geometric measurement and zone analysis`

它可以正式主张二维投影下的长度、直径分布、`foreshortening` 与分区几何；但不应声称已经实现 patient-specific `virtual deployment`、clinical-grade `porosity` 或 `apposition`。
