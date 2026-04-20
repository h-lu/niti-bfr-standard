# NiTi Bend-Free-Recovery Standard 2

面向 NiTi 镍钛“弯曲-自由恢复”试验的最小软件方案仓库。

当前仓库已同时包含两类对象的最小工作流:

- `wire-like`
- `braided`

项目长期目标是把三条算法路线 `A / B / C` 在两类对象上都保留、比较并逐步打通，而不是把仓库收缩成单对象、单 demo 或单一路线。

其中:

- `quicklook` 是快速检查趋势、稳定性和 QC 的输出模式
- `formal Af` 是满足准入条件时才允许给出正式 `Af-95 / Af-tan` 的输出模式
- `formal Af` 是输出模式，不等于“项目只允许保留一种算法”

对 `wire-like`，当前最小实现的三条路线是:

1. 路线 A: 端点/针尖主导的位移或弦长法
2. 路线 B: 主轮廓 + 单弯段拟合法
3. 路线 C: 全局形状 + 时间连续性的增强法

这个最小方案当前同时包含两套输入:

1. `真实视频 quicklook`: 针对 `data/wire-like.mp4` 做基础几何提取
2. `合成基准数据`: 用一个符合物理约束的单弯针模型生成温度曲线、视频和真值，作为系统测试/校准数据

当前仓库也已把第二类 `braided device` 对象纳入一条
`YY/T 1771` 对齐的 BFR 工作流，并同样按 A / B / C 三类几何视角组织量测:

- braided `A = length_axis(T)`
- braided `B = diameter_max(T)`，当前实现定义为 strict body-only `max_s w_orth(s)`，对应 braided 的宽度/直径视角
- braided `C = area_proj(T)`

需要特别区分长期方法学目标和当前实现状态:

- 从长期方法学上, `A / B / C` 是 braided 的三条并行分析路线, 用于交叉验证和稳健性对照。
- 从当前实现状态上, braided 的 `formal Af` 默认仍优先 `A:length_axis(T)`；`B:diameter_max(T)` 当前保留为对象级 formal candidate，`C:area_proj(T)` 则稳定输出自己的 route-level result，并带独立 `status / gate / gate_reason`，但仍不是默认对象级 formal candidate。
- 同理, `wire-like` 当前正式主量默认仍是 `kappa_fit(T)`，这属于当前实现状态，不应误读为项目长期只保留单一 formal 视角。

项目的统一输出口径也应和这个边界保持一致:

- 对每一类对象, `A / B / C` 三条路线都应各自输出结果, 而不是只显示最终被选中的单一路线。
- “有路线结果”表示该路线给出了自己的几何量、值列/恢复列、Af 估计、QC 与可用性判断。
- “有 formal 结果”表示该路线通过了对应准入条件, 可以对外给出正式 `Af-95 / Af-tan`。
- 因此, 同一个 run 里可以同时出现“三条路线都有结果, 但只有其中一条是当前 formal 主量”。
- 同一个 run 里也可以出现“三条路线都只有 quicklook / provisional / formal_blocked 状态, 没有任何一路被正式放行”。

换句话说:

- `route-level result` 不等于 `formal passed`
- `route-level formal_passed` 也不自动等于对象级最终 formal 推荐
- `formal Af` 是某条路线在满足准入条件后的输出状态, 不是把另外两条路线从页面、结果文件或方法学中删除

当前仓库在收口时应逐步把结果页和 summary 统一成“对象级结果 + 路线级结果”两层语义:

- 对象级结果: 当前 run 的请求模式、实际模式、最终 formal 是否放行、推荐展示主量
- 路线级结果: `A / B / C` 每条路线自己的 `metric`、值列/恢复列、`Af-95`、`Af-tan`、`status`、`gate reason`、`warning codes`、路线级 QC 摘要与选择标记

当前 route-level 的稳定结果字段约定为:

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

当前对象级结果字段则应单独读取, 不能回写成某一条 route entry 的替代物:

- 主读路径先用 canonical `object_*` contract
- `object_reportability_status`
- `object_formal_metric_key` / `object_formal_route_alias`
- `object_provisional_metric_key` / `object_provisional_route_alias`
- `object_recommended_metric_key` / `object_recommended_route_alias`
- `object_formal_gate_reason`
- `object_formal_af95_c` / `object_formal_aftan_c`
- `object_provisional_af95_c` / `object_provisional_aftan_c`
- `formal_metric_label`、`provisional_metric_label`、`primary_metric_label` 继续保留为兼容现有 summary/UI 的展示字段, 但不再作为主 contract

当前 route-level QC 摘要的读取口径也应固定:

- 每条路线至少要先读 `reportability_status`、`gate_reason`、`warning_codes`
- 再读 `fit_rmse`、`monotonic_violation_fraction`、`dynamic_range`
- 然后读该路线自己的 `route_qc_summary`
- `route_qc_summary` 的标准骨架固定为:
  `valid_points`、`quality_median`、`monotonic_violation_fraction`、`tail_recovery_median`、`stability_metric`
- `wire-like` 再结合端点稳定性、高温端覆盖、与 B 路线偏差等 gate 语义理解 A/C 的可用性
- `braided` 再结合 `route_qc_summary.stability_metric.key / route_qc_summary.stability_metric.value` 中与路线相关的稳定性指标:
  `A:length_axis` 对应 `centerline_disagreement_median`
  `B:diameter_max` 对应 `axis_peak_position_stability_p95`
  `C:area_proj` 对应 `area_definition_gap_fraction_p95`
- 同时结合 `formal_qc`、`centerline_disagreement_median`、`body_mask_attachment_leak_fraction_median`、`area_proj_definition_gap_px2` 等对象级/benchmark QC 理解 A/B/C 的稳定性
- 也就是说, 当前“route-level QC 摘要”已经有稳定数据骨架, 但展示层仍分布在 `route_results` 与对象级 summary/benchmark summary 两层

这套语义的目标是让用户能同时看到:

- 每条路线都算出了什么
- 每条路线当前处于 `quicklook`、`provisional`、`formal_blocked` 还是 `formal_passed`
- 对象级最终 formal 结论是由哪条路线给出的

因此推荐的阅读顺序应固定为:

1. 先看 `route_results` 里的 `A / B / C` 三条路线各自算出了什么、状态如何、gate 为什么关闭或放行
2. 再看对象级 canonical `object_*` 字段判断 formal / provisional / recommended 的对象级选择
3. 如需兼容旧 summary/UI，再回看 `formal_metric_label`、`provisional_metric_label`、`primary_metric_label`
4. 最后判断对象级这次推荐展示哪一路, 而不是倒过来把对象级 winner 当成唯一输出

这里仍然不能误写成“六条路线都已经 fully formalized”。更准确的说法是:

- 项目长期上保留并展示六条路线
- 当前实现上, 六条路线的 formal 成熟度并不相同
- 文档、前端和结果文件都应把这种“不同行使同一 formal 地位”的事实明确展示出来

这里的“对齐”指:

- 对齐 `YY/T 1771 / ASTM F2082` 的“恢复量 vs 温度 -> Af-95 / Af-tan”大框架
- 不意味着 braided 成品整体已经天然落在原始 `wire / tube / strip` 适用范围内
- 对 braided 的对外口径更适合写成 `YY/T 1771-aligned BFR workflow`
- 不应把它写成“braided finished product 已被原始标准逐字覆盖”

## 最小物理模型

当前把针头近似为:

- 二维平面内运动
- 近似不可伸长
- 近端固支
- 单主弯段主导形状变化
- 升温时曲率 `kappa(T)` 单调下降并最终逼近 0

当前仓库中的 `wire-like` 路线状态是:

- 路线 A: 已实现，直接从针体组件/轮廓提取固定端与自由端，输出 `x_route_a(T)`
- 路线 B: 已实现，提取主骨架路径计算 `x_fit(T)`，并对主弯段拟合得到 `kappa_fit(T)`
- 路线 C: 已实现最小版，在路线 B 基础上加入弱端点约束和时间连续性，输出 `x_route_c(T)` 与 `kappa_route_c(T)`

它们在方法学上的定位是:

- 路线 A: `standard-aligned displacement baseline`
  对应 ASTM / 自动 BFR 流程里“恢复位移 vs 温度”的大框架，是工程基线，不应被表述成唯一标准图像算法。
- 路线 B: `literature-aligned primary method`
  最贴近非接触 BFR 论文里推荐的“整条形状 + 低阶曲率拟合”主线，是当前仓库的正式主方法。
- 路线 C: `engineering temporal enhancement`
  是在路线 B 基础上的工程增强版，用于提升时间稳定性，不应被表述成标准定义方法。

其中路线 B 的主提取流程是:

- 提取主轮廓
- 用 `scikit-image` / `skan` 提取主骨架路径并计算 `x_fit`
- 从主轮廓/主区域采样局部中心线点
- 对单主弯段做二次曲线或圆弧拟合以计算 `kappa_fit`

拟合几何量定义为:

- `x_route_a(T)`

表示路线 A 直接从端点检测得到的固定端到自由端弦长。

- `x_fit(T) = ||tip_fit(T) - clamp_fit(T)||`

也就是路线 B 中，由主骨架路径近端固支点与自由端尖端导出的弦长。对于“由弯到直”的单弯针头，该量应随恢复单调增大并在完全拉直时接近针长 `L`。

- `kappa_fit(T)`

表示单弯段拟合得到的等效曲率。对当前对象，它应随恢复单调减小并在拉直时趋近 0。

- `x_route_c(T)` / `kappa_route_c(T)`

表示路线 C 在路线 B 的几何量基础上，再加入时间连续性约束后的结果。当前最小实现采用:

- 用路线 A 端点作为弱约束
- 对端点轨迹做双向指数平滑
- 对 `x` 和 `kappa` 做单调投影

恢复率定义为:

- 对 `x_fit(T)`:
  `R_x(T) = (x_fit(T) - x_M) / (x_A - x_M)`

- 对 `x_route_a(T)`:
  `R_A(T) = (x_route_a(T) - x_M) / (x_A - x_M)`

- 对 `kappa_fit(T)`:
  `R_k(T) = (kappa_M - kappa_fit(T)) / (kappa_M - kappa_A)`

其中:

- `x_M`, `kappa_M`: 低温/初始马氏体参考状态
- `x_A`, `kappa_A`: 高温/完全恢复奥氏体参考状态

## Af 指标

- `Af-95`: `R(T) = 0.95` 时对应的温度
- `Af-tan`: 在主转变区最大斜率点作切线，与高温平台线交点对应的温度

## 仓库结构

```text
.
├── configs/
│   └── minimal.yaml
├── data/
│   └── wire-like.mp4
├── docs/
│   └── minimal-design.md
├── scripts/
│   ├── analyze_wire_like.py
│   └── run_demo.py
└── src/niti_bfr/
    ├── extract.py
    ├── metrics.py
    ├── model.py
    ├── pipeline.py
    └── synth.py
```

## 快速开始

安装:

```bash
python3 -m pip install -e .
```

运行最小演示:

```bash
python3 scripts/run_demo.py
```

运行第二类对象的最小 demo:

```bash
python3 scripts/run_braided_demo.py
```

运行 braided calibration benchmarks:

```bash
python3 scripts/run_braided_benchmarks.py
```

这会默认生成两个额外的 synthetic benchmark:

- `outputs/braided_demo_asymmetric`
- `outputs/braided_demo_attachment_stress`

它们与 `braided_demo` 使用同一套 truth / analysis / QC 输出口径，但分别强调:

- 峰值偏移与左右不对称下的分区 / 峰位校准
- support / tip attachment 更强、对比度更低时的 body-only / threshold 稳定性校准

若要列出全部 benchmark 或连同基线一起运行:

```bash
python3 scripts/run_braided_benchmarks.py --list
python3 scripts/run_braided_benchmarks.py --all
```

运行 wire-like calibration benchmarks:

```bash
python3 scripts/run_wire_benchmarks.py
```

该命令默认运行 wire 的 calibration benchmark 集合，并写出:

- `outputs/wire_benchmark_suite/benchmark_summary.json`
- `outputs/wire_benchmark_suite/benchmark_summary.csv`

若要列出全部 wire benchmark 或连同基线一起运行:

```bash
python3 scripts/run_wire_benchmarks.py --list
python3 scripts/run_wire_benchmarks.py --all
```

生成 demo 的路线 A / B 计算过程视频:

```bash
python3 scripts/render_demo_process_video.py
```

这会:

1. 生成一组完整恢复的合成温度表和视频
2. 写出 `x_true(T)`、`kappa_true(T)` 及其各自的真值 `Af-95 / Af-tan`
3. 自动提取 `x_route_a(T)`、`x_fit(T)` 和 `kappa_fit(T)`
4. 分别计算恢复曲线并比较 `Af-95`、`Af-tan`
5. 输出图、真值文件和算法对比结果

过程视频会额外输出:

- `outputs/demo/route_a_process.mp4`
- `outputs/demo/route_b_process.mp4`
- `outputs/demo/route_ab_process.mp4`

当前 `run_demo.py` 还会同时输出路线 A / B / C 的数值比较结果。

分析现有部分恢复视频:

```bash
python3 scripts/analyze_wire_like.py
```

注意:

- `wire-like.mp4` 只有部分恢复过程，没有完整高温平台，因此不适合直接作为最终 `Af` 真值
- 它更适合作为视场布局、分割阈值和针体可见性的初始参考
- 脚本会额外输出 `qc_frames/`，用于检查主轮廓、采样中心线、拟合曲线、固支点与自由端识别
- 当前 `wire-like.mp4` 因为没有同步温度和完整高温平台，只适合先比较 `x_route_a(t)`、`x_fit(t)`、`kappa_fit(t)` 的稳定性，不适合直接给出正式 Af
- 在当前样例上，`kappa_fit(t)` / `kappa_route_c(t)` 往往比 `x` 类量更平稳，更适合作为后续 Af 提取候选主量

温度同步格式说明见:

- [temperature-sync.md](/Users/wangxq/Documents/niti-bfr-standard-2/docs/temperature-sync.md)

研究与路线记录见:

- [research-notes.md](/Users/wangxq/Documents/niti-bfr-standard-2/docs/research-notes.md)
  其中新增了“方法对照表”和“商业软件/开源工具参考表”，方便直接比较标准思路、论文方法和当前仓库路线。

第二类对象的独立调研与方案草案见:

- [braided-device-survey.md](/Users/wangxq/Documents/niti-bfr-standard-2/docs/braided-device-survey.md)
  该文档用于记录“编织网状器械 / braided device”这类新对象的资料整理、主量设计，以及预计采用的独立算法路线。
- [braided-video-formal-method.md](/Users/wangxq/Documents/niti-bfr-standard-2/docs/braided-video-formal-method.md)
  该文档用于钉死“只有视频输入”条件下 braided device 的正式方法边界、术语口径，以及当前仓库应采用的主量与不能越界声称的内容。

当前最小方案的封版结论见:

- [v0.1-summary.md](/Users/wangxq/Documents/niti-bfr-standard-2/docs/v0.1-summary.md)
  其中总结了当前针型对象是否算“已解决”、A/B/C 的推荐用法、demo 的理论值，以及真实实验还需要补哪些输入。
- [v0.2-scope-guard.md](docs/v0.2-scope-guard.md)
该文档记录的是 `wire-like v0.2` 阶段的 formal 收敛任务；当前项目级长期原则以 [AGENTS.md](/Users/wangxq/Documents/niti-bfr-standard-2/AGENTS.md) 为准。

第二类对象当前提供独立 quicklook 脚本:

```bash
python3 scripts/analyze_braided_like.py /path/to/video.mp4
```

若已有同步温度文件, 也可直接走 braided 的 `formal Af` 通道:

```bash
python3 scripts/analyze_braided_like.py /path/to/video.mp4 --temperature-csv /path/to/temperature.csv
```

该脚本当前会固定产出 braided `A / B / C` 三条路线的结果:

- 无温度时, 三条路线都保留各自的 quicklook 曲线与 QC 输出
- 有温度时, `length_axis / diameter_max / area_proj` 都会稳定给出各自的 route-level Af 估计、status 与 gate reason
- 其中 `B` 路线固定指向 braided 的宽度/直径视角; 当前实现键为 strict body-only `diameter_max(T)`，并保留为对象级 formal candidate
- `area_proj(T)` 当前稳定输出 route-level result, 承担 provisional / formal_blocked 对照与方法学验证角色
- 同时继续导出最大直径、左右收口定位和 QC 叠加帧

当前 web / JSON / summary 的阅读口径建议固定为:

- 先读 `route_results` 或 `route_results_by_alias`
- 固定按 `A / B / C` 顺序读 `metric_key`、`display_label`、`value_series_col`、`recovery_series_col`、`af95_c`、`aftan_c`、`reportability_status`、`gate_reason`、`warning_codes`
- 若要判断该路线是否属于对象级 formal 候选, 再看 `formal_candidate`、`accepted_as_formal_candidate`
- 若要判断该路线是否被对象级流程选为 primary / formal, 再看 `selected_as_primary`、`selected_as_formal`、`formal_role`
- 若要读取路线级 QC, 再看 `route_qc_summary` 的统一骨架:
  `valid_points`、`quality_median`、`monotonic_violation_fraction`、`tail_recovery_median`、`stability_metric`
- 若要判断对象级最终推荐的是哪一路, 主读路径先看 canonical `object_*` 字段, 旧的 `formal_metric_label` / `provisional_metric_label` / `primary_metric_label` 仅作兼容展示

当满足温度同步、完整高温平台、主轴长度稳定提取等条件时, 该脚本还可输出:

- `length_axis(T)` 主量对应的 `Af-95`
- `length_axis(T)` 主量对应的 `Af-tan`
- `length_axis / diameter_max / area_proj` 的路线级恢复曲线对照

按当前方法学定义，这条 braided 路线的正式目标不是 `virtual deployment`，而是:

- `video-only 2D geometric measurement and zone analysis`

若需要与 `YY/T 1771` 的 Af 框架对齐, 当前 braided 路线采用的正式解释是:

- 用二维投影 `length_axis(T)` 作为 braided 试样的主恢复量
- 把 `R_axis(T) = (L_M - L_axis(T)) / (L_M - L_A)` 作为正式恢复率
- 再按 `Af-95` 与 `Af-tan` 计算正式结果
- `B` 路线始终表示 braided 的宽度/直径视角; 当前实现用 strict body-only `diameter_max(T)` 承载这一路线, 它保留为对象级 formal candidate, 但不是默认 formal 主量
- `area_proj(T)` 当前稳定输出 route-level result, 并带自己的 gate / status / gate_reason, 主要承担 provisional / formal_blocked 对照与方法学验证角色
- `length_env` 和分区量仍主要用于对照与解释层

当前脚本仍是这条正式路线的最小实现 / quicklook 入口；正式口径与后续升级方向以
[braided-video-formal-method.md](/Users/wangxq/Documents/niti-bfr-standard-2/docs/braided-video-formal-method.md)
为准。

## 路线级 QC 与 Benchmark 读取

当前仓库读取 route-level QC 与 benchmark 的建议顺序如下:

1. 单个 run:
   先看 `summary.json` 里的 `route_results` / `route_results_by_alias`, 这是 A/B/C 三条路线的统一入口。
2. 单个 run 的对象级决策:
   主读路径先看 canonical `object_reportability_status`、`object_formal_metric_key`、`object_provisional_metric_key`、`object_recommended_metric_key`、`object_formal_gate_reason`；`reportability_status`、`formal_metric_label`、`provisional_metric_label` 继续保留为兼容展示层。
3. 单对象 benchmark:
   `wire-like` 看各 benchmark 输出目录下的 `summary.json`；
   `braided` 看各 benchmark 输出目录下的 `analysis_metrics.json` 与 `analysis.csv`。
4. benchmark suite:
   `wire-like` 看 `outputs/wire_benchmark_suite/benchmark_summary.json` 与 `benchmark_summary.csv`；
   `braided` 看 `outputs/braided_benchmark_suite/benchmark_summary.json` 与 `benchmark_summary.csv`。

对 wire 与 braided 的 benchmark 汇总文件, 当前推荐的横向比较方式是一致的:

- 先按路线读 suite summary 里的 A/B/C 行为, 包括 `metric_key`、`reportability_status`、`gate_reason`、`accepted_as_formal_candidate`、`selected_as_primary`、`selected_as_formal`
- 再看各路线的 `af95_error_c` / `aftan_error_c`、`dynamic_range`、`fit_rmse`、`monotonic_violation_fraction`
- 再把 `route_qc_summary` 的统一骨架和对象共享 QC 一起读:
  `wire-like` 重点补看端点稳定性、中心线点数、拟合占比等 shared QC；
  `braided` 重点补看 `body_mask_attachment_leak_fraction`、`centerline_disagreement_median`、`axis_peak_position_stability_p95`、`area_proj_definition_gap_px2`
- 最后回到各 benchmark 目录下的 run summary 判断该路线在当前 stressor 下是 `formal_passed`、`provisional` 还是 `formal_blocked`

项目长期目标是让 benchmark 页面或汇总文件能横向比较六条路线:

- wire `A:x_route_a`
- wire `B:kappa_fit`
- wire `C:kappa_route_c`
- braided `A:length_axis`
- braided `B:diameter_max`
- braided `C:area_proj`

当前实现已经为 wire 与 braided 都提供了 suite 级 `benchmark_summary.json/csv` 入口；两侧的单场景文件形态还不完全相同, 但 benchmark summary 的阅读体验应统一为“先路线级比较, 再对象级选择, 最后回到单场景 QC 解释”。因此这里说的“六路线横向比较”既是当前可落地的 summary 体验, 也是后续页面层继续对齐的方向。

## 通俗解释

当前仓库同时计算三条量，它们都在回答“针头恢复了多少”，但看的角度不同。

### 1. `x_route_a(T)`: 直接看端点拉开了多远

可以把它理解成:

- 不去拟合整条针
- 直接从图像里找固定端和自由端
- 再量两点之间的直线距离

它最贴近“位移/弦长法”，也是路线 A 的主量。

### 2. `x_fit(T)`: 看骨架主路径两端拉开了多远

可以把它理解成:

- 固定端到自由端的直线距离

针越弯，这个距离越短。
针越接近拉直，这个距离越长。

所以 `x_fit(T)` 回答的是:

- “这根针整体拉直到什么程度了?”

最简单的示意是:

```text
弯曲时:

anchor o
        \
         \
          )-----o tip

x_fit = anchor 和 tip 之间的直线距离


接近拉直时:

anchor o---------------------------o tip

x_fit 变大
```

当前实现里:

- 先提取针的主骨架路径
- 再取主骨架路径的起点和终点
- 计算这两个点之间的弦长

### 3. `kappa_fit(T)`: 看弯得有多厉害

可以把它理解成:

- 针的“弯曲程度”

如果针弯得很厉害，`kappa_fit` 就大。
如果针越来越直，`kappa_fit` 就越来越小。
接近完全拉直时，它会接近 0。

所以 `kappa_fit(T)` 回答的是:

- “这根针现在还弯得多厉害?”

最简单的示意是:

```text
弯曲大:

anchor o
        \
         )
        /
      o tip

kappa_fit 大


接近直:

anchor o---------------------------o tip

kappa_fit 小，接近 0
```

当前实现里:

- 从针的主轮廓/主区域中采样局部中心线点
- 对主弯段做简单曲线拟合
- 从拟合曲线计算等效曲率

### 两者的区别

- `x_route_a`: 看“直接检测到的端点拉开了多少”
- `x_fit`: 看“沿主骨架路径理解后，两端拉开了多少”
- `kappa_fit`: 看“还弯得多厉害”

可以把它们理解成:

- `x_route_a` 更像“最直接的端点开度”
- `x_fit` 更像“带一点形状约束的开度”
- `kappa_fit` 更像“弯曲强度”指标

对当前 demo 而言，`kappa_fit` 通常比 `x_fit` 更平稳，也更适合作为 Af 提取候选主量。

## 下一步建议

最小方案完成后，建议依次补齐:

1. 夹具/视场标定，把像素量转换成毫米
2. 温度-帧同步
3. 更稳健的中心线提取与骨架追踪
4. 完整实验视频上的 `Af` 评估与批处理
