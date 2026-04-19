# Braided Geometry 优化改造清单

本文把当前 braided 路线的优化工作收敛成一份可执行 checklist，目标是让 `L_axis`、`A_proj`、`D_max` 三条量的定义、实现和验收口径逐步自洽。

当前优先级固定为：

1. `L_axis`
2. `A_proj`
3. `D_max`

## 0. 总原则

- 所有主量共用同一份 `body-only` 定义，不允许 `truth` 和 `measured` 各用一套主体口径。
- `L_axis` 必须表示 `body centerline arc length`，不能再退化成 `PCA span` 或裁剪后的 `L_env`。
- `A_proj` 必须显式声明是 `body-only projected area`，不能默认等于阈值后最大连通域面积。
- `D_max` 暂时保留为对照量，但要并行输出两种定义，避免过早锁死到单一实现。
- 所有新定义都必须同时在 `synthetic braided demo` 和真实视频 quicklook 上验收。

## 1. Phase A: Body-Only 口径统一

### A1. 把 `body_tube_mask` 做成一等公民

- 目标：
  从“最大连通域 + 局部启发式 trim”升级为“centerline 主干约束下的 body-only mask”。
- 影响文件：
  [src/niti_bfr/extract_braided.py](/Users/wangxq/Documents/niti-bfr-standard-2/src/niti_bfr/extract_braided.py)
  [src/niti_bfr/synth_braided.py](/Users/wangxq/Documents/niti-bfr-standard-2/src/niti_bfr/synth_braided.py)
- 改造项：
  - 增加 `body_tube_mask` 或等价数据结构，作为 `L_axis / A_proj / D_max` 的共同输入。
  - 先得到 provisional centerline，再按局部宽度生成 tube mask。
  - 左右端只允许使用同口径裁剪规则，禁止左端 support 和右端 tip cap 走不同启发式。
- 验收：
  - `truth` 和 `measured` 都能输出 `body_mask_area_px2`。
  - 同一帧内 `L_axis / A_proj / D_max` 都能明确说明是否只在 `body-only` 范围内计算。

### A2. 增加附件污染 veto 规则

- 目标：
  不让 rod / support / tip cap / detached blur 被默认计入主体。
- 改造项：
  - 对候选区域输出 `border_touch`、`elongation`、`solidity`、`orientation mismatch`、`distance_to_main_axis` 等特征。
  - 增加端部异常规则：
    `短粗 spur`、`超出 body tube`、`峰值只出现在端点附近` 的区域直接剔除。
- 验收：
  - synthetic demo 中左右端附件不再显著影响 `A_proj` 和 `D_max`。
  - quicklook 叠加图能看见主体与剔除附件的差异。

## 2. Phase B: L_axis 正式化

### B1. 把 `L_axis` 切到主干 centerline 弧长

- 目标：
  `L_axis` 的正式定义变成 `pruned centerline arc length`。
- 影响文件：
  [src/niti_bfr/extract_braided.py](/Users/wangxq/Documents/niti-bfr-standard-2/src/niti_bfr/extract_braided.py)
  [src/niti_bfr/pipeline.py](/Users/wangxq/Documents/niti-bfr-standard-2/src/niti_bfr/pipeline.py)
- 改造项：
  - 在 `body_tube_mask` 内生成 skeleton 或 medial axis。
  - 把 skeleton 图化，识别 `endpoint / branch / spur`。
  - 只保留最长主干路径，再做 spline 或 Savitzky-Golay 平滑。
  - `L_axis = arc_length(centerline)`。
- 推荐实现路线：
  `body mask -> medial_axis/skeletonize -> branch pruning -> longest trunk -> smooth -> arc length`
- 验收：
  - `length_axis_px` 与 `length_env_px` 不再高度重合。
  - `Af-95 / Af-tan` 使用 `L_axis` 时，不再依赖 `L_env` 的名义替代。

### B2. 增加 `L_axis` 双估计护栏

- 目标：
  给 formal 主量加自检，不让单一路径算法失控。
- 改造项：
  - 并行输出：
    - `length_axis_skeleton_px`
    - `length_axis_body_bins_px`
    - `length_axis_medial_px` 或 `length_axis_alt_px`
  - 增加差值指标：
    `length_axis_disagreement_px`
- 验收：
  - 当两条估计分歧过大时，formal gate 拒绝给正式 Af。
  - `summary` 与 `analysis_metrics.json` 里有这项 QC。

### B3. 将 `L_axis` 相关 formal gate 升级

- 目标：
  formal Af 不只看单调性，还要看 centerline 自洽性。
- 改造项：
  - 在 `pipeline.py` 的 braided formal gate 中新增：
    - `centerline_disagreement`
    - `endpoint_jump`
    - `branch_component_count_after_pruning`
    - `axis_peak_position_stability`
- 验收：
  - 对附件干扰明显的序列，formal gate 能主动拒绝结果。

## 3. Phase C: A_proj 统一定义

### C1. 把 `A_proj` 明文定义成 body-only projected area

- 目标：
  truth 和 measured 完全共用同一面积定义。
- 影响文件：
  [src/niti_bfr/extract_braided.py](/Users/wangxq/Documents/niti-bfr-standard-2/src/niti_bfr/extract_braided.py)
  [src/niti_bfr/synth_braided.py](/Users/wangxq/Documents/niti-bfr-standard-2/src/niti_bfr/synth_braided.py)
  [scripts/run_braided_demo.py](/Users/wangxq/Documents/niti-bfr-standard-2/scripts/run_braided_demo.py)
- 改造项：
  - 正式定义：
    `A_proj = integral_body(w_orth(s) ds)`
  - 并行保留对照定义：
    `A_proj_contour_width_integral = contour_width_integral(body_only_contour)`
  - 并行保留对照定义：
    `A_proj_contour = contour_area(body_only_contour)`
  - 输出面积差：
    `area_proj_definition_gap_px2`
- 验收：
  - `truth` 与 `measured` 的 `A_proj` 口径可逐字说明为同一公式。
  - 两种面积定义在 synthetic demo 上偏差处于可解释范围。

### C2. 让 `A_proj` 与 body clip 强绑定

- 目标：
  不允许在全 contour 或全 component 上直接算面积。
- 改造项：
  - 所有面积计算前都必须先获得 `body_tube_mask` 或 `body_only_contour`。
  - quicklook 和 demo 图上明确标注 `A_proj` 使用的 clip 范围。
- 验收：
  - synthetic demo 中 support / tip cap 改变时，`A_proj` 不再大幅漂移。

### C3. 为真实视频预留 segmentation front-end

- 目标：
  给后续真实 fluoroscopy 场景留出升级接口。
- 改造项：
  - 把 `component mask` 与 `body-only mask` 的生成拆成独立步骤。
  - 为后续替换成 learned segmentation 或 active contour 预留函数接口。
- 验收：
  - `extract_braided_geometry()` 不再把“阈值分割”和“几何量测”完全耦死。

## 4. Phase D: D_max 稳定化

### D1. 增加双定义输出

- 目标：
  让 `D_max` 成为可靠对照量，而不是单点单算法结果。
- 影响文件：
  [src/niti_bfr/extract_braided.py](/Users/wangxq/Documents/niti-bfr-standard-2/src/niti_bfr/extract_braided.py)
  [src/niti_bfr/webapp.py](/Users/wangxq/Documents/niti-bfr-standard-2/src/niti_bfr/webapp.py)
  [scripts/run_braided_demo.py](/Users/wangxq/Documents/niti-bfr-standard-2/scripts/run_braided_demo.py)
- 改造项：
  - 输出：
    - `diameter_max_px` (`= diameter_max_orth_px`)
    - `diameter_max_orth_px`
    - `diameter_max_thickness_px`
    - `diameter_max_feret_px` 或 `breadth_ref_px`
    - `diameter_mid_p90_px`
  - 默认展示 `orth + thickness` 两条主值，`feret` 只做参考。
- 验收：
  - `D_max` 在 threshold sweep 下仍明显稳于 `L_axis` 与 `A_proj`。

### D2. 从单点最大值升级为 robust max

- 目标：
  降低单个 noisy spike 或端部伪峰的影响。
- 改造项：
  - 新增统计量：
    `diameter_p95_px`
    `diameter_peak_span_px`
    `diameter_peak_pos_norm`
  - `D_max` 展示时同时展示：
    `max` 与 `p95`
- 验收：
  - 峰值位置在邻帧与邻温区内连续，不长期贴近端点。

## 5. Phase E: Demo 与验收脚本升级

### E1. 扩展 synthetic demo 对照项

- 目标：
  让 demo 不只看数值误差，还看定义误差。
- 改造项：
  - 增加以下输出：
    - `body_mask_area_true_px2`
    - `body_mask_area_measured_px2`
    - `length_axis_alt_true_px`
    - `length_axis_alt_px`

### E2. 修正 demo2 面积 theory 的低温边界条件

- 目标：
  保证 `austenite_fraction = 0` 时几何面积不被额外缩放。
- 影响文件：
  [outputs/braided_demo2/make_niti_heating_video.py](/Users/wangxq/Documents/niti-bfr-standard-2/outputs/braided_demo2/make_niti_heating_video.py)
- 改造项：
  - 去掉会让低温首帧面积先掉到 `< 1.0x` 的额外 `local_scale_y` 经验乘子。
  - 保持 `video` 与 `area_proj_true_px2` 使用同一组几何缩放公式。
- 验收：
  - `austenite_fraction = 0` 时 `area_proj_true_norm = 1.0`
  - `demo2` 的面积 truth 与“零相变时几何不变”解释一致
    - `diameter_peak_pos_norm_true`
    - `diameter_peak_pos_norm`
  - 新增图：
    - `body_mask_vs_temperature.png`
    - `axis_definition_gap_vs_temperature.png`
    - `dmax_peak_position_vs_temperature.png`
- 验收：
  - 每个主量都能同时回答“值对不对”和“定义有没有漂”。

### E2. 把验收指标固定下来

- 建议固定指标：
  - `length_axis_mae_px`
  - `length_axis_af95_error_c`
  - `length_axis_aftan_error_c`
  - `area_proj_mae_px2`
  - `area_proj_af95_error_c`
  - `diameter_max_mae_px`
  - `diameter_threshold_sensitivity`
  - `body_mask_attachment_leak_fraction`
  - `axis_definition_gap_px`
- 通过标准建议：
  - `L_axis`：优先看 `Af` 误差和 definition gap
  - `A_proj`：优先看 attachment leak 和口径一致性
  - `D_max`：优先看 threshold sweep 稳定性

## 6. 文件级改造顺序

1. [src/niti_bfr/extract_braided.py](/Users/wangxq/Documents/niti-bfr-standard-2/src/niti_bfr/extract_braided.py)
   先拆出 `body_tube_mask`、centerline graph、dual Dmax。
2. [src/niti_bfr/synth_braided.py](/Users/wangxq/Documents/niti-bfr-standard-2/src/niti_bfr/synth_braided.py)
   与 measured 同步共享 truth 口径。
3. [src/niti_bfr/pipeline.py](/Users/wangxq/Documents/niti-bfr-standard-2/src/niti_bfr/pipeline.py)
   升级 formal gate 与新 QC 指标。
4. [scripts/run_braided_demo.py](/Users/wangxq/Documents/niti-bfr-standard-2/scripts/run_braided_demo.py)
   固化 benchmark 输出。
5. [scripts/analyze_braided_like.py](/Users/wangxq/Documents/niti-bfr-standard-2/scripts/analyze_braided_like.py)
   把 quicklook 可视化和 summary 对齐到新口径。
6. [src/niti_bfr/webapp.py](/Users/wangxq/Documents/niti-bfr-standard-2/src/niti_bfr/webapp.py)
   追加新的 QC 展示项。

## 7. 里程碑定义

### M1. Formal L_axis 自洽

- 条件：
  - `L_axis != trimmed L_env`
  - `L_axis` 使用主干弧长
  - formal Af 能在 synthetic demo 上稳定收敛

### M2. A_proj 口径统一

- 条件：
  - `truth` 与 `measured` 使用同一 body-only 口径
  - support / tip cap 不再主导 `A_proj`

### M3. D_max 成为可靠对照量

- 条件：
  - 同时存在 `orth width` 与 `thickness` 两条结果
  - threshold / blur 扰动下稳定性可量化

## 8. 当前建议的推进顺序

- 第一步：
  完成 `body_tube_mask + pruned centerline arc length`
- 第二步：
  把 `A_proj` 改成 body-only 正交宽度积分，并保留 contour-area 对照
- 第三步：
  给 `D_max` 增加 `thickness` 与 `p95` 对照
- 第四步：
  升级 formal gate、demo benchmark 和 web/quicklook 展示

## 9. 完成判据

当下面四件事同时成立时，这轮改造可以视为完成：

- `L_axis` 在定义上与 `L_env` 完全拆开
- `A_proj` 的 truth / measured 口径能逐字复述为同一公式
- `D_max` 至少有两条互相校核的实现
- synthetic demo、quicklook、formal Af 三条链路使用同一套主体定义
