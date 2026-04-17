# 研究笔记

更新时间: 2026-04-17

本笔记用于保存当前仓库方案设计时参考过的公开资料、提炼出的工程结论，以及后续算法路线的取舍依据。

## 1. 相关资料

### BFR / NiTi 非接触测试

- [Evaluation Methods for Non-contact Bend and Free Recovery Tests of Thin NiTi Wires and Their Effects on Measured Transformation Temperatures](https://link.springer.com/article/10.1007/s11665-020-05022-2)
- [Comparison of Manual and Automated Bend and Free Recovery](https://www.fwmetals.com/media/jzxdezaj/bfr-whitepaper.pdf)
- [YY/T 1771-2021 说明页](https://www.chinesestandard.net/PDF/English.aspx/YYT1771-2021)
- [ASTM WK97399 - New Guide for Bend and Free Recovery Testing of Finished Products](https://www.astm.org/membership-participation/technical-committees/workitems/workitem-wk97399)

### 细长体 / 导丝 / 可变形线体视觉跟踪

- [Robust Guidewire Tracking in Fluoroscopy](https://perso.telecom-paristech.fr/angelini/MIMED/papers_2010/wang_cvpr_2009.pdf)
- [mBEST: Realtime Deformable Linear Object Detection Through Minimal Bending Energy Skeleton Pixel Traversals](https://arxiv.org/abs/2302.09444)
- [Realtime Robust Shape Estimation of Deformable Linear Object](https://arxiv.org/abs/2403.16146)

## 2. 当前最重要的结论

### 2.1 对 BFR 来说, 最佳实践更偏向“整条线的形状”, 不是只盯针尖

根据 JMEP 2020:

- 非接触 BFR 可以做单点跟踪，也可以做整段曲率评估
- 曲率评估在恢复后段更敏感，更容易分辨转变细节
- 二次多项式和圆弧拟合的曲率表现较稳
- 四次多项式和椭圆段在接近拉直时噪声更高，不适合作为主方法

这意味着:

- “针尖模板匹配”可以做辅助观测
- 但不应成为 Af 提取的主观测量链

### 2.2 对细长体视觉跟踪来说, 端点检测通常是辅助量

导丝/线缆跟踪文献更常见的结构是:

- 线体主体检测
- 样条或路径模型
- 时间连续性/上一帧先验
- 端点检测作为局部观测

也就是说:

- 如果单独做针尖检测，很容易被噪声、遮挡、局部反光或分割断裂影响
- 更稳的路线是“全局形状 + 局部端点弱约束”

### 2.3 对我们这类单根针头恢复到直的场景, 低维单弯拟合是合理的

当前对象具有这些特点:

- 单根针
- 近端固定
- 平面内主导变形
- 单主弯段明显
- 无交叉
- 无复杂自遮挡

因此很适合用:

- 主轮廓采样
- 低维中心线拟合
- 从拟合结果导出 `x_fit(T)` 和 `kappa_fit(T)`

### 2.4 三条路线的“方法地位”应明确区分

按标准、论文和工程实现来区分:

- 路线 A = `standard-aligned displacement baseline`
  它对应 ASTM / 自动 BFR 流程里“恢复位移 vs 温度”的大框架，是工程基线，不是唯一标准图像算法。
- 路线 B = `literature-aligned primary method`
  它最贴近 JMEP 2020 这类非接触 BFR 文献中“整条形状 + 低阶曲率拟合”的推荐主线。
- 路线 C = `engineering temporal enhancement`
  它是路线 B 的工程增强版，用于提高时间稳定性，不应被表述成标准定义方法。

## 3. 三条路线

### 路线 A: 端点/针尖主导

定义:

- 直接检测固定端和自由端
- 用两端点弦长作为主量

优点:

- 实现简单
- 结果直观

缺点:

- 对分割断裂非常敏感
- 针尖局部噪声会直接污染 `x(T)`
- 不够贴近 BFR 文献里更推荐的“整形状”评估思路

结论:

- 适合 quicklook 和工程基线
- 不适合作为长期主方案

### 路线 B: 主轮廓 + 单弯段拟合

定义:

- 从主轮廓或主区域采样出中心线点
- 用二次曲线或圆弧拟合主弯段
- 由拟合曲线导出 `x_fit(T)`、`kappa_fit(T)`

优点:

- 符合 BFR 非接触测试中“整条形状”评估的方向
- 比单点更抗噪
- 很适合当前单根、单弯、固定端场景

缺点:

- 仍然依赖较稳定的轮廓分割
- 对液面折射、背景漂移需要额外校正

结论:

- 这是当前仓库的推荐主路线
- 也是最贴近公开论文最佳实践的一条路线

### 路线 C: 全局形状模型 + 时间连续性 + 局部端点弱约束

定义:

- 在路线 B 基础上加入前一帧形状先验
- 用滤波或时序优化保持连续性
- 针尖模板仅作为辅助观测

优点:

- 最稳健
- 最接近成熟工程系统

缺点:

- 实现复杂度更高
- 需要更多测试视频才能调好

结论:

- 适合作为路线 B 的工程增强版
- 不应替代路线 B 的物理定义

## 4. 方法对照表

下面这张表把“标准/论文里常见的计算方式”“商业软件常见能力”和“对本仓库的适配性”放在一起，便于后续长期参考。

| 方法类别 | 主要计算几何量 | 标准/论文依据 | 常见软件/工具形态 | 优点 | 主要风险 | 对本项目建议 |
| --- | --- | --- | --- | --- | --- | --- |
| 位移/弦长法 | 端点位移、固支点到自由端弦长 `x(T)`、最大挠度 | ASTM F2082 强调测量恢复过程中的 motion / displacement；视觉系统可作为等效测量手段 | 常见于定制视觉脚本、位移传感器替代方案、部分 DIC 后处理 | 直观、易解释、和标准语言最贴近 | 端点一旦找偏，误差会直接传到 `Af`；对针尖噪声和分割断裂敏感 | 适合作为主输出之一，推荐保留 `x_fit(T)`，但不要只靠针尖模板 |
| 角度/转角法 | 局部切线角、末端转角、恢复角度 | 常见于人工判读和部分器械测试流程；更像“恢复量”的另一种表达 | 人工量角、简单视觉量角、旋转位移传感器 | 实现简单，在某些器械结构上好解释 | 对你这类单针头不如曲率/弦长稳；角度参考线定义容易漂 | 不是当前主路线，除非后续某类样件天然以角度验收 |
| 主弯段曲率法 | 等效曲率 `kappa(T)`、局部弯曲半径 | JMEP 2020 明确支持非接触 BFR 中基于整段形状的曲率评估；低阶拟合优于高阶拟合 | 轮廓拟合、样条拟合、DIC 的 curvature / bend line 模块 | 对恢复后段更敏感，通常比单点更稳，更贴近“整条线形状” | 依赖轮廓质量；液面折射、背景漂移会污染局部拟合 | 推荐作为当前主量候选，优先输出 `kappa_fit(T)` |
| 分段模型法 | “弯曲段曲率 + 直线段方向/长度” 或分段中心线参数 | 更符合针头“上端弯、下端近直”的实际几何；属于曲率法的工程增强版 | 自研脚本、部分梁/线体视觉工具 | 比单一圆弧或单一二次曲线更贴近真实针形 | 比当前最小方案更复杂，需要更多 QC 和参数约束 | 作为路线 B 的自然升级方向，暂不急着上 |
| 全场 DIC / 线体 DIC 法 | 位移场、挠度、曲率、bend line | 商业 DIC 软件常提供 deflection / curvature / bend line；适合实验计量 | VIC-2D、ZEISS CORRELATE、X-Sight ALPHA、ZwickRoell 2D DIC | 工具成熟、可视化强、方便做实验报告 | 通常更适合有纹理/散斑的表面；对你这类高对比细线体未必最简 | 可作为后续外部对照方案，但不是当前最小仓库首选 |
| 骨架/中心线路径法 | 主中心线路径、端点、弧长、弦长 | 更接近导丝/线缆/DLO 跟踪文献中的主路径思想 | OpenCV + scikit-image + skan、skeleton-tracing | 对强弯曲时的整条针覆盖更好，适合算 `x(T)` | 骨架分叉、断裂、毛刺会让端点和路径跳动 | 当前 `x_fit` 的推荐实现方式 |
| 针尖模板/局部特征法 | 针尖位置、末端方向 | 更像工程辅助观测，文献中通常与全局形状模型联合使用 | 模板匹配、关键点检测、局部几何规则 | 初始化方便，解释直观 | 对反光、模糊、遮挡、局部噪声很脆弱 | 只建议做辅助观测或 QC，不建议当主链路 |
| 全局形状 + 时间连续性法 | 连续帧中心线、时序平滑后的 `x(T)`/`kappa(T)` | 更接近成熟导丝/细长体跟踪系统；常结合上一帧先验 | 卡尔曼滤波、时序优化、可变形模型跟踪 | 最稳健，能明显降低单帧抖动 | 复杂度更高，需要更多真实视频来调参与验证 | 作为路线 C，等最小方案稳定后再做 |

## 5. 商业软件与开源工具的参考位置

| 类别 | 名称 | 更擅长的能力 | 对本项目的参考价值 |
| --- | --- | --- | --- |
| 商业软件 | VIC-2D / Correlated Solutions | DIC 位移场、应变场、后处理 | 适合做外部实验对照，不是当前最简实现路线 |
| 商业软件 | ZEISS CORRELATE | 2D/3D 相关测量、位移/形变分析 | 适合后续验证实验流程，但对细线体不一定比轮廓法更省事 |
| 商业软件 | X-Sight ALPHA | bend line、deflection、curvature | 对“细长梁/线体”的思路最接近，可作为方法学参考 |
| 商业软件 | ZwickRoell 2D DIC | 实验测量软件链集成 | 更像实验室成套方案参考 |
| 开源工具 | OpenCV | 分割、轮廓、ROI、基础几何 | 当前仓库前处理主工具，继续保留 |
| 开源工具 | scikit-image | skeletonize、medial axis 等形态学操作 | 当前仓库 `x_fit` 主路径提取的重要工具 |
| 开源工具 | skan | 骨架图转路径图、主分支分析 | 当前仓库 `x_fit` 主路径提取的重要工具 |
| 开源工具 | skeleton-tracing | 从骨架图提取 polyline | 可作为后续轻量替代方案参考 |
| 开源工具 | DICe / Ncorr | 开源 DIC | 更适合作为外部对照，不是当前最小实现首选 |

## 6. 当前仓库策略

当前策略采用:

- 路线 A: 已实现，作为 `standard-aligned displacement baseline`
- 路线 B: 已实现，作为 `literature-aligned primary method`
- 路线 C: 已实现最小版，作为 `engineering temporal enhancement`

对 `wire-like.mp4` 这类视频，主输出将逐步切换到:

- `x_route_a(T)`
- `x_fit(T)`
- `kappa_fit(T)`
- `x_route_c(T)`
- `kappa_route_c(T)`

但长期主量仍不建议只依赖“组件端点法”。当前路线 C 采用的是:

- 路线 B 的单帧几何结果
- 路线 A 端点作为弱约束
- 时间方向的双向平滑
- 对 `x` 和 `kappa` 的单调投影
