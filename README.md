# NiTi Bend-Free-Recovery Standard 2

面向 NiTi 镍钛“弯曲-自由恢复”试验的最小软件方案仓库。

当前仅覆盖一类对象: `弯曲针头恢复到直`。

目标是把三条算法路线逐步打通:

1. 路线 A: 端点/针尖主导的位移或弦长法
2. 路线 B: 主轮廓 + 单弯段拟合法
3. 路线 C: 全局形状 + 时间连续性的增强法

这个最小方案同时包含两套输入:

1. `真实视频 quicklook`: 针对 `data/wire-like.mp4` 做基础几何提取
2. `合成基准数据`: 用一个符合物理约束的单弯针模型生成温度曲线、视频和真值，作为系统测试/校准数据

当前仓库也开始把第二类 `braided device` 对象纳入一条
`YY/T 1771` 对齐的 BFR 工作流:

- 对 `wire-like` 对象, 正式主量固定为 `kappa_fit(T)`
- 对 `braided` 对象, 正式主量固定为 `length_axis(T)`

为便于像 `wire-like` 的路线 A/B/C 那样简洁讨论 braided 三条主量, 当前约定:

- braided `A = length_axis(T)`
- braided `B = diameter_max(T)`
- braided `C = area_proj(T)`

其中 `A` 是 formal 主量, `B/C` 是对照量。

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

当前仓库中的路线状态是:

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
  下阶段只收敛 `formal Af` 的边界、准入条件与正式主量，不继续扩算法范围。

第二类对象当前提供独立 quicklook 脚本:

```bash
python3 scripts/analyze_braided_like.py /path/to/video.mp4
```

若已有同步温度文件, 也可直接走 braided 的 `formal Af` 通道:

```bash
python3 scripts/analyze_braided_like.py /path/to/video.mp4 --temperature-csv /path/to/temperature.csv
```

该脚本当前仅用于 `braided device` 风格对象的:

- 包络长度 quicklook
- 主轴长度 quicklook
- 最大直径与左右收口定位
- QC 叠加帧导出

当满足温度同步、完整高温平台、主轴长度稳定提取等条件时, 该脚本还可输出:

- `length_axis(T)` 主量对应的 `Af-95`
- `length_axis(T)` 主量对应的 `Af-tan`
- `length_axis / length_env / diameter_max` 的恢复曲线对照

按当前方法学定义，这条 braided 路线的正式目标不是 `virtual deployment`，而是:

- `video-only 2D geometric measurement and zone analysis`

若需要与 `YY/T 1771` 的 Af 框架对齐, 当前 braided 路线采用的正式解释是:

- 用二维投影 `length_axis(T)` 作为 braided 试样的主恢复量
- 把 `R_axis(T) = (L_M - L_axis(T)) / (L_M - L_A)` 作为正式恢复率
- 再按 `Af-95` 与 `Af-tan` 计算正式结果
- `length_env`、`diameter_max` 和分区量仅作对照, 不参与正式主量竞争

当前脚本仍是这条正式路线的最小实现 / quicklook 入口；正式口径与后续升级方向以
[braided-video-formal-method.md](/Users/wangxq/Documents/niti-bfr-standard-2/docs/braided-video-formal-method.md)
为准。

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
