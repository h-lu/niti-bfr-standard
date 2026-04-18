# 第二类对象调研与方法设计

更新时间: 2026-04-18

本文档用于记录当前仓库“第二类对象”的前期调研结论、公开资料来源，以及预计采用的方法方案。

这里的“第二类对象”指:

- 与当前已实现的“单根弯曲针恢复到直”不同
- 形态更接近编织网状器械
- 其主变化不是单一中心线曲率恢复
- 而是轴向长度与径向鼓包/收口耦合变化

当前判断基于给定图片的外观特征，属于工程命名假设，不是器械型号识别结论。

## 1. 对象判断

从当前两张图看，该对象更像:

- 编织型自膨器械
- braided stent / flow diverter / 网篮类对象
- 具有明显的:
  - 两端收口
  - 中段鼓起
  - 轴向长度变化
  - 局部网孔密度变化

因此它不应继续沿用当前仓库对“针型对象”的主线:

- 单骨架路径
- 单主弯段拟合
- `kappa(T)` 作为核心主量

更合理的建模方式是:

- 把对象看成“轴向主轴 + 宽度包络 + 编织纹理”的二维投影体
- 主量围绕:
  - 长度
  - 直径分布
  - 轴向缩短
  - 局部密度 / 孔隙率代理量

## 2. 预计测量指标

对第二类对象，建议优先定义下面这些量，而不是继续复用针型对象的 `x(T)` / `kappa(T)` 语义。

### 2.1 基础几何量

- `L`
  左右端功能长度
- `Dmax`
  最大外径
- `A_proj`
  投影面积
- `AR = L / Dmax`
  长径比

### 2.2 轴向分布量

- `D(x)`
  沿主轴的宽度分布
- `x_peak`
  最大径位置
- `L_mid`
  中段鼓包主区长度
- `L_taper_left` / `L_taper_right`
  左右端收口长度

### 2.3 形态变化量

- `FS`
  foreshortening / 轴向缩短比
- `BulgeRatio`
  中段最大鼓包相对端部的放大比例
- `Symmetry`
  左右包络对称性

### 2.4 纹理代理量

- `MeshDensity(x)`
  沿轴向分段的边缘或纹理密度代理量
- `PorosityProxy(x)`
  局部孔隙率代理量

其中:

- `L`、`Dmax`、`D(x)`、`FS` 应作为第一批主量
- `MeshDensity(x)` / `PorosityProxy(x)` 建议作为增强量，不建议一开始作为唯一主量

## 3. 方法设计对照表

| 路线 | 对象表示 | 主要输出量 | 核心图像信息 | 优点 | 主要风险 | 当前建议 |
| --- | --- | --- | --- | --- | --- | --- |
| Route A: 纯包络法 | 只看整体外轮廓 | `L`、`Dmax`、`A_proj`、`AR` | 主轮廓、左右端点、最大宽度 | 最稳、最容易先落地、最适合做 QC 基线 | 看不到编织密度，无法反映局部孔隙代理 | 作为基线法保留 |
| Route B: 轴线-宽度剖面法 | 主轴 + `D(x)` | `L`、`D(x)`、`x_peak`、`L_taper`、`FS` | 主轴、法向宽度扫描、端部收口区域 | 最贴近当前对象的几何本质，解释性强 | 轴线如果受局部网纹干扰会抖动；端帽可能干扰边界 | 预计作为主方法 |
| Route C: 局部网孔密度代理法 | 包络 + 编织纹理分区 | `MeshDensity(x)`、`PorosityProxy(x)` | 局部灰度、边缘方向、周期纹理 | 能表达“中段更密、两端更疏”的编织特征 | 对焦、曝光、缩放和莫尔纹敏感 | 作为 Route B 的增强量 |
| Route D: 参数化编织壳拟合法 | 参数化 braided shell | `L`、`D(x)`、编织角、理论 porosity | 包络 + 纹理方向 + 周期结构 | 最接近器械结构模型，可与文献参数直接对接 | 成本高，对先验依赖强，前期容易过拟合 | 作为中长期研究路线 |
| Route E: 虚拟部署/力学预测法 | 几何或力学仿真 | 部署后长度、贴壁、porosity、落点 | 器械参数、边界条件、3D 解剖/工装 | 与学术和商业规划系统高度一致 | 不适合当前 2D 台架图作为首发算法 | 暂不作为当前仓库目标 |

## 4. 预计采用方案

当前建议采用一套与针型对象完全独立的新方案:

### 4.1 主方法

- `Route B: 轴线-宽度剖面法`

理由:

- 第二类对象的核心不是“单条中心线弯曲恢复”
- 而是“沿轴向的直径分布和长度变化”
- `D(x)`、`FS`、收口区长度等量比单一曲率更符合对象本质

### 4.2 基线法

- `Route A: 纯包络法`

用途:

- 提供最稳的 quicklook
- 提供 QC 回退结果
- 用于验证 Route B 是否明显偏离整体外轮廓

### 4.3 增强量

- `Route C: 局部网孔密度代理法`

用途:

- 辅助解释中段与端部的编织密度差异
- 为后续局部 porosity 近似评估做准备

### 4.4 当前不建议作为首批目标的方案

- 不建议直接复用当前针型对象的:
  - 主骨架弦长
  - 单弯段曲率拟合
  - `kappa(T)` 主量链路
- 不建议一开始就做:
  - 完整参数化编织角反演
  - 力学仿真或虚拟部署

## 5. 资料清单

下面的资料按“论文 / GitHub / 商业软件”三类整理。

### 5.1 论文与研究资料

- [Geometrical deployment for braided stent](https://infoscience.epfl.ch/handle/20.500.14299/127817)
  - 关键价值:
    - 用几何模型预测 braided stent 部署后丝线位置、长度与局部 porosity
    - 强调这类对象的核心量是长度、半径和局部几何分布
  - 对当前项目的启发:
    - 第二类对象应优先围绕 `L`、`D(x)`、`foreshortening` 建模

- [The varying porosity of braided self-expanding stents and flow diverters: an experimental study](https://pubmed.ncbi.nlm.nih.gov/22878007/)
  - 关键价值:
    - 明确指出 braided self-expanding stent 在轴向不同位置会出现不同 porosity
    - 压缩后通常中段更致密、两端更疏
  - 对当前项目的启发:
    - 局部密度或孔隙率代理量值得做
    - 但应作为分区增强量，而不是取代长度/直径主量

- [Assessing flow diverter porosity: a comparative analysis of quantification techniques based on imaging and simulation](https://pubmed.ncbi.nlm.nih.gov/39620483/)
  - 关键价值:
    - 比较了 2D 显微图像、3D Dyna-CT 和 ANKYRAS 模拟对 local porosity 的评估
    - 说明“从图像估计 porosity”在方法学上是成立的
  - 对当前项目的启发:
    - 二维图像可以作为局部 porosity 代理量的来源
    - 但最好和整体几何主量并行，而不是单独承担全部判断

- [Image-based mechanical analysis of stent deformation: concept and exemplary implementation for aortic valve stents](https://pubmed.ncbi.nlm.nih.gov/24626769/)
  - 关键价值:
    - 证明了“从图像反推支架形变状态”这条路可行
  - 对当前项目的启发:
    - 第二类对象做图像几何提取是合理方向，不必一开始就转向纯仿真

- [Machine learning and reduced order modelling for the simulation of braided stent deployment](https://pmc.ncbi.nlm.nih.gov/articles/PMC10090671/)
  - 关键价值:
    - 展示了 braided stent 当前更成熟的高阶研究方向是部署预测
  - 对当前项目的启发:
    - 当前仓库更适合先做 2D 台架几何量测
    - 之后再考虑是否往部署预测或 reduced-order 模型延伸

### 5.2 GitHub 与开源实现

- [bisighinibeatrice/BraidedStentsGeometry](https://github.com/bisighinibeatrice/BraidedStentsGeometry)
  - 用途:
    - 参数化生成 braided stent 几何
  - 参考价值:
    - 有助于理解编织器械的几何变量和参数命名

- [bisighinibeatrice/EndoBeams.jl](https://github.com/bisighinibeatrice/EndoBeams.jl)
  - 用途:
    - beam-to-surface contact 的有限元框架
    - 自带 braided stent deployment 示例
  - 参考价值:
    - 更偏高保真仿真，不是当前 2D 视觉算法模板

- [bisighinibeatrice/ROMforBraidedStents](https://github.com/bisighinibeatrice/ROMforBraidedStents)
  - 用途:
    - 基于 FE 数据的 reduced-order / ML 部署预测
  - 参考价值:
    - 说明学术界已有成熟的“部署结果预测”方向

- [jeffbli/VirtualCathLab](https://github.com/jeffbli/VirtualCathLab)
  - 用途:
    - 面向血管 mesh 和 centerline 的虚拟 stent deployment
  - 参考价值:
    - 适合参考交互式部署与后几何分析

当前调研结论是:

- 开源世界里，现成成熟的是“几何生成 + 部署仿真”
- 暂未发现现成的“桌面灰度图 -> 直接输出 `L / D(x) / FS / density proxy`”开源流水线

### 5.3 商业软件与临床/工业方案

- [Mentice Ankyras](https://www.mentice.com/ankyras)
  - 官方强调:
    - flow diverter foreshortening
    - device sizing
    - wall apposition
    - local porosity
  - 参考价值:
    - 与第二类对象的指标体系最接近

- [Medtronic / Sim&Cure Sim&Size for Pipeline Flex](https://www.medtronic.com/en-us/l/e/neurovascular-partnerships/sim-cure-software.html)
  - 官方强调:
    - segmentation
    - centerline detection
    - device sizing
    - proximal / distal landing zone visualization
  - 参考价值:
    - 说明商业路径也很重视尺寸、中心线和落点，而不是只看单张图的局部纹理

- [Mentice VIST Ankyras Flow Diverter Rehearsal](https://www.mentice.com/software/vist-ankyras-flow-diverter-rehearsal)
  - 官方强调:
    - patient-specific rehearsal
    - realistic flow diverter deployment
    - push-pull effect
  - 参考价值:
    - 更偏排练和部署行为模拟，不是当前项目的首批目标

- [Siemens ClearStent](https://www.siemens-healthineers.com/angio/options-and-upgrades/clinical-software-applications/clearstent)
  - 官方强调:
    - stent visibility enhancement
    - ROI detection
    - live fluoro overlay
  - 参考价值:
    - 更接近“显示增强”而不是几何建模
    - 说明商业上非常重视先把器械看清

- [Philips StentBoost Live](https://www.philips.com.eg/healthcare/product/HCOPT12/stentboost-enhanced-visualization-software)
  - 参考价值:
    - 与 ClearStent 类似，偏术中增强可视化

- [GE Healthcare 3DStent](https://www.gehealthcare.com/products/image-guiding-solutions/3dstent/)
  - 参考价值:
    - 同样偏 stent visualization / assessment，不是当前 2D 台架算法模板

## 6. 当前阶段的工程结论

### 6.1 第二类对象必须独立于针型对象建模

原因:

- 针型对象本质是“单线体恢复”
- 第二类对象本质是“编织壳体形态变化”
- 两者主量、噪声源、先验和可解释性都不同

### 6.2 第一批主量应优先几何包络和轴向剖面

优先级建议:

1. `L`
2. `Dmax`
3. `D(x)`
4. `FS`
5. `x_peak`
6. `L_taper`

### 6.3 局部 porosity 不应一开始就成为唯一主量

原因:

- 文献确实重视 local porosity
- 但在二维图像里它对照明、对焦、缩放和网纹 aliasing 非常敏感
- 更适合作为增强解释量

### 6.4 当前仓库的最小目标

如果后续正式开做第二类对象，最小目标建议定义为:

- 从图像中稳定提取:
  - `L`
  - `Dmax`
  - `D(x)`
  - `FS`
- 能对鼓包与收口区域做 QC 可视化
- 暂不要求:
  - 真实 porosity 定量
  - 力学部署预测
  - 与具体商业器械型号一一映射

## 7. 一句话方案

第二类对象建议采用一套完全独立于针型对象的新链路:

- 基线: `纯包络法`
- 主方法: `轴线-宽度剖面法`
- 增强: `局部网孔密度代理法`

也就是说:

- 先把“长度和直径分布”稳定测出来
- 再考虑“局部编织密度/孔隙率代理”
- 不要一开始就把问题定义成“曲率恢复”或“高保真部署仿真”
