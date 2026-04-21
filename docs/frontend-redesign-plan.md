# 前端重做实施说明

## Summary
将当前偏研究和 benchmark 控制台的前端，重构为面向真实使用的分析台。新页面只服务真实分析流程：选择物体类型、上传视频、设置抽帧步长、上传温度 CSV、执行分析、导出 1 个总览标注视频，并展示三条路线 `A / B / C` 的曲线、`Af-95`、`Af-tan` 和下载产物。测试校准、sample run、benchmark 汇总不再出现在用户可见页面里。

## 1. 页面与信息架构
- 首页改为单一真实分析入口，只保留：
  `物体类型`
  `视频文件`
  `抽帧步长`
  `温度 CSV（可选）`
- 去掉前端中的：
  `quicklook/formal Af` 手动模式选择
  sample run 入口
  benchmark 页面入口
  测试和校准说明
- 保留历史页，仅用于查看真实分析 run。
- 结果页固定展示：
  输入信息
  对象级结果摘要
  A/B/C 三条路线卡片
  曲线区
  标注视频
  下载区

## 2. 运行逻辑与输入 contract
- 前端提交表单字段固定为：
  `preset`
  `video_file`
  `frame_stride`
  `temperature_file`
- 删除 `requested_mode` 的用户输入。
- 模式判定规则固定为：
  无温度 CSV 时，自动 quicklook
  有温度 CSV 时，尝试 formal；若 gate 不通过则自动降级 quicklook 或 provisional
- `frame_stride` 语义固定为：
  每隔 `N` 帧分析一帧
  `1` 表示逐帧分析
- run 和 summary 元数据新增：
  `frame_stride`
  `original_frame_count`
  `analyzed_frame_count`
  `annotated_video_filename`

## 3. 后端分析能力接入
- `wire` 与 `braided` 两套分析入口都新增 `frame_stride` 参数。
- 抽帧后仍必须计算三条路线 `A / B / C`，不能只跑单一路线。
- 温度 CSV 对齐规则固定为：
  若 CSV 使用 `frame` 列，则匹配被分析的原始帧号
  若 CSV 使用 `time_sec` 列，则对被分析帧时间戳插值
  缺少 `temperature_c` 时直接报错
- 继续复用现有 canonical `object_*` 和 `route_results` contract 作为结果页主数据源。

## 4. 曲线与指标展示
- 前端始终展示三条路线 `A / B / C` 的 route-level 结果。
- 每条路线卡片固定显示：
  metric label
  status
  gate reason
  `Af-95`
  `Af-tan`
  warning codes
- 曲线展示固定为两层：
  路线 metric 趋势图：按 A / B / C 分开显示
  recovery vs temperature 图：仅在有温度 CSV 时显示，A / B / C 同图叠加
- 无温度 CSV 时：
  不展示 Af 数值
  不展示温度 recovery 图
  明确提示“当前为 quicklook，仅展示几何趋势”

## 5. 标注视频导出
- 前端默认只导出 1 个总览标注视频：
  `annotated_overview.mp4`
- 视频内容固定为同一画面整合 A / B / C 三条路线的标注信息。
- 画面中至少包含：
  当前帧号和时间
  A 路线关键标注
  B 路线关键标注
  C 路线关键标注
  当前指标摘要
- 若 `frame_stride > 1`，导出视频只包含被分析帧。
- 导出 fps 规则固定为：
  `max(input_fps / frame_stride, 1.0)`
- 结果页支持浏览器内预览和下载该视频。

## 6. 用户可见范围收口
- benchmark 页面与导航从前端移除。
- sample run 从首页移除。
- 测试和校准继续通过脚本和代码执行完成，不进入真实使用页面。
- 允许保留 benchmark/sample 相关后端代码与脚本，但不再暴露给前端用户。

## Public Interfaces / Types
- `POST /runs` 表单：
  删除 `requested_mode`
  新增 `frame_stride`
- 分析入口：
  `analyze_video(..., frame_stride=1)`
  `analyze_braided_video_quicklook(..., frame_stride=1)`
- run 和 summary 字段新增：
  `frame_stride`
  `original_frame_count`
  `analyzed_frame_count`
  `annotated_video_filename`
- 结果产物新增标准文件名：
  `annotated_overview.mp4`

## Test Plan
- 表单提交与入库：
  四个输入项能正确提交
  `frame_stride` 非法值会被拒绝
- 抽帧行为：
  `frame_stride=1` 与现有逐帧结果一致
  `frame_stride>1` 时分析帧数正确
  抽帧后温度 CSV 仍能正确对齐
- 结果页：
  A/B/C 三条路线始终展示
  无温度 CSV 时不显示 Af
  有温度 CSV 时显示对象级与路线级 Af
- 标注视频：
  `wire` 和 `braided` 都能生成 `annotated_overview.mp4`
  视频可预览、可下载
- 页面收口：
  首页不再出现 sample/benchmark/test 校准入口
  历史页仍可查看真实分析任务

## Assumptions
- 继续使用现有 FastAPI + Jinja，不引入新的前端框架。
- “选择帧数”按“抽帧步长”实现。
- 标注视频默认只导出 1 个总览视频。
- 历史页保留。
- 温度 CSV 仍为可选输入；缺失时只提供 quicklook 结果。
