# 温度同步最小约定

最小仓库当前支持两种温度文件:

## 1. 按 `frame` 对齐

适合已经把温度日志重采样到视频帧的情况。

```csv
frame,temperature_c
0,20.0
1,20.1
2,20.2
```

## 2. 按 `time_sec` 对齐

适合采集卡/温度计独立记录的情况。

```csv
time_sec,temperature_c
0.0,20.0
0.5,20.4
1.0,20.8
```

分析时会把温度序列线性插值到视频时间轴。

## 最小同步建议

推荐先用这套最小规则:

1. 视频开始录制时，让热台或水浴温度稳定 2 到 3 秒
2. 在视频第一帧里留下一个可见事件作为同步点，例如夹具轻触或 LED 亮灭
3. 温度日志保留绝对时间或相对秒数
4. 若视频与温度日志存在固定延迟，在分析入口传入 `temperature_time_offset_sec`

## 当前仓库中的模板

模板文件位于:

- [temperature_template.csv](/Users/wangxq/Documents/niti-bfr-standard-2/data/temperature_template.csv)

这只是格式示意，不是实验真值。
