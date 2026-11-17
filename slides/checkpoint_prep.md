# 11/17 Lab checkpoint 准备清单

对应老师要求的四样东西。PPT 是 `Final_Project_Checkpoint_AoHe.pptx`(10 页,每页备注栏有英文讲稿)。第 7 页是真实输出文件 `outputs_new/all_locations_summary.csv` 的完整表格。

## 1. One example of your input data(PPT 第 3、4 页)

现场最好能打开原始文件,而不只是看幻灯片:

- 一对原始抓拍:`IM_24976.JPG` 和 `IM_24977.JPG`(第 3 页用的就是这一对,09:13:46 和 09:13:49)。在本地 `D:\0_MAIN_BIKE_DATASETS_clean` 对应 loc_04 的文件夹里,拷到笔记本上。
- 一份标注文件:`configs/locations_new/loc_04.json`(第 4 页右侧的 JSON 就是它的缩写)。

## 2. A working path from input toward output(PPT 第 5–8 页)

- 打开 `notebooks/run_rider_count_new.ipynb`,Cell 2 设 `LOC = "loc_04"`,保持 `reuse_detections=True`(复用 detections_raw.csv,不用现场跑 YOLO,几十秒出结果)。
- 能展示的产物:`outputs_new/loc_04/riders.csv`、`direction_check/` 里的 `WW_` / `OK_` 验证图、`scene_summary.json`。
- 提前在本地跑一遍确认不报错。

## 3. One thing that currently fails(PPT 第 9 页)

自动方向只覆盖 14.7% 的事件(268/1,823)。原因分解(来自 `outputs_new/pair_diagnostic.csv`,1,545 个方向未知事件):45% 磁盘上没有第二张图,27% 有图对但漏检,22% 被保守门控扣下,6% 位移太小。注意论文里写的是 48/26/25/1,和这份文件对不上。
试过的修法(信任被扣下的图对)失败:Loc 04 新增 16/16 全逆行,Loc 15 新增 23/23 全顺流,原因是排队的两个骑行者在两张图之间互换。已回滚。

## 4. A draft plan for your final presentation(PPT 第 10 页)

按 10 分钟假设排的六段。页面上留了两个要问老师的问题:时长是不是 10 分钟,demo 要现场跑还是录屏。

## 可能被问到的问题

- **为什么不用视频?** 相机是低功耗野生动物相机,只能拍静态图。好处是便宜、能长期多点布设;代价就是第 9 页的方向覆盖率问题。
- **检测器训练过吗?** 没有,YOLOv8s 直接零样本使用,置信度阈值 0.10,输入 1280 像素。
- **92.9% 是 recall 吗?** 不是严格的 recall,是计数比(1,823/1,963)。没有逐框标注,所以算不了 mAP 或事件级 precision/recall。超计数主要来自同一骑行者被切成两个事件(Loc 17 多 38 个)。
- **人行道 49.6% 算违规吗?** 不直接算。人行道没有方向标线,这个数字衡量的是相对近侧车流的方向混行,不是违规率。
- **手工计数可靠吗?** 单人计数,没有做第二标注者一致性检验,这是论文里写明的局限。
