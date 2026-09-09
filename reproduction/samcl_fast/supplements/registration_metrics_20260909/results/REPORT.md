# 当前配准复现：补充指标

只使用 native-v4 实验记录。seed 42，原持续序列每任务 10,000 步；不使用论文表格数值。

| 方法 | 指标组 | BWTR | RMA |
|---|---|---:|---:|
| mer | Dice | -0.087097 | 独立参照训练中 |
| mer | TRE_mm | -0.067017 | 独立参照训练中 |
| samcl | Dice | -0.073888 | 独立参照训练中 |
| samcl | TRE_mm | -0.089172 | 独立参照训练中 |

BWTR：Dice 使用 final/initial - 1；TRE 使用 initial/final - 1，均越大越好。
RMA：Dice 使用 initial/independent；TRE 使用 independent/initial。参照均为同配置的独立普通 Adam 训练。
Dice-BWTR 汇总 OASIS、CTCT；Dice-RMA 汇总 CTCT、MRCT。NLST 单列，不跨 Dice/TRE 求总分。
第一个任务不纳入 RMA；最后一个任务不纳入 BWTR。缺失参照不以零代替。
该 RMA 比较固定外层步数下相对普通单任务 Adam 的建模水平，不单独分离 SAM、回放、额外内层更新与历史约束的作用。

资源测量尚未完成。

逐任务明细见 task_metrics.csv；指标状态见 metric_status.json。
独立参照完成后，后台流程自动重新生成本报告。所有测试使用固定预算的最终模型，无测试集选优。
此处报告当前复现实验口径，保留现有数据划分、前景 Dice、物理 RMS-TRE；不声称与其他实验协议相同。
