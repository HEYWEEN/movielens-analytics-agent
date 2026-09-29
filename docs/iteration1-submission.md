# 迭代一系统与实验提交说明

## 1. 系统说明

本轮交付的是 MovieLens 1M 数据治理 Agent：用户在网页提交中文清洗与评估请求，Agent 选择已登记的治理工具，依次运行清洗前评分、清洗、清洗后评分三个 Hadoop Streaming 作业，再展示任务状态、五维质量对比、异常处置及报告。Agent 负责组织任务和解释已保存的结果；清洗与评分计算由 Hadoop 作业执行。

系统由 `index.html` 前端、`server.py` Agent/API、`runner.py` 执行适配、`streaming.py` Hadoop 入口、`pipeline.py` 规则与评分、SQLite 任务状态和文件产物组成。架构边界及后续迭代接入约定见 [架构说明](architecture.md)。本轮仅完成数据治理能力，算法分析和知识图谱属于后续迭代。

## 2. 运行方法

将课程提供的 `users.dat`、`movies.dat`、`ratings.dat` 放入仓库根目录的 `ml-1m/`。需要 Python 3.9+；正式 Hadoop 演示还需要 Java、Hadoop 和 Hadoop Streaming JAR。数据与运行产物被 Git 忽略，不随仓库提交。

```bash
./start.sh --hadoop
# 浏览器打开 http://127.0.0.1:8765，提交一次默认清洗请求
python3 verify_iteration1.py runs/<task_id>
python3 -m unittest discover -p 'test_*.py' -v
```

`./start.sh` 会自动选择可用引擎；`./start.sh --local` 仅用于本地规则验证。现场证明 Hadoop 执行时使用 `--hadoop`，并核对报告中的 `engine`、`hadoop_mode` 与 `phase_jobs`。端口可用 `LAB2_PORT` 指定。完整环境查找顺序及其他启动方式见 [README](../README.md#运行)。

## 3. 工具接口与任务产物

当前唯一登记工具为 `governance.clean_and_assess`，要求输入原始数据版本，输出清洗数据、质量报告及未核验冲突证据。请求可指定 `data_version`、`rule_version`、`score_config_version`；省略时使用登记的默认版本。规则路由支持约定范围内的中文发起任务、查询进度和围绕报告追问，不是通用自然语言理解。

| 接口 | 用途 |
| --- | --- |
| `GET /api/config` | 获取已登记版本与默认值 |
| `POST /api/chat` | 以 `message` 发起任务或追问，可附 `task_id` 和三个版本字段 |
| `GET /api/tasks/<id>` | 查询任务状态、阶段与失败原因 |
| `GET /api/tasks/<id>/report` | 获取评估报告 |
| `GET /api/tasks/<id>/manifest` | 校验并获取产物清单；摘要不符返回 409 |
| `GET /api/tasks/<id>/sample?table=ratings` | 获取清洗数据样例；`table` 可为 `users`、`movies`、`ratings` |
| `GET /api/tasks/<id>/unresolved` | 获取保留但未核验的冲突证据 |

成功任务在 `runs/<task_id>/` 输出三份 `*_clean.dat`、`report.json`、`manifest.json`，以及异常证据文件 `quarantine.jsonl`、`unresolved_conflicts.jsonl`。`manifest.json` 记录版本、时间边界、引擎和文件 SHA-256；后续迭代应先校验清单再读取数据。

## 4. 数据、规则与模型版本

| 项目 | 本轮版本 |
| --- | --- |
| 课程异常输入数据 | `04a56a90fa50af01` |
| 清洗规则 | `ml1m-rules-v4` |
| 五维评分配置 | `ml1m-quality-v2` |
| 本次清洗数据 | `8891cc6d797569c0` |
| 模型 | 本轮未训练或调用大模型，因此无模型版本；Agent 使用规则路由 |

版本在 `registry.json` 登记。`T1=2000-12-02T14:52:18Z`，`T2=2000-12-29T23:43:34Z`；后续训练、验证和测试沿用同一数据版本及边界。

## 5. 实验配置与评价结果

正式实验任务 `e95ea259cb07` 使用 Hadoop 3.4.2 单机 standalone 模式、单 reducer，依次执行清洗前评分、清洗、清洗后评分三个独立 Hadoop Streaming 作业。五维分数前后使用同一评分配置，满分 100。

| 维度 | 清洗前 | 清洗后 |
| --- | ---: | ---: |
| Accurate | 91.211 | 99.968 |
| Complete | 99.232 | 100.000 |
| Unique | 95.690 | 100.000 |
| Up-to-date | 2.066 | 2.168 |
| Consistent | 97.503 | 100.000 |

用户、电影、评分表分别从 6,946、4,465、1,150,241 行变为 6,040、3,883、1,000,209 行。共移除 151,520 行：完全重复 34,742、冲突变体 15,325、无效或孤立记录 101,453。另有 322 条同频冲突保留但标为未核验，不计入移除；本次规范化后最终保留的修复行数为 0。公式、分子分母、作业 ID、复核过程和结果解释见 [迭代一实验记录](iteration1-experiment.md)。

## 6. 已知限制

- Accurate 等五维分数是按项目规则计算的代理指标，无法证明用户属性、电影信息或评分内容在真实世界中正确；Up-to-date 使用固定历史参照，不代表数据已经更新到今天。
- 322 条同频冲突虽然有稳定保留值，仍缺少外部证据确认其真实性。隔离记录不等于修复记录。
- Hadoop 作业目前由单 reducer 汇总，证明了实际 Hadoop 执行，但没有多节点扩展性能证据。
- 规则路由仅覆盖约定的中文请求及报告追问；未接入大模型，也不支持任意自然语言指令或任意清洗规则编辑。
- 原始课程数据和完整运行产物不提交 Git。复现实验必须取得同一课程输入，并通过版本和产物摘要校验；官方原版 MovieLens 1M 不等同于本次异常输入。
