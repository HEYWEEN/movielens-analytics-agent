# MovieLens 1M 数据治理 Agent（迭代一）

一次中文请求可触发 Hadoop Streaming 的清洗前评分、数据清洗、清洗后复评分三个作业，并在网页查看五维质量分数、数据处置、未核验冲突、版本和 `T1/T2`。Agent 使用规则路由，不调用大模型 API。

## 准备与运行

需要 Python 3.9+。将课程提供的 `users.dat`、`movies.dat`、`ratings.dat` 放在**仓库同级**的 `ml-1m/` 目录；程序会核对输入摘要 `04a56a90fa50af01`，不匹配时明确拒绝。本仓库不附带数据或运行产物：数据集自带许可证禁止未经许可再分发。仅下载官方原版也无法复现课程提供的含异常输入，需要获得课程原始文件并自行配置。

```bash
python3 -m unittest discover -p 'test_*.py' -v
python3 runner.py --engine local        # 本地逻辑验证
python3 server.py                       # 本地 Agent，访问 http://127.0.0.1:8765
```

Hadoop 演示可使用 `./hadoop_local.sh run` 或 `./hadoop_local.sh server`。该脚本默认读取同级 `../.runtime/hadoop-3.4.2` 和本机 OpenJDK 17；换机器须配置对应 Hadoop、Java 环境。三阶段作业在单机 Hadoop 模式验证过；每阶段仍使用单个 reducer，尚未验证 HDFS/YARN 多节点部署。

输出写入 `runs/<task_id>/`，包括三个 UTF-8 清洗文件、`quarantine.jsonl`、`unresolved_conflicts.jsonl`、阶段报告和最终 `report.json`。`GET /api/config` 返回登记版本；`POST /api/chat` 发起任务或追问；`GET /api/tasks/<id>`、`/report`、`/sample?table=ratings`、`/unresolved` 分别提供状态、报告、样例和未核验冲突明细。

## 规则与评分

当前输入、清洗规则、评分配置版本分别为 `04a56a90fa50af01`、`ml1m-rules-v4`、`ml1m-quality-v2`。规则校验字段与取值、跨表引用和时间范围；可确定的空格及毫秒时间戳会规范化，重复记录去重，无效记录隔离。冲突值按频次选择；最高频次打平时以字典序稳定输出，**但保留项标记为未核验，不视为修复或确认正确**。下游使用清洗数据时应同时读取 `unresolved_conflicts.jsonl`。

五维评分清洗前后使用相同口径，实际分子、分母见报告的 `score_evidence`。Accurate 从合规记录数中扣除无效引用和同频冲突未核验行；Complete 衡量必需字段非空；Unique 衡量重复业务键；Up-to-date 以 2003-02-28 日终前 365 天为固定历史参照；Consistent 衡量结构、冲突和引用一致性。它们是课程设计的代理指标，不能证明人口属性、电影信息或评分内容真实。

## 已验证结果与边界

本机全量 Hadoop 任务 `f73af400f895` 和 Agent 单次请求任务 `0d7617dfc153` 均完成三阶段作业，计数与评分一致；六项单元测试通过。

| 表 | 原始行 | 清洗后 | 同频冲突保留但未核验 |
| --- | ---: | ---: | ---: |
| users | 6,946 | 6,040 | 217 |
| movies | 4,465 | 3,883 | 105 |
| ratings | 1,150,241 | 1,000,209 | 0 |

五维分数（Accurate / Complete / Unique / Up-to-date / Consistent）从 `91.211 / 99.232 / 95.690 / 2.066 / 97.503` 变为 `99.968 / 100 / 100 / 2.168 / 100`。训练截止 `T1=2000-12-02T14:52:18Z`，验证截止 `T2=2000-12-29T23:43:34Z`，测试期在 T2 之后。

与已校验的官方基准比对，评分记录完全一致，但用户 126 条、电影 48 条内容不同；这些 ID 全部在 322 条未核验集合内。原因和核查方式见[基准对照说明](官方基准对照核查.md)。本版没有按官方答案回填，仍不能确认同频冲突的真实值；也未提供通用自然语言理解、任意规则编辑或多节点扩展能力。
