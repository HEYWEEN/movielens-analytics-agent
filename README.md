# MovieLens Analytics Agent

以 MovieLens 1M 为基础，逐轮建设可追溯的数据治理与分析 Agent。**当前实现范围是迭代一**：一次中文请求触发数据检查、清洗前评分、Hadoop Streaming 清洗、清洗后复评分，并在网页展示五维质量、异常处置、证据和 `T1/T2`。第二轮算法分析及第三轮知识图谱尚未实现。

Agent 目前采用规则路由，不调用大模型 API，也不生成模拟评分。`pipeline.py` 是清洗和评分规则的唯一实现；本地模式与 Hadoop 模式调用同一套规则。Hadoop 模式使用三个独立作业，当前为单机 Hadoop / 单 reducer，并未验证多节点扩展。

## 运行

需要 Python 3.9+。把课程提供的 `users.dat`、`movies.dat`、`ratings.dat` 放在仓库**同级**的 `ml-1m/`。程序会核对输入文件合成摘要 `04a56a90fa50af01`，不匹配则拒绝运行。仓库不包含数据或运行产物；仅下载官方原版无法复现课程提供的含异常输入。

```bash
python3 -m unittest discover -p 'test_*.py' -v
python3 runner.py --engine local      # 仅验证本地算法逻辑
python3 server.py                     # 默认本地模式，访问 http://127.0.0.1:8765
./hadoop_local.sh run                 # 实际调用 Hadoop Streaming
./hadoop_local.sh server              # 使用 Hadoop 的网页演示入口
```

`hadoop_local.sh` 默认读取同级 `../.runtime/hadoop-3.4.2` 和 `/opt/homebrew/opt/openjdk@17`。换机器需配置 Hadoop、Java、`HADOOP_STREAMING_JAR` 等环境。若要证明一次演示由 Hadoop 执行，应使用 `./hadoop_local.sh server`，并在报告中核对 `engine`、`hadoop_mode` 和 `phase_jobs`。普通 `python3 server.py` **不会**调用 Hadoop。

## 当前架构

```text
index.html ──HTTP──> server.py (规则路由、任务状态、结果服务)
                         │
                         ├── movielens_agent/tools/registry.py (工具契约登记)
                         ├── movielens_agent/storage/task_store.py (SQLite 任务状态)
                         └── runner.py (本地/Hadoop 执行适配)
                                  ├── streaming.py (Hadoop 入口)
                                  ├── pipeline.py (治理规则与评分)
                                  └── movielens_agent/storage/artifacts.py (产物校验)
```

为保持现有演示命令和 Hadoop worker 导入路径稳定，第一轮核心文件仍在仓库根目录。新增模块承担跨迭代的任务、工具和产物边界；未来算法按工具接入，而非继续把业务逻辑写进 HTTP Handler。详细设计与接入约定见 [架构说明](docs/architecture.md)。

## 任务与产物

`POST /api/chat` 发起任务或追问；`GET /api/tasks/<id>` 查询状态；`/report`、`/sample?table=ratings`、`/unresolved`、`/manifest` 分别获取报告、样例、未核验冲突及产物清单。`/manifest` 会重新核验列出的文件摘要，文件变化时返回 409。

任务状态和阶段事件写在 `runs/tasks.sqlite3`。服务重启时，未完成的 `queued/running` 任务标记为失败，避免继续显示为运行中；用户可重新提交。当前一次只运行一个清洗任务，其他请求等待执行。SQLite 只保存任务元数据，数据文件仍位于 `runs/<task_id>/`。

每次成功运行输出 `users_clean.dat`、`movies_clean.dat`、`ratings_clean.dat`、`report.json`、`manifest.json`；存在异常时还会输出 `quarantine.jsonl` 与 `unresolved_conflicts.jsonl`。`clean_data_version` 由三个清洗文件的实际内容摘要组成，`manifest.json` 记录输入版本、规则版本、评分配置、`T1/T2`、执行引擎和各产物 SHA-256。后续迭代接入数据前必须校验清单，且同时读取未核验冲突标记。

## 规则、评价与已知边界

当前输入、清洗规则和评分配置分别为 `04a56a90fa50af01`、`ml1m-rules-v4`、`ml1m-quality-v2`。规则校验字段、取值、跨表引用和时间范围；可确定的空格及毫秒时间戳会规范化，重复记录去重，无效记录隔离。冲突值按频次选择，最高频次打平时按字典序稳定输出，**但保留项仍标记为未核验**。

五维评分清洗前后使用相同口径。Accurate 从合规记录数中扣除无效引用和同频冲突未核验行；Complete 衡量必需字段非空；Unique 衡量重复业务键；Up-to-date 以 2003-02-28 日终前 365 天为固定历史参照；Consistent 衡量结构、冲突和引用一致性。它们是代理指标，不能证明人口属性、电影信息或评分内容真实。

先前使用课程输入完成的 Hadoop 任务记录显示，原始用户、电影、评分行数分别为 6,946、4,465、1,150,241；清洗后为 6,040、3,883、1,000,209。五维分数从 `91.211 / 99.232 / 95.690 / 2.066 / 97.503` 变为 `99.968 / 100 / 100 / 2.168 / 100`。`T1=2000-12-02T14:52:18Z`，`T2=2000-12-29T23:43:34Z`。这些是旧版运行记录；更新架构后的全量 Hadoop 运行需要在具备课程数据的机器上重新执行和核验。

与官方基准相比，评分记录一致，用户 126 条、电影 48 条内容不同；这些 ID 均在 322 条未核验集合内，见 [基准对照说明](官方基准对照核查.md)。未按官方答案回填。当前系统也未提供通用自然语言理解、任意规则编辑、多节点 Hadoop 扩展、第二轮算法或第三轮图谱能力。
