# MovieLens Analytics Agent

以 MovieLens 1M 为基础，逐轮建设可追溯的数据治理与分析 Agent。**当前实现范围是迭代一**：一次中文请求触发数据检查、清洗前评分、Hadoop Streaming 清洗、清洗后复评分，并在网页展示五维质量、异常处置、证据和 `T1/T2`。第二轮算法分析及第三轮知识图谱尚未实现。

Agent 目前采用规则路由，不调用大模型 API，也不生成模拟评分。`pipeline.py` 是清洗和评分规则的唯一实现；本地模式与 Hadoop 模式调用同一套规则。Hadoop 模式使用三个独立作业，当前为单机 Hadoop / 单 reducer，并未验证多节点扩展。

## 运行

需要 Python 3.9+。把课程提供的 `users.dat`、`movies.dat`、`ratings.dat` 放在仓库**根目录下**的 `ml-1m/`。程序会核对输入文件合成摘要 `04a56a90fa50af01`，不匹配则拒绝运行。仓库不包含数据或运行产物；仅下载官方原版无法复现课程提供的含异常输入。

```bash
./start.sh                           # 一键启动网页，默认 http://127.0.0.1:8765
./start.sh --hadoop                  # 必须使用 Hadoop；未配置时明确退出
./start.sh --local                   # 明确使用本地验证模式
LAB2_PORT=9000 ./start.sh            # 自定义端口
python3 -m unittest discover -p 'test_*.py' -v
python3 verify_iteration1.py runs/<task_id>  # 核验正式 Hadoop 产物
```

`start.sh` 会优先检查已配置的 `HADOOP_STREAMING_JAR`、`hadoop`、`hdfs`，然后检查同级 `../.runtime/hadoop-3.4.2` 与可用的 Java 运行时。两者都不可用时，默认以**本地验证模式**启动并在终端明确提示；它不会把本地结果标成 Hadoop 结果。即使课程数据尚未放好，网页仍可启动，但任务会返回缺失数据文件的原因。按 `Ctrl+C` 停止服务。若要现场证明 Hadoop 执行，请使用 `./start.sh --hadoop`，核对报告中的 `engine`、`hadoop_mode` 和 `phase_jobs`。

单独运行或调试仍可使用：

```bash
python3 runner.py --engine local      # 仅验证本地算法逻辑
python3 server.py                     # 默认本地模式
./hadoop_local.sh run                 # 实际调用 Hadoop Streaming
./hadoop_local.sh server              # 使用 Hadoop 的网页演示入口
```

`hadoop_local.sh` 默认读取同级 `../.runtime/hadoop-3.4.2`，通过 `JAVA_HOME` 或 macOS 的 `/usr/libexec/java_home` 定位 Java。换机器需配置 Hadoop、Java、`HADOOP_STREAMING_JAR` 等环境。普通 `python3 server.py` **不会**调用 Hadoop。

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

为保持现有演示命令和 Hadoop worker 导入路径稳定，第一轮核心文件仍在仓库根目录。新增模块承担跨迭代的任务、工具和产物边界；未来算法按工具接入，而非继续把业务逻辑写进 HTTP Handler。详细设计与接入约定见 [架构说明](docs/architecture.md)，三轮任务顺序与验收点见 [项目路线](docs/project-roadmap.md)。

## Hadoop 与非 Hadoop 的职责

| 环节 | 负责方 | 本轮边界 |
| --- | --- | --- |
| 自然语言请求、工具选择、任务状态、失败提示、追问解释 | Agent / API | 组织任务并解释已保存的结果，不在 Agent 内计算或编造质量分数。 |
| 原始数据解析与检查、异常隔离、去重和规范化 | Hadoop Streaming | 正式演示时由 Hadoop 作业执行，输出清洗数据和异常证据。 |
| 清洗前与清洗后的五维指标计算 | Hadoop Streaming | 两次评分使用同一套 `pipeline.py` 规则，并由独立作业核对结果。 |
| 版本校验、产物摘要、清单、任务持久化 | Python 服务层 | 管理输入与输出的可信边界，不替代 Hadoop 的数据清洗和评分。 |
| 网页输入、进度、五维对比、报告与证据展示 | 前端 | 仅展示真实任务状态和报告；未完成时不显示占位分数。 |
| 本地模式 | Python 验证入口 | 复用相同规则做开发与回归验证；不能作为“Hadoop 已执行”的证据。 |

当前 Hadoop Streaming 的 mapper 将记录送往单个 reducer。它满足本轮实际调用 Hadoop 的要求，但不是可横向扩展的并行清洗架构；若后续数据规模增大，再按表和业务键拆分计算与聚合。

## 当前目录

```text
movielens-analytics-agent/
├── start.sh                     # 推荐的网页启动入口：自动选 Hadoop 或本地模式
├── hadoop_local.sh              # 固定 Hadoop 3.4.2 单机环境的运行入口
├── index.html                   # 对话、任务进度、评分及证据页面
├── server.py                    # HTTP API、规则式 Agent、任务调度
├── runner.py                    # 本地 / Hadoop 三阶段任务执行
├── streaming.py                 # Hadoop Streaming mapper / reducer 入口
├── pipeline.py                  # MovieLens 解析、清洗与五维评分规则
├── verify_iteration1.py         # 正式运行产物、版本与三阶段证据验收
├── registry.json               # 已登记的数据、规则与评分配置版本
├── movielens_agent/
│   ├── core/                    # 后续跨迭代契约的预留包；当前无业务实现
│   ├── tools/registry.py        # 已实现工具的显式登记
│   └── storage/
│       ├── task_store.py        # SQLite 任务状态与阶段事件
│       └── artifacts.py         # 产物清单与 SHA-256 校验
├── docs/
│   ├── architecture.md          # 架构与后续迭代接入约定
│   ├── iteration1-experiment.md # 本轮真实 Hadoop 实验结果与局限
│   ├── project-roadmap.md        # 三轮任务顺序与验收点
│   └── ui-prototype.png          # 已批准的前端原型图
├── test_pipeline.py             # 清洗规则、Streaming、API 错误测试
├── test_architecture.py         # 任务持久化与产物校验测试
├── 官方基准对照核查.md            # 与官方 MovieLens 1M 的历史对照
├── ml-1m/                      # 课程输入，已忽略：users.dat / movies.dat / ratings.dat
└── runs/                        # 运行时生成，不提交：任务 SQLite 与各任务产物
```

课程输入位于仓库内的 `ml-1m/`，但被 `.gitignore` 排除，不会提交到 Git。`runs/` 和数据集由 `.gitignore` 排除；同级 `../.runtime/` 位于仓库之外。第二、三轮工具尚未实现，目录不会用空工具冒充已完成能力。

## 任务与产物

`POST /api/chat` 发起任务或追问；`GET /api/tasks/<id>` 查询状态；`/report`、`/sample?table=ratings`、`/unresolved`、`/manifest` 分别获取报告、样例、未核验冲突及产物清单。`/manifest` 会重新核验列出的文件摘要，文件变化时返回 409。

任务状态和阶段事件写在 `runs/tasks.sqlite3`。服务重启时，未完成的 `queued/running` 任务标记为失败，避免继续显示为运行中；用户可重新提交。当前一次只运行一个清洗任务，其他请求等待执行。SQLite 只保存任务元数据，数据文件仍位于 `runs/<task_id>/`。

每次成功运行输出 `users_clean.dat`、`movies_clean.dat`、`ratings_clean.dat`、`report.json`、`manifest.json`；存在异常时还会输出 `quarantine.jsonl` 与 `unresolved_conflicts.jsonl`。`clean_data_version` 由三个清洗文件的实际内容摘要组成，`manifest.json` 记录输入版本、规则版本、评分配置、`T1/T2`、执行引擎和各产物 SHA-256。后续迭代接入数据前必须校验清单，且同时读取未核验冲突标记。

## 规则、评价与已知边界

当前输入、清洗规则和评分配置分别为 `04a56a90fa50af01`、`ml1m-rules-v4`、`ml1m-quality-v2`。规则校验字段、取值、跨表引用和时间范围；可确定的空格及毫秒时间戳会规范化，重复记录去重，无效记录隔离。冲突值按频次选择，最高频次打平时按字典序稳定输出，**但保留项仍标记为未核验**。

五维评分清洗前后使用相同口径。Accurate 从合规记录数中扣除无效引用和同频冲突未核验行；Complete 衡量必需字段非空；Unique 衡量重复业务键；Up-to-date 以 2003-02-28 日终前 365 天为固定历史参照；Consistent 衡量结构、冲突和引用一致性。它们是代理指标，不能证明人口属性、电影信息或评分内容真实。

2026-09-26 已使用更新后的架构、课程输入和一键启动入口完成三阶段全量 Hadoop Streaming 实验。任务 ID 为 `e95ea259cb07`，清洗数据版本为 `8891cc6d797569c0`；原始用户、电影、评分行数分别为 6,946、4,465、1,150,241，清洗后为 6,040、3,883、1,000,209。五维分数从 `91.211 / 99.232 / 95.690 / 2.066 / 97.503` 变为 `99.968 / 100 / 100 / 2.168 / 100`。`T1=2000-12-02T14:52:18Z`，`T2=2000-12-29T23:43:34Z`。本机的 `runs/e95ea259cb07/` 已通过 `verify_iteration1.py` 核验；本地验证模式与 Hadoop 模式的三个清洗文件 SHA-256 完全一致。实验配置、处置统计和评价局限见 [迭代一实验记录](docs/iteration1-experiment.md)。

与官方基准相比，评分记录一致，用户 126 条、电影 48 条内容不同；这些 ID 均在 322 条未核验集合内，见 [基准对照说明](官方基准对照核查.md)。未按官方答案回填。当前系统也未提供通用自然语言理解、任意规则编辑、多节点 Hadoop 扩展、第二轮算法或第三轮图谱能力。
