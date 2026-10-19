# MovieLens Analytics Agent

以 MovieLens 1M 为基础，逐轮建设可追溯的数据治理与分析 Agent。当前已实现**迭代一数据治理**和**迭代二分类、聚类、降维**：先校验和清洗课程数据，再用清洗数据训练六个独立算法工具，在网页中发起分析、查看真实报告、二维坐标和证据，并通过 DeepSeek 围绕已发布结果对话。第三轮知识图谱尚未实现。

迭代二对话使用 DeepSeek tool calling：用户直接输入自然语言，模型根据目标发现已有任务、连续调用已登记工具，再解释实际结果。普通问候、方法讨论和澄清问题可以直接对话，查询整体报告不要求填写用户 ID。实际预测、指标和版本来自分析产物；未配置密钥、缺少模型或版本校验失败时会明确返回不可用。迭代一原有的 `/api/chat` 规则入口仍可单独使用。`pipeline.py` 是清洗和五维评分规则的唯一实现；本地模式与 Hadoop 模式调用同一套规则。Hadoop 模式使用三个独立作业，当前为单机 Hadoop / 单 reducer，尚未验证多节点扩展。

## 运行

需要 Python 3.10+。先安装分析依赖；把课程提供的 `users.dat`、`movies.dat`、`ratings.dat` 放在仓库**根目录下**的 `ml-1m/`。治理程序会核对输入文件合成摘要 `04a56a90fa50af01`，不匹配则拒绝运行。仓库不包含数据或运行产物；仅下载官方原版无法复现课程提供的含异常输入。

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-analysis.txt
export DEEPSEEK_API_KEY='你的密钥'  # 仅在启动服务的终端设置；不要写进仓库
./start.sh                           # 一键启动网页，默认 http://127.0.0.1:8765
./start.sh --hadoop                  # 必须使用 Hadoop；未配置时明确退出
./start.sh --local                   # 明确使用本地验证模式
LAB2_PORT=9000 ./start.sh            # 自定义端口
python3 -m unittest discover -p 'test_*.py' -v
python3 verify_iteration1.py runs/<task_id>  # 核验正式 Hadoop 产物
# 用下方分析命令生成新任务后，再运行 verify_iteration2.py 核验对应目录
```

`DEEPSEEK_API_KEY` 只由服务端读取。未设置时网页仍可查看已发布报告，也能从“开始分析”按钮执行算法；模型对话会显示不可用。默认模型为 `deepseek-flash`，可用 `DEEPSEEK_MODEL` 调整。首次分析前须先有通过清单校验的迭代一任务；网站会列出可用来源。改动聚类数、目标维数或信息保留阈值需要新建分析任务，不会更改已发布模型；指定 UserID 集合只筛选已发布结果，不重训模型。

不通过网页时可直接运行和核验一个完整分析任务：

```bash
analysis_id=$(python3 -c 'import uuid; print(uuid.uuid4().hex[:12])')
python3 analysis_pipeline.py --source runs/e95ea259cb07 --output "runs/iteration2/$analysis_id"
python3 verify_iteration2.py "runs/iteration2/$analysis_id"
# 也可增加 --n-clusters 4 --target-dimensions 3；信息保留阈值用 --information-threshold 0.6
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
index.html ──HTTP──> server.py (DeepSeek 工具调度、任务状态、结果服务)
                         │
                         ├── movielens_agent/core/deepseek_client.py (模型调用)
                         ├── movielens_agent/tools/registry.py (六个算法工具契约)
                         ├── movielens_agent/storage/task_store.py (SQLite 任务状态)
                         ├── analysis_pipeline.py + movielens_agent/analysis/ (迭代二)
                         └── runner.py (迭代一本地/Hadoop 执行适配)
                                  ├── streaming.py (Hadoop 入口)
                                  ├── pipeline.py (治理规则与评分)
                                  └── movielens_agent/storage/artifacts.py (产物校验)
```

为保持现有演示命令和 Hadoop worker 导入路径稳定，第一轮核心文件仍在仓库根目录。第二轮算法在独立模块中实现，HTTP Handler 只负责输入校验、任务调度和结果读取。详细设计见 [架构说明](docs/architecture.md)，三轮验收顺序见 [项目路线](docs/project-roadmap.md)。

## Hadoop 与非 Hadoop 的职责

| 环节 | 负责方 | 本轮边界 |
| --- | --- | --- |
| 自然语言请求、工具选择、任务状态、失败提示、追问解释 | Agent / API | 组织任务并解释已保存的结果，不在 Agent 内计算或编造质量分数。 |
| 原始数据解析与检查、异常隔离、去重和规范化 | Hadoop Streaming | 正式演示时由 Hadoop 作业执行，输出清洗数据和异常证据。 |
| 清洗前与清洗后的五维指标计算 | Hadoop Streaming | 两次评分使用同一套 `pipeline.py` 规则，并由独立作业核对结果。 |
| 版本校验、产物摘要、清单、任务持久化 | Python 服务层 | 管理输入与输出的可信边界，不替代 Hadoop 的数据清洗和评分。 |
| 网页输入、进度、五维对比、报告与证据展示 | 前端 | 仅展示真实任务状态和报告；未完成时不显示占位分数。 |
| 本地模式 | Python 验证入口 | 复用相同规则做开发与回归验证；不能作为“Hadoop 已执行”的证据。 |
| 分类、聚类与降维 | Python 分析工具 | 读取已验证的清洗产物，按 `T1/T2` 切分并训练；不把非 Hadoop 的机器学习训练称为 Hadoop 作业。 |
| DeepSeek 对话 | Agent / API | 选择工具并说明已返回的证据；不参与模型训练，也不能编造缺失的预测。 |

当前 Hadoop Streaming 的 mapper 将记录送往单个 reducer。它满足本轮实际调用 Hadoop 的要求，但不是可横向扩展的并行清洗架构；若后续数据规模增大，再按表和业务键拆分计算与聚合。

## 当前目录

```text
movielens-analytics-agent/
├── start.sh                     # 推荐的网页启动入口：自动选 Hadoop 或本地模式
├── hadoop_local.sh              # 固定 Hadoop 3.4.2 单机环境的运行入口
├── index.html                   # 三栏 Agent 工作台、任务、分析结果及证据
├── server.py                    # HTTP API、DeepSeek 工具调用、任务调度
├── analysis_pipeline.py         # 分类、聚类、降维训练与已发布结果查询
├── verify_iteration2.py         # 迭代二产物、版本和上游链校验
├── requirements-analysis.txt    # 第二轮分析依赖
├── runner.py                    # 本地 / Hadoop 三阶段任务执行
├── streaming.py                 # Hadoop Streaming mapper / reducer 入口
├── pipeline.py                  # MovieLens 解析、清洗与五维评分规则
├── verify_iteration1.py         # 正式运行产物、版本与三阶段证据验收
├── registry.json               # 已登记的数据、规则与评分配置版本
├── movielens_agent/
│   ├── core/deepseek_client.py  # 服务端 DeepSeek Chat Completions 客户端
│   ├── analysis/                # 数据切分、分类特征、画像/聚类/降维
│   ├── tools/registry.py        # 已实现工具的显式登记
│   └── storage/
│       ├── task_store.py        # SQLite 任务状态与阶段事件
│       └── artifacts.py         # 产物清单与 SHA-256 校验
├── docs/
│   ├── architecture.md          # 架构与后续迭代接入约定
│   ├── iteration1-experiment.md # 本轮真实 Hadoop 实验结果与局限
│   ├── iteration2-experiment.md # 第二轮真实分析实验与评价边界
│   ├── project-roadmap.md        # 三轮任务顺序与验收点
│   └── ui-prototype.png          # 已批准的前端原型图
├── test_pipeline.py             # 清洗规则、Streaming、API 错误测试
├── test_architecture.py         # 任务持久化与产物校验测试
├── test_analysis_pipeline.py    # 时间特征、参数与迭代二清单验证
├── test_agent_api.py            # 工具调度与服务端密钥边界
├── 官方基准对照核查.md            # 与官方 MovieLens 1M 的历史对照
├── ml-1m/                      # 课程输入，已忽略：users.dat / movies.dat / ratings.dat
└── runs/                        # 运行时生成，不提交：任务 SQLite 与各任务产物
```

课程输入位于仓库内的 `ml-1m/`，但被 `.gitignore` 排除，不会提交到 Git。`runs/` 和数据集由 `.gitignore` 排除；同级 `../.runtime/` 位于仓库之外。迭代二六个工具已实现；第三轮工具尚未实现。

## 任务与产物

迭代一原有 `POST /api/chat` 发起任务或追问；`GET /api/tasks/<id>` 查询状态；`/report`、`/sample?table=ratings`、`/unresolved`、`/manifest` 分别获取报告、样例、未核验冲突及产物清单。迭代二使用 `POST /api/agent/chat` 对话；网页使用 `POST /api/agent/stream`，以 NDJSON 实时接收阶段、工具事件、模型文本增量与最终回答。模型端使用 DeepSeek SSE 流，网页边接收边显示，回复完成后保存会话；中断时保留已接收文字并提示未完成。`GET /api/source-tasks` 选择已验证来源，`POST /api/analysis/tasks` 发起训练，`GET /api/analysis/tasks/<id>` 读取进度；其 `/report` 与 `/artifacts/<name>` 提供已核验结果与公开产物。清单校验失败时拒绝读取。

任务状态和阶段事件分别写在 `runs/tasks.sqlite3` 与 `runs/analysis_tasks.sqlite3`。服务重启时，未完成的 `queued/running` 任务标记为失败，避免继续显示为运行中；用户可重新提交。当前一次只执行一个耗时任务，其他请求等待执行。SQLite 只保存任务元数据，数据文件仍位于 `runs/<task_id>/` 或 `runs/iteration2/<analysis_task_id>/`。

每次成功运行输出 `users_clean.dat`、`movies_clean.dat`、`ratings_clean.dat`、`report.json`、`manifest.json`；存在异常时还会输出 `quarantine.jsonl` 与 `unresolved_conflicts.jsonl`。`clean_data_version` 由三个清洗文件的实际内容摘要组成，`manifest.json` 记录输入版本、规则版本、评分配置、`T1/T2`、执行引擎和各产物 SHA-256。后续迭代接入数据前必须校验清单，且同时读取未核验冲突标记。

迭代二任务输出 `report.json`、`models.joblib`、`profiles.json`、`clusters.json`、`coordinates.json` 和 `manifest.json`；后者记录上游清单摘要、模型版本、参数与每个文件的 SHA-256。训练只用 `timestamp <= T1` 的数据；验证用 `(T1,T2]`，测试用 `>T2`。画像保存 37 维数值向量，降维坐标保存实际目标维数，网页绘制其前两维。真实配置、指标与局限见 [迭代二实验记录](docs/iteration2-experiment.md)。

## 规则、评价与已知边界

当前输入、清洗规则和评分配置分别为 `04a56a90fa50af01`、`ml1m-rules-v4`、`ml1m-quality-v2`。规则校验字段、取值、跨表引用和时间范围；可确定的空格及毫秒时间戳会规范化，重复记录去重，无效记录隔离。冲突值按频次选择，最高频次打平时按字典序稳定输出，**但保留项仍标记为未核验**。

五维评分清洗前后使用相同口径。Accurate 从合规记录数中扣除无效引用和同频冲突未核验行；Complete 衡量必需字段非空；Unique 衡量重复业务键；Up-to-date 以 2003-02-28 日终前 365 天为固定历史参照；Consistent 衡量结构、冲突和引用一致性。它们是代理指标，不能证明人口属性、电影信息或评分内容真实。

2026-09-26 已使用更新后的架构、课程输入和一键启动入口完成三阶段全量 Hadoop Streaming 实验。任务 ID 为 `e95ea259cb07`，清洗数据版本为 `8891cc6d797569c0`；原始用户、电影、评分行数分别为 6,946、4,465、1,150,241，清洗后为 6,040、3,883、1,000,209。五维分数从 `91.211 / 99.232 / 95.690 / 2.066 / 97.503` 变为 `99.968 / 100 / 100 / 2.168 / 100`。`T1=2000-12-02T14:52:18Z`，`T2=2000-12-29T23:43:34Z`。本机的 `runs/e95ea259cb07/` 已通过 `verify_iteration1.py` 核验；本地验证模式与 Hadoop 模式的三个清洗文件 SHA-256 完全一致。实验配置、处置统计和评价局限见 [迭代一实验记录](docs/iteration1-experiment.md)。

与官方基准相比，评分记录一致，用户 126 条、电影 48 条内容不同；这些 ID 均在 322 条未核验集合内，见 [基准对照说明](官方基准对照核查.md)。未按官方答案回填。迭代二继承这份清洗输入及其未核验标记，不宣称重新核实了原始事实。目前尚未实现第三轮图谱和多节点 Hadoop 扩展；聚类与降维选型尚待对应课程 PPT 核对。
