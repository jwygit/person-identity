# 独立人物身份关联实验

第一阶段原型：独立于 m3-agent 和 Grove 的可回放身份图。当前只实现观测输入、同模态候选检索、关联审核、撤销、人物图和事件导出。**尚未实现视频提取器或自动音画匹配，合成数据通过不代表真实识别有效。**

## 为什么拆分

m3-agent 当前实现路径：

- `mmagent/face_processing.py`：InsightFace buffalo_l 提取脸部向量，聚类后匹配已有 face 节点。
- `mmagent/voice_processing.py`：Gemini 分段和转录，ERes2NetV2 提取声纹，再匹配 voice 节点。模型加载和推理写死 CUDA，导入时加载权重。
- `mmagent/prompts.py` 的 `prompt_generate_semantic_memory_with_ids_sft_equivalence`：根据时间、上下文、音画一致性生成 face/voice 等价关系。
- `mmagent/videograph.py` 的 `refresh_equivalences`：收集等价关系，使用并查集生成 character。每次刷新重新编号，不适合作为外部长期稳定人物主键。

face 与 voice 不在同一向量空间，不能直接算相似度确定是否同一人。画面中有人且同时有声音也不能证明该人在说话。

## 目标流水线

带音轨的视频 → 带时间戳的人脸轨迹、说话片段 → 同模态跨片段检索 → 音画对应候选 → 人物图 → Grove 适配层。

视频提取层后续需输出：

- face：裁剪图引用、检测框、轨迹、质量、向量和模型版本。
- voice：说话区间、音频引用、转录、重叠说话标记、向量和模型版本。
- 跨模态候选：两个观测 ID、视频区间、方法、模型版本、证据及原始分数。VLM 或主动说话人检测结果先作为候选；分数经标注集校准后再考虑自动确认。

“人物图”第一阶段指身份图：Person ↔ Face/Voice observations ↔ 来源片段。姓名可后续人工标注，不从外观猜测。人与人的社会关系推断不在当前范围。

## 运行现有原型

只需 Python 标准库。在本目录执行：

```sh
python3 -m unittest discover -s tests -v
python3 identity.py examples/synthetic.json --out runs/synthetic
```

输出目录必须不存在，防止覆盖实验。`graph.json` 保存归一化观测、关联和审核日志；`people.json` 保存人物列表、别名、来源事件和关联状态。

输入格式见示例；时间使用各源视频的秒数，ID 必须全局唯一，建议带视频命名空间。实际跨视频时间线还需采集起始时间映射。

Python API：`observe`、`candidates`、`propose`、`review`、`people`、`export`。用 `IdentityGraph(json.load(...))` 恢复自己的可信图数据。`candidates` 只检索同模态、同模型、同维度向量，不自动认定身份。

人物 ID 以最早插入的观测为锚；新增不相关观测不会改号。确认关联会合并组件，被吸收 ID 成为别名；撤销后重新计算组件，拆出的旧 ID 恢复。下游应同时存 observation ID 和图 revision，不能把合并结果永久写死。另一条确认路径仍存在时，撤销一条边不会拆开组件。

所有关联初始是 candidate，只有明确 review 为 confirmed 才参与合并。当前审核是人工/实验标注机制，不具备自动冲突检测、置信度校准或生产级并发持久化。

## 真实视频验证顺序

1. 准备同一批人物的多个视频，人工标注谁在何时出现/说话及 face↔voice 对应；按视频划分调参集与测试集。
2. 独立验证人脸重识别和说话人重识别，统计错误合并、错误拆分。
3. 验证 face↔voice：单人正面讲话、多人同框、画外音、遮挡、交替说话、重叠语音；画外音和不确定情况应允许未关联。
4. 统计跨模态关联 precision、recall、覆盖率及错误合并率，不能只报告“人物数量正确”。基于真实样本确定阈值，保留原始证据便于逐条复核。
5. 验证新增视频仍复用身份、错误关联可撤销，最后才接 Grove。

已下载 AMI IS1009a 会议测试素材，见 [内网 GPU 部署准备](DEPLOY.md)。本机无 GPU，已预留设备与离线模型路径配置；尚未安装或下载识别模型，未跑真实视频识别。

## Grove 接入边界（设计，未实现）

在感知/音频描述进入记忆之前附加 `person_id`、`observation_id`、时间区间、来源与关联 revision；保留原始 speaker 标签。Grove 仍负责事件、片段、长期记忆；身份图独立维护并提供按人物查询事件的能力。

检查到 `Grove/hypervideo/mmlifelong_processing.py` 的 AudioContextProvider 可消费 speaker/time_span/text；但 `modules/audio_cues/__init__.py` 引用的 asr、diarization、audio_fact_injector 文件在当前 checkout 中缺失，不能假定其音频管线可直接运行。

Grove 的在线查询有时间因果约束：未来视频才确认的身份关联不能用于回填当时的在线评测。正式适配需为关联增加 observed_at/confirmed_at 和按时刻查询；当前 revision 仅用于实验追踪，不足以支持历史时间查询。
