# 内网 GPU 部署准备

当前可部署的是身份图实验和视频样本。GPU 提取/音画关联适配器尚未实现，不能将本目录描述为已可端到端识别的视频服务。无需在 Mac 安装 CUDA 或下载大模型。

## 已准备内容

两个派生 MP4 已随 Git 仓库提供，克隆后可直接使用；原始 AVI/WAV 不入库。

- AMI IS1009a 原始中央摄像机画面与 Mix-Headset 混合音轨。
- 从原始会议 60–120 秒和 180–240 秒生成的两段 H.264/AAC MP4。
- manifest.json：来源、授权说明、时间偏移、文件 SHA256 和流信息。
- configs/gpu.example.json：GPU 设备、本地权重位置、输入文件、待实现适配器。
- preflight.py：仅检查文件、依赖和 CUDA，不联网、不自动下载模型。当前会报告 inference_ready=false 并返回 1，避免把预留配置误当作已完成推理管线。

## 传到服务器后可立即验证

复制整个 person-identity 目录即可；图核心没有第三方 Python 依赖。媒体准备和检查需安装 FFmpeg/FFprobe；建议 Python 3.11。

```sh
python3 -m unittest discover -s tests -v
python3 identity.py examples/synthetic.json --out runs/server-smoke
python3 preflight.py --config configs/gpu.example.json
```

`runs/server-smoke` 需为新目录。第一次命令检查图逻辑，第二次回放合成数据，第三次列出 GPU 接入缺项；均不执行真实人物识别。

## GPU 接入约定

人脸模型目录：`models/buffalo_l/*.onnx`；声纹检查点：`models/pretrained_eres2netv2.ckpt`。使用和 m3-agent 一致的模型作为首个对照方案，但不用其整体 setup.sh（含 sudo、多个 torch 安装以及训练依赖）。

实际部署前需按服务器的 Linux 架构、Python、NVIDIA 驱动和 CUDA 组合确定 torch/torchaudio/onnxruntime-gpu 版本。在匹配服务器的联网 Linux 环境构建 wheelhouse、下载权重并记录哈希，再整体搬入内网；Mac 下载的二进制 wheel 不能直接当作 Linux GPU 安装包。本次未制作未经验证的依赖锁定文件或声称已打包权重。

后续实现顺序：

1. 人脸提取器：延迟加载模型、显式 device/model_dir、输出标准 observation。
2. 声纹提取器：移除导入时 CUDA 初始化；分离语音分段与 embedding；先支持人工提供的说话区间，再接离线说话人分离。
3. 音画关联：本地 VLM 或主动说话人检测适配器，将结果写为 candidate；保留画外音/不确定，不把时间重叠直接当成身份确认。
4. 同人物不同片段重识别与人工核对，评估后再设置自动确认策略。

## 样本来源和局限

来源：AMI Meeting Corpus / AMI Consortium，CC BY 4.0。

- https://groups.inf.ed.ac.uk/ami/
- https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml
- https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IS1009a/

本次修改：中央摄像机视频与混合音轨合并、截取两段、转码为 MP4。原始记录和派生文件关系在 manifest.json 中。

官方给出的 IS1009a 对照：Closeup1 / Headset-0 / A，Closeup2 / Headset-1 / B，Closeup3 / Headset-2 / C，Closeup4 / Headset-3 / D。中央摄像机画面中的人物位置仍需人工映射，不可直接把左右位置视为官方身份标签。

这是同一会议的低分辨率片段，适合流程冒烟测试；未人工标注说话区间，未人工确认音画同步，也不能代表跨天/跨设备识别性能。需要更可靠的人脸评估时，再下载 Closeup*_orig.avi 和转录标注。
