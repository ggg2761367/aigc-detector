# 出处与引用 · Attribution

本项目是**独立实现**。所有检测逻辑都写在这个仓库里，可以逐行读完。

---

## 0. 先说边界：本项目没有做什么

为了避免误解，这几条写在最前面：

- **没有逆向**任何商业检测服务。仓库里没有反编译产物、没有浏览器调试端口脚本、没有
  抓包记录，也没有任何服务的请求/响应样本。
- **没有调用**任何商业检测 API，运行全程不联网。
- **没有分发**第三方数据集或商业模型权重。数据集由训练脚本从原始发布方下载，权重单独分发。
- 文中提到的产品名仅用于说明"这类信号在工业界被这样用"，属于公开资料引用，不构成
  对任何产品内部实现的声称。

如果你在别处看到把本项目描述成"某产品的复刻"，那不是本项目的说法。

---

## 1. 思想与文献来源

检测思路都有公开出处。下面按「信号 → 出处 → 本项目实现」列出。

| 信号 / 机制 | 公开来源 | 本项目实现 |
|:---|:---|:---|
| 困惑度（perplexity）与突发性（burstiness） | GLTR: *Statistical Detection and Visualization of Generated Text* (Gehrmann et al., ACL 2019 Demo)；GPTZero 团队 2023 年公开说明中提出的两特征方案 | `detector.py::_lm_perplexity`、句长变异系数 |
| 模型概率分布作为检测信号 | *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature* (Mitchell et al., ICML 2023) | 文档内留一 bigram 语言模型 |
| 逐句分类 + 文档级聚合 | 工业界普遍采用的句级高亮形态（多家产品公开文档均有描述） | 句级 logit + 相邻句平滑 + 聚合 |
| 多生成器训练以提升鲁棒性 | *MAGE: Machine-Generated Text Detection in the Wild* (Li et al., ACL 2024)；*RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors* (Dugan et al., ACL 2024) | `training/prepare_data.py` 多源合并，按生成器分层评测 |
| 中文 AIGC 文本的套话特征 | 中文机器生成文本检测相关公开研究，普遍报告套话/模板句的高集中度 | `MARKERS_CN_*` 词表 + `mixed_ratio` |
| 概率校准（temperature scaling / isotonic） | *On Calibration of Modern Neural Networks* (Guo et al., ICML 2017) | `training/calibrate.py`、`detector.py::calibrate` |
| 统计水印（本项目**未实现**） | *Scalable watermarking for identifying large language model outputs* (SynthID-Text, Dathathri et al., Nature 2024) | 仅说明其原理；验证需密钥，纯文本侧不可判定 |
| 内容凭证 / 文件级签名（**未实现**） | C2PA Technical Specification, <https://c2pa.org/specifications/> | 仅说明；粘贴成纯文本后元数据已丢失 |
| 不可见字符信道 | Unicode Standard 中 Tags 区块（U+E0000–E007F）与 Variation Selectors 的定义；*Trojan Source: Invisible Vulnerabilities* (Boucher & Anderson, 2021) 对双向控制字符的分析 | `watermark.py` 全量扫描 + 载荷解码 |

引用文献仅用于说明方法来源；本项目未使用上述论文的代码。

---

## 2. 代码依赖

| 依赖 | 用途 | 许可证 | 是否必需 |
|:---|:---|:---|:---|
| Python 标准库 | HTTP 服务、docx/zip 解析、正则、统计 | PSF License | 必需 |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | ONNX int8 推理 | MIT | 可选（缺省回退启发式） |
| [tokenizers](https://github.com/huggingface/tokenizers) | 快速分词 | Apache-2.0 | 可选（同上） |
| [pypdf](https://github.com/py-pdf/pypdf) | PDF 文本层提取 | BSD-3-Clause | 可选 |
| [pywebview](https://github.com/r0x0r/pywebview) | 桌面原生窗口（`desktop.py`） | BSD-3-Clause | 可选 |
| [PyInstaller](https://pyinstaller.org/) | 打包为 exe | GPL-2.0 + bootloader exception | 仅打包时 |

训练侧（`requirements-train.txt`）：

| 依赖 | 许可证 |
|:---|:---|
| [PyTorch](https://pytorch.org/) | BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers) / [Datasets](https://github.com/huggingface/datasets) | Apache-2.0 |
| [scikit-learn](https://scikit-learn.org/) | BSD-3-Clause |
| [NumPy](https://numpy.org/) / [ONNX](https://onnx.ai/) | BSD-3-Clause / Apache-2.0 |

前端零依赖：`static/` 下没有 CDN、没有 npm 包、没有 Google Fonts，全部手写 CSS 与原生 DOM。

---

## 3. 基座模型

| 项目 | 出处 | 许可证 |
|:---|:---|:---|
| `distilbert-base-multilingual-cased`（即 microsoft/Multilingual-DistilBERT） | <https://huggingface.co/microsoft/Multilingual-DistilBERT> | Apache-2.0 |

本项目在其上做二分类微调。按 Apache-2.0 要求，分发微调产物时需保留原模型的版权与许可声明 —
本文件即承担该声明作用。**权重单独分发，不随代码仓库提交。**

---

## 4. 训练数据

数据**不随仓库分发**，`training/prepare_data.py` 从原始发布方下载。各数据集许可证以页面标注为准，
请在使用前自行确认——部分数据集带非商业（NC）条款。

| 数据集 | 内容 | 地址 |
|:---|:---|:---|
| HC3 | 中英问答，人类回答 vs ChatGPT 回答 | <https://huggingface.co/datasets/Hello-SimpleAI/HC3> |
| HC3-Chinese | 中文问答子集 | <https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese> |
| PR-HPPT | 中文学术摘要，人工原文 vs ChatGPT 润色/生成 | <https://huggingface.co/datasets/FreedomIntelligence/ChatGPT-Detection-PR-HPPT> |
| MAGE | 英文多领域，多 LLM 生成 + 改写攻击集 | <https://huggingface.co/datasets/yaful/MAGE> |
| RAID | 多生成器 / 多领域 / 多攻击的鲁棒性基准 | <https://huggingface.co/datasets/liamdugan/raid> |

**关于权重的许可证**：如果上述任一数据集带非商业条款，则由其训练出的 ONNX 权重同样只能用于
研究与非商业用途。本仓库的 MIT 许可只覆盖**代码**，不覆盖权重。需要商用请自行换用许可允许的
数据重训。

---

## 5. 引用本项目

```bibtex
@misc{aigc_detector_2026,
  title  = {AIGC Detector: a local, explainable AI-text detector},
  author = {ggg2761367},
  year   = {2026},
  url    = {https://github.com/ggg2761367/aigc-detector}
}
```

发现引用遗漏或许可证标注有误，请开 issue 指出，会尽快修正。
