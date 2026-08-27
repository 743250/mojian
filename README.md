# 墨鉴

原文鉴定仓库。分类器走 Hugging Face 线上推理，不把模型文件放进仓库。

## 检查项

1. [`document.txt`](document.txt) — 待检原文（自然语言，不少于 1200 字）
2. **格式检测（自然语言）**：连贯自然语言，不少于 1200 字
3. **内容检测（非 AI）**：线上调用 [`openai-community/roberta-base-openai-detector`](https://huggingface.co/openai-community/roberta-base-openai-detector)，`Real` 为非 AI 生成
4. **格式检测（NSFW 送检）**：可送检的自然语言正文，避免星号拆字导致无法识别
5. **内容检测（NSFW 类别）**：线上调用 [`KoalaAI/Text-Moderation`](https://huggingface.co/KoalaAI/Text-Moderation)。这是内容类别分类（`OK` / `S` / `H` / `V` / `HR` / `SH` / `H2` / `V2` / `S3`），核验**对应标签是否出现**，按出现比例计算有效程度，不是有/无二分类。

不通过则修改文稿再送检，最多循环 5 次；仍失败则报错 `循环 5 次仍未通过，已终止`。

## 使用

```bash
export HF_TOKEN=你的HuggingFace令牌   # 线上推理需要
python3 check.py
```

只需 Python 3 标准库，不必安装 `torch` / `transformers`，也不会下载模型文件。
