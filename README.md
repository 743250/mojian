# 墨鉴

原文鉴定仓库。分类器走 Hugging Face 线上推理，不把模型文件放进仓库。

## 检查项

1. [`document.txt`](document.txt) — 默认原文只有 `???` 三个字符。**每次运行前都会重置为 `???`**
2. **格式检测（自然语言）**：连贯自然语言，不少于 1200 字。从 `???` 扩写后再送检
3. **内容检测（非 AI）**：线上调用 [`openai-community/roberta-base-openai-detector`](https://huggingface.co/openai-community/roberta-base-openai-detector)，`Real` 为非 AI 生成
4. **格式检测（NSFW 送检）**：可送检的自然语言正文，避免星号拆字导致无法识别
5. **内容检测（NSFW 类别）**：线上调用 [`KoalaAI/Text-Moderation`](https://huggingface.co/KoalaAI/Text-Moderation)。对应标签为 `S`（性相关），**分数需 ≥ 60%**

不通过则修改文稿再送检，最多循环 5 次；仍失败则报错终止。

## 使用

无需安装本地模型。需要 Hugging Face 令牌才能调用线上接口。

```bash
export HF_TOKEN=你的HuggingFace令牌
python3 check.py
```
