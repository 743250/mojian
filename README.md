# 墨鉴

原文鉴定仓库。包含一篇自然语言文稿，以及两项自动检查。

## 仓库内容

1. [`document.txt`](document.txt) — 待检原文（自然语言，不少于 1200 字）
2. Hugging Face AI 内容分类器 [`openai-community/roberta-base-openai-detector`](https://huggingface.co/openai-community/roberta-base-openai-detector)
3. **内容检测**：把原文提交给分类器，`Real` 表示识别为不是 AI 生成内容
4. **格式检测**：必须是连贯自然语言，且不少于 1200 字

## 使用

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python check.py
```

`check.py` 会先做格式检测，再把 `document.txt` 分段提交给上述分类器，并打印是否判定为非 AI 生成。
