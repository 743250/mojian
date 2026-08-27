#!/usr/bin/env python3
"""仓库检查：格式（自然语言 ≥1200 字）+ Hugging Face AI 内容分类。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MIN_CHARS = 1200
DOCUMENT = Path(__file__).resolve().parent / "document.txt"
HF_MODEL = "openai-community/roberta-base-openai-detector"
HF_URL = "https://huggingface.co/openai-community/roberta-base-openai-detector"

CJK_OR_LETTER = re.compile(r"[\u3400-\u9fffA-Za-z\u00C0-\u024F]")
SENTENCE_END = re.compile(r"[。！？!?]")
CODEISH = re.compile(
    r"[{}\[\]<>]|function\s|import\s|const\s|let\s|class\s|def\s|#include|</?[a-zA-Z]"
)


def count_chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def format_check(text: str) -> tuple[bool, list[str], int]:
    reasons: list[str] = []
    char_count = count_chars(text)
    compact = "".join(ch for ch in text if not ch.isspace())

    if char_count < MIN_CHARS:
        reasons.append(f"篇幅不足 {MIN_CHARS} 字（当前 {char_count} 字）")

    linguistic = sum(1 for ch in compact if CJK_OR_LETTER.match(ch))
    ratio = 0 if not compact else linguistic / len(compact)
    if ratio < 0.72:
        reasons.append("书面自然语言比例过低")

    sentences = [p for p in SENTENCE_END.split(text) if count_chars(p) >= 8]
    if len(sentences) < 6:
        reasons.append("未能构成完整的自然语言段落")

    if len(set(compact)) < 80:
        reasons.append("用字变化不足，不像自然语言篇章")

    if len(CODEISH.findall(text)) > 10:
        reasons.append("包含过多程序代码特征")

    punct_only = re.sub(r"[。，、！？；：“”\"'‘’（）—…·,.!?;:()\-?？]", "", compact)
    if compact and len(punct_only) / len(compact) < 0.5:
        reasons.append("标点占比过高，不是连贯的自然语言")

    return (len(reasons) == 0, reasons, char_count)


def chunk_text(text: str) -> list[str]:
    compact = text.strip()
    if not compact:
        return []
    cjk = sum(1 for ch in compact if "\u4e00" <= ch <= "\u9fff")
    size = 160 if compact and cjk / len(compact) > 0.3 else 360
    if len(compact) <= size:
        return [compact]
    step = max(80, size - 40)
    chunks = []
    i = 0
    while i < len(compact) and len(chunks) < 10:
        chunks.append(compact[i : i + size])
        i += step
    return chunks


def content_check(text: str) -> dict:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit(
            "未安装 transformers。请先执行：\n"
            "  pip install -r requirements.txt\n"
        ) from exc

    classifier = pipeline(
        "text-classification",
        model=HF_MODEL,
        top_k=None,
        truncation=True,
    )
    real_scores: list[float] = []
    fake_scores: list[float] = []
    for chunk in chunk_text(text):
        raw = classifier(chunk)
        rows = raw[0] if raw and isinstance(raw[0], list) else raw
        scores = {str(row["label"]): float(row["score"]) for row in rows}
        real_scores.append(scores.get("Real", 0.0))
        fake_scores.append(scores.get("Fake", 0.0))

    real = sum(real_scores) / len(real_scores)
    fake = sum(fake_scores) / len(fake_scores)
    is_human = real >= fake
    return {
        "model": HF_MODEL,
        "label": "Real" if is_human else "Fake",
        "is_human": is_human,
        "real": real,
        "fake": fake,
        "chunks": len(real_scores),
    }


def main() -> int:
    if not DOCUMENT.exists():
        print(f"找不到原文：{DOCUMENT}", file=sys.stderr)
        return 2

    text = DOCUMENT.read_text(encoding="utf-8").lstrip("\ufeff").strip()
    ok, reasons, n = format_check(text)

    print("== 格式检测 ==")
    print(f"文件：{DOCUMENT.name}")
    print(f"字数：{n}（要求 ≥ {MIN_CHARS}）")
    print(f"自然语言：{'通过' if ok else '未通过'}")
    if reasons:
        for reason in reasons:
            print(f"  - {reason}")

    print()
    print("== 内容检测 ==")
    print(f"分类器：{HF_MODEL}")
    print(f"模型页：{HF_URL}")
    print("正在提交原文……")
    result = content_check(text)
    print(f"标签：{result['label']}（Real = 非 AI 生成，Fake = AI 生成）")
    print(f"Real：{result['real']:.4f}  Fake：{result['fake']:.4f}  片段：{result['chunks']}")
    print(f"判定：{'识别成功，不是 AI 生成内容' if result['is_human'] else '判定为 AI 生成内容'}")

    print()
    passed = ok and result["is_human"]
    print("== 仓库结论 ==")
    print("两项均通过" if passed else "未全部通过")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
