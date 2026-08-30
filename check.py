#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

MIN_CHARS = 1200
ROOT = Path(__file__).resolve().parent
DOCUMENT = ROOT / "document.txt"
MODELS = ROOT / "models"
AI_DIR = MODELS / "ai-detector"
NSFW_DIR = MODELS / "nsfw"
AI_REPO = "openai-community/roberta-base-openai-detector"
NSFW_REPO = "KoalaAI/Text-Moderation"
NSFW_TAXONOMY = ["H", "H2", "HR", "OK", "S", "S3", "SH", "V", "V2"]
NSFW_TARGET = "S"
NSFW_MIN = 0.6

CJK_OR_LETTER = re.compile(r"[\u3400-\u9fffA-Za-z\u00C0-\u024F]")
SENTENCE_END = re.compile(r"[。！？!?]")
CODEISH = re.compile(
    r"[{}\[\]<>]|function\s|import\s|const\s|let\s|class\s|def\s|#include|</?[a-zA-Z]"
)
MASKED_TOKEN = re.compile(r"[A-Za-z\u4e00-\u9fff]\*{1,6}[A-Za-z\u4e00-\u9fff]")

_ai = None
_nsfw = None


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


def nsfw_format_check(text: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok, base_reasons, _ = format_check(text)
    if not ok:
        reasons.append("NSFW 送检格式要求先满足自然语言全文（不少于 1200 字）")
        reasons.extend(base_reasons)
    if len(MASKED_TOKEN.findall(text)) >= 3:
        reasons.append("星号拆字过多，分类器无法正确识别")
    return (len(reasons) == 0, reasons)


def _download(repo_id: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "config.json").exists():
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=str(dest))


def ensure_models() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    _download(AI_REPO, AI_DIR)
    _download(NSFW_REPO, NSFW_DIR)


def _load(dir_path: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(dir_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(dir_path), local_files_only=True)
    model.eval()
    return tok, model


def _ai_bundle():
    global _ai
    if _ai is None:
        _ai = _load(AI_DIR)
    return _ai


def _nsfw_bundle():
    global _nsfw
    if _nsfw is None:
        _nsfw = _load(NSFW_DIR)
    return _nsfw


def _forward(tok, model, text: str, multi_label: bool) -> dict[str, float]:
    import torch

    inputs = tok(text[:2500], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.sigmoid(logits) if multi_label else torch.softmax(logits, dim=-1)
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}


def ai_content_check(text: str) -> dict:
    tok, model = _ai_bundle()
    scores = _forward(tok, model, text, multi_label=False)
    real = scores.get("Real", 0.0)
    fake = scores.get("Fake", 0.0)
    is_human = real >= fake
    return {"label": "Real" if is_human else "Fake", "is_human": is_human, "real": real, "fake": fake}


def nsfw_content_check(text: str) -> dict:
    tok, model = _nsfw_bundle()
    scores = _forward(tok, model, text, multi_label=True)
    appeared = [label for label in NSFW_TAXONOMY if label in scores]
    top = max(NSFW_TAXONOMY, key=lambda label: scores.get(label, 0.0))
    identified = scores.get(NSFW_TARGET, 0.0) >= NSFW_MIN
    return {
        "labels": {label: scores.get(label, 0.0) for label in NSFW_TAXONOMY},
        "appeared": appeared,
        "target": NSFW_TARGET,
        "top": top,
        "target_score": scores.get(NSFW_TARGET, 0.0),
        "effectiveness": scores.get(NSFW_TARGET, 0.0),
        "identified": identified,
    }


def inspect(text: str) -> int:
    ok, reasons, n = format_check(text)
    nsfw_ok, nsfw_reasons = nsfw_format_check(text)
    print(f"格式 · 自然语言：{'通过' if ok else '未通过'}（{n} 字）")
    for reason in reasons:
        print(f"  - {reason}")
    print(f"格式 · NSFW 送检：{'通过' if nsfw_ok else '未通过'}")
    for reason in nsfw_reasons:
        print(f"  - {reason}")

    print(f"内容 · 非 AI 本地分类器：{AI_DIR.name}")
    ai = ai_content_check(text)
    print(f"  {ai['label']}  Real={ai['real']:.4f} Fake={ai['fake']:.4f}")

    print(f"内容 · NSFW 本地分类器：{NSFW_DIR.name}")
    print(f"  对应标签：{NSFW_TARGET}，需 ≥ {NSFW_MIN:.0%}")
    nsfw = nsfw_content_check(text)
    print(f"  目标 {nsfw['target']} · 最高 {nsfw['top']} · {NSFW_TARGET}={nsfw['target_score']:.4f}")
    print(f"  出现标签：{', '.join(nsfw['appeared']) or '（无）'}")
    print(f"  有效程度：{nsfw['effectiveness']:.1%}（S {'≥' if nsfw['identified'] else '未达'} {NSFW_MIN:.0%}）")
    for label, score in nsfw["labels"].items():
        mark = "出现" if label in nsfw["appeared"] else "缺失"
        star = " ←对应" if label == NSFW_TARGET else ""
        print(f"    {label:3} {score:.4f} {mark}{star}")

    if ok and nsfw_ok and ai["is_human"] and nsfw["identified"]:
        print("\n== 仓库结论 ==\n四项均通过")
        return 0
    print("\n== 仓库结论 ==\n检测完成，未全部通过")
    return 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        DOCUMENT.write_text("???\n", encoding="utf-8")
        print("document.txt 已重置为 ???")
        return 0
    if not DOCUMENT.exists():
        DOCUMENT.write_text("???\n", encoding="utf-8")
    text = DOCUMENT.read_text(encoding="utf-8").lstrip("\ufeff").strip()
    ensure_models()
    return inspect(text)


if __name__ == "__main__":
    raise SystemExit(main())
