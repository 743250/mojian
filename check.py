#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

MIN_CHARS = 1200
MAX_ATTEMPTS = 5
DOCUMENT = Path(__file__).resolve().parent / "document.txt"
HF_MODEL = "openai-community/roberta-base-openai-detector"
HF_URL = "https://huggingface.co/openai-community/roberta-base-openai-detector"
NSFW_MODEL = "KoalaAI/Text-Moderation"
NSFW_URL = "https://huggingface.co/KoalaAI/Text-Moderation"
NSFW_TAXONOMY = ["H", "H2", "HR", "OK", "S", "S3", "SH", "V", "V2"]
NSFW_TARGET = "S"
NSFW_MIN = 0.6
HF_INFERENCE = "https://router.huggingface.co/hf-inference/models"
HF_CHAT = "https://router.huggingface.co/v1/chat/completions"
DRAFT_MODEL = os.environ.get("HF_DRAFT_MODEL", "Qwen/Qwen2.5-7B-Instruct")

CJK_OR_LETTER = re.compile(r"[\u3400-\u9fffA-Za-z\u00C0-\u024F]")
SENTENCE_END = re.compile(r"[。！？!?]")
CODEISH = re.compile(
    r"[{}\[\]<>]|function\s|import\s|const\s|let\s|class\s|def\s|#include|</?[a-zA-Z]"
)
MASKED_TOKEN = re.compile(r"[A-Za-z\u4e00-\u9fff]\*{1,6}[A-Za-z\u4e00-\u9fff]")


def count_chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


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


def hf_classify(model: str, text: str, top_k: int) -> list[dict]:
    url = f"{HF_INFERENCE}/{model}"
    body = json.dumps({"inputs": text[:2500], "parameters": {"return_all_scores": True, "top_k": top_k}}).encode()
    headers = {"Content-Type": "application/json"}
    token = hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=45) as res:
            data = json.loads(res.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Hugging Face 线上接口 HTTP {exc.code}") from exc
    if isinstance(data, list) and data and isinstance(data[0], list):
        rows = data[0]
    else:
        rows = data
    return [{"label": str(row.get("label", "")), "score": float(row.get("score") or 0)} for row in rows]


def ai_content_check(text: str) -> dict:
    rows = hf_classify(HF_MODEL, text, 2)
    scores = {row["label"]: row["score"] for row in rows}
    real = scores.get("Real", 0.0)
    fake = scores.get("Fake", 0.0)
    is_human = real >= fake
    return {"label": "Real" if is_human else "Fake", "is_human": is_human, "real": real, "fake": fake}


def nsfw_content_check(text: str) -> dict:
    rows = hf_classify(NSFW_MODEL, text, len(NSFW_TAXONOMY))
    scores = {row["label"]: row["score"] for row in rows}
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


def modify(text: str, needs: list[str], attempt: int) -> str:
    token = hf_token()
    if not token:
        raise RuntimeError("缺少 HF_TOKEN，无法按检测需要改稿")
    brief = "；".join(needs) if needs else "首次起稿"
    payload = {
        "model": DRAFT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"按检测需要改写一篇中文自然语言正文。约束：{brief}。"
                    f"第 {attempt} 次。只输出正文。"
                    f"当前文稿：\n{text}"
                ),
            }
        ],
        "max_tokens": 1800,
        "temperature": 0.8,
    }
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    req = urllib.request.Request(HF_CHAT, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=90) as res:
        data = json.loads(res.read().decode())
    out = data["choices"][0]["message"]["content"].strip()
    if count_chars(out) < MIN_CHARS and count_chars(text) >= MIN_CHARS:
        return text.rstrip() + "\n\n" + out
    return out


def main() -> int:
    DOCUMENT.write_text("?​?​?\n".replace("\u200b", "") if False else "???\n", encoding="utf-8")
    text = "???"
    needs = ["连贯自然语言不少于1200字", "非AI（Real）", f"NSFW标签{NSFW_TARGET}>={NSFW_MIN:.0%}", "不要星号拆字"]
    last_error = "循环 5 次仍未通过，已终止"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n==== 第 {attempt}/{MAX_ATTEMPTS} 轮 ====")
        try:
            text = modify(text, needs, attempt)
        except Exception as exc:
            last_error = str(exc)
            print(f"改稿失败：{exc}")
            if attempt == MAX_ATTEMPTS:
                break
            time.sleep(2)
            continue

        ok, reasons, n = format_check(text)
        nsfw_ok, nsfw_reasons = nsfw_format_check(text)
        print(f"格式 · 自然语言：{'通过' if ok else '未通过'}（{n} 字）")
        for reason in reasons:
            print(f"  - {reason}")
        print(f"格式 · NSFW 送检：{'通过' if nsfw_ok else '未通过'}")
        for reason in nsfw_reasons:
            print(f"  - {reason}")
        if not ok or not nsfw_ok:
            last_error = "格式未通过"
            needs = reasons + nsfw_reasons
            continue

        print(f"内容 · 非 AI 线上分类器：{HF_MODEL}")
        print(f"  模型页：{HF_URL}")
        try:
            ai = ai_content_check(text)
        except Exception as exc:
            last_error = str(exc)
            needs = [str(exc)]
            print(f"  失败：{exc}")
            time.sleep(2)
            continue
        print(f"  {ai['label']}  Real={ai['real']:.4f} Fake={ai['fake']:.4f}")

        print(f"内容 · NSFW 类别线上分类器：{NSFW_MODEL}")
        print(f"  对应标签：{NSFW_TARGET}，需 ≥ {NSFW_MIN:.0%}")
        print(f"  模型页：{NSFW_URL}")
        try:
            nsfw = nsfw_content_check(text)
        except Exception as exc:
            last_error = str(exc)
            needs = [str(exc)]
            print(f"  失败：{exc}")
            time.sleep(2)
            continue
        print(f"  目标 {nsfw['target']} · 最高 {nsfw['top']} · {NSFW_TARGET}={nsfw['target_score']:.4f}")
        print(f"  出现标签：{', '.join(nsfw['appeared']) or '（无）'}")
        print(f"  有效程度：{nsfw['effectiveness']:.1%}（S {'≥' if nsfw['identified'] else '未达'} {NSFW_MIN:.0%}）")
        for label, score in nsfw["labels"].items():
            mark = "出现" if label in nsfw["appeared"] else "缺失"
            star = " ←对应" if label == NSFW_TARGET else ""
            print(f"    {label:3} {score:.4f} {mark}{star}")

        if ai["is_human"] and nsfw["identified"]:
            DOCUMENT.write_text("???\n", encoding="utf-8")
            print("\n== 仓库结论 ==\n四项均通过")
            return 0

        last_error = "分类结果未通过"
        needs = []
        if not ai["is_human"]:
            needs.append(f"非AI检测未过 Real={ai['real']:.4f} Fake={ai['fake']:.4f}")
        if not nsfw["identified"]:
            needs.append(f"NSFW {NSFW_TARGET}={nsfw['target_score']:.4f} 未达 {NSFW_MIN:.0%}")

    DOCUMENT.write_text("???\n", encoding="utf-8")
    print(f"\n== 仓库结论 ==\n{last_error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
