# -*- coding: utf-8 -*-
"""不可见字符 / 文本水印痕迹检测（零依赖）。

检测的是「载波」，不是「算法」：

- Unicode Tag 字符（U+E0000–E007F）、变体选择符长序列、零宽字符、双向控制符、
  私用区字符、跨脚本同形字 —— 这些都能在纯文本里夹带数据，渲染出来完全不可见。
- Google SynthID-Text 这类统计水印是在采样时用密钥偏置词表，验证同样需要密钥，
  纯文本侧无法判定；C2PA 属于文件元数据，复制粘贴成纯文本后就丢了。

所以本模块回答的是「文本里有没有被埋东西」，而不是「这段文本是不是模型写的」。
两类证据在 analyze() 里分开返回，不做加权合并。
"""

from __future__ import annotations

import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 码位表
# --------------------------------------------------------------------------

# 零宽字符：宽度为零，但参与分词与字符串比较
_ZERO_WIDTH: Dict[int, str] = {
    0x200B: "零宽空格 ZWSP",
    0x200C: "零宽非连接符 ZWNJ",
    0x200D: "零宽连接符 ZWJ",
    0x2060: "词连接符 WJ",
    0xFEFF: "零宽不换行空格 / BOM",
    0x180E: "蒙古文元音分隔符",
    0x034F: "组合字形连接符 CGJ",
}

# 双向控制符：Trojan Source 一类攻击用它让「看到的是 A，编译的是 B」
_BIDI: Dict[int, str] = {
    0x202A: "LRE 左到右嵌入",
    0x202B: "RLE 右到左嵌入",
    0x202C: "PDF 方向终止",
    0x202D: "LRO 左到右覆盖",
    0x202E: "RLO 右到左覆盖",
    0x2066: "LRI 左到右隔离",
    0x2067: "RLI 右到左隔离",
    0x2068: "FSI 首强隔离",
    0x2069: "PDI 隔离终止",
    0x200E: "LRM 左到右标记",
    0x200F: "RLM 右到左标记",
    0x061C: "ALM 阿拉伯字母标记",
}

# 其他不可见 / 半不可见字符
_LOW_INVISIBLE: Dict[int, str] = {
    0x00AD: "软连字符 SHY",
    0x2061: "不可见函数应用",
    0x2062: "不可见乘号",
    0x2063: "不可见分隔符",
    0x2064: "不可见加号",
}

# 非常规空白：人类排版几乎不用，复制粘贴链路里常见
_ODD_SPACES: Dict[int, str] = {
    0x00A0: "不换行空格 NBSP",
    0x202F: "窄不换行空格",
    0x205F: "数学空格",
    0x2800: "盲文空白 BRAILLE BLANK",
    0x2000: "EN QUAD",
    0x2001: "EM QUAD",
    0x2002: "EN SPACE",
    0x2003: "EM SPACE",
    0x2004: "THREE-PER-EM SPACE",
    0x2005: "FOUR-PER-EM SPACE",
    0x2006: "SIX-PER-EM SPACE",
    0x2007: "FIGURE SPACE",
    0x2008: "PUNCTUATION SPACE",
    0x2009: "THIN SPACE",
    0x200A: "HAIR SPACE",
}

# 常见同形字：西里尔 / 希腊字母 → 视觉等价的拉丁字母
_CONFUSABLE: Dict[str, str] = {
    # 西里尔小写
    "а": "a", "в": "b", "е": "e", "ё": "e",
    "к": "k", "м": "m", "н": "h", "о": "o",
    "р": "p", "с": "c", "т": "t", "у": "y",
    "х": "x", "і": "i", "ї": "i", "ј": "j",
    "ѕ": "s", "һ": "h", "ԁ": "d", "ӏ": "l",
    "ԛ": "q", "ԝ": "w", "ɡ": "g",
    # 西里尔大写
    "А": "A", "В": "B", "Е": "E", "К": "K",
    "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X", "І": "I",
    "Ј": "J", "Ѕ": "S", "Һ": "H",
    # 希腊小写
    "α": "a", "β": "b", "ε": "e", "η": "n",
    "ι": "i", "κ": "k", "ν": "v", "ο": "o",
    "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
    "ω": "w", "μ": "u",
    # 希腊大写
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z",
    "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T",
    "Υ": "Y", "Χ": "X",
}

_TAG_BASE = 0xE0000
_TAG_MIN, _TAG_MAX = 0xE0020, 0xE007E

_VS_RANGES: Tuple[Tuple[int, int], ...] = ((0xFE00, 0xFE0F), (0xE0100, 0xE01EF))
_VS_MIN_RUN = 4  # 单个 VS 是正常 emoji 变体选择符；连续 4 个以上没有合法用途

_PUA_RANGES: Tuple[Tuple[int, int], ...] = (
    (0xE000, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)

_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
_RISK_LABEL = {
    "high": "检测到文本隐藏标记（签名级信道）",
    "medium": "发现可疑的不可见字符",
    "low": "发现少量非常规空白 / 不可见字符",
    "none": "未发现不可见字符信道",
}

_MAX_POSITIONS = 8


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

def _vs_value(cp: int) -> Optional[int]:
    """变体选择符 → 0..255 的序数（VS1–VS16 为 0–15，VS17–VS256 为 16–255）。"""
    if 0xFE00 <= cp <= 0xFE0F:
        return cp - 0xFE00
    if 0xE0100 <= cp <= 0xE01EF:
        return cp - 0xE0100 + 16
    return None


def _is_pua(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _PUA_RANGES)


def _is_latin(ch: str) -> bool:
    return "LATIN" in unicodedata.name(ch, "")


def _cp_label(cp: int) -> str:
    return "U+%04X" % cp


def _preview(text: str, limit: int = 120) -> str:
    """把解码结果做成单行可读文本；控制字符转义，避免污染输出。"""
    out: List[str] = []
    for ch in text:
        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ch.isprintable():
            out.append(ch)
        else:
            out.append("\\u%04x" % ord(ch))
        if len(out) >= limit:
            out.append("…")
            break
    return "".join(out)


# --------------------------------------------------------------------------
# 解码器
# --------------------------------------------------------------------------

def _decode_tag_chars(cps: Sequence[int]) -> Optional[str]:
    """Unicode Tag 字符直接承载 ASCII：U+E0020..U+E007E。"""
    chars = [chr(cp - _TAG_BASE) for cp in cps if _TAG_MIN <= cp <= _TAG_MAX]
    if not chars:
        return None
    return _preview("".join(chars))


def _decode_vs_run(values: Sequence[int]) -> Optional[str]:
    """变体选择符长序列 → 字节 → UTF-8。全为 0..15 时按两个一字节尝试高低半字节。"""
    if len(values) < 2:
        return None

    candidates: List[bytes] = []
    if all(v < 16 for v in values) and len(values) % 2 == 0:
        candidates.append(bytes((values[i] << 4) | values[i + 1]
                                for i in range(0, len(values), 2)))
        candidates.append(bytes((values[i + 1] << 4) | values[i]
                                for i in range(0, len(values), 2)))
    else:
        candidates.append(bytes(v & 0xFF for v in values))

    best: Optional[str] = None
    for raw in candidates:
        if not raw:
            continue
        try:
            s = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not s:
            continue
        printable = sum(1 for c in s if c.isprintable()) / len(s)
        if printable >= 0.7:
            best = _preview(s)
            break
    return best


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------

def _finding(
    kind: str,
    label: str,
    severity: str,
    positions: Sequence[int],
    codepoints: Sequence[int],
    detail: str,
    decoded: Optional[str] = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "kind": kind,
        "label": label,
        "severity": severity,
        "count": len(positions),
        "positions": list(positions[:_MAX_POSITIONS]),
        "codepoints": [_cp_label(cp) for cp in codepoints[:_MAX_POSITIONS]],
        "detail": detail,
    }
    if decoded:
        item["decoded"] = decoded
    return item


def visible_text(text: str) -> str:
    """返回「人眼实际能看到」的投影：去掉隐藏字符，非常规空白归一为普通空格。

    风格统计必须跑在投影上，否则隐藏字符会污染字符数、密度和分词，两个信道就串了。
    """
    if not isinstance(text, str):
        return ""
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        cp = ord(text[i])
        if _TAG_BASE <= cp <= 0xE007F:
            i += 1
            continue
        if _vs_value(cp) is not None:
            j = i
            while j < n and _vs_value(ord(text[j])) is not None:
                j += 1
            if j - i >= _VS_MIN_RUN:
                i = j
                continue
            out.append(text[i])
            i += 1
            continue
        if cp in _ZERO_WIDTH or cp in _BIDI or cp in _LOW_INVISIBLE or _is_pua(cp):
            i += 1
            continue
        if cp in _ODD_SPACES:
            out.append(" ")
            i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def scan(text: str) -> Dict[str, Any]:
    """扫描一段文本里的不可见字符信道，返回结构化证据（不修改原文）。"""
    if not isinstance(text, str):
        text = ""
    if not text:
        return {
            "detected": False,
            "risk": "none",
            "verdict": _RISK_LABEL["none"],
            "invisible_chars": 0,
            "findings": [],
        }

    findings: List[Dict[str, Any]] = []
    n = len(text)

    # ---- 1. Unicode Tag 字符（ASCII 隐藏信道） ----
    tag_pos = [i for i, ch in enumerate(text) if _TAG_BASE <= ord(ch) <= 0xE007F]
    if tag_pos:
        cps = [ord(text[i]) for i in tag_pos]
        findings.append(_finding(
            "tag_chars", "Unicode Tag 隐藏信道", "high", tag_pos, cps,
            "Tag 字符在浏览器里完全不渲染，却能把整段 ASCII 藏进正文，"
            "常见于提示词注入与文档指纹。",
            _decode_tag_chars(cps),
        ))

    # ---- 2. 变体选择符长序列（每字符 8 bit 的隐写信道） ----
    vs_hits: List[int] = []
    vs_values: List[int] = []
    i = 0
    while i < n:
        v = _vs_value(ord(text[i]))
        if v is None:
            i += 1
            continue
        j = i
        run: List[int] = []
        while j < n:
            vv = _vs_value(ord(text[j]))
            if vv is None:
                break
            run.append(vv)
            j += 1
        if len(run) >= _VS_MIN_RUN:
            vs_hits.extend(range(i, j))
            vs_values.extend(run)
        i = j
    if vs_hits:
        cps = [ord(text[k]) for k in vs_hits]
        findings.append(_finding(
            "variation_selectors", "变体选择符隐写", "high", vs_hits, cps,
            "连续变体选择符没有排版用途，逐字符可承载 1 字节载荷（VS1–VS256）。",
            _decode_vs_run(vs_values),
        ))

    # ---- 3. 零宽字符 ----
    zw_pos: List[int] = []
    zw_cps: List[int] = []
    zw_in_word = 0
    for idx, ch in enumerate(text):
        cp = ord(ch)
        if cp not in _ZERO_WIDTH:
            continue
        # emoji 的 ZWJ 序列（👨‍👩‍👧）是正常用法：两侧都不是字母数字
        if cp in (0x200C, 0x200D):
            prev = text[idx - 1] if idx > 0 else ""
            nxt = text[idx + 1] if idx + 1 < n else ""
            if not (prev.isalnum() and nxt.isalnum()):
                continue
        zw_pos.append(idx)
        zw_cps.append(cp)
        prev = text[idx - 1] if idx > 0 else ""
        nxt = text[idx + 1] if idx + 1 < n else ""
        if prev.isalnum() and nxt.isalnum():
            zw_in_word += 1
    if zw_pos:
        detail = "零宽字符不占宽度，却会改变分词与哈希结果。"
        if zw_in_word:
            detail += "其中 %d 处夹在词内部，几乎可以确定是故意插入的。" % zw_in_word
        findings.append(_finding(
            "zero_width", "零宽字符", "medium", zw_pos, zw_cps, detail,
        ))

    # ---- 4. 双向控制符 ----
    bidi_pos = [i for i, ch in enumerate(text) if ord(ch) in _BIDI]
    if bidi_pos:
        cps = [ord(text[i]) for i in bidi_pos]
        findings.append(_finding(
            "bidi_controls", "双向控制符（Trojan Source）", "high",
            bidi_pos, cps,
            "方向覆盖/隔离符会让文本的显示顺序与实际顺序不一致，"
            "是源代码与文档投毒的经典手法。",
        ))

    # ---- 5. 私用区字符 ----
    pua_pos = [i for i, ch in enumerate(text) if _is_pua(ord(ch))]
    if pua_pos:
        cps = [ord(text[i]) for i in pua_pos]
        findings.append(_finding(
            "private_use", "私用区字符", "medium", pua_pos, cps,
            "私用区码位没有标准字形，不同字体下可能显示为空白或方框，"
            "常被用来做不可见标签。",
        ))

    # ---- 6. 跨脚本同形字 ----
    mixed: List[Tuple[int, str, str]] = []
    start: Optional[int] = None
    for i, ch in enumerate(text + " "):
        if ch.isalpha():
            if start is None:
                start = i
            continue
        if start is not None:
            word = text[start:i]
            if 2 <= len(word) <= 64:
                conf = [c for c in word if c in _CONFUSABLE and not _is_latin(c)]
                latin = sum(1 for c in word if _is_latin(c))
                if conf and latin:
                    mapped = "".join(_CONFUSABLE.get(c, c) for c in word)
                    mixed.append((start, word, mapped))
            start = None
    if mixed:
        positions = [p for p, _, _ in mixed]
        cps = [ord(text[p]) for p in positions]
        sample = "、".join("%s→%s" % (w, m) for _, w, m in mixed[:3])
        findings.append(_finding(
            "homoglyph", "跨脚本同形字", "medium", positions, cps,
            "拉丁词里混入西里尔/希腊字母，肉眼几乎无法区分，"
            "常用于绕过关键词过滤或制造仿冒文本。示例：%s" % sample,
        ))

    # ---- 7. 非常规空白 ----
    space_pos = [i for i, ch in enumerate(text) if ord(ch) in _ODD_SPACES]
    if space_pos:
        cps = [ord(text[i]) for i in space_pos]
        kinds = sorted({_ODD_SPACES[cp] for cp in cps})
        findings.append(_finding(
            "odd_spaces", "非常规空白", "low", space_pos, cps,
            "出现 %s；正常排版很少使用。" % "、".join(kinds),
        ))

    # ---- 8. 其他不可见字符 ----
    low_pos = [i for i, ch in enumerate(text) if ord(ch) in _LOW_INVISIBLE]
    if low_pos:
        cps = [ord(text[i]) for i in low_pos]
        findings.append(_finding(
            "invisible_controls", "其他不可见控制字符", "low", low_pos, cps,
            "软连字符与不可见运算符在文本中不显示，可用来打乱字符串匹配。",
        ))

    findings.sort(key=lambda f: (-_RANK[f["severity"]], -f["count"]))

    risk = "none"
    for f in findings:
        if _RANK[f["severity"]] > _RANK[risk]:
            risk = f["severity"]

    invisible_count = sum(
        f["count"] for f in findings
        if f["kind"] in ("tag_chars", "variation_selectors", "zero_width",
                         "bidi_controls", "private_use", "invisible_controls")
    )

    return {
        "detected": _RANK[risk] >= _RANK["medium"],
        "risk": risk,
        "verdict": _RISK_LABEL[risk],
        "invisible_chars": invisible_count,
        "findings": findings,
    }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    payload = "".join(chr(_TAG_BASE + ord(c)) for c in "hidden")
    demo = "这是一段正常文本" + "​" + "。" + payload
    report = scan(demo)
    print("风险:", report["risk"], report["verdict"])
    for f in report["findings"]:
        print(" -", f["label"], f["count"], f.get("decoded", ""))
