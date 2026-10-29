# -*- coding: utf-8 -*-
"""AIGC 文本检测引擎 —— 可解释的启发式评分（零第三方依赖）。

信号来源
--------
下面三类思路都来自公开的论文、技术博客与产品公开说明，本项目按自己的方式重新实现，
没有使用任何商业服务的接口、权重或内部数据：

困惑度 / 突发性     : 用参考语言模型衡量"这句话有多好猜"，再看句长方差
                      -> _lm_perplexity / 句长变异系数
逐句分类 + 文档聚合 : 每句给一个概率，再聚合成整篇分数，便于定位可疑段落
                      -> 句级 logit + 相邻句平滑 + 文档聚合
中文套话词表 + 占比 : 中文 AI 输出套话集中度高，词表法在该语种上性价比高
                      -> MARKERS_CN + mixed_ratio

声明：本文件的手工权重（SENT_WEIGHTS / analyze）是按上述信号方向标定的，不是训练出来的；
商业服务的权重来自大规模标注语料上的分类器，两者不可比。检测结果只作参考，不构成任何
学术、招聘或法律结论的依据。完整出处见 ATTRIBUTION.md。
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from watermark import scan as scan_watermark, visible_text

# --------------------------------------------------------------------------
# 1. 词表：AI 文本的高频"指纹"
# --------------------------------------------------------------------------

MARKERS_CN_STRONG = [
    "综上所述", "总而言之", "总的来说", "值得注意的是", "需要指出的是",
    "由此可见", "不难发现", "扮演着重要角色", "具有重要意义", "深入探讨",
    "深入分析", "赋能", "闭环", "抓手", "至关重要", "不可或缺",
    "蓬勃发展", "与此同时", "在此基础上", "在当今社会", "从某种意义上说",
    "在一定程度上", "多角度", "全方位", "进一步提升", "有效提升",
]

MARKERS_CN_WEAK = [
    "首先", "其次", "再次", "此外", "另外", "然而", "因此", "维度", "层面",
    "日益", "进一步", "极大", "显著", "充分", "深刻", "广泛", "持续",
    "越来越", "一方面", "另一方面", "不仅仅", "既要", "又要", "从而",
]

MARKERS_EN = [
    "delve", "tapestry", "landscape", "realm", "crucial", "pivotal", "intricate",
    "seamlessly", "moreover", "furthermore", "additionally", "in conclusion",
    "it is important to note", "plays a vital role", "in today's", "a testament to",
    "navigate the complexities", "underscore", "foster", "leverage", "robust",
    "comprehensive", "multifaceted", "paramount", "myriad", "nuanced", "holistic",
    "significantly", "substantial", "in essence", "ultimately", "consequently",
    "notably", "underscores", "fostering", "leveraging", "it is worth noting",
    "in the realm of", "when it comes to",
]

_EN_MARKER_RE = re.compile(
    r"\b(" + "|".join(re.escape(m).replace(r"\ ", r"\s+") for m in MARKERS_EN) + r")\b",
    re.IGNORECASE,
)

# 句首型话语连接词（AI 偏爱，人类也用，故只做弱信号）
DISCOURSE_OPENERS_CN = [
    "首先", "其次", "再次", "最后", "此外", "另外", "然而", "因此", "所以",
    "总的来说", "综上所述", "总而言之", "与此同时", "不仅如此", "值得注意的是",
    "需要指出的是", "由此可见", "换言之", "也就是说", "在此基础上",
]
DISCOURSE_OPENERS_EN = [
    "however", "moreover", "furthermore", "additionally", "therefore",
    "consequently", "in conclusion", "firstly", "secondly", "finally",
    "in addition", "overall", "notably", "importantly", "ultimately",
]

# --------------------------------------------------------------------------
# 2. 分句
# --------------------------------------------------------------------------

_SENT_END = "。！？!?；;…"
_CLOSERS = '"”’）)]】」』'
_ABBREV = ("mr.", "mrs.", "ms.", "dr.", "prof.", "e.g.", "i.e.", "etc.",
           "vs.", "fig.", "al.", "no.", "approx.", "st.", "jr.", "sr.")


def split_sentences(text: str) -> List[str]:
    """中英混排分句。换行视为硬边界；跳过英文缩写与小数点。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out: List[str] = []

    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue

        n = len(para)
        start = 0
        i = 0
        while i < n:
            ch = para[i]
            cut_at = -1

            if ch in _SENT_END:
                j = i + 1
                while j < n and (para[j] in _SENT_END or para[j] in _CLOSERS):
                    j += 1
                cut_at = j
            elif ch == ".":
                nxt = para[i + 1] if i + 1 < n else ""
                prev_win = para[max(0, i - 8):i + 1].lower()
                is_abbrev = prev_win.endswith(_ABBREV)
                is_decimal = i > 0 and para[i - 1].isdigit() and nxt.isdigit()
                is_insentence = bool(nxt) and not (nxt.isspace() or nxt in _CLOSERS)
                if not (is_abbrev or is_decimal or is_insentence):
                    cut_at = i + 1

            if cut_at > 0:
                seg = para[start:cut_at].strip()
                if seg:
                    out.append(seg)
                start = cut_at
                i = cut_at
            else:
                i += 1

        tail = para[start:].strip()
        if tail:
            out.append(tail)

    return out


# --------------------------------------------------------------------------
# 3. 分词（中文按字，英文按词）
# --------------------------------------------------------------------------

_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    word: List[str] = []
    for ch in text:
        if _CJK_RE.match(ch):
            if word:
                tokens.append("".join(word))
                word = []
            tokens.append(ch)
        elif ch.isalnum():
            word.append(ch.lower())
        else:
            if word:
                tokens.append("".join(word))
                word = []
    if word:
        tokens.append("".join(word))
    return tokens


# --------------------------------------------------------------------------
# 4. 数学 / 统计小工具
# --------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 60.0:
        return 1.0
    if x <= -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


def _cv(xs: Sequence[float]) -> float:
    """变异系数 = 标准差 / 均值。AI 文本句长更均匀 -> 更小。"""
    m = _mean(xs)
    if m <= 1e-9:
        return 0.0
    return _stdev(xs) / m


def _zscore(vals: Sequence[float]) -> List[float]:
    n = len(vals)
    if n < 2:
        return [0.0] * n
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / n)
    if sd < 1e-9:
        return [0.0] * n
    return [(v - m) / sd for v in vals]


def _ngrams(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    c: Dict[Tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i:i + n])
        c[g] = c.get(g, 0) + 1
    return c


# --------------------------------------------------------------------------
# 5. 困惑度：文档内自适应 n-gram 语言模型（留一法）
# --------------------------------------------------------------------------
# 商用检测器用预训练 LM 算 PPL；这里没有语料，改成"用文档自身建模型"，再用留一法
# 扣掉句子自己的贡献 —— 衡量的是"这句话相对于全文还剩多少新信息"。
# AI 文本信息密度低、自我重复高 -> 留一困惑度偏低。

def _lm_perplexity(
    sent_tokens: Sequence[str],
    doc_uni: Dict[str, int],
    doc_bi: Dict[Tuple[str, str], int],
    vocab_size: int,
    doc_len: int,
    k_bi: float = 0.5,
    k_uni: float = 0.05,
) -> Optional[float]:
    n = len(sent_tokens)
    if n == 0:
        return None

    own_uni: Dict[str, int] = {}
    for t in sent_tokens:
        own_uni[t] = own_uni.get(t, 0) + 1
    own_bi = _ngrams(sent_tokens, 2)

    n_adj = doc_len - n
    if n_adj < 5:
        return None  # 文档太短，留一法退化，由句内压缩率兜底

    adj_uni: Dict[str, int] = dict(doc_uni)
    for t, c in own_uni.items():
        adj_uni[t] = adj_uni.get(t, 0) - c

    adj_bi: Dict[Tuple[str, str], int] = dict(doc_bi)
    for g, c in own_bi.items():
        adj_bi[g] = adj_bi.get(g, 0) - c

    row_sum: Dict[str, int] = {}
    for (a, _b), c in adj_bi.items():
        if c > 0:
            row_sum[a] = row_sum.get(a, 0) + c

    v_eff = max(vocab_size, 1)
    total = 0.0
    prev: Optional[str] = None

    for w in sent_tokens:
        p_uni = (adj_uni.get(w, 0) + k_uni) / (n_adj + k_uni * v_eff)
        if prev is None:
            p = p_uni
        else:
            row = row_sum.get(prev, 0)
            p = (adj_bi.get((prev, w), 0) + k_bi * p_uni) / (row + k_bi)
        total += -math.log2(max(p, 1e-12))
        prev = w

    return 2.0 ** (total / n)


# --------------------------------------------------------------------------
# 6. 句级特征
# --------------------------------------------------------------------------

def _compress_ratio(s: str) -> float:
    """zlib 压缩比。低 = 冗余高 = 更像模型输出。"""
    raw = s.encode("utf-8", "ignore")
    if len(raw) < 8:
        return 1.0
    return len(zlib.compress(raw, 9)) / len(raw)


def _marker_hits(text: str) -> Tuple[float, Dict[str, int]]:
    """返回 (加权命中数, 命中明细)。强标记 1.0，弱标记 0.4。"""
    hits: Dict[str, int] = {}
    score = 0.0
    for m in MARKERS_CN_STRONG:
        c = text.count(m)
        if c:
            hits[m] = c
            score += c * 1.0
    for m in MARKERS_CN_WEAK:
        c = text.count(m)
        if c:
            hits[m] = c
            score += c * 0.4
    for mt in _EN_MARKER_RE.finditer(text):
        k = mt.group(0).lower()
        hits[k] = hits.get(k, 0) + 1
        score += 1.0
    return score, hits


def _punct_density(s: str) -> float:
    """每 100 字符里"AI 偏好标点"的数量：分号 / 冒号 / 破折号 / 括号补充。"""
    if not s:
        return 0.0
    n = (s.count("；") + s.count(";") + s.count("：") + s.count(":")
         + s.count("——") * 2 + s.count("(") + s.count("（"))
    return n / len(s) * 100.0


def _markdown_density(text: str) -> float:
    """每 100 字符的 Markdown 残留：粗体/**、标题 #、列表、表格。"""
    if not text:
        return 0.0
    n = 0
    n += len(re.findall(r"\*\*[^*]+\*\*", text))
    n += len(re.findall(r"^#{1,6}\s+", text, re.M))
    n += len(re.findall(r"^\s*[-*+]\s+", text, re.M))
    n += len(re.findall(r"^\s*\d+[.、]\s+", text, re.M))
    n += text.count("|") // 3
    return n / max(len(text), 1) * 100.0


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _shannon_norm(counts: Sequence[int]) -> float:
    """归一化香农熵（0..1）。句首多样性低 -> 更像模板。"""
    total = sum(counts)
    if total <= 0 or len(counts) < 2:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h / math.log2(len(counts))


def _starts_with_opener(s: str) -> bool:
    head = s.lstrip("\"'“”‘’(（[【 ").lower()
    for op in DISCOURSE_OPENERS_CN:
        if head.startswith(op):
            return True
    for op in DISCOURSE_OPENERS_EN:
        if head.startswith(op):
            return True
    return False


def _bigram_set(tokens: Sequence[str]) -> set:
    if len(tokens) >= 2:
        return {tuple(tokens[i:i + 2]) for i in range(len(tokens) - 1)}
    return {(t,) for t in tokens}


def _max_pair_jaccard(sent_tokens: Sequence[Sequence[str]],
                      bigram_sets: Sequence[set]) -> List[float]:
    """每句与其它句的最大 bigram Jaccard。

    小文档精确两两比较；大文档改为「完全重复句 + 局部窗口 + 均匀抽样」，
    避免 O(n^2) 把十万字文档拖到几十秒。AI 复读主要发生在相邻句与模板重复句上。
    """
    n = len(bigram_sets)
    if n <= 1:
        return [0.0] * n

    best = [0.0] * n
    groups: Dict[tuple, List[int]] = {}
    for i, toks in enumerate(sent_tokens):
        groups.setdefault(tuple(toks), []).append(i)
    for ids in groups.values():
        if len(ids) > 1:
            for i in ids:
                best[i] = 1.0

    if n <= 600:
        for i in range(n):
            if best[i] >= 1.0:
                continue
            bi = bigram_sets[i]
            m = 0.0
            for j in range(n):
                if j != i:
                    v = _jaccard(bi, bigram_sets[j])
                    if v > m:
                        m = v
            best[i] = m
        return best

    step = max(1, n // 80)
    sample = list(range(0, n, step))[:80]
    for i in range(n):
        if best[i] >= 1.0:
            continue
        bi = bigram_sets[i]
        m = best[i]
        lo, hi = max(0, i - 16), min(n, i + 17)
        for j in range(lo, hi):
            if j == i:
                continue
            v = _jaccard(bi, bigram_sets[j])
            if v > m:
                m = v
        for j in sample:
            if lo <= j < hi:
                continue
            v = _jaccard(bi, bigram_sets[j])
            if v > m:
                m = v
        best[i] = m
    return best


MAX_HEATMAP_SENTS = 6000


def _limit_sentences(scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """超长文档按等距抽样返回，避免一次回十几 MB JSON 拖垮浏览器。"""
    n = len(scored)
    if n <= MAX_HEATMAP_SENTS:
        return scored
    step = n / MAX_HEATMAP_SENTS
    return [scored[min(n - 1, int(i * step))] for i in range(MAX_HEATMAP_SENTS)]


# --------------------------------------------------------------------------
# 7. 权重（手工标定；商业服务是训练出来的）
# --------------------------------------------------------------------------

# 相对信号：文档内 z 分数（这句话相对全文有多反常）
SENT_WEIGHTS: Dict[str, float] = {
    "ppl": -0.55,      # 困惑度低 -> AI
    "zratio": -0.50,   # 压缩比低（冗余高）-> AI
    "punct": 0.30,     # 分号/破折号/冒号密度高 -> AI
    "rep": 0.35,       # 与别句高度重叠 -> AI
    "ttr": -0.25,      # 词汇多样性低 -> AI
}
SENT_Z_KEYS = ("ppl", "zratio", "punct", "rep", "ttr")

# 绝对信号：不看全文，只看这一句本身。z 分数均值为 0，缺了这两项每句都会
# 收敛到 sigmoid(-0.45)≈0.39，热力图就失去区分度。
SENT_MARKER_W = 1.40   # 套话密度，以 1.25 个/百字为中心，2.5 个饱和
SENT_OPENER_W = 0.55   # 句首是话语连接词

SENT_BIAS = -0.45
LEN_UNIFORMITY_WEIGHT = 0.25  # 句长越接近全文均值越像 AI


# --------------------------------------------------------------------------
# 8. 主分析入口
# --------------------------------------------------------------------------

def analyze(text: str) -> Dict[str, Any]:
    # 水印/隐藏字符扫描必须用原文：strip() 会吃掉首尾的 NBSP 之类证据
    raw_text = text or ""
    wm = scan_watermark(raw_text)

    # 风格统计跑在「可见投影」上，隐藏字符不参与分词与密度计算
    text = visible_text(raw_text).strip()
    if not text:
        err = "文本为空"
        if wm["detected"]:
            err = "文本中没有可见字符（仅含隐藏字符，见 watermark 字段）"
        return {"ok": False, "error": err, "watermark": wm}

    paragraphs = [p for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    sentences = split_sentences(text)
    if not sentences:
        return {"ok": False, "error": "未能分出任何句子", "watermark": wm}

    sents = [{"text": s, "tokens": tokenize(s)} for s in sentences]
    sents = [s for s in sents if s["tokens"]]
    if not sents:
        err = "文本中没有可分析的字符"
        if wm["detected"]:
            err += "（但检测到隐藏字符，见 watermark 字段）"
        return {"ok": False, "error": err, "watermark": wm}

    doc_tokens: List[str] = []
    for s in sents:
        doc_tokens.extend(s["tokens"])

    doc_uni: Dict[str, int] = {}
    for t in doc_tokens:
        doc_uni[t] = doc_uni.get(t, 0) + 1
    doc_bi = _ngrams(doc_tokens, 2)
    vocab = len(doc_uni)

    # ---- 句级原始特征 ----
    raw: List[Dict[str, float]] = []
    bigram_sets = [_bigram_set(s["tokens"]) for s in sents]
    rep_vals = _max_pair_jaccard([s["tokens"] for s in sents], bigram_sets)

    for idx, s in enumerate(sents):
        st = s["tokens"]
        ppl = _lm_perplexity(st, doc_uni, doc_bi, vocab, len(doc_tokens))
        mk, _ = _marker_hits(s["text"])
        max_j = rep_vals[idx]

        raw.append({
            "ppl": ppl if ppl is not None else 0.0,
            "ppl_ok": 1.0 if ppl is not None else 0.0,
            "zratio": _compress_ratio(s["text"]),
            "markers": mk / max(len(s["text"]), 40) * 100.0,
            "punct": _punct_density(s["text"]),
            "rep": max_j,
            "ttr": len(set(st)) / len(st) if st else 0.0,
            "opener": 1.0 if _starts_with_opener(s["text"]) else 0.0,
            "len": float(len(s["text"])),
        })

    zs = {k: _zscore([r[k] for r in raw]) for k in SENT_Z_KEYS}

    lens = [r["len"] for r in raw]
    mlen = _mean(lens)
    sdlen = _stdev(lens) or 1.0

    scored: List[Dict[str, Any]] = []
    for i, s in enumerate(sents):
        dev = abs(raw[i]["len"] - mlen) / sdlen
        uniform = _clamp(1.0 - dev / 2.0)

        logit = SENT_BIAS
        contrib: List[Tuple[str, float]] = []

        for key in SENT_Z_KEYS:
            z = zs[key][i]
            if key == "ppl" and raw[i]["ppl_ok"] == 0.0:
                z = 0.0
            c = SENT_WEIGHTS[key] * z
            logit += c
            contrib.append((key, c))

        # 以 1.25 个/百字为中心：零套话的句子要被往下压，否则人类句永远停在中性 0.5
        c = SENT_MARKER_W * (2.0 * _clamp(raw[i]["markers"] / 2.5) - 1.0)
        logit += c
        contrib.append(("markers", c))

        c = SENT_OPENER_W * raw[i]["opener"]
        logit += c
        contrib.append(("opener", c))

        c = LEN_UNIFORMITY_WEIGHT * (uniform * 2.0 - 1.0)
        logit += c
        contrib.append(("uniform", c))

        contrib.sort(key=lambda x: abs(x[1]), reverse=True)
        scored.append({
            "i": i,
            "text": s["text"],
            "base_logit": logit,
            "signals": [k for k, _ in contrib[:3]],
        })

    # ---- 文档级特征 ----
    total_chars = max(len(text), 1)
    sent_chars = [len(s["text"]) for s in sents]
    all_mk, mk_hits = _marker_hits(text)
    # 分母下限 120：10 个字里出现一个"具有重要意义"不该被算成 10/百字
    marker_density = all_mk / max(total_chars, 120) * 100.0
    md_density = _markdown_density(text)
    emdash = (text.count("——") / total_chars) * 100.0

    sent_len_cv = _cv(sent_chars)
    para_cv = _cv([len(p) for p in paragraphs])

    grams4 = _ngrams(doc_tokens, 4)
    rep4 = (sum(c for c in grams4.values() if c > 1) / len(grams4)) if grams4 else 0.0

    W = 50  # MATTR 窗口，长度无关的词汇多样性
    if len(doc_tokens) <= W:
        mattr = len(set(doc_tokens)) / len(doc_tokens) if doc_tokens else 0.0
    else:
        span = len(doc_tokens) - W + 1
        step = max(1, span // 4000)  # 长文等距采样，避免逐窗口扫全文
        mattr = _mean([len(set(doc_tokens[i:i + W])) / W
                       for i in range(0, span, step)])

    opener_ratio = _mean([r["opener"] for r in raw])
    head_counts: Dict[str, int] = {}
    for s in sents:
        head = s["text"][:2]
        head_counts[head] = head_counts.get(head, 0) + 1
    head_entropy = _shannon_norm(list(head_counts.values()))

    # ---- 文档级 logit ----
    parts: List[Tuple[str, str, float, float]] = []

    def add(key: str, label: str, value: float, contribution: float) -> None:
        parts.append((key, label, round(value, 4), round(contribution, 4)))

    logit = -1.45

    v = _clamp(marker_density / 3.0, 0.0, 1.5)
    c = 2.6 * v
    logit += c
    add("marker_density", "AI 套话密度", marker_density, c)

    # 句数 < 6 时方差不稳，按样本量打折，避免短文本被"均匀"误伤
    n_factor = _clamp(len(scored) / 6.0, 0.25, 1.0)
    v = 1.0 - _clamp(sent_len_cv / 0.75)
    c = 1.6 * v * n_factor
    logit += c
    add("sentence_uniformity", "句长均匀度", sent_len_cv, c)

    v = _clamp((rep4 - 0.05) / 0.35)
    c = 1.4 * v
    logit += c
    add("repetition_4gram", "4-gram 重复率", rep4, c)

    v = _clamp((0.72 - mattr) / 0.30)
    c = 1.2 * v
    logit += c
    add("lexical_diversity", "词汇多样性 MATTR", mattr, c)

    v = _clamp(opener_ratio / 0.5)
    c = 0.9 * v
    logit += c
    add("opener_ratio", "连接词句首占比", opener_ratio, c)

    v = _clamp(emdash / 1.5)
    c = 0.8 * v
    logit += c
    add("emdash", "破折号密度", emdash, c)

    v = _clamp(md_density / 2.0)
    c = 0.7 * v
    logit += c
    add("markdown", "Markdown 残留", md_density, c)

    v = _clamp((head_entropy - 0.55) / 0.45)
    c = -0.6 * v
    logit += c
    add("opener_entropy", "句首熵", head_entropy, c)

    # 段落数 < 3 时这个量没有意义（单段落 CV 恒为 0）
    p_factor = _clamp((len(paragraphs) - 2) / 3.0, 0.0, 1.0)
    c = 0.35 * (1.0 - _clamp(para_cv / 1.0)) * p_factor
    logit += c
    add("paragraph_cv", "段落长度变异", para_cv, c)

    # ---- 句级在文档先验下重标定 ----
    # 句级特征全是"文档内 z 分数"，表达不了"整篇都像 AI"这种全局信号。
    # 在概率空间以 28% 权重注入文档级证据：既让热力图与总分方向一致，
    # 又不会把混写文本里明显的人类句拖进"可疑"。
    PRIOR_W = 0.28
    doc_prior_p = _sigmoid(logit)
    for s in scored:
        s["raw_logit"] = round(s["base_logit"], 4)
        s["score"] = round((1.0 - PRIOR_W) * _sigmoid(s["base_logit"])
                           + PRIOR_W * doc_prior_p, 4)

    # ---- 相邻句平滑（商业产品同样做句间平滑，避免单句抖动）----
    for i, s in enumerate(scored):
        nb = [s["score"]]
        if i > 0:
            nb.append(scored[i - 1]["score"])
        if i + 1 < len(scored):
            nb.append(scored[i + 1]["score"])
        v = sum(nb) / len(nb) * 0.7 + s["score"] * 0.3
        s["score_smooth"] = round(v, 4)
        s["tier"] = "high" if v >= CALIBRATION["tier_high"] else (
            "mid" if v >= CALIBRATION["tier_mid"] else "low")

    mean_sent_score = _mean([s["score_smooth"] for s in scored])
    # 与热力图分档保持同一口径（都用平滑后的分数），否则会出现"占比 50% 但全是绿色"
    mixed_ratio = sum(1 for s in scored
                      if s["score_smooth"] >= CALIBRATION["tier_mid"]) / len(scored)

    c = 1.5 * (mean_sent_score - 0.5) * 2.0
    logit += c
    add("sentence_aggregate", "句级分类器聚合", mean_sent_score, c)

    # ---- 校准 ----
    prob = calibrate(logit, len(text), len(scored))

    if prob >= 0.85:
        verdict, tier = "高度疑似 AI 生成", "high"
    elif prob >= 0.60:
        verdict, tier = "疑似 AI 生成", "mid"
    elif prob >= 0.35:
        verdict, tier = "人机混写可能性", "lowmid"
    else:
        verdict, tier = "更可能为人类撰写", "low"

    parts.sort(key=lambda p: abs(p[3]), reverse=True)

    return {
        "ok": True,
        "doc": {
            "chars": len(text),
            "paragraphs": len(paragraphs),
            "sentences": len(scored),
            "sentences_returned": min(len(scored), MAX_HEATMAP_SENTS),
            "heatmap_sampled": len(scored) > MAX_HEATMAP_SENTS,
            "logit": round(logit, 4),
            "probability": round(prob, 4),
            "percent": round(prob * 100, 1),
            "verdict": verdict,
            "tier": tier,
            "mixed_ratio": round(mixed_ratio, 4),
            "sentence_mean": round(mean_sent_score, 4),
            "confidence": ("high" if len(text) >= 600
                           else ("medium" if len(text) >= 200 else "low")),
            "contributions": [
                {"key": k, "label": lbl, "value": val, "contribution": cc}
                for k, lbl, val, cc in parts
            ],
            "metrics": {
                "marker_density": round(marker_density, 3),
                "marker_hits": mk_hits,
                "sentence_len_cv": round(sent_len_cv, 3),
                "paragraph_len_cv": round(para_cv, 3),
                "repetition_4gram": round(rep4, 3),
                "mattr": round(mattr, 3),
                "opener_ratio": round(opener_ratio, 3),
                "opener_entropy": round(head_entropy, 3),
                "emdash_density": round(emdash, 3),
                "markdown_density": round(md_density, 3),
                "mean_ppl": round(_mean([r["ppl"] for r in raw if r["ppl_ok"] == 1.0]), 2),
                "mean_compress_ratio": round(_mean([r["zratio"] for r in raw]), 3),
            },
        },
        "sentences": _limit_sentences(scored),
        "watermark": wm,
    }


# --------------------------------------------------------------------------
# 9. 校准层 —— 全程唯一真正"可调"的地方
# --------------------------------------------------------------------------

CALIBRATION = {
    # <1 更激进（更容易报"疑似AI"）；>1 更保守。默认 0.85 略偏激进。
    "temperature": 0.85,
    # 短文本最低保留强度。0.0 = 一句话也能报 0%/100%；1.0 = 极短文本永远 50%。
    "min_shrink": 0.25,
    # 短文本收缩的参考长度：达到该字符数即完全采信模型输出
    "full_confidence_chars": 200.0,
    # 句级热力图分档阈值
    "tier_mid": 0.50,
    "tier_high": 0.75,
}


def calibrate(logit: float, n_chars: int, n_sent: int) -> float:
    """把文档级 logit 映射成对外报告的百分比。

    这一环叫 calibration：同一个 logit，不同产品的报告口径完全不同
    （有的偏保守、有的偏激进）。这里做两件事：

      1. 温度缩放：<1 放大差异（激进），>1 压缩差异（保守）。
      2. 样本量收缩：文本越短，越往 0.5 拉，避免"一句话定罪"。
    """
    temp = CALIBRATION["temperature"]
    p = _sigmoid(logit / temp)

    shrink = _clamp(n_chars / CALIBRATION["full_confidence_chars"], 0.0, 1.0)
    shrink = max(shrink, CALIBRATION["min_shrink"])
    p = 0.5 + (p - 0.5) * shrink

    return _clamp(p, 0.001, 0.999)


# --------------------------------------------------------------------------
# 10. 独立自测
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    demo_ai = (
        "首先，人工智能技术在教育领域的应用具有重要意义。其次，人工智能能够显著提升教学效率，"
        "为学生提供个性化的学习体验。此外，人工智能还可以赋能教师，使教师能够更加聚焦于学生的"
        "全面发展。值得注意的是，随着技术的不断发展，人工智能在教育中的作用将日益凸显。"
        "综上所述，人工智能技术在教育领域扮演着重要角色，具有广阔的发展前景。"
    )
    demo_human = (
        "昨天下午我在楼下便利店买了瓶水，收银的阿姨正在看一部很老的家庭剧，声音开得挺大。"
        "她一边找零一边跟我抱怨说这剧重播第八遍了，但她还是每天准点守着。"
        "我拎着水走出去，外面下起了小雨，没带伞，就站在屋檐下站了十来分钟。"
        "后来雨小了，我骑车回家，路上想不起晚饭该吃什么。"
    )

    for name, sample in (("AI 风格样例", demo_ai), ("人类风格样例", demo_human)):
        r = analyze(sample)
        print(f"===== {name} =====")
        print(f"  概率: {r['doc']['percent']}%  判定: {r['doc']['verdict']}")
        print(f"  句数: {r['doc']['sentences']}  混写占比: {r['doc']['mixed_ratio']}")
        print(f"  Top 贡献: {[c['label'] for c in r['doc']['contributions'][:3]]}")
        print(f"  隐藏标记: {r['watermark']['verdict']}")

    demo_hidden = demo_ai + "".join(chr(0xE0000 + ord(c)) for c in "leak")
    r = analyze(demo_hidden)
    wm = r["watermark"]
    print("\n===== 隐藏标记样例 =====")
    print(f"  风格分: {r['doc']['percent']}%")
    print(f"  风险: {wm['risk']}  结论: {wm['verdict']}")
    for f in wm["findings"]:
        extra = f"  解码: {f['decoded']}" if f.get("decoded") else ""
        print(f"   - {f['label']} ×{f['count']}{extra}")
    print("\n自测完成。")
