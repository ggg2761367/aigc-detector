/* AIGC 检测器前端 —— 无框架，纯 DOM。 */

const $ = (id) => document.getElementById(id);

const RING_CIRC = 2 * Math.PI * 52;

const TIER_COLOR = {
  high: "#ef4444",
  mid: "#f97316",
  lowmid: "#eab308",
  low: "#22c55e",
};

const SIGNAL_LABEL = {
  ppl: "低困惑度",
  zratio: "高冗余",
  markers: "套话词",
  punct: "标点模板",
  rep: "句间重复",
  ttr: "词汇重复",
  opener: "句首连接词",
  uniform: "句长均匀",
};

const SAMPLE_AI = `首先，人工智能技术在教育领域的应用具有重要意义。其次，人工智能能够显著提升教学效率，为学生提供个性化的学习体验。此外，人工智能还可以赋能教师，使教师能够更加聚焦于学生的全面发展。值得注意的是，随着技术的不断发展，人工智能在教育中的作用将日益凸显。与此同时，教育机构也需要在数据安全与伦理规范层面建立相应的保障机制。综上所述，人工智能技术在教育领域扮演着重要角色，具有广阔的发展前景。`;

const SAMPLE_HUMAN = `昨天下午我在楼下便利店买了瓶水，收银的阿姨正在看一部很老的家庭剧，声音开得挺大。她一边找零一边跟我抱怨说这剧重播第八遍了，但她还是每天准点守着。我拎着水走出去，外面下起了小雨，没带伞，就站在屋檐下站了十来分钟。后来雨小了，我骑车回家，路上想不起晚饭该吃什么。晚上翻冰箱，只有半棵白菜和两个鸡蛋，最后煮了碗面。`;

// ---------------------------------------------------------------- 工具

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function updateCounter() {
  const n = $("input").value.length;
  $("counter").textContent = `${n} 字`;
}

// ---------------------------------------------------------------- 渲染

function render(data) {
  const doc = data.doc;

  $("empty").classList.add("hidden");
  $("result").classList.remove("hidden");

  // 主分数：模型原始置信度（softmax）；跨语料校准值只作参考展示。
  // 校准表在高分窄区间会被「正式人类文体」拉低，不能直接当 AI 率展示。
  const hasModel = data.model && typeof data.model.percent === "number";
  const calPct = hasModel ? Math.max(0, Math.min(100, data.model.percent)) : null;
  const modelPct = hasModel
    ? Math.max(0, Math.min(100,
        typeof data.model.raw_probability === "number"
          ? data.model.raw_probability * 100
          : data.model.percent))
    : null;
  const pct = hasModel ? modelPct : Math.max(0, Math.min(100, doc.percent));
  const tier = hasModel
    ? (modelPct >= 85 ? "high" : modelPct >= 60 ? "mid" : modelPct >= 35 ? "lowmid" : "low")
    : doc.tier;
  const verdict = hasModel
    ? ({ high: "高度疑似 AI 生成", mid: "疑似 AI 生成", lowmid: "人机混写可能性", low: "更可能为人类撰写" })[tier]
    : doc.verdict;

  const color = TIER_COLOR[tier] || "#94a3b8";
  const ring = $("ringFg");
  ring.style.stroke = color;
  ring.style.strokeDasharray = `${(pct / 100) * RING_CIRC} ${RING_CIRC}`;
  ring.style.strokeDashoffset = "0";

  $("scoreNum").innerHTML = `${pct.toFixed(1)}<span>%</span>`;
  $("scoreNum").style.color = color;
  $("verdict").textContent = verdict;
  $("verdict").style.color = color;

  const CONF = { high: "置信度：高", medium: "置信度：中", low: "置信度：低（文本偏短）" };
  const calNote = hasModel && calPct !== null
    ? ` · 跨语料校准 ${calPct.toFixed(1)}%`
    : "";
  $("confidence").textContent =
    `${hasModel ? "神经分类器 · " : ""}${CONF[doc.confidence] || ""} · ` +
    `${doc.chars} 字 / ${doc.sentences} 句${calNote}`;

  // 统计卡片
  const m = doc.metrics;
  const stats = [
    ["疑似片段占比", `${(doc.mixed_ratio * 100).toFixed(1)}%`, "句级分数 ≥ 50% 的句子比例"],
    ["套话密度", `${m.marker_density.toFixed(2)}/百字`, "AI 高频连接词与套话"],
    ["句长变异系数", m.sentence_len_cv.toFixed(3), "越低越均匀，越像机器"],
    ["4-gram 重复率", `${(m.repetition_4gram * 100).toFixed(1)}%`, "自我复读程度"],
    ["词汇多样性", m.mattr.toFixed(3), "MATTR，窗口 50"],
    ["平均困惑度", m.mean_ppl > 0 ? m.mean_ppl.toFixed(1) : "—", "越低越可预测"],
  ];
  if (hasModel) {
    const thr = typeof data.model.threshold === "number"
      ? ` · 1% 误报阈值 ${(data.model.threshold * 100).toFixed(1)}%`
      : "";
    stats.unshift([
      "风格启发式",
      `${doc.percent.toFixed(1)}%`,
      "规则+特征工程分数，仅作参考；仿人写作会明显偏低",
    ]);
    stats.push([
      "模型原始置信度",
      `${modelPct.toFixed(1)}%`,
      `softmax 原始输出 · 主分 · ONNX int8${thr}`,
    ]);
    if (calPct !== null) {
      stats.push([
        "跨语料校准",
        `${calPct.toFixed(1)}%`,
        "按混合基准校准；正式人类文体可能拉低此值，故不作主分",
      ]);
    }
  } else {
    stats.unshift([
      "风格启发式",
      `${doc.percent.toFixed(1)}%`,
      "未加载神经模型，当前只有启发式分数",
    ]);
  }
  const statsBox = $("stats");
  statsBox.textContent = "";
  for (const [k, v, hint] of stats) {
    const card = el("div", "stat");
    card.appendChild(el("div", "stat-k", k));
    card.appendChild(el("div", "stat-v", v));
    card.title = hint;
    statsBox.appendChild(card);
  }

  // 特征贡献
  const maxAbs = Math.max(0.001, ...doc.contributions.map((c) => Math.abs(c.contribution)));
  const box = $("contribs");
  box.textContent = "";
  for (const c of doc.contributions) {
    const row = el("div", "contrib");
    row.appendChild(el("div", "contrib-label", c.label));

    const track = el("div", "contrib-track");
    const bar = el("div", "contrib-bar");
    const ratio = Math.abs(c.contribution) / maxAbs;
    bar.style.width = `${Math.max(2, ratio * 100)}%`;
    bar.style.background = c.contribution >= 0 ? "#ef4444" : "#38bdf8";
    if (c.contribution < 0) bar.style.marginLeft = "auto";
    track.appendChild(bar);
    row.appendChild(track);

    row.appendChild(el("div", "contrib-val",
      (c.contribution >= 0 ? "+" : "") + c.contribution.toFixed(2)));
    box.appendChild(row);
  }

  // 句级热力图
  const hm = $("heatmap");
  hm.textContent = "";
  if (doc.heatmap_sampled) {
    hm.appendChild(el("div", "hm-note muted small",
      `长文：热力图等距展示 ${doc.sentences_returned} / ${doc.sentences} 句，文档级分数基于全文`));
  }
  for (const s of data.sentences) {
    const span = el("span", `sent tier-${s.tier}`, s.text + " ");
    const labels = (s.signals || []).map((k) => SIGNAL_LABEL[k] || k).join(" / ");
    span.title = `AI 倾向 ${(s.score_smooth * 100).toFixed(0)}%\n主要信号：${labels}`;
    hm.appendChild(span);
  }

  // 命中词表
  const mk = $("markers");
  mk.textContent = "";
  const hits = Object.entries(m.marker_hits || {}).sort((a, b) => b[1] - a[1]);
  if (!hits.length) {
    mk.appendChild(el("span", "muted", "未命中任何 AI 高频词"));
  } else {
    for (const [term, count] of hits) {
      const chip = el("span", "chip");
      chip.appendChild(el("span", "chip-t", term));
      if (count > 1) chip.appendChild(el("span", "chip-n", `×${count}`));
      mk.appendChild(chip);
    }
  }

  // 隐藏标记 / 水印痕迹
  renderWatermark(data.watermark);
}

function renderWatermark(wm) {
  const box = $("watermark");
  box.textContent = "";
  if (!wm) return;

  const risk = wm.risk || "none";
  box.appendChild(el("div", `wm-badge wm-${risk}`, wm.verdict || "—"));
  box.appendChild(el("div", "wm-meta muted small",
    `风险等级 ${risk}${wm.invisible_chars ? ` · 不可见字符 ${wm.invisible_chars} 个` : ""}`));

  for (const f of wm.findings || []) {
    const item = el("div", "wm-item");

    const head = el("div", "wm-head");
    head.appendChild(el("span", `wm-sev wm-${f.severity}`, f.severity));
    head.appendChild(el("span", "wm-label", f.label));
    head.appendChild(el("span", "wm-count muted small", `×${f.count}`));
    item.appendChild(head);

    item.appendChild(el("div", "wm-detail muted small", f.detail));

    if (f.codepoints && f.codepoints.length) {
      item.appendChild(el("div", "wm-cps", f.codepoints.join("  ")));
    }

    if (f.decoded) {
      const payload = el("div", "wm-payload");
      payload.appendChild(el("span", "muted small", "解码载荷 "));
      // 载荷来自用户输入：只能用 textContent，绝不能用 innerHTML
      payload.appendChild(el("code", "", f.decoded));
      item.appendChild(payload);
    }

    box.appendChild(item);
  }
}

// ---------------------------------------------------------------- 请求

async function runAnalyze() {
  const text = $("input").value.trim();
  if (!text) {
    $("input").focus();
    return;
  }

  const btn = $("btnAnalyze");
  const old = btn.textContent;
  btn.disabled = true;
  btn.textContent = "检测中…";

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!data.ok) {
      alert(data.error || "分析失败");
      return;
    }
    render(data);
    if (window.innerWidth < 900) {
      $("result").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (e) {
    alert("请求失败：" + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
}

// ---------------------------------------------------------------- 文档上传

const MAX_INPUT = 400000;

function setFileInfo(text) {
  $("fileinfo").textContent = text || "";
}

async function loadFile(file) {
  if (!file) return;

  const btn = $("btnFile");
  const old = btn.textContent;
  btn.disabled = true;
  btn.textContent = "解析中…";
  setFileInfo(`正在解析 ${file.name}…`);

  try {
    const res = await fetch(`/api/extract?name=${encodeURIComponent(file.name)}`, {
      method: "POST",
      body: file,
    });
    const raw = await res.text();
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      throw new Error("服务返回异常，可能是旧版进程仍在运行；请关闭所有检测器窗口后重新打开");
    }
    if (!data.ok) {
      const msg = data.error === "not found"
        ? "服务版本过旧（旧服务进程仍在运行）。请关闭所有检测器窗口和命令行后重新打开"
        : (data.error || "解析失败");
      throw new Error(msg);
    }

    let text = data.text || "";
    let note = `${data.filename || file.name} · ${data.kind} · ${data.chars} 字`;
    if (text.length > MAX_INPUT) {
      text = text.slice(0, MAX_INPUT);
      note += ` · 超出上限，仅检测前 ${MAX_INPUT} 字`;
    }

    $("input").value = text;
    updateCounter();
    setFileInfo(note);
    await runAnalyze();
  } catch (e) {
    setFileInfo("");
    alert("文档解析失败：" + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
}

function bindDrop(node) {
  node.addEventListener("dragover", (e) => {
    e.preventDefault();
    node.classList.add("dragover");
  });
  node.addEventListener("dragleave", () => node.classList.remove("dragover"));
  node.addEventListener("drop", (e) => {
    e.preventDefault();
    node.classList.remove("dragover");
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) loadFile(f);
  });
}

// ---------------------------------------------------------------- 绑定

$("input").addEventListener("input", updateCounter);
$("btnAnalyze").addEventListener("click", runAnalyze);

$("btnFile").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) loadFile(f);
  e.target.value = "";
});
bindDrop($("dropzone"));
bindDrop($("input"));

$("btnClear").addEventListener("click", () => {
  $("input").value = "";
  updateCounter();
  $("result").classList.add("hidden");
  $("empty").classList.remove("hidden");
  $("input").focus();
});

$("btnSampleAI").addEventListener("click", () => {
  $("input").value = SAMPLE_AI;
  updateCounter();
});

$("btnSampleHuman").addEventListener("click", () => {
  $("input").value = SAMPLE_HUMAN;
  updateCounter();
});

$("btnHow").addEventListener("click", () => $("how").classList.remove("hidden"));
$("btnCloseHow").addEventListener("click", () => $("how").classList.add("hidden"));
$("how").addEventListener("click", (e) => {
  if (e.target === $("how")) $("how").classList.add("hidden");
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") $("how").classList.add("hidden");
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runAnalyze();
});

updateCounter();
