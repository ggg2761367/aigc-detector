# -*- coding: utf-8 -*-
"""文档文本提取：.docx / .pdf / .txt / .md / .rtf → 纯文本。

设计原则
--------
- docx 是 zip + XML，用标准库解析，零依赖；
- pdf 走可选依赖 pypdf，缺失时给出明确提示（系统 Python 零依赖模式仍可用 docx/txt）；
- 所有解析都在内存中完成，不落盘，不改动原文件。
"""

from __future__ import annotations

import io
import re
import zipfile
from typing import Dict
from xml.etree import ElementTree as ET

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".log", ".json"}


class ExtractError(ValueError):
    """可展示给用户的提取失败原因。"""


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "gb18030", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _from_docx(raw: bytes) -> str:
    if raw.startswith(_OLE2_MAGIC):
        raise ExtractError(
            "这是加密的 Office 文档或旧版 .doc（OLE2 容器），不是真正的 .docx。"
            "请取消密码保护，或用 Word/WPS 另存为 .docx 后重试"
        )
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise ExtractError("不是有效的 .docx 文件（zip 结构损坏或后缀被改过）")

    try:
        xml = zf.read("word/document.xml")
    except KeyError:
        raise ExtractError("这不是 Word 文档：缺少 word/document.xml")

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise ExtractError(f"docx XML 解析失败: {e}")

    paras = []
    for p in root.iter(_W + "p"):
        buf = []
        for node in p.iter():
            if node.tag == _W + "t" and node.text:
                buf.append(node.text)
            elif node.tag == _W + "tab":
                buf.append("\t")
            elif node.tag in (_W + "br", _W + "cr"):
                buf.append("\n")
        line = "".join(buf).strip()
        if line:
            paras.append(line)
    return "\n\n".join(paras)


def _from_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ExtractError("当前环境未安装 pypdf，无法解析 PDF（打包版已内置；"
                           "也可先把 PDF 导出为 .docx/.txt）")

    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as e:
        raise ExtractError(f"PDF 解析失败: {e}")

    text = "\n\n".join(p for p in pages if p)
    if not text.strip():
        raise ExtractError("PDF 中没有可提取的文本层（可能是扫描件/图片，需要 OCR）")
    return text


def _from_rtf(text: str) -> str:
    # \'hh 十六进制字节、\uN Unicode、\par 换行，其余控制字剥掉
    text = re.sub(r"\\'([0-9a-fA-F]{2})",
                  lambda m: bytes.fromhex(m.group(1)).decode("latin-1"), text)
    text = re.sub(r"\\u(-?\d+)\??",
                  lambda m: chr(int(m.group(1)) % 65536), text)
    text = re.sub(r"\\par[d]?", "\n", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    return text.replace("{", "").replace("}", "")


def extract(raw: bytes, filename: str = "") -> Dict:
    """把上传文件转成纯文本；失败抛 ExtractError（消息可直接展示）。"""
    if not raw:
        raise ExtractError("文件为空")

    name = (filename or "").strip().lower()
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""

    if ext == ".docx":
        text, kind = _from_docx(raw), "docx"
    elif ext == ".pdf":
        text, kind = _from_pdf(raw), "pdf"
    elif ext == ".rtf":
        text, kind = _from_rtf(_decode(raw)), "rtf"
    elif ext in _TEXT_EXT:
        text, kind = _decode(raw), ext.lstrip(".")
    elif ext in (".doc", ".wps"):
        raise ExtractError("旧版 .doc/.wps 不支持，请在 Word/WPS 里另存为 .docx，或导出 PDF")
    elif raw[:4] == b"%PDF":
        text, kind = _from_pdf(raw), "pdf"
    elif raw[:2] == b"PK":
        text, kind = _from_docx(raw), "docx"
    else:
        text, kind = _decode(raw), "text"

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 10:
        raise ExtractError("提取到的文本太少（可能全是图片或空白）")

    return {"text": text, "kind": kind, "chars": len(text)}
