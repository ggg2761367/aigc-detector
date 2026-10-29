# -*- coding: utf-8 -*-
"""文档提取测试：内存构造 docx，不依赖外部样本文件。"""

import io
import unittest
import zipfile

from document import ExtractError, extract

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def make_docx(paragraphs):
    paras = "".join(
        f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs
    )
    xml = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W}"><w:body>{paras}</w:body></w:document>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", xml)
    return buf.getvalue()


class TestExtract(unittest.TestCase):
    def test_docx_paragraphs(self):
        raw = make_docx(["第一段文本内容足够长。", "第二段：人工智能技术的应用。"])
        out = extract(raw, "论文.docx")
        self.assertEqual(out["kind"], "docx")
        self.assertIn("第一段文本内容足够长。", out["text"])
        self.assertIn("第二段：人工智能技术的应用。", out["text"])
        self.assertEqual(out["chars"], len(out["text"]))

    def test_txt_utf8(self):
        out = extract("第一行内容\n第二行内容".encode("utf-8"), "a.txt")
        self.assertEqual(out["kind"], "txt")
        self.assertIn("第二行内容", out["text"])

    def test_txt_gbk_fallback(self):
        out = extract("中文编码兼容测试内容".encode("gb18030"), "old.txt")
        self.assertIn("中文编码兼容测试内容", out["text"])

    def test_legacy_doc_rejected(self):
        with self.assertRaises(ExtractError):
            extract(b"\xd0\xcf\x11\xe0whatever", "old.doc")

    def test_broken_docx(self):
        with self.assertRaises(ExtractError):
            extract(b"PK\x03\x04not-a-real-zip", "broken.docx")

    def test_sniff_docx_without_extension(self):
        raw = make_docx(["没有扩展名也要能识别。"])
        out = extract(raw, "upload.bin")
        self.assertEqual(out["kind"], "docx")


if __name__ == "__main__":
    unittest.main()
