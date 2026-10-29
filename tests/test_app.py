# -*- coding: utf-8 -*-
"""HTTP 接口测试：临时端口起真实 ThreadingHTTPServer，不依赖外部服务。"""

import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

import app


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def test_health(self):
        status, data = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        obj = json.loads(data)
        self.assertTrue(obj["ok"])
        self.assertIn("temperature", obj["calibration"])
        self.assertIn("model", obj)

    def test_index_page(self):
        status, data = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"AIGC", data)

    def test_analyze_ok(self):
        body = json.dumps(
            {"text": "首先，人工智能具有重要意义。其次，它显著提升了效率。"},
            ensure_ascii=False,
        ).encode("utf-8")
        status, data = self.request(
            "POST", "/api/analyze", body,
            {"Content-Type": "application/json; charset=utf-8"},
        )
        self.assertEqual(status, 200)
        obj = json.loads(data)
        self.assertTrue(obj["ok"])
        self.assertIn("doc", obj)
        self.assertIn("sentences", obj)
        self.assertIn("watermark", obj)

    def test_analyze_hidden_payload(self):
        hidden = "".join(chr(0xE0000 + ord(c)) for c in "leak")
        body = json.dumps({"text": "正常文本。" + hidden},
                          ensure_ascii=False).encode("utf-8")
        status, data = self.request("POST", "/api/analyze", body,
                                    {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        obj = json.loads(data)
        self.assertTrue(obj["watermark"]["detected"])
        self.assertEqual(obj["watermark"]["findings"][0]["decoded"], "leak")

    def test_bad_json(self):
        status, _ = self.request("POST", "/api/analyze", b"{",
                                 {"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_empty_body(self):
        status, _ = self.request("POST", "/api/analyze", b"",
                                 {"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_missing_text_field(self):
        body = json.dumps({"foo": 1}).encode("utf-8")
        status, _ = self.request("POST", "/api/analyze", body,
                                 {"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_unknown_post_path(self):
        status, _ = self.request("POST", "/api/nope", b"{}",
                                 {"Content-Type": "application/json"})
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
