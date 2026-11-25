"""Unit and Integration Tests for core.voice and core.tunnel
"""

import sys
import unittest

sys.path.insert(0, ".")

from core.voice import clean_text_for_speech, transcribe_audio_bytes
from core.tunnel import (
    get_local_ip,
    get_all_local_ips,
    get_local_url,
    generate_qr_code_data_uri,
    get_available_tunnel_services,
    is_tunnel_running,
    stop_tunnel,
)


class TestVoiceModule(unittest.TestCase):
    def test_clean_text_for_speech(self):
        sample = """
        ### Section 4: Fire Tariff Deductibles
        In accordance with **All India Fire Tariff (AIFT)**, the excess is 5% w/o deductible.
        Visit [IRDAI Guidelines](https://irdai.gov.in) for details.
        ```python
        rate = 0.05
        ```
        - Item 1: Storage risk
        - Item 2: Manufacturing risk
        > Note: [Doc: Fire tariff.pdf, Page 12] applies.
        """
        cleaned = clean_text_for_speech(sample)
        self.assertNotIn("###", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("```", cleaned)
        self.assertNotIn("https://irdai.gov.in", cleaned)
        self.assertIn("Section 4: Fire Tariff Deductibles", cleaned)
        self.assertIn("percent", cleaned)
        self.assertIn("without deductible", cleaned)
        self.assertNotIn("[Doc: Fire tariff.pdf, Page 12]", cleaned)

    def test_clean_text_empty(self):
        self.assertEqual(clean_text_for_speech(""), "")
        self.assertEqual(clean_text_for_speech(None), "")

    def test_transcribe_audio_empty(self):
        res = transcribe_audio_bytes(b"")
        self.assertEqual(res, "")


class TestTunnelModule(unittest.TestCase):
    def test_get_local_ip(self):
        ip = get_local_ip()
        self.assertIsInstance(ip, str)
        self.assertTrue(len(ip.split(".")) == 4)

    def test_get_all_local_ips(self):
        ips = get_all_local_ips()
        self.assertIsInstance(ips, list)
        self.assertTrue(len(ips) >= 1)

    def test_get_local_url(self):
        url = get_local_url(port=8501)
        self.assertTrue(url.startswith("http://"))
        self.assertTrue(url.endswith(":8501"))

    def test_generate_qr_code_data_uri(self):
        uri = generate_qr_code_data_uri("http://192.168.1.50:8501")
        self.assertTrue(uri.startswith("data:image/"))
        self.assertTrue(len(uri) > 100)

    def test_tunnel_status_initially_stopped(self):
        self.assertFalse(is_tunnel_running())
        self.assertFalse(stop_tunnel())

    def test_available_services(self):
        services = get_available_tunnel_services()
        self.assertIn("cloudflared", services)
        self.assertIn("localtunnel", services)
        self.assertIn("ngrok", services)


if __name__ == "__main__":
    unittest.main()
