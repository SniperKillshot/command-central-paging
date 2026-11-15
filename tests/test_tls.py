"""HTTPS: the local CA and server certificate, and serving over them."""
import http.client
import os
import socket
import ssl
import tempfile
import unittest

from cryptography import x509

from src import tls
from src.web_ui import create_app


def load(path):
    with open(path, 'rb') as f:
        return x509.load_pem_x509_certificate(f.read())


class CertificateTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, 'tls')

    def tearDown(self):
        self.dir.cleanup()

    def test_ca_signs_a_certificate_for_this_machine(self):
        crt, key, ca = tls.ensure_certificate(self.path, ['paging.example'])
        server, authority = load(crt), load(ca)
        server.verify_directly_issued_by(authority)
        names = tls._san(server)
        for n in ('localhost', '127.0.0.1', socket.gethostname().lower(), 'paging.example'):
            self.assertIn(n, names)
        self.assertTrue(authority.extensions.get_extension_for_class(x509.BasicConstraints).value.ca)

    def test_reused_then_reissued_by_the_same_ca(self):
        crt, _, ca = tls.ensure_certificate(self.path)
        first, first_ca = load(crt).serial_number, load(ca).serial_number
        tls.ensure_certificate(self.path)
        self.assertEqual(load(crt).serial_number, first)                 # nothing changed
        tls.ensure_certificate(self.path, ['10.99.0.1'])
        self.assertNotEqual(load(crt).serial_number, first)              # new address
        self.assertIn('10.99.0.1', tls._san(load(crt)))
        self.assertEqual(load(ca).serial_number, first_ca)               # same CA: no reinstall


class ServeTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        ctx, self.ca = tls.setup({}, self.dir.name)
        self.app = create_app(https_port=1, ca_cert=self.ca)
        self.httpd = tls.serve(self.app, '127.0.0.1', 0, ctx)
        self.port = self.httpd.server_port

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def get(self, path):
        client_ctx = ssl.create_default_context(cafile=self.ca)          # verifies name and CA
        conn = http.client.HTTPSConnection('127.0.0.1', self.port, context=client_ctx, timeout=5)
        try:
            conn.request('GET', path)
            r = conn.getresponse()
            return r.status, r.read()
        finally:
            conn.close()

    def test_https_trusted_through_the_ca(self):
        status, body = self.get('/health')
        self.assertEqual(status, 200)
        self.assertIn(b'healthy', body)

    def test_stalled_client_does_not_block_others(self):
        with socket.create_connection(('127.0.0.1', self.port)):          # connects, says nothing
            self.assertEqual(self.get('/health')[0], 200)

    def test_ca_download(self):
        status, body = self.get('/ca.crt')
        self.assertEqual(status, 200)
        with open(self.ca, 'rb') as f:
            self.assertEqual(body, f.read())
        self.assertEqual(create_app().test_client().get('/ca.crt').status_code, 404)


if __name__ == '__main__':
    unittest.main()
