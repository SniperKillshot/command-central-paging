"""HTTPS for the web UI, so a browser on another PC can use its microphone
(browsers only allow it on localhost or over HTTPS).

The first start makes a small local certificate authority (ca.crt/ca.key)
and a server certificate it signs for this machine's hostname and IP
addresses. Install ca.crt as trusted once on each PC that talks (the web UI
serves it at /ca.crt); the server certificate is reissued from the same CA
whenever the names or addresses change or it nears expiry, with nothing to
reinstall. A certificate of your own (web.https_cert/https_key) is used
instead if set.
"""
import datetime
import ipaddress
import logging
import os
import socket
import ssl
import subprocess
import threading
from typing import Iterable, List, Optional, Set, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

logger = logging.getLogger(__name__)

CA_DAYS = 3650
SERVER_DAYS = 825               # the most Apple devices accept
RENEW_DAYS = 30


def local_names() -> Set[str]:
    """This machine's hostname and IP addresses (LAN, VPN, loopback)."""
    host = socket.gethostname()
    names = {'localhost', '127.0.0.1', '::1', host, f'{host}.local'}
    try:
        names.update(socket.gethostbyname_ex(host)[2])        # all of them, on Windows
    except OSError:
        pass
    try:                                                      # Linux: every interface
        out = subprocess.run(['ip', '-o', 'addr', 'show'], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) > 3 and parts[2] in ('inet', 'inet6'):
                addr = ipaddress.ip_address(parts[3].split('/')[0])
                if not addr.is_link_local:
                    names.add(str(addr))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:                                                      # the address of the default route
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('192.0.2.1', 9))                       # nothing is sent
            names.add(s.getsockname()[0])
    except OSError:
        pass
    return names


def _general_names(names: Iterable[str]) -> List[x509.GeneralName]:
    out = []
    for n in sorted(names):
        try:
            out.append(x509.IPAddress(ipaddress.ip_address(n)))
        except ValueError:
            out.append(x509.DNSName(n))
    return out


def _san(cert: x509.Certificate) -> Set[str]:
    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return set()
    return {str(v) for v in ext.get_values_for_type(x509.DNSName)} | \
           {str(v) for v in ext.get_values_for_type(x509.IPAddress)}


def _normal(names: Iterable[str]) -> Set[str]:
    out = set()
    for n in names:
        try:
            out.add(str(ipaddress.ip_address(n)))
        except ValueError:
            out.add(n.lower())
    return out


def _write(path: str, data: bytes, private: bool = False) -> None:
    with open(path, 'wb') as f:
        f.write(data)
    if private:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def _key_pem(key) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _load_ca(directory: str):
    crt, key = os.path.join(directory, 'ca.crt'), os.path.join(directory, 'ca.key')
    if os.path.exists(crt) and os.path.exists(key):
        with open(crt, 'rb') as f:
            cert = x509.load_pem_x509_certificate(f.read())
        with open(key, 'rb') as f:
            pkey = serialization.load_pem_private_key(f.read(), None)
        if cert.not_valid_after_utc > _now() + datetime.timedelta(days=RENEW_DAYS):
            return cert, pkey
        logger.warning('HTTPS: local CA is about to expire - making a new one (reinstall ca.crt)')
    pkey = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,
                                         f'Command Central Paging CA ({socket.gethostname()})')])
    now = _now()
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(pkey.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=CA_DAYS))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False,
                                         key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, key_cert_sign=True, crl_sign=True,
                                         encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(pkey.public_key()), critical=False)
            .sign(pkey, hashes.SHA256()))
    _write(key, _key_pem(pkey), private=True)
    _write(crt, cert.public_bytes(serialization.Encoding.PEM))
    logger.info(f'HTTPS: made a local CA ({crt}) - install it as trusted on the PCs that talk')
    return cert, pkey


def ensure_certificate(directory: str, extra_names: Iterable[str] = ()) -> Tuple[str, str, str]:
    """Make (or reuse) the local CA and a server certificate for this
    machine; returns (server cert, server key, CA cert) paths."""
    os.makedirs(directory, exist_ok=True)
    ca_cert, ca_key = _load_ca(directory)
    crt, key = os.path.join(directory, 'server.crt'), os.path.join(directory, 'server.key')
    names = _normal(local_names() | {n.strip() for n in extra_names if str(n).strip()})
    if os.path.exists(crt) and os.path.exists(key):
        with open(crt, 'rb') as f:
            cert = x509.load_pem_x509_certificate(f.read())
        if (cert.issuer == ca_cert.subject and _normal(_san(cert)) >= names
                and cert.not_valid_after_utc > _now() + datetime.timedelta(days=RENEW_DAYS)):
            return crt, key, os.path.join(directory, 'ca.crt')
    pkey = ec.generate_private_key(ec.SECP256R1())
    now = _now()
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, socket.gethostname())]))
            .issuer_name(ca_cert.subject)
            .public_key(pkey.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=SERVER_DAYS))
            .add_extension(x509.SubjectAlternativeName(_general_names(names)), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                           critical=False)
            .sign(ca_key, hashes.SHA256()))
    _write(key, _key_pem(pkey), private=True)
    _write(crt, cert.public_bytes(serialization.Encoding.PEM))
    logger.info(f"HTTPS: server certificate for {', '.join(sorted(names))}")
    return crt, key, os.path.join(directory, 'ca.crt')


def server_context(cert: str, key: str) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert, key)
    return ctx


def serve(app, host: str, port: int, ctx: ssl.SSLContext):
    """Serve the app over HTTPS on a background thread; returns the server."""
    from werkzeug.serving import make_server
    httpd = make_server(host, port, app, threaded=True)
    # Handshake on each connection's own thread (werkzeug's ssl_context does
    # it in the accept loop, where one stalled client would block the port).
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True, do_handshake_on_connect=False)
    httpd.ssl_context = ctx         # werkzeug: https URLs; TLS errors logged, not raised
    threading.Thread(target=httpd.serve_forever, name='https', daemon=True).start()
    return httpd


def setup(web: dict, data_dir: str) -> Tuple[Optional[ssl.SSLContext], Optional[str]]:
    """The SSL context for the HTTPS port and the CA certificate to offer at
    /ca.crt (None for a certificate of your own), from the web: settings."""
    cert, key = (web.get('https_cert') or '').strip(), (web.get('https_key') or '').strip()
    if cert and key:
        return server_context(cert, key), None
    names = web.get('https_names') or []
    if isinstance(names, str):
        names = [names]
    crt, key, ca = ensure_certificate(os.path.join(data_dir, 'tls'), names)
    return server_context(crt, key), ca
