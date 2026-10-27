"""Passphrase encryption for scans and results stored on Kaggle.

Used by the app (encrypt scans before upload, decrypt results after download) and by the
Kaggle notebook (decrypt scans, encrypt results). Needs only the `cryptography` package.
The Decrypter app keeps an identical copy: change both together.

File format (all integers big-endian):
    MAGIC (8) | salt (16) | nonce prefix (8) | chunks...
    chunk = ciphertext length (4) | AES-256-GCM ciphertext + tag

The key is scrypt(passphrase, salt). Chunk i uses nonce = prefix + i (4 bytes) and the header,
the index and a "last chunk" flag as associated data, so a wrong key, tampering, reordering
or truncation all fail loudly instead of producing garbage.
"""
import hashlib
import os
import struct
from pathlib import Path

MAGIC = b"MCSENC1\n"
CHUNK = 4 * 1024 * 1024
EXTENSION = ".mcsenc"
# Name of the key file inside the temporary per-run key dataset on Kaggle.
KEY_FILE = "mcs-key.txt"


class DecryptError(Exception):
    pass


def fingerprint(passphrase: str) -> str:
    """Short, non-secret identifier of a key: lets the notebook check it has the same key
    as the app without either side revealing it."""
    return hashlib.sha256(b"mcs-fingerprint:" + passphrase.encode("utf-8")).hexdigest()[:12]


def _derive(passphrase: str, salt: bytes) -> bytes:
    return hashlib.scrypt(passphrase.encode("utf-8"), salt=salt, n=2 ** 15, r=8, p=1, maxmem=64 * 1024 * 1024, dklen=32)


def _aad(header: bytes, index: int, last: bool) -> bytes:
    return header + struct.pack(">I?", index, last)


def encrypt_file(src, dst, passphrase: str, progress=None):
    """progress, if given, is called with (bytes done, total bytes) after each chunk."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not passphrase:
        raise ValueError("An encryption key is required.")
    salt, prefix = os.urandom(16), os.urandom(8)
    header = MAGIC + salt + prefix
    aes = AESGCM(_derive(passphrase, salt))
    size = Path(src).stat().st_size
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(header)
        index, done = 0, 0
        while True:
            block = fin.read(CHUNK)
            done += len(block)
            last = done >= size
            ct = aes.encrypt(prefix + struct.pack(">I", index), block, _aad(header, index, last))
            fout.write(struct.pack(">I", len(ct)) + ct)
            if progress:
                progress(done, size)
            index += 1
            if last:
                break


def decrypt_file(src, dst, passphrase: str, progress=None):
    """progress, if given, is called with (bytes done, total bytes) after each chunk."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    size = Path(src).stat().st_size
    with open(src, "rb") as fin:
        header = fin.read(len(MAGIC) + 24)
        if len(header) < len(MAGIC) + 24 or not header.startswith(MAGIC):
            raise DecryptError("Not an encrypted Manga Color archive.")
        salt, prefix = header[len(MAGIC):len(MAGIC) + 16], header[len(MAGIC) + 16:]
        aes = AESGCM(_derive(passphrase, salt))
        tmp = Path(str(dst) + ".part")
        try:
            with open(tmp, "wb") as fout:
                index = 0
                while True:
                    raw_len = fin.read(4)
                    if len(raw_len) < 4:
                        raise DecryptError("The encrypted archive is truncated.")
                    (n,) = struct.unpack(">I", raw_len)
                    ct = fin.read(n)
                    nonce = prefix + struct.pack(">I", index)
                    last = fin.tell() >= size
                    try:
                        block = aes.decrypt(nonce, ct, _aad(header, index, last))
                    except InvalidTag:
                        if index == 0:
                            raise DecryptError("Wrong encryption key (or the archive is damaged).")
                        raise DecryptError("The encrypted archive is damaged or was tampered with.")
                    fout.write(block)
                    if progress:
                        progress(fin.tell(), size)
                    index += 1
                    if last:
                        break
            tmp.replace(dst)
        finally:
            if tmp.exists():
                tmp.unlink()
