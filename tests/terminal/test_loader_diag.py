"""A Windows engine that cannot load a DLL ends with 0xC0000135 and no message. The start error must name the missing file
(tester report 2026-10-17: cublas64_13.dll was not in the engine component and the app blamed memory)."""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve import loader_diag  # noqa: E402
from serve.server import start_exit_note  # noqa: E402


def make_pe(imports, delay=()):
    """A minimal PE32+ file with a .idata section that imports the named DLLs (and delay-loads `delay`)."""
    va, raw = 0x1000, 0x400
    body = bytearray()
    descriptors = (len(imports) + 1) * 20
    delays = (len(delay) + 1) * 32
    names_at = descriptors + delays
    name_rvas = []
    blob = bytearray()
    for n in list(imports) + list(delay):
        name_rvas.append(va + names_at + len(blob))
        blob += n.encode("ascii") + b"\0"
    for i, _ in enumerate(imports):
        body += struct.pack("<IIIII", 0, 0, 0, name_rvas[i], 0)
    body += b"\0" * 20
    for j, _ in enumerate(delay):
        body += struct.pack("<II", 1, name_rvas[len(imports) + j]) + b"\0" * 24
    body += b"\0" * 32
    body += blob
    size = len(body)
    dos = bytearray(0x40)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x40)
    coff = b"PE\0\0" + struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 240, 0x22)
    opt = bytearray(240)
    struct.pack_into("<H", opt, 0, 0x20B)
    struct.pack_into("<II", opt, 112 + 8, va if imports else 0, descriptors if imports else 0)
    struct.pack_into("<II", opt, 112 + 13 * 8, va + descriptors if delay else 0, delays if delay else 0)
    section = b".idata\0\0" + struct.pack("<IIIIIIHHI", size, va, size, raw, 0, 0, 0, 0, 0x40000040)
    head = bytes(dos) + coff + bytes(opt) + section
    return head + b"\0" * (raw - len(head)) + bytes(body)


class Imports(unittest.TestCase):
    def test_pe_reader_lists_load_time_and_delay_load_imports(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.exe"
            p.write_bytes(make_pe(["cublas64_13.dll", "KERNEL32.dll"], delay=["cudart64_13.dll"]))
            self.assertEqual(loader_diag.pe_imports(p), ["cublas64_13.dll", "KERNEL32.dll", "cudart64_13.dll"])


class MissingFiles(unittest.TestCase):
    def folder(self, d, *, with_cublas):
        d = Path(d)
        (d / "strata.exe").write_bytes(make_pe(["cublas64_13.dll", "KERNEL32.dll", "api-ms-win-crt-runtime-l1-1-0.dll"]))
        (d / "strata_kq.dll").write_bytes(make_pe(["KERNEL32.dll"]))
        if with_cublas:
            (d / "cublas64_13.dll").write_bytes(make_pe(["cublasLt64_13.dll", "KERNEL32.dll"]))
            (d / "cublasLt64_13.dll").write_bytes(make_pe(["KERNEL32.dll"]))
        return d / "strata.exe"

    def test_names_the_missing_dll(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as sysdir:
            exe = self.folder(d, with_cublas=False)
            (Path(sysdir) / "KERNEL32.dll").write_bytes(b"MZ")
            self.assertEqual(loader_diag.missing_files(exe, [Path(sysdir)]), ["cublas64_13.dll"])

    def test_transitive_import_is_named(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as sysdir:
            exe = self.folder(d, with_cublas=True)
            (exe.parent / "cublasLt64_13.dll").unlink()
            (Path(sysdir) / "KERNEL32.dll").write_bytes(b"MZ")
            self.assertEqual(loader_diag.missing_files(exe, [Path(sysdir)]), ["cublasLt64_13.dll"])

    def test_complete_folder_names_nothing(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as sysdir:
            exe = self.folder(d, with_cublas=True)
            (Path(sysdir) / "KERNEL32.dll").write_bytes(b"MZ")
            self.assertEqual(loader_diag.missing_files(exe, [Path(sysdir)]), [])

    def test_unreadable_file_never_raises(self):
        self.assertEqual(loader_diag.missing_files(Path("no-such-folder") / "strata.exe"), [])

    def test_start_note_only_for_the_loader_status(self):
        with tempfile.TemporaryDirectory() as d:
            exe = self.folder(d, with_cublas=False)
            note = start_exit_note(0xC0000135, exe)
            self.assertIn("0xC0000135", note)
            self.assertIn("cublas64_13.dll", note)
            self.assertIn("Reinstall the engine in Settings > Engine", note)
            self.assertNotIn("cublas64_13.dll", start_exit_note(1, exe))
            self.assertNotIn("missing a file", start_exit_note(0xC0000135, None))


if __name__ == "__main__":
    unittest.main()
