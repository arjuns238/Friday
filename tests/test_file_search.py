"""Unit tests for file search helpers (no ripgrep required)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from friday.file_search import (
    _expand_glob_candidates,
    _find_files_fuzzy,
    build_rg_search_argv,
    find_files_python,
    normalize_search_path,
    open_file_sync,
    resolve_search_directory,
    search_files_python,
)


class TestBuildRgSearchArgv(unittest.TestCase):
    def test_leading_dash_pattern_uses_dash_e(self) -> None:
        argv = build_rg_search_argv(
            "-foo",
            output_mode="content",
            glob=None,
            file_type=None,
            case_insensitive=False,
            context_lines=None,
            multiline=False,
        )
        self.assertEqual(argv[0], "rg")
        self.assertIn("-e", argv)
        self.assertEqual(argv[argv.index("-e") + 1], "-foo")
        self.assertEqual(argv[-1], ".")

    def test_files_with_matches_has_l(self) -> None:
        argv = build_rg_search_argv(
            "needle",
            output_mode="files_with_matches",
            glob="*.py",
            file_type=None,
            case_insensitive=True,
            context_lines=None,
            multiline=False,
        )
        self.assertIn("-l", argv)
        self.assertIn("-i", argv)
        self.assertIn("*.py", argv)

    def test_content_context(self) -> None:
        argv = build_rg_search_argv(
            "x",
            output_mode="content",
            glob=None,
            file_type="py",
            case_insensitive=False,
            context_lines=2,
            multiline=False,
        )
        self.assertIn("-n", argv)
        self.assertEqual(argv[argv.index("-C") + 1], "2")
        self.assertEqual(argv[argv.index("--type") + 1], "py")


class TestResolveSearchDirectory(unittest.TestCase):
    def test_requires_directory(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(ValueError):
                resolve_search_directory(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_resolves_existing_dir(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            r = resolve_search_directory(d)
            self.assertTrue(r.is_dir())


class TestPathNormalization(unittest.TestCase):
    def test_normalizes_downloads_alias(self) -> None:
        self.assertEqual(
            normalize_search_path("downloads"),
            str(Path.home() / "Downloads"),
        )

    def test_rewrites_other_user_home_folder(self) -> None:
        got = normalize_search_path("/Users/arjunsriram/Downloads")
        self.assertEqual(got, str(Path.home() / "Downloads"))


class TestGlobExpansion(unittest.TestCase):
    def test_space_name_expands_separator_variants(self) -> None:
        candidates = _expand_glob_candidates("*Arjun Sriram*")
        joined = "\n".join(candidates)
        self.assertIn("Arjun_Sriram", joined)
        self.assertIn("Arjun-Sriram", joined)
        self.assertIn("*Arjun*Sriram*", joined)

    def test_adds_common_extensions_when_missing(self) -> None:
        candidates = _expand_glob_candidates("Arjun Sriram")
        self.assertTrue(any(c.endswith(".pdf") for c in candidates))


class TestPythonFallback(unittest.TestCase):
    def test_search_files_content(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.txt").write_text("hello world\n", encoding="utf-8")
            out = search_files_python(
                root,
                r"world",
                output_mode="content",
                glob="*.txt",
                case_insensitive=False,
                head_limit=10,
                offset=0,
                context_lines=None,
                multiline=False,
            )
            self.assertIn("a.txt", out)
            self.assertIn("world", out)

    def test_find_files_glob(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "x.py").write_text("#", encoding="utf-8")
            (root / "y.md").write_text("", encoding="utf-8")
            out = find_files_python(root, "*.py", head_limit=20, offset=0)
            self.assertIn("x.py", out)
            self.assertNotIn("y.md", out)

    def test_find_files_matches_underscore_name(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "Arjun_Sriram.pdf").write_text("resume", encoding="utf-8")
            for pat in _expand_glob_candidates("Arjun Sriram"):
                out = find_files_python(root, pat, head_limit=20, offset=0)
                if "Arjun_Sriram.pdf" in out:
                    break
            else:
                self.fail("Expanded patterns did not match Arjun_Sriram.pdf")

    def test_fuzzy_finds_minor_name_drift(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "Arjun_Sriram.pdf").write_text("resume", encoding="utf-8")
            out = _find_files_fuzzy(root, "*Arjun Shriram*.pdf", head_limit=20, offset=0)
            self.assertIn("Arjun_Sriram.pdf", out)


class TestOpenFile(unittest.TestCase):
    def test_open_file_requires_existing_file(self) -> None:
        out = open_file_sync({"path": "/tmp/definitely-not-a-real-file-xyz.txt"})
        self.assertIn("Error: file does not exist:", out)

    def test_open_file_invokes_open_command(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "note.txt"
            f.write_text("hello", encoding="utf-8")
            with patch("friday.file_search.subprocess.run") as run_mock:
                run_mock.return_value.returncode = 0
                out = open_file_sync({"path": str(f), "reveal": False})
                self.assertIn("Opened:", out)
                run_mock.assert_called_once()
