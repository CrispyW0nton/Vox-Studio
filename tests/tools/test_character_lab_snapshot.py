"""Verify recovery source bytes and syntax without importing any lab module."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import unittest

ROOT = Path(__file__).resolve().parents[2] / 'tools/character_lab'


class CharacterLabSourceTests(unittest.TestCase):
    def test_complete_source_snapshot(self):
        manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['schemaVersion'], 1)
        self.assertEqual(len(manifest['files']), 63)
        names = set()
        for entry in manifest['files']:
            name = PurePosixPath(entry['path'])
            self.assertFalse(name.is_absolute())
            self.assertNotIn('..', name.parts)
            self.assertEqual(name.suffix, '.py')
            self.assertNotIn(str(name), names)
            names.add(str(name))
            path = ROOT / str(name)
            self.assertFalse(path.is_symlink())
            raw = path.read_bytes()
            self.assertEqual(len(raw), entry['bytes'], str(name))
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry['sha256'], str(name))
            ast.parse(raw.decode('utf-8-sig'), str(name))
        actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.py')}
        self.assertEqual(actual, names)


if __name__ == '__main__':
    unittest.main()
