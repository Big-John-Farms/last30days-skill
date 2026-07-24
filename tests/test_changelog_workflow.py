"""Contract tests for towncrier + lockstep release preparation."""

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_prepare_release():
    path = ROOT / ".github" / "scripts" / "prepare_release.py"
    spec = importlib.util.spec_from_file_location("prepare_release", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestChangelogWorkflow(unittest.TestCase):
    def test_towncrier_config_present(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("[tool.towncrier]", text)
        self.assertIn('directory = "changelog.d"', text)
        self.assertIn('filename = "CHANGELOG.md"', text)
        for fragment_type in (
            "security",
            "removed",
            "deprecated",
            "added",
            "changed",
            "fixed",
        ):
            self.assertIn(f'directory = "{fragment_type}"', text)

    def test_changelog_has_towncrier_start_marker(self) -> None:
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("<!-- towncrier release notes start -->", text)
        self.assertNotIn("## [Unreleased]", text)

    def test_changelog_d_readme_exists(self) -> None:
        self.assertTrue((ROOT / "changelog.d" / "README.md").is_file())

    def test_prepare_release_script_exists(self) -> None:
        self.assertTrue((ROOT / ".github" / "scripts" / "prepare_release.py").is_file())

    def test_release_workflows_exist(self) -> None:
        workflows = ROOT / ".github" / "workflows"
        for name in (
            "prepare-release.yml",
            "tag-release.yml",
            "changelog-guard.yml",
        ):
            self.assertTrue((workflows / name).is_file(), msg=name)

    def test_pr_template_has_agent_and_relationship_sections(self) -> None:
        text = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("## Agent disclosure", text)
        self.assertIn("### Relationship to this change", text)
        self.assertIn("changelog.d/", text)
        self.assertIn("What does this PR do?", text)
        self.assertTrue((ROOT / "CONTRIBUTING.md").is_file())

    def test_next_version_bumps(self) -> None:
        mod = _load_prepare_release()
        self.assertEqual(mod.next_version("3.18.1", "patch"), "3.18.2")
        self.assertEqual(mod.next_version("3.18.1", "minor"), "3.19.0")
        self.assertEqual(mod.next_version("3.18.1", "major"), "4.0.0")

    def test_bump_all_updates_lockstep_surfaces(self) -> None:
        mod = _load_prepare_release()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Minimal fixtures mirroring the lockstep layout.
            (tmp_path / "skills" / "last30days").mkdir(parents=True)
            (tmp_path / ".claude-plugin").mkdir()
            (tmp_path / ".codex-plugin").mkdir()
            (tmp_path / ".grok-plugin").mkdir()

            (tmp_path / "pyproject.toml").write_text(
                '[project]\nname = "last30days-skill"\nversion = "3.18.1"\n',
                encoding="utf-8",
            )
            (tmp_path / "skills" / "last30days" / "SKILL.md").write_text(
                '---\nversion: "3.18.1"\n---\n\n# last30days v3.18.1: Title\n',
                encoding="utf-8",
            )
            for rel in (
                ".claude-plugin/plugin.json",
                ".codex-plugin/plugin.json",
                ".grok-plugin/plugin.json",
                "gemini-extension.json",
            ):
                (tmp_path / rel).write_text(
                    json.dumps({"name": "last30days", "version": "3.18.1"}, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
            for rel in (
                ".claude-plugin/marketplace.json",
                ".grok-plugin/marketplace.json",
            ):
                (tmp_path / rel).write_text(
                    json.dumps(
                        {
                            "name": "last30days-skill",
                            "plugins": [{"name": "last30days", "version": "3.18.1"}],
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            (tmp_path / "uv.lock").write_text(
                'version = 1\n\n[[package]]\nname = "last30days-skill"\n'
                'version = "3.18.1"\nsource = { virtual = "." }\n',
                encoding="utf-8",
            )

            # Point module paths at the temp tree.
            mod.ROOT = tmp_path
            mod.SKILL_MD = tmp_path / "skills" / "last30days" / "SKILL.md"
            mod.PYPROJECT = tmp_path / "pyproject.toml"
            mod.UV_LOCK = tmp_path / "uv.lock"
            mod.JSON_VERSION_FILES = (
                tmp_path / ".claude-plugin" / "plugin.json",
                tmp_path / ".codex-plugin" / "plugin.json",
                tmp_path / ".grok-plugin" / "plugin.json",
                tmp_path / "gemini-extension.json",
            )
            mod.MARKETPLACE_FILES = (
                tmp_path / ".claude-plugin" / "marketplace.json",
                tmp_path / ".grok-plugin" / "marketplace.json",
            )

            touched = mod.bump_all("9.9.9")
            self.assertEqual(len(touched), 9)

            pyproject = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
            self.assertIn('version = "9.9.9"', pyproject)

            skill = (tmp_path / "skills" / "last30days" / "SKILL.md").read_text(
                encoding="utf-8"
            )
            self.assertIn('version: "9.9.9"', skill)
            self.assertIn("# last30days v9.9.9:", skill)

            for rel in (
                ".claude-plugin/plugin.json",
                ".codex-plugin/plugin.json",
                ".grok-plugin/plugin.json",
                "gemini-extension.json",
            ):
                data = json.loads((tmp_path / rel).read_text(encoding="utf-8"))
                self.assertEqual(data["version"], "9.9.9", msg=rel)

            for rel in (
                ".claude-plugin/marketplace.json",
                ".grok-plugin/marketplace.json",
            ):
                data = json.loads((tmp_path / rel).read_text(encoding="utf-8"))
                self.assertEqual(data["plugins"][0]["version"], "9.9.9", msg=rel)

            uv_lock = (tmp_path / "uv.lock").read_text(encoding="utf-8")
            self.assertRegex(
                uv_lock,
                re.compile(
                    r'(?ms)^\[\[package\]\]\nname = "last30days-skill"\nversion = "9\.9\.9"',
                ),
            )


if __name__ == "__main__":
    unittest.main()
