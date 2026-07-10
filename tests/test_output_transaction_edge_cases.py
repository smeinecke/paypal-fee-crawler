from __future__ import annotations

import os
from pathlib import Path

import pytest

from paypal_fee_crawler.exceptions import ValidationError as CrawlerValidationError
from paypal_fee_crawler.output import OutputPublisher, _JournalEntry


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_rollback_restores_live_after_backup_when_install_fails(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "json" / "old.json", "old")
    backup = root / "json.old"

    publisher = OutputPublisher(root)
    entry = _JournalEntry(
        managed_name="json",
        live_path=root / "json",
        backup_path=backup,
        staged_path=tmp_path / "staging" / "json",
        action="replaced",
        original_existed=True,
    )

    # Simulate: live path was already moved to backup, but staged->live failed
    # before the old code appended a journal entry / marked swapped=True.
    os.rename(entry.live_path, entry.backup_path)
    entry.backup_created = True
    entry.live_installed = False

    publisher._rollback_live([entry])

    assert (root / "json" / "old.json").read_text(encoding="utf-8") == "old"
    assert not backup.exists()


def test_rollback_restores_removed_path_even_without_live_install(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "change-report.json", "old report")
    backup = root / "change-report.json.old"

    publisher = OutputPublisher(root)
    entry = _JournalEntry(
        managed_name="change-report.json",
        live_path=root / "change-report.json",
        backup_path=backup,
        staged_path=None,
        action="removed",
        original_existed=True,
    )

    # Simulate removing an obsolete managed file before a later managed path
    # fails. There is no staged replacement, so live_installed remains False.
    os.rename(entry.live_path, entry.backup_path)
    entry.backup_created = True

    publisher._rollback_live([entry])

    assert (root / "change-report.json").read_text(encoding="utf-8") == "old report"
    assert not backup.exists()


def test_commit_does_not_replace_repository_root_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    staging = tmp_path / "repo" / ".staging-test"

    _write(root / ".git" / "sentinel", "git")
    _write(root / ".github" / "workflows" / "update.yml", "workflow")
    _write(root / "README.md", "readme")
    _write(root / "LICENSE", "license")
    _write(root / "crawler" / "sentinel", "crawler")
    _write(root / "json" / "old.json", "old-json")

    _write(staging / "json" / "new.json", "new-json")
    _write(staging / "meta" / "dummy.json", "meta")
    _write(staging / "schemas" / "dummy.json", "schema")

    # Focus this test on transaction behavior, not JSON schema construction.
    monkeypatch.setattr("paypal_fee_crawler.output.validate_output_tree", lambda _root: [])

    real_rename = os.rename
    calls = {"count": 0}

    def failing_rename(src: Path | str, dst: Path | str) -> None:
        calls["count"] += 1
        # First rename backs up json; second rename tries to install new json.
        if calls["count"] == 2:
            raise OSError("injected staged-install failure")
        real_rename(src, dst)

    monkeypatch.setattr("paypal_fee_crawler.output.os.rename", failing_rename)

    publisher = OutputPublisher(root)
    with pytest.raises(CrawlerValidationError):
        publisher.commit(staging)

    assert (root / "json" / "old.json").read_text(encoding="utf-8") == "old-json"
    assert not (root / "json.old").exists()
    assert (root / ".git" / "sentinel").read_text(encoding="utf-8") == "git"
    assert (root / ".github" / "workflows" / "update.yml").read_text(encoding="utf-8") == "workflow"
    assert (root / "README.md").read_text(encoding="utf-8") == "readme"
    assert (root / "LICENSE").read_text(encoding="utf-8") == "license"
    assert (root / "crawler" / "sentinel").read_text(encoding="utf-8") == "crawler"


def test_rollback_leaves_original_live_untouched_if_no_mutation_happened(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write(root / "json" / "old.json", "original")
    backup = root / "json.old"

    publisher = OutputPublisher(root)
    entry = _JournalEntry(
        managed_name="json",
        live_path=root / "json",
        backup_path=backup,
        staged_path=tmp_path / "staging" / "json",
        action="replaced",
        original_existed=True,
        backup_created=False,
        live_installed=False,
    )

    publisher._rollback_live([entry])

    assert (root / "json" / "old.json").read_text(encoding="utf-8") == "original"
    assert not backup.exists()


def test_commit_preserves_original_live_when_backup_rename_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    staging = tmp_path / "staging"

    _write(root / "json" / "old.json", "original")
    _write(staging / "json" / "new.json", "new-json")
    _write(staging / "meta" / "dummy.json", "meta")
    _write(staging / "schemas" / "dummy.json", "schema")

    monkeypatch.setattr("paypal_fee_crawler.output.validate_output_tree", lambda _root: [])

    real_rename = os.rename

    def failing_rename(src: Path | str, dst: Path | str) -> None:
        if str(dst) == str(root / "json.old"):
            raise OSError("injected backup rename failure")
        real_rename(src, dst)

    monkeypatch.setattr("paypal_fee_crawler.output.os.rename", failing_rename)

    publisher = OutputPublisher(root)
    with pytest.raises(CrawlerValidationError):
        publisher.commit(staging)

    assert (root / "json" / "old.json").read_text(encoding="utf-8") == "original"
    assert not (root / "json.old").exists()


def test_commit_restores_backup_when_install_fails_after_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    staging = tmp_path / "staging"

    _write(root / "json" / "old.json", "original")
    _write(staging / "json" / "new.json", "new-json")
    _write(staging / "meta" / "dummy.json", "meta")
    _write(staging / "schemas" / "dummy.json", "schema")

    monkeypatch.setattr("paypal_fee_crawler.output.validate_output_tree", lambda _root: [])

    real_rename = os.rename
    calls = {"count": 0}

    def failing_rename(src: Path | str, dst: Path | str) -> None:
        calls["count"] += 1
        # First rename backs up json; second rename tries to install new json.
        if calls["count"] == 2:
            raise OSError("injected staged-install failure")
        real_rename(src, dst)

    monkeypatch.setattr("paypal_fee_crawler.output.os.rename", failing_rename)

    publisher = OutputPublisher(root)
    with pytest.raises(CrawlerValidationError):
        publisher.commit(staging)

    assert (root / "json" / "old.json").read_text(encoding="utf-8") == "original"
    assert not (root / "json.old").exists()
