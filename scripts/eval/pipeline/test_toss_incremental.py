"""The merge only ever appends: old rows untouched, ids never duplicated."""

import json

import toss_incremental as ti


def _rows(*ids_dates):
    return [{"id": i, "createdAt": f"{d}T10:00:00", "title": i} for i, d in ids_dates]


def _write(path, rows, trailing_newline=True):
    text = "\n".join(json.dumps(r) for r in rows)
    path.write_text(text + ("\n" if trailing_newline else ""))


def test_appends_only_new_ids_and_keeps_old_rows_byte_identical(tmp_path, monkeypatch):
    arch, stage = tmp_path / "arch", tmp_path / "stage"
    arch.mkdir()
    stage.mkdir()
    monkeypatch.setattr(ti, "ARCHIVE", arch)
    _write(arch / "000001.jsonl", _rows(("a", "2026-09-15"), ("b", "2026-09-16")))
    before = (arch / "000001.jsonl").read_text()
    _write(stage / "000001.jsonl", _rows(("b", "2026-09-16"), ("c", "2026-09-17")))

    added = ti.merge(stage, ["000001"])

    after = (arch / "000001.jsonl").read_text()
    assert after.startswith(before)
    ids = [json.loads(line)["id"] for line in after.splitlines()]
    assert ids == ["a", "b", "c"]
    assert dict(added) == {"2026-09-17": 1}


def test_archive_file_without_trailing_newline_is_not_corrupted(tmp_path, monkeypatch):
    arch, stage = tmp_path / "arch", tmp_path / "stage"
    arch.mkdir()
    stage.mkdir()
    monkeypatch.setattr(ti, "ARCHIVE", arch)
    _write(arch / "000002.jsonl", _rows(("a", "2026-09-15")), trailing_newline=False)
    _write(stage / "000002.jsonl", _rows(("b", "2026-09-16")))
    ti.merge(stage, ["000002"])
    lines = (arch / "000002.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["a", "b"]


def test_a_code_new_to_the_archive_gets_a_file(tmp_path, monkeypatch):
    arch, stage = tmp_path / "arch", tmp_path / "stage"
    arch.mkdir()
    stage.mkdir()
    monkeypatch.setattr(ti, "ARCHIVE", arch)
    _write(stage / "000003.jsonl", _rows(("x", "2026-09-18")))
    assert dict(ti.merge(stage, ["000003"])) == {"2026-09-18": 1}
    assert (arch / "000003.jsonl").exists()


def test_rerunning_the_same_staging_adds_nothing(tmp_path, monkeypatch):
    arch, stage = tmp_path / "arch", tmp_path / "stage"
    arch.mkdir()
    stage.mkdir()
    monkeypatch.setattr(ti, "ARCHIVE", arch)
    _write(stage / "000004.jsonl", _rows(("x", "2026-09-18"), ("y", "2026-09-18")))
    ti.merge(stage, ["000004"])
    first = (arch / "000004.jsonl").read_text()
    assert dict(ti.merge(stage, ["000004"])) == {}
    assert (arch / "000004.jsonl").read_text() == first
