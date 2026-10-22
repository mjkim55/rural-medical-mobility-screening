#!/usr/bin/env python3
"""로컬 최종 결과와 코드 ZIP이 같은 분석판인지 확인함. 업로드는 수행하지 않음."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent


class MapDocument(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_payload = False
        self.payload = []
        self.external_assets = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("id") == "analysis-data":
            self.in_payload = True
        if tag in {"script", "img", "iframe", "link"}:
            source = attrs.get("src") or attrs.get("href")
            if source:
                self.external_assets.append(source)

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_payload = False

    def handle_data(self, data):
        if self.in_payload:
            self.payload.append(data)


def main():
    config = json.loads((ROOT / "config.json").read_text())
    results = ROOT / config["output_dir"]
    summary = json.loads((results / "run_summary.json").read_text())
    assert summary["analysis_version"] == config["analysis_version"]
    assert summary["failed_error_checks"] == 0
    assert summary["all_cases_route_analysis_complete"]
    clean = json.loads((ROOT / "analysis.ipynb").read_text())
    assert all(not cell.get("outputs") and cell.get("execution_count") is None
               for cell in clean["cells"] if cell["cell_type"] == "code")
    executed = json.loads((ROOT / "analysis_executed.ipynb").read_text())
    cells = [cell for cell in executed["cells"] if cell["cell_type"] == "code"]
    assert cells and all(cell.get("execution_count") is not None for cell in cells)
    assert not any(out["output_type"] == "error" for cell in cells for out in cell.get("outputs", []))
    parser = MapDocument()
    map_path = results / "national_interactive_map.html"
    parser.feed(map_path.read_text())
    assert not parser.external_assets, "독립 지도에 외부 리소스가 있음"
    payload = json.loads("".join(parser.payload))
    assert payload["analysis_version"] == config["analysis_version"]
    assert len(payload["national"]) == summary["national_region_count"]
    for case in summary["case_status"]:
        assert len(payload["local"][case["case_slug"]]) == case["grid_count"]
    archive_path = ROOT.parent / f"{ROOT.name}_code_only.zip"
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert not name.startswith(("data/", "results/", "/"))
            assert ".." not in Path(name).parts
            assert not any(part in {"__pycache__", ".env"} for part in Path(name).parts)
            assert "executed" not in name and not name.endswith((".pyc", ".gpkg", ".xlsx"))
            content = archive.read(name)
            private_prefixes = [b"/" + b"Users" + b"/", b"/private" + b"/var/"]
            assert not any(prefix in content for prefix in private_prefixes), f"개인 경로: {name}"
            assert content == (ROOT / name).read_bytes(), f"최신 코드와 ZIP 불일치: {name}"
        file_count = len(archive.namelist())
    report = {"analysis_version": config["analysis_version"], "all_code_cells_executed": len(cells),
              "notebook_error_outputs": 0, "code_zip_file_count": file_count,
              "zip_crc_ok": True, "zip_matches_current_sources": True,
              "public_notebook_outputs_cleared": True, "map_external_assets": [],
              "national_map_regions": len(payload["national"]),
              "case_map_grid_counts": {key: len(value) for key, value in payload["local"].items()},
              "open_data_warning_checks": summary["open_warning_checks"],
              "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [archive_path, map_path, ROOT / "analysis_executed.ipynb"]}}
    (results / "release_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
