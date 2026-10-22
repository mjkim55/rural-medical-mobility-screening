#!/usr/bin/env python3
"""원자료·결과·인증정보를 제외한 GitHub 업로드 준비 ZIP 생성. 외부 업로드 안 함."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent


def main():
    destination = ROOT.parent / f"{ROOT.name}_code_only.zip"
    allowed = ["mobility", "tests", "docs", ".github"]
    files = [ROOT / name for name in ["README.md", "config.json", "requirements.txt", ".gitignore", "analysis.ipynb", "run_analysis.py", "prepare_inputs.py", "prepare_additional_cases.py", "collect_tago_routes.py", "audit_tago_downloads.py", "build_notebook.py", "package_code.py", "verify_release.py"]]
    for directory in allowed:
        files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if not path.exists():
                raise FileNotFoundError(f"필수 코드 파일 없음: {path.name}")
            archive.write(path, path.relative_to(ROOT))
    print(f"코드 전용 ZIP 생성: {destination.name}. GitHub 업로드는 수행하지 않았음.")


if __name__ == "__main__":
    main()
