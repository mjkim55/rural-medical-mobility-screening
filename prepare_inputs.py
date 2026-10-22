#!/usr/bin/env python3
"""로컬 원자료를 분석 입력으로 준비한다. 네트워크·로그인·공개 업로드는 수행하지 않는다.

예: python prepare_inputs.py --project-root /path/to/original-project --downloads /path/to/downloads
--mode raw: KTDB 원본 3개와 행안부 원본을 다시 피벗·결합한다.
--mode prepared: 기존 전국 결합표에서 원변수만 가져온다. 점수·군집·등급은 읽지 않는다.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd

from mobility.common import csv, json_file, require_unique, sha256
from mobility.raw import PRIMARY_TYPES, build_accessibility_table, join_population, load_hira, make_emd_wide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--downloads", required=True, type=Path)
    parser.add_argument("--mode", choices=["raw", "prepared"], default="raw")
    parser.add_argument("--destination", type=Path, default=Path(__file__).parent / "data/inputs")
    args = parser.parse_args()
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    project, downloads = args.project_root, args.downloads
    manifest = []

    def record(path, role, period, origin):
        manifest.append({"file_name": path.name, "role": role, "period": period,
                         "origin": origin, "sha256": sha256(path), "bytes": path.stat().st_size})

    boundary_source = downloads / "bnd_dong_00_2023_4Q.zip"
    boundary = gpd.read_file(f"zip://{boundary_source}").to_crs(5179)
    boundary["HDCD"] = boundary["ADM_CD"].astype("string")
    require_unique(boundary, ["HDCD"], "SGIS 읍면동")
    boundary = boundary[["HDCD", "ADM_NM", "geometry"]]
    boundary.to_file(dest / "national_boundary.gpkg", layer="boundary", driver="GPKG")
    record(boundary_source, "읍면동 경계", "2023-4Q", "SGIS 원본")

    if args.mode == "raw":
        ktdb_dir = project / "tmp/mobility_access/ktdb_2023"
        access = build_accessibility_table(ktdb_dir)
        csv(access, dest / "ktdb_facility_mode_pairs.csv")
        national = join_population(make_emd_wide(access), downloads / "downloadExcelAge2 (2).xls")
        area = boundary[["HDCD"]].copy()
        area["area_km2"] = boundary.geometry.area / 1_000_000
        national = national.merge(area, on="HDCD", how="left", validate="one_to_one")
        for path in sorted(ktdb_dir.glob("*.xlsx")):
            record(path, "KTDB 복합키·교통수단 원자료", "2023", "KTDB 원본")
        record(downloads / "downloadExcelAge2 (2).xls", "주민등록 고령·전체 인구", "2023-12", "행정안전부 원본")
    else:
        path = project / "deliverables/national_model/results/2023_전국_읍면동_이동인구취약성_분석.csv"
        source = pd.read_csv(path, dtype={"HDCD": "string"}, low_memory=False)
        names = ["HDCD", "HDCD_SD_NM", "HDCD_SGG_NM", "HDCD_EMD_NM", "total_population", "elderly_population", "area_km2"]
        metric_columns = [col for col in source if col[:2] in {"21", "22", "23", "31", "32", "41", "42"} and "facility_name" not in col]
        national = source[names + metric_columns].copy()
        record(path, "점수·군집을 제외한 결합 원변수", "2023", "기존 결합표; 원자료에서 재결합한 실행은 아님")
    csv(national, dest / "national_base.csv")

    hira_dir = project / "tmp/mobility_access/hira_2025_12/전국 병의원 및 약국 현황 2025.12"
    for file_name, sheet, output in [
        ("1.병원정보서비스(2025.12.).xlsx", "hospBasisList", "primary_care"),
        ("2.약국정보서비스(2025.12.).xlsx", "parmacyBasisList", "pharmacy"),
    ]:
        path = hira_dir / file_name
        points, invalid = load_hira(path, sheet)
        csv(invalid, dest / f"{output}_invalid_coordinates.csv")
        if output == "primary_care":
            points = points.loc[points["종별코드명"].isin(PRIMARY_TYPES)].copy()
        points.to_file(dest / f"{output}.gpkg", layer=output, driver="GPKG")
        record(path, f"HIRA {output}", "2025-12", "건강보험심사평가원 원본")

    grid_source = downloads / "iksan_grid_access_v2.gpkg"
    raw_grid = gpd.read_file(grid_source, layer="iksan_access").to_crs(5179)
    # 공급·거리·정책 열을 가져오지 않고 SGIS 인구와 격자 공간정보만 보존한다.
    keep = ["grid_id", "total_population", "elderly_population", "population_status", "population_year", "geometry"]
    raw_grid[keep].to_file(dest / "iksan_population.gpkg", layer="population", driver="GPKG")
    record(grid_source, "SGIS 격자인구·공간정보", "2024", "윤지 제공 파일의 iksan_access 레이어")
    # 경계묶음은 파일명을 재해석하지 않고 확인된 SHP를 GeoPandas로 읽는다.
    path = downloads / "drive-download-20261002T065454Z-1-001.zip"
    local_boundary = gpd.read_file(f"zip://{path}").to_crs(5179)
    local_boundary["HDCD"] = local_boundary["ADM_CD"].astype("string")
    local_boundary[["HDCD", "ADM_NM", "geometry"]].to_file(dest / "iksan_boundary.gpkg", layer="boundary", driver="GPKG")
    record(path, "익산 읍면동 경계", "2025-2Q (파일 기준, 기존 표의 2024 표기와 다름)", "SGIS 경계 제공묶음")
    for source_name, target in [("익산시_버스정류장 (1).csv", "iksan_stops.csv"), ("익산시_TAGO_버스자료.xlsx", "iksan_tago.xlsx")]:
        path = downloads / source_name
        shutil.copy2(path, dest / target)
        record(path, "익산 버스 정류장·노선", "수집 2026-10, 통계기준일 미확인", "팀 제공 파일; 정류장 원 출처 문서 추가 확인 필요")
    csv(pd.read_csv(downloads / "2024_1K_격자_인구_분석용.csv"), dest / "national_grid_population_2024.csv")
    record(downloads / "2024_1K_격자_인구_분석용.csv", "전국 격자인구 표(경계 별도 필요)", "2024", "SGIS, 팀 제공")
    csv(pd.DataFrame(manifest), dest / "input_manifest.csv")
    json_file({"mode": args.mode, "network_used": False, "original_files_modified": False,
               "notice": "공통·익산 입력 준비 단계. 추가지역 상태는 prepare_additional_cases.py 및 실행 결과 case_completion_status.csv에서 확인."}, dest / "preparation_status.json")
    print(f"공통·익산 입력 준비 완료: {dest}. 추가지역은 별도 준비·검증 단계에서 확인.")


if __name__ == "__main__":
    main()
