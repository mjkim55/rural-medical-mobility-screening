#!/usr/bin/env python3
"""공식 SGIS 2025 격자 ZIP과 보유한 2024 인구표로 추가 사례 입력을 만든다.

네트워크·로그인·신청 제출은 하지 않는다. 원본 ZIP과 인구표를 수정하지 않는다.
실제 ZIP의 GRID_CD를 확인해 grid_id와 일대일 연결하며 좌표를 추정하지 않는다.
인구표에 없는 격자를 0명으로 대체하지 않는다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from mobility.common import csv, json_file, read_config, require_columns, require_unique, sha256


def join_grid_population(boundaries, population):
    """원자료 필드명을 검증한 뒤 결합. 결합 누락과 0명은 별도로 보존한다."""
    require_columns(boundaries, ["GRID_CD", "geometry"], "SGIS 공식 격자")
    require_columns(population, ["year", "grid_id", "total_population", "elderly_population"], "SGIS 인구표")
    require_unique(boundaries, ["GRID_CD"], "SGIS 격자")
    require_unique(population, ["grid_id"], "SGIS 인구표")
    if set(pd.to_numeric(population["year"], errors="raise")) != {2024}:
        raise ValueError("이 어댑터가 검증한 인구표는 2024년 단일 자료임")
    geometry = boundaries[["GRID_CD", "geometry"]].rename(columns={"GRID_CD": "grid_id"})
    joined = geometry.merge(
        population[["grid_id", "year", "total_population", "elderly_population"]],
        on="grid_id", how="left", validate="one_to_one", indicator=True,
    )
    joined["population_record_present"] = joined.pop("_merge").eq("both")
    joined["population_year"] = joined.pop("year").astype("Int64")
    joined["grid_boundary_year"] = 2025
    total = pd.to_numeric(joined["total_population"], errors="coerce")
    joined["population_status"] = np.select(
        [~joined["population_record_present"], total.isna(), total.lt(0), total.eq(0)],
        ["인구표 미수록", "인구값 결측", "인구값 음수", "인구 0명"], default="유효",
    )
    # 미수록에는 실제 무인·비밀보호·누락 등이 섞일 수 있어 원인을 단정하지 않는다.
    return gpd.GeoDataFrame(joined, geometry="geometry", crs=boundaries.crs)


def official_stop_points(path, audit_dir):
    """15067528의 실제 CP949 원본 열을 공통 입력 계약에 명시적으로 매핑한다."""
    raw = pd.read_csv(path, encoding="cp949", dtype="string")
    require_columns(raw, ["정류장번호", "정류장명", "위도", "경도", "정보수집일", "도시코드", "도시명"], "국토부 정류장 CSV")
    require_unique(raw, ["정류장번호"], "국토부 정류장 CSV")
    if set(raw["정보수집일"].dropna()) != {"2025-10-31"}:
        raise ValueError("다른 수집일 원본은 별도 검증 후 사용해야 함")
    stops = raw.rename(columns={"정류장번호": "nodeid", "정류장명": "nodenm", "경도": "gpslong", "위도": "gpslati"}).copy()
    stops["gpslong"] = pd.to_numeric(stops["gpslong"], errors="coerce")
    stops["gpslati"] = pd.to_numeric(stops["gpslati"], errors="coerce")
    valid = stops["gpslong"].between(124, 132) & stops["gpslati"].between(33, 39)
    # 위경도 뒤바뀜처럼 보여도 임의 교정하지 않고 원행을 감사표에 남긴다.
    csv(raw.loc[~valid], audit_dir / "national_stop_invalid_coordinates.csv")
    points = gpd.GeoDataFrame(stops.loc[valid].reset_index(drop=True),
                              geometry=gpd.points_from_xy(stops.loc[valid, "gpslong"], stops.loc[valid, "gpslati"]),
                              crs=4326).to_crs(5179)
    return points, {"raw_rows": len(raw), "valid_coordinate_rows": int(valid.sum()),
                    "invalid_coordinate_rows": int((~valid).sum()), "snapshot": "2025-10-31",
                    "metadata_row_count": 206022,
                    "metadata_row_count_matches_actual": len(raw) == 206022,
                    "official_url": "https://www.data.go.kr/data/15067528/fileData.do",
                    "file_name": path.name, "sha256": sha256(path),
                    "coverage_note": "TAGO 연계 지자체 자료임. 지역별 정류장 전수·현재 운행은 보증하지 않음. SGIS 코드와 도시코드 직접 결합 금지."}


def prepare(config_path, zip_paths, stops_path=None):
    config = read_config(config_path)
    inputs = config_path.parent / config["input_dir"]
    audit_dir = inputs / "additional_case_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    population_path = inputs / "national_grid_population_2024.csv"
    population = pd.read_csv(population_path, dtype={"grid_id": "string"})
    boundary = gpd.read_file(inputs / "national_boundary.gpkg").to_crs(5179)
    stops, stop_manifest, stop_tree = None, None, None
    if stops_path:
        stops, stop_manifest = official_stop_points(stops_path, audit_dir)
        stop_tree = cKDTree(np.c_[stops.geometry.x, stops.geometry.y])
        raw_dir = config_path.parent / "data/raw/official_stops"
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stops_path, raw_dir / "national_bus_stops_20251031.csv")
    parts, manifest = [], []
    for path in zip_paths:
        with ZipFile(path) as archive:
            invalid = archive.testzip()
            if invalid:
                raise ValueError(f"ZIP 무결성 오류: {path.name}, {invalid}")
            members = [name for name in archive.namelist() if name.endswith("_1K.shp")]
            if len(members) != 1:
                raise ValueError(f"1K SHP가 한 개가 아님: {path.name}: {members}")
        frame = gpd.read_file(f"/vsizip/{path.resolve()}/{members[0]}")
        require_columns(frame, ["GRID_CD", "geometry"], path.name)
        if frame.crs is None:
            raise ValueError("공식 격자 좌표계 누락")
        frame = frame.to_crs(5179)
        if (frame.geometry.isna().any() or frame.geometry.is_empty.any()
                or not frame.geometry.is_valid.all()
                or not np.allclose(frame.geometry.area, 1_000_000, atol=.01)):
            raise ValueError(f"1km 폴리곤 무결성 오류: {path.name}")
        parts.append(frame[["GRID_CD", "geometry"]])
        manifest.append({"file_name": path.name, "sha256": sha256(path),
                         "bytes": path.stat().st_size, "grid_boundary_year": 2025,
                         "one_k_grid_count": len(frame), "crs": str(frame.crs),
                         "official_url": "https://sgis.mods.go.kr/view/pss/downloadList",
                         "usage_condition": "출처 명시 및 연구 완료 후 결과물 사본 또는 URL 제출"})
    polygons = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=5179)
    require_unique(polygons, ["GRID_CD"], "4개 타일 결합")
    rows = []
    for case in config["cases"]:
        if not case.get("HDCD"):
            continue
        emd = boundary.loc[boundary["HDCD"].eq(case["HDCD"])]
        if len(emd) != 1:
            raise ValueError(f"사례 읍면동 코드가 한 개가 아님: {case['HDCD']}")
        area = emd.geometry.iloc[0]
        # 다른 분석과 동일한 중심점 기준. 걸친 격자를 자르거나 인구를 면적배분하지 않는다.
        selected = polygons.loc[polygons.geometry.centroid.within(area)].copy()
        if selected.empty:
            raise ValueError(f"해당 사례 격자 없음: {case['name']}")
        grid = join_grid_population(selected, population)
        grid["emd_boundary_period"] = case["boundary_year"]
        grid["HDCD_source"] = case["HDCD"]
        grid.to_file(inputs / case["population_file"], layer="population", driver="GPKG")
        csv(grid.drop(columns="geometry"), audit_dir / f"{case['slug']}_population_join.csv")
        stop_count, stop_count_inside = None, None
        if stops is not None:
            centers = grid.geometry.centroid
            _, nearest_indexes = stop_tree.query(np.c_[centers.x, centers.y])
            # 500m 배후권의 모든 정류장 + 전국 자료 중 각 중심점의 최근접 정류장.
            # 지역경계 밖 정류장을 잘라 최근접 거리·500m 개수를 왜곡하지 않는다.
            selected_indexes = set(stops.index[stops.geometry.intersects(area.buffer(500))]) | set(nearest_indexes)
            case_stops = stops.loc[sorted(selected_indexes)].copy()
            stop_count_inside = int(stops.geometry.within(area).sum())
            if stop_count_inside == 0:
                raise ValueError(f"{case['name']}: 지역 안 정류장 수록을 확인할 수 없음. 무공급으로 처리 금지")
            csv(case_stops.drop(columns="geometry"), inputs / case["stops_file"])
            stop_count = len(case_stops)
        matched = grid["population_record_present"]
        valid = grid["population_status"].eq("유효")
        rows.append({
            "case_slug": case["slug"], "case_name": case["name"], "HDCD": case["HDCD"],
            "centroid_grid_count": len(grid), "population_record_matched": int(matched.sum()),
            "population_record_absent": int((~matched).sum()),
            "valid_population_grid_count": int(valid.sum()),
            "zero_population_grid_count": int(grid["total_population"].eq(0).sum()),
            "elderly_missing_in_valid_grid": int((valid & grid["elderly_population"].isna()).sum()),
            "elderly_gt_total_flag_count": int((valid & grid["elderly_population"].gt(grid["total_population"])).sum()),
            "total_population_sgis_estimate": grid.loc[valid, "total_population"].sum(),
            "elderly_population_sgis_estimate": grid.loc[valid, "elderly_population"].sum(min_count=1),
            "grid_boundary_year": 2025, "population_year": 2024,
            "emd_boundary_period": case["boundary_year"],
            "partial_boundary_grid_count": int((grid.geometry.intersection(area).area / 1_000_000 < .99).sum()),
            "grid_input_file": case["population_file"],
            "stop_input_available": (inputs / case["stops_file"]).exists(),
            "route_input_available": (inputs / case["routes_file"]).exists(),
            "official_stop_subset_count": stop_count,
            "official_stop_count_inside_emd": stop_count_inside,
        })
    csv(pd.DataFrame(rows), audit_dir / "grid_preparation_summary.csv")
    json_file({"grid_zip_files": manifest, "official_stops": stop_manifest, "population_file": population_path.name,
               "population_sha256": sha256(population_path),
               "boundary_sha256": sha256(inputs / "national_boundary.gpkg"),
               "original_files_modified": False, "grid_ids_invented": False,
               "missing_population_filled_with_zero": False,
               "note": "2025 격자판·2024 인구·2023Q4 읍면동 경계를 구분. 경계 격자의 중심점 기준 배정이며 정확한 읍면동 인구가 아님."},
              audit_dir / "official_grid_manifest.json")
    status_path = inputs / "preparation_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status.update({"additional_cases_have_grid_geometry": True,
                   "additional_cases_have_stops": all(row["stop_input_available"] for row in rows),
                   "additional_cases_have_routes": all(row["route_input_available"] for row in rows),
                   "notice": "공식 2025 격자·2024 인구 결합 완료. 노선관계 자료는 별도 필요. 추가 감사표는 additional_case_audit 참조."})
    json_file(status, status_path)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--grid-zips", type=Path, nargs="+", required=True)
    parser.add_argument("--stops-csv", type=Path, help="국토부 15067528의 CP949 2025-10-31 원본 CSV")
    args = parser.parse_args()
    prepare(args.config.resolve(), args.grid_zips, args.stops_csv)


if __name__ == "__main__":
    main()
