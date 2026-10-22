"""실제 수집 CSV·기존 TAGO 통합 XLSX를 같은 노선 입력으로 읽고 충돌을 격리한다."""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd

from .common import csv, json_file, require_columns, require_unique, sha256


def quarantine_relations(source):
    """동일 키의 내용 충돌을 탐지하고 그 정류장 ID의 모든 관계를 보수적으로 격리한다.

    원본 source를 수정하지 않는다. 숫자를 평균하거나 임의 첫 행을 선택하지 않는다.
    완전히 같은 레코드만 분석용에서 제거한다. 원본·충돌 행은 별도로 보존한다.
    """
    keys = ["routeid", "nodeid", "nodeord"]
    require_columns(source, keys, "TAGO 경유정류소")
    if source[keys].isna().any().any():
        raise ValueError("경유정류소 결합키 결측")
    distinct = source.drop_duplicates().copy()
    fields = [name for name in ["nodenm", "gpslati", "gpslong", "updowncd"] if name in distinct]
    if fields:
        conflict_keys = distinct.groupby(keys, dropna=False)[fields].nunique(dropna=False).gt(1).any(axis=1)
        ambiguous = pd.MultiIndex.from_frame(distinct[keys]).isin(conflict_keys.index[conflict_keys])
    else:
        ambiguous = np.zeros(len(distinct), dtype=bool)
    conflicts = distinct.loc[ambiguous].copy()
    ids = set(conflicts["nodeid"].astype(str))
    excluded = distinct["nodeid"].astype(str).isin(ids)
    valid_coordinate = pd.Series(True, index=distinct.index)
    if {"gpslong", "gpslati"} <= set(distinct):
        valid_coordinate = pd.to_numeric(distinct["gpslong"], errors="coerce").between(124, 132) & pd.to_numeric(distinct["gpslati"], errors="coerce").between(33, 39)
    cleaned = distinct.loc[~excluded & valid_coordinate].copy()
    return cleaned, conflicts, distinct.loc[excluded | ~valid_coordinate].copy(), ids


def load_case_routes(case, inputs, output):
    path = inputs / case["routes_file"]
    if not path.exists():
        return None, None, {"route_data_available": False, "route_source_complete": False}, set()
    if path.suffix.lower() == ".csv":
        source = pd.read_csv(path, dtype="string")
        list_path = inputs / case["route_list_file"]
        route_list = pd.read_csv(list_path, dtype="string")
    else:
        source = pd.read_excel(path, sheet_name="노선별_정류소", dtype="string")
        route_list = pd.read_excel(path, sheet_name="노선목록", dtype="string")
        list_path = path
    require_unique(route_list, ["routeid"], "TAGO 노선목록")
    require_columns(route_list, ["routeid", "routetp"], "TAGO 노선목록")
    if not set(source["routeid"]) <= set(route_list["routeid"]):
        raise ValueError("경유자료에 노선목록 밖 routeid 있음")
    cleaned, conflicts, excluded, ids = quarantine_relations(source)
    directory = output / "cases" / case["slug"]
    csv(conflicts, directory / "route_input_conflicts.csv")
    csv(excluded, directory / "route_input_quarantine.csv")
    metadata = {}
    if case.get("route_metadata_file"):
        metadata = json.loads((inputs / case["route_metadata_file"]).read_text(encoding="utf-8"))
        if len(source) != metadata["stop_records"] or len(route_list) != metadata["downloaded_routes"]:
            raise ValueError("노선 수집 메타데이터와 실제 CSV 건수 불일치")
    summary = {
        "case_slug": case["slug"], "route_data_available": True,
        "route_list_count": len(route_list), "routes_with_raw_stop_records": source["routeid"].nunique(),
        "raw_relation_records": len(source), "analysis_relation_records": len(cleaned),
        "conflict_key_groups": conflicts.groupby(["routeid", "nodeid", "nodeord"]).ngroups,
        "conflicting_stop_id_count": len(ids), "quarantined_relation_records": len(excluded),
        "route_source_complete": source["routeid"].nunique() == len(route_list) and metadata.get("failed_routes", 0) == 0,
        "route_snapshot": case.get("route_snapshot", "2026년 수집, 상세 기준일 미확인"),
        "stop_snapshot": case.get("transit_snapshot", "통계 기준일 미확인"),
        "route_file": case["routes_file"], "route_sha256": sha256(path), "route_list_sha256": sha256(list_path),
        "note": "군 등록 노선의 관측자료. 수집 성공이 현재 운행·모든 인접군 노선·운행빈도를 보증하지 않음",
    }
    json_file(summary, directory / "route_input_summary.json")
    return cleaned, route_list, summary, ids


def stop_coordinate_review(stops, relations):
    """같은 ID의 기존 정류장과 API 좌표 사이 최소 차이. 새 좌표로 몰래 교체하지 않음."""
    result = pd.Series(np.nan, index=stops.index, dtype=float)
    if relations is None or not {"gpslong", "gpslati"} <= set(relations):
        return result
    known = relations[["nodeid", "gpslong", "gpslati"]].drop_duplicates().copy()
    x, y = pd.to_numeric(known["gpslong"], errors="coerce"), pd.to_numeric(known["gpslati"], errors="coerce")
    known = gpd.GeoDataFrame(known, geometry=gpd.points_from_xy(x, y), crs=4326).to_crs(5179)
    points = gpd.GeoSeries(gpd.points_from_xy(stops["gpslong"], stops["gpslati"]), crs=4326).to_crs(5179)
    positions = {str(key): group.geometry for key, group in known.groupby("nodeid")}
    for index, row in stops.iterrows():
        if str(row["nodeid"]) in positions:
            result.loc[index] = positions[str(row["nodeid"])].distance(points.loc[index]).min()
    return result
