#!/usr/bin/env python3
"""수집한 네 군의 노선·경유자료와 기존 정류장 ID를 검사한다. 분석결과는 갱신하지 않음."""
from pathlib import Path
import json
from zipfile import ZipFile, ZIP_DEFLATED

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from mobility.common import csv, json_file, require_columns, require_unique, sha256

ROOT = Path(__file__).resolve().parent


def main():
    raw = ROOT / "data/raw/tago_20261022"
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    rows, collection = [], []
    for case in config["cases"]:
        if not case.get("HDCD"):
            continue
        directory = raw / case["slug"]
        summary = json.loads((directory / "collection_summary.json").read_text(encoding="utf-8"))
        routes = pd.read_csv(directory / "route_list.csv", dtype="string")
        relations = pd.read_csv(directory / "route_stops.csv", dtype="string")
        status = pd.read_csv(directory / "route_collection_status.csv", dtype={"routeid": "string"})
        failures = pd.read_csv(directory / "failures.csv", dtype="string")
        require_columns(routes, ["routeid", "routeno", "routetp"], "수집한 노선목록")
        require_unique(routes, ["routeid"], "수집한 노선목록")
        require_columns(relations, ["routeid", "nodeid", "nodeord", "gpslati", "gpslong"], "수집 경유정류소")
        require_unique(status, ["routeid"], "노선별 수집상태")
        assert len(routes) == summary["route_list_totalCount"] == len(status)
        assert set(relations["routeid"]) <= set(routes["routeid"])
        assert status["totalCount"].fillna(-1).eq(status["downloaded_rows"]).all()
        assert len(relations) == summary["stop_records"]
        assert relations[["routeid", "nodeid", "nodeord"]].notna().all().all()
        coordinate_valid = pd.to_numeric(relations["gpslong"], errors="coerce").between(124, 132) & pd.to_numeric(relations["gpslati"], errors="coerce").between(33, 39)
        csv(relations.loc[~coordinate_valid], directory / "invalid_route_stop_coordinates.csv")
        master = pd.read_csv(ROOT / "data/inputs" / case["stops_file"], dtype={"nodeid": "string"})
        require_unique(master, ["nodeid"], "기존 공식 정류장")
        known_ids = set(relations["nodeid"])
        joined = master.copy()
        joined["matched_route_stop_id"] = joined["nodeid"].isin(known_ids)
        # 분석결과를 재계산하지 않고 기존 유효인구 격자의 반경 안 결합 누락만 검사한다.
        grid = gpd.read_file(ROOT / "results/cases" / case["slug"] / "grid_results.gpkg")
        centers = grid.loc[grid["valid_population"]].geometry.centroid
        points = gpd.GeoSeries(gpd.points_from_xy(master["gpslong"], master["gpslati"]), crs=4326).to_crs(5179)
        nearby = cKDTree(np.c_[points.x, points.y]).query_ball_point(np.c_[centers.x, centers.y], 500)
        used = {index for indexes in nearby for index in indexes}
        joined["within_500m_of_valid_population_grid"] = joined.index.isin(used)
        relevant = joined["within_500m_of_valid_population_grid"]
        csv(joined, directory / "stop_id_join_audit.csv")
        csv(joined.loc[~joined["matched_route_stop_id"]], directory / "unmatched_selected_stops.csv")
        conflict = relations.loc[relations.duplicated(["routeid", "nodeid", "nodeord"], keep=False)].copy()
        # 실제 관측된 동일키·다른 명칭/좌표를 방향 차이라고 추정하거나 첫 행으로 덮지 않는다.
        csv(conflict, directory / "conflicting_route_stop_records.csv")
        conflict_ids = set(conflict["nodeid"])
        rows.append({
            "case_name": case["name"], "case_slug": case["slug"], "cityCode": summary["cityCode"],
            "county_route_count": len(routes), "routes_with_stop_records": relations["routeid"].nunique(),
            "route_stop_records": len(relations), "unique_route_stop_ids": relations["nodeid"].nunique(),
            "failed_routes": len(failures), "empty_routes": int(status["empty_route_flag"].sum()),
            "duplicate_exact_records": int(relations.duplicated().sum()),
            "duplicate_route_node_order": int(relations.duplicated(["routeid", "nodeid", "nodeord"]).sum()),
            "conflicting_key_record_count": len(conflict),
            "conflicting_key_group_count": conflict.groupby(["routeid", "nodeid", "nodeord"]).ngroups,
            "conflicting_stop_ids_in_used_500m": int((relevant & joined["nodeid"].isin(conflict_ids)).sum()),
            "invalid_coordinate_records": int((~coordinate_valid).sum()),
            "selected_2025_stop_count": len(master), "selected_stop_id_matched": int(joined["matched_route_stop_id"].sum()),
            "stops_used_within_500m_valid_grid": int(relevant.sum()),
            "used_500m_stop_id_matched": int((relevant & joined["matched_route_stop_id"]).sum()),
            "used_500m_stop_id_unmatched": int((relevant & ~joined["matched_route_stop_id"]).sum()),
            "route_reference_period": "2026-10-22 수집시점", "stop_master_reference_period": "2025-10-31",
            "analysis_results_recalculated": False,
        })
        collection.append(summary)
    result = pd.DataFrame(rows)
    csv(result, raw / "download_and_join_summary.csv")
    csv(pd.DataFrame(collection), raw / "collection_summary.csv")
    report = ["# 추가 네 사례 TAGO 노선자료 다운로드", "", "공식 API의 조회시점 자료를 2026-10-22에 수집함. 기존 익산·전국 분석결과와 입력은 변경하지 않음.", "",
              "| 사례 | 군 전체 노선 수 | 경유 레코드 | 정류장 ID 종류 | 최종 실패 노선 |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        report.append(f"| {row['case_name']} | {row['county_route_count']:,} | {row['route_stop_records']:,} | {row['unique_route_stop_ids']:,} | {row['failed_routes']} |")
    report.extend(["", "## 포함 파일", "", "각 지역 폴더의 route_list.csv는 노선목록, route_stops.csv는 노선별 경유정류소임. 원 응답은 raw/에 보존함. 노선별 수집 건수와 실패, 정류장 ID 연결 검사는 별도 CSV에 있음. 각 군 전체 노선을 수집했으며 해당 면만의 노선 수가 아님.", "",
                   "## 검증과 남은 작업", "", "유효인구 격자의 500m 안 기존 정류장은 총 " + str(int(result['stops_used_within_500m_valid_grid'].sum())) + "개이며 ID 미연결 " + str(int(result['used_500m_stop_id_unmatched'].sum())) + "개임. 주변 전체 선택 정류장과 이 분모는 다름. 인접 군 정류장·미정차 지점 등 나머지 미연결은 unmatched_selected_stops.csv에 보존함.", "",
                   "도시목록에 홍성군이 없었지만 공식 정류장 파일에서 확인한 34360으로 직접 조회해 180개 노선을 확인함. 도시목록 누락만으로 자료가 없다고 판정하지 않음.", "",
                   "초기 서버 오류 3개 노선은 성공한 응답을 재사용하면서 실패 건만 재요청해 해결함. raw/의 error_latest.json은 초기 오류 증빙이며 최종 실패 현황과 구분함.", "",
                   "홍성군 응답에서는 같은 routeid·nodeid·nodeord인데 명칭과 좌표가 다른 64개 키 그룹(128행·정류장 ID 48개·노선 4개)이 발견됨. 방향코드도 같아 방향 차이로 설명되지 않음. conflicting_route_stop_records.csv에 원문을 보존하며 임의 첫 행 선택·좌표 평균·중복 삭제는 수행하지 않음. 현재 결성면 유효인구 격자 500m 범위의 정류장 ID와 겹치는 충돌 ID는 0개임. 다른 지역 확대나 전체 노선 경로 분석에는 그대로 사용하면 안 됨.", "",
                   "2026년 조회시점 노선과 2025-10-31 정류장 파일의 시점이 다름. ID 일치는 실제 현재 운행·배차간격·운행횟수 또는 주변 모든 노선의 완전성을 보증하지 않음. 좌표 변화와 인접 군 경유노선은 분석 반영 전에 추가 확인해야 함.", "",
                   "다운로드와 ID 결합검사만 완료했으며 기존 분석표·지도는 아직 이 노선자료로 재계산하지 않음. 다음 분석 반영에서는 CSV 입력 연결, 500m 내 고유 routeid 합집합 계산과 미연결 품질표시가 필요함.", "",
                   "출처: https://www.data.go.kr/data/15098529/openapi.do", ""])
    (raw / "README.md").write_text("\n".join(report), encoding="utf-8")
    manifest = [{"path": str(path.relative_to(raw)), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in sorted(raw.rglob("*")) if path.is_file() and path.name != "manifest.json"]
    json_file({"official_source": "https://www.data.go.kr/data/15098529/openapi.do", "files": manifest,
               "original_analysis_inputs_modified": False, "analysis_results_recalculated": False,
               "warnings": ["TAGO 수집시점 2026년과 정류장 스냅샷 2025년은 다름", "ID 일치가 좌표·실제 운행·모든 경유노선의 완전성을 증명하지 않음", "군 등록 노선목록의 수집이며 인접 군 등록 노선까지 전국 전수 수집한 것은 아님", "배차간격·운행횟수·승하차 수요를 계산하지 않음"]}, raw / "manifest.json")
    destination = ROOT.parent / "추가4지역_TAGO_노선자료_20261022.zip"
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(raw.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(raw))
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
    print(result.to_string(index=False))
    print(f"다운로드 묶음: {destination.name}, {destination.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
