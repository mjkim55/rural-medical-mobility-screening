"""공통 1km 격자 분석. 익산에서 검증하고 다른 사례에도 같은 입력 계약을 적용한다.

도로거리·실제 보행경로·운행빈도·승하차 수요는 계산하지 않는다.
KTDB 읍면동 맥락값은 격자별 이동시간으로 해석하지 않는다.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .common import csv, require_columns, require_unique
from .routes import load_case_routes, stop_coordinate_review

MEDICAL_TARGET_TYPES = ["고령·기초의료 접근 우선검토형", "고령·의료·정류장 복합고위험형"]
POLICY_MAP = {
    "고령·의료·정류장 복합고위험형": "의료목적 DRT·공공택시 및 안전한 승하차지점 검토",
    "고령·기초의료 접근 우선검토형": "공공택시·의료목적 DRT 비교 검토",
    "정류장 접근 우선검토형": "기존 노선 연결·보행안전 확인 후 승하차지점 검토",
    "노선공급 우선검토형": "노선·환승·시간표 검토(운행횟수 미확인)",
    "읍면동 대중교통 장시간형": "읍면동 맥락값 확인 후 의료목적 직결서비스 검토",
    "현행 유지·추가 모니터링": "현행 서비스·미충족 수요 확인",
    "인구 0명·분석 제외": "인구 결측·비밀보호·0명 구분 확인",
}


def nearest_points(origins, destinations, prefix):
    if destinations.empty:
        raise ValueError(f"{prefix}: 목적지 좌표 자료가 없음")
    points = destinations.to_crs(5179).reset_index(drop=True)
    distance, index = cKDTree(np.c_[points.geometry.x, points.geometry.y]).query(np.c_[origins.x, origins.y])
    result = pd.DataFrame({f"nearest_{prefix}_distance_m": distance})
    for field, column in [("id", "암호화요양기호"), ("name", "요양기관명"), ("type", "종별코드명")]:
        result[f"nearest_{prefix}_{field}"] = points.iloc[index][column].to_numpy()
    return result


def stop_supply(centers, stops, route_stops=None, radius=500, conflict_ids=None, coordinate_tolerance=100):
    """정류장 수와 노선 합집합 수를 분리. 관계 레코드 수는 운행횟수가 아니다."""
    require_columns(stops, ["nodeid", "gpslong", "gpslati", "nodenm"], "정류장")
    require_unique(stops, ["nodeid"], "정류장")
    if not stops["gpslong"].between(124, 132).all() or not stops["gpslati"].between(33, 39).all():
        raise ValueError("정류장 좌표 누락·범위 오류")
    points = gpd.GeoDataFrame(stops.copy(), geometry=gpd.points_from_xy(stops["gpslong"], stops["gpslati"]), crs=4326).to_crs(5179).reset_index(drop=True)
    if points.empty:
        raise ValueError("정류장 자료가 비어 있음. 0개 공급으로 간주하지 않음")
    conflict_ids = conflict_ids or set()
    points["route_coordinate_min_shift_m"] = stop_coordinate_review(stops.reset_index(drop=True), route_stops)
    points["route_position_review_flag"] = points["route_coordinate_min_shift_m"].gt(coordinate_tolerance)
    points["route_conflict_id_flag"] = points["nodeid"].astype(str).isin(conflict_ids)
    tree = cKDTree(np.c_[points.geometry.x, points.geometry.y])
    xy = np.c_[centers.x, centers.y]
    distance, nearest_index = tree.query(xy)
    nearby = tree.query_ball_point(xy, radius)
    known_routes = route_stops is not None
    routes_by_stop = {} if not known_routes else route_stops.groupby("nodeid")["routeid"].agg(lambda x: set(x.dropna().astype(str))).to_dict()
    relations = {} if not known_routes else route_stops.groupby("nodeid").size().to_dict()
    rows = []
    for idx, indexes in enumerate(nearby):
        nodes = points.iloc[indexes]["nodeid"].astype(str).tolist()
        nearest = points.iloc[int(nearest_index[idx])]
        routes = set().union(*(routes_by_stop.get(node, set()) for node in nodes)) if nodes else set()
        rows.append({"nearest_stop_id_recalculated": nearest["nodeid"], "nearest_stop_name_recalculated": nearest["nodenm"],
                     "nearest_stop_distance_m_recalculated": float(distance[idx]), "stops_within_500m": len(nodes),
                     "nearest_stop_route_count": len(routes_by_stop[str(nearest["nodeid"])]) if known_routes and str(nearest["nodeid"]) in routes_by_stop else np.nan,
                     "routes_within_500m": len(routes) if known_routes else np.nan,
                     "route_stop_relations_within_500m": sum(relations.get(n, 0) for n in nodes) if known_routes else np.nan,
                     "unmatched_stop_count_within_500m": sum(n not in routes_by_stop for n in nodes) if known_routes else np.nan,
                     "conflict_stop_count_within_500m": int(points.iloc[indexes]["route_conflict_id_flag"].sum()),
                     "position_review_stop_count_within_500m": int(points.iloc[indexes]["route_position_review_flag"].sum()),
                     "route_data_available": known_routes})
    return pd.DataFrame(rows), points


def classify_grid(grid, config):
    grid = grid.copy()
    valid = grid["valid_population"]
    sample = grid.loc[valid]
    ratio_threshold = sample["elderly_ratio_analysis_pct"].quantile(.75)
    elderly_threshold = sample["elderly_population"].quantile(.75)
    distance_threshold = sample["nearest_primary_care_distance_m"].quantile(config["medical_far_quantile"])
    grid["elderly_high"] = grid["elderly_ratio_analysis_pct"].ge(ratio_threshold) | (~grid["elderly_gt_total_flag"] & grid["elderly_population"].ge(elderly_threshold))
    grid["primary_care_relative_far"] = grid["nearest_primary_care_distance_m"].ge(distance_threshold)
    grid["stop_access_over_500m"] = grid["nearest_stop_distance_m_recalculated"].gt(config["stop_radius_m"])
    grid["route_coverage_unknown_flag"] = grid["stops_within_500m"].gt(0) & (grid["routes_within_500m"].isna() | grid["unmatched_stop_count_within_500m"].gt(0))
    for field in ["conflict_stop_count_within_500m", "position_review_stop_count_within_500m"]:
        if field in grid:
            grid["route_coverage_unknown_flag"] |= grid[field].gt(0)
    grid["route_supply_low_within_500m"] = ~grid["route_coverage_unknown_flag"] & grid["routes_within_500m"].gt(0) & grid["routes_within_500m"].le(config["route_poverty_max"])
    grid["bus_supply_type"] = np.select(
        [grid["stops_within_500m"].eq(0), grid["routes_within_500m"].le(config["route_poverty_max"]), grid["routes_within_500m"].ge(10)],
        ["정류장 미비", "노선 빈곤", "풍부 공급"], default="일반 공급")
    grid.loc[grid["routes_within_500m"].isna() & grid["stops_within_500m"].gt(0), "bus_supply_type"] = "노선자료 미확보"
    grid.loc[grid["route_coverage_unknown_flag"] & grid["routes_within_500m"].notna(), "bus_supply_type"] = "노선 연결 추가확인"
    grid.loc[~valid, "bus_supply_type"] = "인구 0명·분석 제외"
    grid["ktdb_emd_pt_time_over_30min"] = grid["22_pt_time_min"].gt(30)
    complex_mask = valid & grid["elderly_high"] & grid["primary_care_relative_far"] & grid["stop_access_over_500m"] & grid["ktdb_emd_pt_time_over_30min"]
    medical_mask = valid & grid["elderly_high"] & grid["primary_care_relative_far"] & ~complex_mask
    stop_mask = valid & grid["stop_access_over_500m"] & ~complex_mask & ~medical_mask
    route_mask = valid & ~grid["stop_access_over_500m"] & grid["route_supply_low_within_500m"] & grid["ktdb_emd_pt_time_over_30min"] & ~medical_mask
    long_time = valid & ~grid["stop_access_over_500m"] & grid["ktdb_emd_pt_time_over_30min"] & ~route_mask & ~medical_mask
    grid["recalibrated_priority_type"] = "현행 유지·추가 모니터링"
    for mask, label in [(long_time, "읍면동 대중교통 장시간형"), (route_mask, "노선공급 우선검토형"), (stop_mask, "정류장 접근 우선검토형"), (medical_mask, MEDICAL_TARGET_TYPES[0]), (complex_mask, MEDICAL_TARGET_TYPES[1]), (~valid, "인구 0명·분석 제외")]:
        grid.loc[mask, "recalibrated_priority_type"] = label
    grid["recalibrated_policy_candidate"] = grid["recalibrated_priority_type"].map(POLICY_MAP)
    grid["ktdb_context_note"] = "2023 읍면동 평균 맥락값. 격자별·개인별 의료여행시간이 아님"
    grid["medical_relative_threshold_m"] = distance_threshold
    return grid


def analyze_case(case, inputs, output, national, config):
    boundary_path = inputs / case["boundary_file"] if case.get("boundary_file") else None
    boundary = gpd.read_file(boundary_path) if boundary_path else national.loc[national["HDCD"].eq(case["HDCD"]), ["HDCD", "geometry"]].copy()
    boundary = boundary.to_crs(5179)[["HDCD", "geometry"]]
    if boundary.empty:
        raise ValueError(f"{case['name']}: 경계 코드 불일치")
    require_unique(boundary, ["HDCD"], "사례 경계")
    population = gpd.read_file(inputs / case["population_file"])
    if population.crs is None:
        raise ValueError("격자 좌표계 누락. 임의 지정하지 않음")
    population = population.to_crs(5179)
    require_columns(population, ["grid_id", "total_population", "elderly_population", "geometry"], "격자인구")
    require_unique(population, ["grid_id"], "격자인구")
    if "population_year" in population:
        periods = set(pd.to_numeric(population["population_year"], errors="coerce").dropna().astype(int))
        if periods and periods != {case["population_year"]}:
            raise ValueError(f"{case['name']}: 실제 인구연도 {periods}와 설정 연도 불일치")
    if population.geometry.isna().any() or population.geometry.is_empty.any() or not population.geometry.is_valid.all():
        raise ValueError("격자 공간정보 결측·빈 도형·유효성 오류")
    if not population.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("격자 인구는 폴리곤이어야 함. 중심점만으로 격자 도형을 만들지 않음")
    if not population.geometry.area.between(950000, 1050000).all():
        raise ValueError("1km 격자 면적이 아님. 경계로 잘린 격자·다른 해상도인지 확인")
    population = population[[col for col in ["grid_id", "total_population", "elderly_population", "population_status", "population_year", "population_record_present", "grid_boundary_year", "emd_boundary_period", "geometry"] if col in population]].copy()
    centers = population[["grid_id", "geometry"]].copy()
    centers.geometry = centers.geometry.centroid
    joined = gpd.sjoin(centers, boundary, how="left", predicate="within").drop(columns="index_right")
    require_unique(joined, ["grid_id"], "격자 중심점 행정구역 배정")
    population["HDCD"] = joined["HDCD"].to_numpy()
    grid = population.loc[population["HDCD"].notna()].reset_index(drop=True)
    if grid.empty:
        raise ValueError(f"{case['name']}: 해당 지역에 속하는 격자가 없음. 익산 격자 재사용 금지")
    for col in ["total_population", "elderly_population"]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    grid["valid_population"] = grid["total_population"].gt(0)
    if "population_status" in grid:
        grid["valid_population"] &= grid["population_status"].eq("유효")
    grid["elderly_gt_total_flag"] = grid["elderly_population"].gt(grid["total_population"])
    grid["elderly_missing_or_negative_flag"] = grid["elderly_population"].isna() | grid["elderly_population"].lt(0)
    grid["elderly_ratio_analysis_pct"] = grid["elderly_population"].div(grid["total_population"].where(grid["valid_population"])) * 100
    grid.loc[grid["elderly_gt_total_flag"] | grid["elderly_missing_or_negative_flag"], "elderly_ratio_analysis_pct"] = np.nan
    grid["boundary_overlap_ratio"] = grid.geometry.intersection(boundary.geometry.union_all()).area / grid.geometry.area
    grid["boundary_partial_grid"] = grid["boundary_overlap_ratio"].lt(.99)
    grid["case_slug"], grid["case_name"] = case["slug"], case["name"]
    grid["population_year"], grid["transit_year"], grid["boundary_period"] = case["population_year"], case["transit_year"], case["boundary_year"]
    grid["transit_snapshot"] = case.get("transit_snapshot", "통계 기준일 미확인")
    centers = grid.geometry.centroid
    stops = pd.read_csv(inputs / case["stops_file"], dtype={"nodeid": "string"})
    route_stops, route_list, route_summary, conflict_ids = load_case_routes(case, inputs, output)
    # 안내서와 불일치한 배차간격·0000 시각은 운행횟수로 변환하지 않는다.
    supply, stop_points = stop_supply(centers, stops, route_stops, config["stop_radius_m"], conflict_ids, config.get("route_coordinate_review_tolerance_m", 100))
    primary = gpd.read_file(inputs / "primary_care.gpkg")
    pharmacy = gpd.read_file(inputs / "pharmacy.gpkg")
    grid = gpd.GeoDataFrame(pd.concat([grid, supply, nearest_points(centers, primary, "primary_care"), nearest_points(centers, pharmacy, "pharmacy")], axis=1), geometry="geometry", crs=5179)
    context_columns = ["HDCD", "22_pt_time_min", "22_car_time_min", "22_time_gap_ratio", "22_reach30_gap_pct", "22_pt_time_min_censored", "22_car_time_min_censored"]
    grid = grid.merge(pd.DataFrame(national[context_columns]), on="HDCD", how="left", validate="many_to_one")
    grid = classify_grid(grid, config)
    grid["route_snapshot"] = route_summary.get("route_snapshot", "미확보")
    grid["route_source_collection_complete"] = route_summary.get("route_source_complete", False)
    grid["route_source_conflict_key_groups"] = route_summary.get("conflict_key_groups", 0)
    grid["route_count_interpretation"] = np.where(grid["route_coverage_unknown_flag"], "관측된 고유노선 하한·연결 및 좌표 추가확인", "확보한 등록 노선 범위의 고유노선 합집합·현재 운행 보증 아님")
    # 시외버스 제외는 별도 민감도 열로 유지. 기존 전체노선 결과를 몰래 교체하지 않는다.
    if route_stops is not None and "routetp" in route_list:
        ids = set(route_list.loc[route_list["routetp"].ne("시외버스"), "routeid"])
        local_relations = route_stops.loc[route_stops["routeid"].isin(ids)]
        local_nodes = set(local_relations["nodeid"])
        local_stops = stops.loc[stops["nodeid"].isin(local_nodes)]
        if len(local_stops):
            local_supply, _ = stop_supply(centers, local_stops, local_relations, config["stop_radius_m"], conflict_ids, config.get("route_coordinate_review_tolerance_m", 100))
            grid["local_stops_within_500m"] = local_supply["stops_within_500m"]
            grid["local_routes_within_500m"] = local_supply["routes_within_500m"]
    case_output = output / "cases" / case["slug"]
    case_output.mkdir(parents=True, exist_ok=True)
    csv(grid.drop(columns="geometry"), case_output / "grid_results.csv")
    grid.to_file(case_output / "grid_results.gpkg", layer="grid_results", driver="GPKG")
    valid = grid.loc[grid["valid_population"]]
    summary = valid.groupby("recalibrated_priority_type", dropna=False).agg(
        grid_count=("grid_id", "size"), total_population=("total_population", "sum"), elderly_population=("elderly_population", "sum")) .reset_index()
    summary["grid_share_pct"] = summary["grid_count"] / len(valid) * 100
    summary["population_share_pct"] = summary["total_population"] / valid["total_population"].sum() * 100
    summary["elderly_ratio_sgis_estimate_pct"] = summary["elderly_population"] / summary["total_population"] * 100
    csv(summary, case_output / "policy_type_summary.csv")
    csv(valid.groupby("bus_supply_type").agg(grid_count=("grid_id", "size"), total_population=("total_population", "sum"), elderly_population=("elderly_population", "sum")).reset_index(), case_output / "bus_supply_summary.csv")
    distance_rows = []
    for threshold in [1000, 3000, 5000]:
        chosen = valid["nearest_primary_care_distance_m"].gt(threshold)
        distance_rows.append({"distance_threshold_m": threshold, "grid_count": chosen.sum(), "population_sgis_estimate": valid.loc[chosen, "total_population"].sum()})
    csv(pd.DataFrame(distance_rows), case_output / "medical_distance_sensitivity.csv")
    csv(stop_points.drop(columns="geometry"), case_output / "stop_route_link_quality.csv")
    return grid, stop_points, primary, pharmacy


def run_cases(inputs, output, national, config):
    status, completed, map_context = [], {}, {}
    for case in config["cases"]:
        required = [case["population_file"], case["stops_file"]] + ([case["boundary_file"]] if case.get("boundary_file") else [])
        missing = [name for name in required if not (inputs / name).exists()]
        if missing:
            status.append({"case_slug": case["slug"], "case_name": case["name"], "HDCD": case["HDCD"], "status": "미확보", "missing_inputs": "; ".join(missing), "grid_analysis_complete": False})
            continue
        grid, stops, primary, pharmacy = analyze_case(case, inputs, output, national, config)
        completed[case["slug"]] = grid
        map_context[case["slug"]] = (stops, primary, pharmacy)
        status.append({"case_slug": case["slug"], "case_name": case["name"], "HDCD": case["HDCD"], "status": "격자·정류장 분석 완료" if grid["route_data_available"].all() else "부분 가능: 노선자료 미확보", "missing_inputs": "" if grid["route_data_available"].all() else case["routes_file"], "grid_analysis_complete": True,
                       "grid_count": len(grid), "valid_population_grid_count": grid["valid_population"].sum(), "elderly_gt_total_flag_count": grid["elderly_gt_total_flag"].sum(), "transit_period_unverified": not bool(case.get("transit_snapshot")),
                       "route_analysis_complete": bool(grid["route_data_available"].all()),
                       "route_source_collection_complete": bool(grid["route_source_collection_complete"].all()),
                       "route_conflict_key_groups": int(grid["route_source_conflict_key_groups"].iloc[0]),
                       "valid_grids_route_coverage_unknown": int((grid["valid_population"] & grid["route_coverage_unknown_flag"]).sum()),
                       "route_snapshot": grid["route_snapshot"].iloc[0],
                       "population_missing_grid_count": int(grid["total_population"].isna().sum())})
    csv(pd.DataFrame(status), output / "case_completion_status.csv")
    return completed, map_context, pd.DataFrame(status)
