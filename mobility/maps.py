"""외부 지도 타일·CDN·API 키 없이 열리는 전국/사례 클릭 지도 생성.

실제 SGIS 도형을 EPSG:5179로 투영한 SVG를 사용한다. 웹 단순화는 표시만 바꾸며
거리·인구·면적 계산은 원 도형으로 수행한다. 지도 결과는 정책 확정 명령이 아니다.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def clean_value(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def svg_path(geometry, x0, y1, scale):
    if geometry.geom_type == "Polygon":
        polygons = [geometry]
    elif geometry.geom_type == "MultiPolygon":
        polygons = list(geometry.geoms)
    else:
        return ""
    pieces = []
    for polygon in polygons:
        for ring in [polygon.exterior] + list(polygon.interiors):
            coords = list(ring.coords)
            pieces.append("M" + "L".join(f"{(x-x0)*scale:.2f},{(y1-y)*scale:.2f}" for x, y, *_ in coords) + "Z")
    return "".join(pieces)


def build(national, completed, status, output, config):
    x0, y0, x1, y1 = national.total_bounds
    scale = min(1100 / (x1 - x0), 1400 / (y1 - y0))
    width, height = (x1 - x0) * scale, (y1 - y0) * scale
    paths, records = [], {}
    columns = ["HDCD", "HDCD_SD_NM", "HDCD_SGG_NM", "HDCD_EMD_NM", "total_population", "elderly_population", "elderly_ratio_pct",
               "population_vulnerability_score", "location_vulnerability_score", "overall_vulnerability_score", "balanced_cluster_type",
               "balanced_isolation_forest_anomaly_5pct", "revised_priority_tier", "population_signal", "medical_signal", "transport_signal",
               "core_signal_count", "nearest_primary_care_distance_m", "primary_care_count_2025", "primary_care_count_strict_within",
               "22_pt_time_min", "22_time_gap_ratio", "22_pt_time_min_censored", "selection_frequency", "policy_candidate_primary", "incomplete_indicator_flag"]
    for _, row in national.iterrows():
        code = str(row["HDCD"])
        records[code] = {key: clean_value(row[key]) for key in columns}
        geometry = row.geometry.simplify(config["map_simplify_m"], preserve_topology=True)
        paths.append(f'<path data-code="{html.escape(code)}" d="{svg_path(geometry, x0, y1, scale)}"/>')
    local_paths, local_records, local_boxes = {}, {}, {}
    for slug, grid in completed.items():
        values, shapes = {}, []
        for _, row in grid.iterrows():
            gid = str(row["grid_id"])
            keep = ["grid_id", "case_name", "population_year", "transit_year", "boundary_period", "total_population", "elderly_population",
                    "elderly_ratio_analysis_pct", "elderly_gt_total_flag", "boundary_partial_grid", "nearest_stop_distance_m_recalculated",
                    "stops_within_500m", "routes_within_500m", "nearest_primary_care_name", "nearest_primary_care_distance_m",
                    "nearest_pharmacy_name", "nearest_pharmacy_distance_m", "22_pt_time_min", "22_pt_time_min_censored", "22_time_gap_ratio",
                    "recalibrated_priority_type", "recalibrated_policy_candidate", "route_coverage_unknown_flag",
                    "population_status", "grid_boundary_year", "transit_snapshot", "route_data_available",
                    "route_snapshot", "route_source_collection_complete", "route_source_conflict_key_groups",
                    "route_count_interpretation", "bus_supply_type", "unmatched_stop_count_within_500m",
                    "conflict_stop_count_within_500m", "position_review_stop_count_within_500m", "valid_population"]
            values[gid] = {key: clean_value(row.get(key)) for key in keep}
            shapes.append(f'<path data-grid="{html.escape(gid)}" data-case="{html.escape(slug)}" d="{svg_path(row.geometry, x0, y1, scale)}"/>')
        bx0, by0, bx1, by1 = grid.total_bounds
        local_boxes[slug] = [(bx0 - x0) * scale, (y1 - by1) * scale, (bx1 - bx0) * scale, (by1 - by0) * scale]
        local_paths[slug], local_records[slug] = "".join(shapes), values
    thresholds = json.loads((output / "screening_thresholds.json").read_text(encoding="utf-8"))
    carbon = pd.read_csv(output / "carbon_distance_scenarios.csv")
    carbon = carbon.loc[carbon["target_group"].eq("의료지원 대상 전체") & carbon["road_distance_factor"].eq(1.3)]
    carbon_columns = ["case_slug", "scenario", "target_grid_count", "eligible_elderly_sgis_estimate", "expected_users_assumed", "annual_round_trips_assumed", "diesel_drt_net_change_tco2_year", "electric_drt_net_change_tco2e_year"]
    payload = json.dumps({"national": records, "local": local_records, "local_paths": local_paths, "local_boxes": local_boxes, "thresholds": thresholds,
                          "status": status.replace({np.nan: None}).to_dict("records"), "cases": config["cases"],
                          "carbon": carbon[carbon_columns].to_dict("records"), "analysis_version": config["analysis_version"]}, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    template = Path(__file__).with_name("map_template.html").read_text(encoding="utf-8")
    replacements = {"__WIDTH__": str(width), "__HEIGHT__": str(height), "__PADDED_WIDTH__": str(width + 20), "__PADDED_HEIGHT__": str(height + 20), "__PATHS__": "".join(paths), "__DATA__": payload}
    for key, value in replacements.items():
        template = template.replace(key, value)
    (output / "national_interactive_map.html").write_text(template, encoding="utf-8")
