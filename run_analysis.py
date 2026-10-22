#!/usr/bin/env python3
"""원변수에서 전국 유형화·선별·사례 상세·민감도·지도를 순서대로 재생성한다."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".cache/matplotlib"))

import numpy as np
import pandas as pd

from mobility import eda, local, maps, national, scenarios
from mobility.common import csv, json_file, read_config, sha256
from mobility.definitions import variable_dictionary


def audit(nation, rural, completed, status, carbon, output):
    checks = []
    def add(name, passed, observed, severity="error"):
        checks.append({"check": name, "passed": bool(passed), "observed": str(observed), "severity": severity})
    add("전국 코드 유일성", not nation["HDCD"].duplicated().any(), len(nation))
    included = nation["model_included"].fillna(False)
    add("군집 입력 두 점수의 유효성", np.isfinite(nation.loc[included, national.SCORE_COLUMNS].to_numpy(dtype=float)).all(), int(included.sum()))
    add("전국 점수 0~100", nation.loc[included, national.SCORE_COLUMNS].apply(lambda x: x.between(0, 100)).all().all(), "0~100")
    focus = rural["revised_priority_tier"].eq("집중검토")
    priority = rural["revised_priority_tier"].eq("우선검토")
    add("집중·우선 등급 상호배타", not (focus & priority).any(), rural["revised_priority_tier"].value_counts().to_dict())
    add("집중검토 3신호 충족", rural.loc[focus, "core_signal_count"].eq(3).all(), int(focus.sum()))
    add("정책선별 대상은 비수도권 읍면", rural["policy_eligible"].all(), len(rural))
    add("추가 사례 실제자료 확보", status["grid_analysis_complete"].all(), status.loc[~status["grid_analysis_complete"], "case_name"].tolist(), "warning")
    if "route_analysis_complete" in status:
        add("사례 노선관계 자료 확보", status["route_analysis_complete"].fillna(False).all(), status.loc[~status["route_analysis_complete"].fillna(False), "case_name"].tolist(), "warning")
    if "transit_period_unverified" in status:
        add("사례 정류장 통계 기준일 확인", not status["transit_period_unverified"].fillna(True).any(), status.loc[status["transit_period_unverified"].fillna(True), "case_name"].tolist(), "warning")
    add("사례 노선목록 전체 수집 확인", status["route_source_collection_complete"].fillna(False).all(), status.loc[~status["route_source_collection_complete"].fillna(False), "case_name"].tolist(), "warning")
    for slug, grid in completed.items():
        flagged = grid["elderly_gt_total_flag"]
        add(f"{slug}: 고령인구 비율 이상값 결측", grid.loc[flagged, "elderly_ratio_analysis_pct"].isna().all(), int(flagged.sum()))
        add(f"{slug}: 격자 키 중복 없음", not grid["grid_id"].duplicated().any(), len(grid))
        valid = grid["valid_population"]
        uncertain = grid["route_coverage_unknown_flag"] & valid
        add(f"{slug}: 불완전 노선연결을 빈곤으로 확정하지 않음", not grid.loc[uncertain, "bus_supply_type"].eq("노선 빈곤").any(), int(uncertain.sum()))
        add(f"{slug}: 분석 범위 노선연결 확인", not uncertain.any(), int(uncertain.sum()), "warning")
        conflicts = int(grid["route_source_conflict_key_groups"].iloc[0])
        add(f"{slug}: 원 응답 동일키 충돌 없음", conflicts == 0, f"{conflicts}개 충돌키; 원문 보존·충돌 ID 격리", "warning")
        if slug == "iksan":
            # 발표 최종 감사에서 검증한 값. 변경은 숨기지 않고 실패로 노출한다.
            add("익산 보정 격자 재현", len(grid) == 508 and grid["valid_population"].sum() == 488, f"{len(grid)} / {grid['valid_population'].sum()}")
            add("익산 SGIS 품질표시 재현", flagged.sum() == 13, int(flagged.sum()))
            add("익산 의료지원 대상 재현", (grid["valid_population"] & grid["recalibrated_priority_type"].isin(local.MEDICAL_TARGET_TYPES)).sum() == 57, int(grid["recalibrated_priority_type"].isin(local.MEDICAL_TARGET_TYPES).sum()))
    if len(carbon):
        add("공차 정의: 추가비율을 전체비율로 환산", np.allclose(carbon["deadhead_fraction_of_total"], carbon["drt_deadhead_share"] / (1 + carbon["drt_deadhead_share"])), "d/(1+d)")
        add("순배출 = 정책배출 - 대체 자동차배출", np.allclose(carbon["electric_drt_net_change_tco2e_year"], carbon["electric_drt_emissions_tco2e_year"] - carbon["baseline_car_emissions_tco2_year"]), len(carbon))
    table = pd.DataFrame(checks)
    csv(table, output / "quality_checks.csv")
    return table


def run(config_path: Path, strict=False, skip_map=False):
    config_path = config_path.resolve()
    config = read_config(config_path)
    root = config_path.parent
    inputs, output = root / config["input_dir"], root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    csv(variable_dictionary(), output / "variable_dictionary.csv")
    required = ["national_base.csv", "national_boundary.gpkg", "primary_care.gpkg", "pharmacy.gpkg"]
    missing = [x for x in required if not (inputs / x).exists()]
    if missing:
        raise FileNotFoundError(f"필수 입력 없음: {missing}. prepare_inputs.py 또는 docs/INPUTS.md 참조")
    print("1/6 원변수 재점수화·전국 2축 K-means·규칙 선별", flush=True)
    nation, rural, validation, profiles, stability = national.run(inputs, output, config)
    print("2/6 동일 절차로 사례지역 격자 분석", flush=True)
    completed, map_context, status = local.run_cases(inputs, output, nation, config)
    case_comparison = []
    for slug, grid in completed.items():
        valid = grid.loc[grid["valid_population"]]
        total = valid["total_population"].sum()
        far_stop = valid["stop_access_over_500m"]
        far_medical = valid["nearest_primary_care_distance_m"].gt(3000)
        case_comparison.append({"case_slug": slug, "case_name": grid["case_name"].iloc[0],
            "grid_count": len(grid), "valid_population_grid_count": len(valid), "total_population_sgis_estimate": total,
            "stop_over_500m_grid_count": int(far_stop.sum()), "stop_over_500m_population_estimate": valid.loc[far_stop, "total_population"].sum(),
            "stop_over_500m_population_share_pct": valid.loc[far_stop, "total_population"].sum() / total * 100,
            "medical_over_3km_grid_count": int(far_medical.sum()), "medical_over_3km_population_estimate": valid.loc[far_medical, "total_population"].sum(),
            "medical_target_grid_count": int(valid["recalibrated_priority_type"].isin(local.MEDICAL_TARGET_TYPES).sum()),
            "route_coverage_unknown_grid_count": int(valid["route_coverage_unknown_flag"].sum()),
            "population_year": grid["population_year"].iloc[0], "stop_year": grid["transit_year"].iloc[0], "route_snapshot": grid["route_snapshot"].iloc[0],
            "note": "1km 중심점 기준·선택 격자인구 추정치, 공식 읍면동 총인구·주민별 실제 접근률 아님"})
    csv(pd.DataFrame(case_comparison), output / "case_comparison.csv")
    case_rows = []
    for case in config["cases"]:
        if case["HDCD"]:
            row = rural.loc[rural["HDCD"].eq(case["HDCD"])]
            if len(row) != 1:
                raise ValueError(f"추가 사례가 비수도권 읍면 선별범위에 없음: {case['name']}")
            item = row.iloc[0]
            case_rows.append({"case_name": case["name"], "HDCD": case["HDCD"], "revised_priority_tier": item["revised_priority_tier"],
                              "population_signal": item["population_signal"], "medical_signal": item["medical_signal"], "transport_signal": item["transport_signal"],
                              "overall_vulnerability_score": item["overall_vulnerability_score"], "selection_basis": "서로 다른 신호 조합의 목적표집, 전국 대표표본·4군집 대표 아님"})
    csv(pd.DataFrame(case_rows), output / "four_case_selection_basis.csv")
    print("3/6 거리계수·운영조건별 조건부 탄소·지원 이동량", flush=True)
    carbon = scenarios.run(completed, output, config)
    print("4/6 원자료 EDA·기준 민감도 그림", flush=True)
    eda.run(nation, rural, inputs, output, validation, profiles, carbon)
    print("5/6 품질 검사·재현성 기록", flush=True)
    checks = audit(nation, rural, completed, status, carbon, output)
    json_file(config, output / "config_used.json")
    versions = {name: importlib.metadata.version(name) for name in ["pandas", "numpy", "geopandas", "scipy", "scikit-learn", "matplotlib", "shapely"]}
    json_file({"python": sys.version, "packages": versions}, output / "runtime_versions.json")
    input_hashes = [{"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(inputs.iterdir()) if path.is_file()]
    for case in config["cases"]:
        for field in ["routes_file", "route_list_file", "route_metadata_file"]:
            if case.get(field):
                path = (inputs / case[field]).resolve()
                if path.exists() and not any(row["sha256"] == sha256(path) for row in input_hashes):
                    input_hashes.append({"name": str(path.relative_to(root)), "sha256": sha256(path), "bytes": path.stat().st_size})
    json_file(input_hashes, output / "input_checksums.json")
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "analysis_version": config["analysis_version"],
               "national_region_count": len(nation), "national_modeled_count": int(nation["model_included"].sum()),
               "rural_policy_scope_count": len(rural), "policy_tier_counts": rural["revised_priority_tier"].value_counts().to_dict(),
               "selected_candidate_count": int(rural["revised_priority_tier"].ne("관찰").sum()),
               "k_selected_for_interpretability": config["selected_k"], "k_best_silhouette": int(validation.sort_values("silhouette_score", ascending=False).iloc[0]["k"]),
               "minimum_seed_ARI": stability["ARI_vs_reference"].min(),
               "additional_cases_complete": bool(status["grid_analysis_complete"].all()),
               "all_cases_route_analysis_complete": bool(status.get("route_analysis_complete", pd.Series(False, index=status.index)).fillna(False).all()),
               "case_status": status.replace({np.nan: None}).to_dict("records"),
               "failed_error_checks": int((~checks["passed"] & checks["severity"].eq("error")).sum()),
               "open_warning_checks": int((~checks["passed"] & checks["severity"].eq("warning")).sum()),
               "limitations": ["선별은 진단 정답·인과효과 검증이 아님", "사례 격자·정류장 분석과 노선분석 완료를 별도로 표시함", "비수도권 읍면은 정책연구 적용 범위이지 수도권 무취약의 증거 아님", "서비스 시간단축·건강 개선·예산 최적 정책 미추정", "원자료 이용조건 확인 전 공개 저장소에 업로드하지 않음"]}
    json_file(summary, output / "run_summary.json")
    if not skip_map:
        print("6/6 최종 결과 기반 전국·사례 클릭 지도 (외부 연결 불필요)", flush=True)
        maps.build(nation, completed, status, output, config)
    if summary["failed_error_checks"]:
        raise RuntimeError("품질검사 실패. results/quality_checks.csv 확인")
    if strict and (not summary["additional_cases_complete"] or not summary["all_cases_route_analysis_complete"]):
        raise RuntimeError("strict 모드: 사례 격자 또는 노선 계산 미완료. 완료로 제출하지 말 것")
    print(f"가용자료 분석 완료: {summary['selected_candidate_count']}곳 후보. 추가 사례 전체 완료={summary['additional_cases_complete']}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--strict", action="store_true", help="추가 네 지역 자료가 미확보면 성공 종료하지 않음")
    parser.add_argument("--skip-map", action="store_true")
    args = parser.parse_args()
    run(args.config, strict=args.strict, skip_map=args.skip_map)
