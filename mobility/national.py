"""전국 2축 유형화와 비수도권 읍·면 규칙 선별. 두 절차를 혼동하지 않는다."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .common import csv, json_file, require_columns, require_unique

POPULATION_FEATURES = ["elderly_ratio_pct", "inverse_log_population_density"]
MOBILITY_FEATURES = ["22_pt_time_min", "22_time_gap_ratio", "22_reach30_gap_pct", "23_pt_time_min", "daily_life_pt_time_min", "regional_hub_pt_time_min"]
SCORE_COLUMNS = ["population_vulnerability_score", "location_vulnerability_score"]
CAPITAL_PROVINCES = {"서울특별시", "경기도", "인천광역시"}


def score_national(frame, population_weight=0.5):
    """큰 값일수록 취약하도록 정렬한 원지표의 전국 백분위를 영역별로 평균한다.

    Overall은 군집 입력이 아님. 순위화 앞의 단조 로그변환은 순위를 바꾸지 않음.
    결측은 0으로 채우지 않는다. 이용 지표 수를 별도 열로 공개한다.
    """
    frame = frame.copy()
    require_unique(frame, ["HDCD"], "전국 원변수")
    require_columns(frame, ["total_population", "elderly_population", "area_km2"], "전국 원변수")
    frame["elderly_ratio_pct"] = frame["elderly_population"].div(frame["total_population"].where(frame["total_population"].gt(0))) * 100
    frame["population_density_per_km2"] = frame["total_population"].div(frame["area_km2"].where(frame["area_km2"].gt(0)))
    frame["inverse_log_population_density"] = -np.log1p(frame["population_density_per_km2"])
    frame["daily_life_pt_time_min"] = frame[["31_pt_time_min", "32_pt_time_min"]].min(axis=1, skipna=True)
    frame["regional_hub_pt_time_min"] = frame[["41_pt_time_min", "42_pt_time_min"]].min(axis=1, skipna=True)
    frame["population_data_invalid_flag"] = frame["elderly_population"].lt(0) | frame["elderly_population"].gt(frame["total_population"])
    # 원래 기준모형처럼 유효 인구가 있는 지역 안에서 백분위를 계산한다.
    frame["model_included"] = frame["total_population"].gt(0) & frame["area_km2"].gt(0) & ~frame["population_data_invalid_flag"]
    selected = frame["model_included"]
    for features, output, count_col in [
        (POPULATION_FEATURES, SCORE_COLUMNS[0], "population_indicator_available_count"),
        (MOBILITY_FEATURES, SCORE_COLUMNS[1], "mobility_indicator_available_count"),
    ]:
        values = frame.loc[selected, features].replace([np.inf, -np.inf], np.nan)
        ranks = values.rank(pct=True, method="average") * 100
        frame[output] = np.nan
        frame.loc[selected, output] = ranks.mean(axis=1, skipna=True)
        frame[count_col] = frame[features].notna().sum(axis=1)
        for column in features:
            frame[f"{column}_percentile"] = np.nan
            frame.loc[selected, f"{column}_percentile"] = ranks[column]
    frame["incomplete_indicator_flag"] = frame["population_indicator_available_count"].ne(2) | frame["mobility_indicator_available_count"].ne(6)
    frame["model_included"] &= frame[SCORE_COLUMNS].notna().all(axis=1)
    frame["overall_vulnerability_score"] = population_weight * frame[SCORE_COLUMNS[0]] + (1 - population_weight) * frame[SCORE_COLUMNS[1]]
    return frame


def cluster_national(frame, config):
    result = frame.copy()
    modeled = result.loc[result["model_included"]].copy()
    # K-means와 Isolation Forest 모두 이 두 열만 입력한다. Overall은 넣지 않는다.
    scaled = StandardScaler().fit_transform(modeled[SCORE_COLUMNS])
    models, validation = {}, []
    for k in config["k_candidates"]:
        model = KMeans(n_clusters=k, n_init=config["n_init"], random_state=config["random_state"])
        labels = model.fit_predict(scaled)
        models[k] = model
        counts = pd.Series(labels).value_counts()
        validation.append({"k": k, "silhouette_score": silhouette_score(scaled, labels),
                           "smallest_cluster_n": counts.min(), "largest_cluster_n": counts.max(),
                           "selected": k == config["selected_k"]})
    chosen = models[config["selected_k"]]
    modeled["balanced_kmeans_cluster_raw"] = chosen.labels_
    profiles = modeled.groupby("balanced_kmeans_cluster_raw").agg(
        region_count=("HDCD", "size"), population_vulnerability_score=(SCORE_COLUMNS[0], "mean"),
        location_vulnerability_score=(SCORE_COLUMNS[1], "mean"), overall_vulnerability_score=("overall_vulnerability_score", "mean"))
    # 이름은 중심값을 읽은 사후 해석이다. 네 사분면을 강제한 정답 라벨이 아니다.
    low, high = profiles["overall_vulnerability_score"].idxmin(), profiles["overall_vulnerability_score"].idxmax()
    names = {low: "상대 저취약형", high: "인구·위치 복합취약형"}
    for key, row in profiles.iterrows():
        if key not in names:
            names[key] = "인구취약 우세형" if row[SCORE_COLUMNS[0]] >= row[SCORE_COLUMNS[1]] else "위치취약 우세형"
    modeled["balanced_cluster_type"] = modeled["balanced_kmeans_cluster_raw"].map(names)
    profiles["balanced_cluster_type"] = profiles.index.map(names)
    stability = []
    for seed in config["stability_seeds"]:
        labels = KMeans(n_clusters=config["selected_k"], n_init=config["n_init"], random_state=seed).fit_predict(scaled)
        stability.append({"seed": seed, "ARI_vs_reference": adjusted_rand_score(chosen.labels_, labels)})
    isolation = IsolationForest(n_estimators=config["if_estimators"], contamination=config["if_contamination"], random_state=config["random_state"])
    modeled["balanced_isolation_forest_anomaly_5pct"] = isolation.fit_predict(scaled) == -1
    modeled["balanced_isolation_forest_score"] = -isolation.decision_function(scaled)
    added = ["balanced_kmeans_cluster_raw", "balanced_cluster_type", "balanced_isolation_forest_anomaly_5pct", "balanced_isolation_forest_score"]
    result = result.merge(modeled[["HDCD"] + added], on="HDCD", how="left", validate="one_to_one")
    return result, pd.DataFrame(validation), profiles.reset_index(), pd.DataFrame(stability)


def add_medical_context(national, boundary, primary, pharmacy):
    """전국 시설 최근접 거리. 엄격한 경계 내부 집계와 기존 경계 보완을 구분한다."""
    boundary = boundary.to_crs(5179).copy()
    require_unique(boundary, ["HDCD"], "전국 경계")
    geo = boundary[["HDCD", "geometry"]].merge(national, on="HDCD", how="left", validate="one_to_one")
    origins = geo.geometry.representative_point()
    assignments = []
    for facilities, prefix in [(primary, "primary_care"), (pharmacy, "pharmacy")]:
        points = facilities.to_crs(5179).reset_index(drop=True)
        if points.empty:
            raise ValueError(f"{prefix}: 유효 시설이 없음")
        distance, indexes = cKDTree(np.c_[points.geometry.x, points.geometry.y]).query(np.c_[origins.x, origins.y])
        geo[f"nearest_{prefix}_distance_m"] = distance
        for field, source in [("id", "암호화요양기호"), ("name", "요양기관명"), ("type", "종별코드명")]:
            geo[f"nearest_{prefix}_{field}"] = points.iloc[indexes][source].to_numpy()
        inside = gpd.sjoin(points[["암호화요양기호", "geometry"]], boundary[["HDCD", "geometry"]], how="left", predicate="within").drop(columns="index_right")
        inside = inside.sort_values(["암호화요양기호", "HDCD"]).drop_duplicates("암호화요양기호")
        counts = inside.groupby("HDCD").size()
        geo[f"{prefix}_count_strict_within"] = geo["HDCD"].map(counts).fillna(0).astype(int)
        inside["assignment_method"] = "within_2023_boundary"
        # 기존 분석의 5km 이내 경계 보완을 재현하되 실제 내부 시설이라고 단정하지 않는다.
        missing = inside.loc[inside["HDCD"].isna(), ["암호화요양기호", "geometry"]]
        nearest = gpd.sjoin_nearest(missing, boundary[["HDCD", "geometry"]], how="left", max_distance=5000, distance_col="boundary_distance_m") if len(missing) else pd.DataFrame()
        if len(nearest):
            nearest = nearest.sort_values(["암호화요양기호", "boundary_distance_m", "HDCD"]).drop_duplicates("암호화요양기호")
            nearest["assignment_method"] = "nearest_boundary_within_5km_legacy"
            assigned = pd.concat([inside.loc[inside["HDCD"].notna()], nearest], ignore_index=True)
        else:
            assigned = inside
        legacy_counts = assigned.groupby("HDCD").size()
        geo[f"{prefix}_count_2025"] = geo["HDCD"].map(legacy_counts).fillna(0).astype(int)
        assigned["facility_group"] = prefix
        assignments.append(pd.DataFrame(assigned.drop(columns=["geometry", "index_right"], errors="ignore")))
    geo["primary_care_absent_within_emd"] = geo["primary_care_count_2025"].eq(0)
    geo["medical_count_boundary_assignment_diff_flag"] = geo["primary_care_count_2025"].ne(geo["primary_care_count_strict_within"])
    geo["is_noncapital"] = ~geo["HDCD_SD_NM"].isin(CAPITAL_PROVINCES)
    geo["is_eup_myeon"] = geo["HDCD_EMD_NM"].str.endswith(("읍", "면"), na=False)
    geo["is_noncapital_eup_myeon"] = geo["is_noncapital"] & geo["is_eup_myeon"]
    geo["policy_eligible"] = geo["is_noncapital_eup_myeon"] & geo["model_included"].fillna(False)
    return geo, pd.concat(assignments, ignore_index=True)


def screen_rural(rural, config, signal_quantile=None, population_weight=None, strict_counts=False):
    """연구자 기준. 군집·이상치 결과를 선별 조건에 넣지 않는다."""
    rural = rural.copy()
    q = config["signal_quantile"] if signal_quantile is None else signal_quantile
    weight = config["population_weight"] if population_weight is None else population_weight
    rural["overall_vulnerability_score"] = weight * rural[SCORE_COLUMNS[0]] + (1 - weight) * rural[SCORE_COLUMNS[1]]
    thresholds = {name: float(rural[col].quantile(q)) for name, col in [
        ("elderly_ratio_p", "elderly_ratio_pct"), ("primary_distance_p_m", "nearest_primary_care_distance_m"),
        ("pt_time_p_min", "22_pt_time_min"), ("time_gap_p", "22_time_gap_ratio")]}
    count_column = "primary_care_count_strict_within" if strict_counts else "primary_care_count_2025"
    rural["population_signal"] = rural["elderly_ratio_pct"].ge(thresholds["elderly_ratio_p"])
    rural["medical_signal"] = rural["nearest_primary_care_distance_m"].ge(thresholds["primary_distance_p_m"]) | rural[count_column].eq(0)
    rural["transport_signal"] = rural["22_pt_time_min"].ge(thresholds["pt_time_p_min"]) | rural["22_time_gap_ratio"].ge(thresholds["time_gap_p"])
    rural["core_signal_count"] = rural[["population_signal", "medical_signal", "transport_signal"]].sum(axis=1)
    thresholds["overall_p80"] = float(rural["overall_vulnerability_score"].quantile(config["priority_quantile"]))
    thresholds["overall_p90"] = float(rural["overall_vulnerability_score"].quantile(config["focus_quantile"]))
    focus = rural["overall_vulnerability_score"].ge(thresholds["overall_p90"]) & rural["core_signal_count"].eq(3)
    priority = rural["overall_vulnerability_score"].ge(thresholds["overall_p80"]) & rural["core_signal_count"].ge(2) & ~focus
    rural["revised_priority_tier"] = np.select([focus, priority], ["집중검토", "우선검토"], default="관찰")
    rural["policy_candidate_primary"] = "현장 수요·기존 서비스 확인"
    rural.loc[rural["transport_signal"], "policy_candidate_primary"] = "기존 버스의 노선·환승·시간표 개선 검토"
    rural.loc[rural["medical_signal"], "policy_candidate_primary"] = "공공택시·의료목적 합승 DRT 비교 검토"
    rural.loc[rural["medical_signal"] & rural["transport_signal"], "policy_candidate_primary"] = "의료목적 DRT·공공택시와 기존 버스 연계 비교"
    rural["policy_assessment_note"] = "정책 확정 아님. 도로·수요·운영능력·기존 지원제도 확인 필요"
    return rural, thresholds


def sensitivity(rural, baseline, config):
    base_ids = set(baseline.loc[baseline["revised_priority_tier"].ne("관찰"), "HDCD"])
    rows, membership = [], []
    for q in config["signal_quantiles_sensitivity"]:
        for weight in config["population_weights_sensitivity"]:
            variant, thresholds = screen_rural(rural, config, q, weight)
            ids = set(variant.loc[variant["revised_priority_tier"].ne("관찰"), "HDCD"])
            rows.append({"signal_quantile": q, "population_weight": weight,
                         "selected_n": len(ids), "focus_n": variant["revised_priority_tier"].eq("집중검토").sum(),
                         "retained_baseline_n": len(ids & base_ids), "jaccard_vs_baseline": len(ids & base_ids) / len(ids | base_ids) if ids | base_ids else 1,
                         **thresholds})
            membership.append(pd.DataFrame({"HDCD": variant["HDCD"], "signal_quantile": q, "population_weight": weight, "selected": variant["revised_priority_tier"].ne("관찰")}))
    long = pd.concat(membership, ignore_index=True)
    frequency = long.groupby("HDCD")["selected"].agg(["sum", "count", "mean"]).reset_index().rename(columns={"sum": "selected_scenario_count", "count": "tested_scenario_count", "mean": "selection_frequency"})
    return pd.DataFrame(rows), long, frequency


def run(inputs, output, config):
    raw = pd.read_csv(inputs / "national_base.csv", dtype={"HDCD": "string"}, low_memory=False)
    scored = score_national(raw, config["population_weight"])
    modeled, validation, profiles, stability = cluster_national(scored, config)
    boundary = gpd.read_file(inputs / "national_boundary.gpkg")
    primary = gpd.read_file(inputs / "primary_care.gpkg")
    pharmacy = gpd.read_file(inputs / "pharmacy.gpkg")
    national, assignment = add_medical_context(modeled, boundary, primary, pharmacy)
    rural, thresholds = screen_rural(national.loc[national["policy_eligible"]], config)
    sens, membership, frequency = sensitivity(rural, rural, config)
    rural = rural.merge(frequency, on="HDCD", validate="one_to_one")
    strict, _ = screen_rural(rural, config, strict_counts=True)
    csv(pd.DataFrame({"HDCD": rural["HDCD"], "legacy_tier": rural["revised_priority_tier"], "strict_within_tier": strict["revised_priority_tier"], "changed": rural["revised_priority_tier"].to_numpy() != strict["revised_priority_tier"].to_numpy()}), output / "medical_boundary_assignment_sensitivity.csv")
    policy_cols = ["HDCD", "population_signal", "medical_signal", "transport_signal", "core_signal_count", "revised_priority_tier", "policy_candidate_primary", "policy_assessment_note", "selection_frequency"]
    national = national.merge(pd.DataFrame(rural[policy_cols]), on="HDCD", how="left", validate="one_to_one")
    national["revised_priority_tier"] = national["revised_priority_tier"].fillna("정책선별 범위 밖")
    national.loc[~national["model_included"].fillna(False), "revised_priority_tier"] = "분석 제외"
    national["balanced_cluster_type"] = national["balanced_cluster_type"].fillna("분석 제외")
    for name, table in [("national_results", national.drop(columns="geometry")), ("rural_policy_candidates", rural.drop(columns="geometry")),
                        ("kmeans_validation", validation), ("cluster_profiles", profiles), ("cluster_stability", stability),
                        ("threshold_weight_sensitivity", sens), ("sensitivity_membership", membership), ("facility_boundary_assignment", assignment)]:
        csv(table, output / f"{name}.csv")
    csv(rural.loc[rural["revised_priority_tier"].ne("관찰")].drop(columns="geometry"), output / "selected_policy_candidates.csv")
    national.to_file(output / "national_results.gpkg", layer="national_results", driver="GPKG")
    json_file(thresholds, output / "screening_thresholds.json")
    return national, rural, validation, profiles, stability
