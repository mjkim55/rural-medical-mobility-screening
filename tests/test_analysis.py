"""합성 입력으로 수식·경계조건을 검사한다. 합성값을 실제 분석결과로 사용하지 않음."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from mobility.common import read_config, require_unique
from mobility.local import classify_grid, stop_supply
from mobility.national import SCORE_COLUMNS, score_national, screen_rural
from mobility.raw import KEY_COLUMNS, parse_censored, pivot_modes
from mobility.scenarios import calculate_scenario

ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_config(ROOT / "config.json")


def test_actual_kmeans_inputs_have_only_two_scores():
    assert SCORE_COLUMNS == ["population_vulnerability_score", "location_vulnerability_score"]
    assert "overall_vulnerability_score" not in SCORE_COLUMNS


def test_censored_value_not_exact_travel_time():
    assert parse_censored(">120") == (120, True)
    assert parse_censored(120) == (120, False)
    assert np.isnan(parse_censored(None)[0])


def test_duplicate_keys_raise():
    with pytest.raises(ValueError):
        require_unique(pd.DataFrame({"HDCD": ["123", "123"]}), ["HDCD"])


def test_duplicate_composite_mode_key_raise():
    frame = pd.DataFrame([{**dict.fromkeys(KEY_COLUMNS, "same"), "Mode": "1_PC"}] * 2)
    with pytest.raises(ValueError):
        pivot_modes(frame, "time_min")


def test_monotone_log_does_not_change_percentile_ranks():
    density = pd.Series([0, 10, 100, 10000])
    assert np.allclose((-density).rank(pct=True), (-np.log1p(density)).rank(pct=True))


def test_domain_average_prevents_six_to_two_count_weighting():
    n = 10
    frame = pd.DataFrame({"HDCD": [str(i) for i in range(n)], "total_population": [100.] * n,
                          "elderly_population": np.arange(1, n + 1) * 5., "area_km2": np.arange(1, n + 1),
                          "22_pt_time_min": np.arange(n) + 1., "22_time_gap_ratio": np.arange(n) + 1.,
                          "22_reach30_gap_pct": np.arange(n) + 1., "23_pt_time_min": np.arange(n) + 1.,
                          "31_pt_time_min": np.arange(n) + 1., "32_pt_time_min": np.arange(n) + 2.,
                          "41_pt_time_min": np.arange(n) + 1., "42_pt_time_min": np.arange(n) + 2.})
    result = score_national(frame, .5)
    assert np.allclose(result["overall_vulnerability_score"], (result[SCORE_COLUMNS[0]] + result[SCORE_COLUMNS[1]]) / 2)
    weighted = score_national(frame, .3)
    assert np.allclose(weighted["overall_vulnerability_score"], .3 * result[SCORE_COLUMNS[0]] + .7 * result[SCORE_COLUMNS[1]])
    assert result["population_indicator_available_count"].eq(2).all()
    assert result["mobility_indicator_available_count"].eq(6).all()
    frame.loc[0, "22_pt_time_min"] = np.nan
    missing = score_national(frame)
    assert missing.loc[0, "incomplete_indicator_flag"]
    assert pd.isna(missing.loc[0, "22_pt_time_min"])


def target():
    return pd.DataFrame({"elderly_population": [100.0], "nearest_primary_care_distance_m": [1000.0], "elderly_gt_total_flag": [False]})


def test_carbon_occupancy_and_additional_deadhead():
    assumptions = CONFIG["carbon"]["scenarios"][1]
    r = calculate_scenario(target(), assumptions, 1.3, CONFIG["carbon"])
    assert r["annual_passenger_km"] == pytest.approx(100 * .25 * 12 * 2 * 1 * 1.3)
    assert r["occupied_vehicle_km"] == pytest.approx(r["annual_passenger_km"] / 3)
    assert r["deadhead_vehicle_km"] == pytest.approx(r["occupied_vehicle_km"] * .25)
    assert r["deadhead_fraction_of_total"] == pytest.approx(.20)
    assert r["total_drt_vehicle_km"] == pytest.approx(r["annual_passenger_km"] / 3 * 1.25)


def test_distance_factor_scales_net_emissions_not_sign():
    scenario = CONFIG["carbon"]["scenarios"][1]
    a = calculate_scenario(target(), scenario, 1.1, CONFIG["carbon"])
    b = calculate_scenario(target(), scenario, 1.5, CONFIG["carbon"])
    assert b["electric_drt_net_change_tco2e_year"] == pytest.approx(a["electric_drt_net_change_tco2e_year"] * 1.5 / 1.1)


def test_zero_substitution_does_not_claim_car_avoidance():
    assumptions = {**CONFIG["carbon"]["scenarios"][1], "car_substitution_share": 0}
    result = calculate_scenario(target(), assumptions, 1.3, CONFIG["carbon"])
    assert result["baseline_car_emissions_tco2_year"] == 0
    assert result["electric_drt_net_change_tco2e_year"] > 0
    assert np.isnan(result["electric_break_even_occupancy_at_assumed_substitution"])


def test_anomalous_elderly_never_silently_capped():
    frame = target()
    frame["elderly_gt_total_flag"] = True
    with pytest.raises(ValueError):
        calculate_scenario(frame, CONFIG["carbon"]["scenarios"][1], 1.3, CONFIG["carbon"])


@pytest.mark.parametrize("field,value", [("adoption_rate", 1.1), ("drt_average_occupancy", 0), ("drt_deadhead_share", -.1)])
def test_invalid_carbon_inputs_raise(field, value):
    assumptions = {**CONFIG["carbon"]["scenarios"][1], field: value}
    with pytest.raises(ValueError):
        calculate_scenario(target(), assumptions, 1.3, CONFIG["carbon"])


def test_route_counts_are_union_not_sum_or_nearest_only():
    stops = pd.DataFrame({"nodeid": ["S1", "S2"], "nodenm": ["A", "B"], "gpslong": [127, 127.001], "gpslati": [36, 36]})
    centers = gpd.GeoSeries(gpd.points_from_xy([127], [36]), crs=4326).to_crs(5179)
    routes = pd.DataFrame({"nodeid": ["S1", "S1", "S2", "S2"], "routeid": ["R1", "R2", "R2", "R3"]})
    result, _ = stop_supply(centers, stops, routes)
    assert result.iloc[0]["routes_within_500m"] == 3
    assert result.iloc[0]["nearest_stop_route_count"] == 2
    assert result.iloc[0]["route_stop_relations_within_500m"] == 4


def test_missing_routes_are_na_not_zero():
    stops = pd.DataFrame({"nodeid": ["S1"], "nodenm": ["A"], "gpslong": [127], "gpslati": [36]})
    centers = gpd.GeoSeries(gpd.points_from_xy([127], [36]), crs=4326).to_crs(5179)
    result, _ = stop_supply(centers, stops, None)
    assert result["routes_within_500m"].isna().all()
    assert not result["route_data_available"].any()


def test_screening_independent_of_ai_labels():
    n = 100
    frame = pd.DataFrame({"HDCD": [str(i) for i in range(n)], "population_vulnerability_score": np.arange(n), "location_vulnerability_score": np.arange(n),
                          "elderly_ratio_pct": np.arange(n), "nearest_primary_care_distance_m": np.arange(n) * 100,
                          "22_pt_time_min": np.arange(n), "22_time_gap_ratio": np.arange(n) / 10, "primary_care_count_2025": 1})
    a, thresholds = screen_rural(frame, CONFIG)
    frame["balanced_cluster_type"] = "전혀 다른 군집 이름"
    frame["balanced_isolation_forest_anomaly_5pct"] = True
    b, _ = screen_rural(frame, CONFIG)
    assert a["revised_priority_tier"].equals(b["revised_priority_tier"])
    equal = a.loc[a["revised_priority_tier"].eq("집중검토")]
    assert equal["core_signal_count"].eq(3).all()


def test_actual_results_regression_when_available():
    path = ROOT / "results/national_results.csv"
    if not path.exists():
        pytest.skip("원자료 비공개 CI: 실제자료 회귀검사는 로컬 전체 실행 후 수행")
    national = pd.read_csv(path)
    tiers = pd.read_csv(ROOT / "results/rural_policy_candidates.csv")
    grid = pd.read_csv(ROOT / "results/cases/iksan/grid_results.csv")
    assert len(national) == 3530
    assert national["model_included"].sum() == 3525
    assert len(tiers) == 1247
    assert tiers["revised_priority_tier"].eq("집중검토").sum() == 19
    assert tiers["revised_priority_tier"].eq("우선검토").sum() == 132
    assert len(grid) == 508 and grid["valid_population"].sum() == 488
    assert grid["elderly_gt_total_flag"].sum() == 13
    # 서로 다른 분모인 격자비율과 인구비율을 혼동하지 않도록 별도 확인한다.
    poor = grid["valid_population"] & grid["bus_supply_type"].isin(["정류장 미비", "노선 빈곤"])
    assert poor.sum() == 241
    assert grid.loc[poor, "total_population"].sum() == 19834
