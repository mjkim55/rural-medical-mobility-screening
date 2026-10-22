"""노선 동일키 충돌과 연결 미확인을 임의 0·첫 행으로 숨기지 않는지 검사한다."""
import geopandas as gpd
import pandas as pd

from mobility.local import stop_supply, classify_grid
from mobility.routes import quarantine_relations


def test_conflicting_stop_id_is_quarantined_not_first_selected():
    source = pd.DataFrame({"routeid": ["test_R1", "test_R1", "test_R2", "test_R3"],
        "nodeid": ["test_S1", "test_S1", "test_S1", "test_S2"], "nodeord": [1, 1, 2, 1],
        "nodenm": ["A", "B", "A", "C"], "gpslong": [127., 128., 127., 127.], "gpslati": [36.] * 4})
    cleaned, conflicts, quarantined, ids = quarantine_relations(source)
    assert len(conflicts) == 2 and len(quarantined) == 3
    assert ids == {"test_S1"} and cleaned["nodeid"].tolist() == ["test_S2"]
    assert len(source) == 4


def test_unmatched_nearest_stop_count_unknown_not_zero():
    stops = pd.DataFrame({"nodeid": ["test_S1"], "nodenm": ["A"], "gpslong": [127.], "gpslati": [36.]})
    origins = gpd.GeoSeries(gpd.points_from_xy([127.], [36.]), crs=4326).to_crs(5179)
    routes = pd.DataFrame(columns=["routeid", "nodeid", "nodeord"])
    result, _ = stop_supply(origins, stops, routes)
    assert pd.isna(result.loc[0, "nearest_stop_route_count"])
    assert result.loc[0, "unmatched_stop_count_within_500m"] == 1


def test_unmatched_zero_observed_routes_not_called_route_poverty():
    data = pd.DataFrame({"valid_population": [True], "elderly_ratio_analysis_pct": [20.], "elderly_population": [2.],
        "elderly_gt_total_flag": [False], "nearest_primary_care_distance_m": [500.],
        "nearest_stop_distance_m_recalculated": [10.], "stops_within_500m": [1], "routes_within_500m": [0.],
        "unmatched_stop_count_within_500m": [1], "22_pt_time_min": [40.]})
    result = classify_grid(data, {"medical_far_quantile": .75, "stop_radius_m": 500, "route_poverty_max": 2})
    assert result.loc[0, "bus_supply_type"] == "노선 연결 추가확인"
    assert not result.loc[0, "route_supply_low_within_500m"]
