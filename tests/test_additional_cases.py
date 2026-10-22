"""추가 사례 어댑터: 합성 단위검사와 실제 파일이 있는 로컬 회귀검사를 구분한다."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree
from shapely.geometry import box

from prepare_additional_cases import join_grid_population

ROOT = Path(__file__).resolve().parents[1]


def test_missing_population_not_zero_and_source_unchanged():
    boundaries = gpd.GeoDataFrame({"GRID_CD": ["test_A", "test_B", "test_C"]},
                                  geometry=[box(i * 1000, 0, (i + 1) * 1000, 1000) for i in range(3)], crs=5179)
    population = pd.DataFrame({"year": [2024, 2024], "grid_id": ["test_A", "test_B"],
                               "total_population": [10, 0], "elderly_population": [12, 0]})
    result = join_grid_population(boundaries, population)
    assert result["population_status"].tolist() == ["유효", "인구 0명", "인구표 미수록"]
    assert result["population_record_present"].tolist() == [True, True, False]
    assert pd.isna(result.loc[2, "total_population"])
    assert result.loc[0, "elderly_population"] == 12  # 임의 클리핑 없음
    assert result["grid_boundary_year"].eq(2025).all()
    assert "grid_id" not in boundaries.columns


def test_duplicate_population_id_raises():
    boundaries = gpd.GeoDataFrame({"GRID_CD": ["test_A"]}, geometry=[box(0, 0, 1000, 1000)], crs=5179)
    population = pd.DataFrame({"year": [2024, 2024], "grid_id": ["test_A", "test_A"],
                               "total_population": [10, 10], "elderly_population": [5, 5]})
    with pytest.raises(ValueError):
        join_grid_population(boundaries, population)


def test_additional_cases_against_full_official_stop_file_when_available():
    path = ROOT / "data/raw/official_stops/national_bus_stops_20251031.csv"
    if not path.exists() or not (ROOT / "results/cases/gabuk/grid_results.gpkg").exists():
        pytest.skip("공개 CI에는 공식 원자료를 포함하지 않음")
    raw = pd.read_csv(path, encoding="cp949", dtype="string")
    x = pd.to_numeric(raw["경도"], errors="coerce")
    y = pd.to_numeric(raw["위도"], errors="coerce")
    valid = x.between(124, 132) & y.between(33, 39)
    points = gpd.GeoSeries(gpd.points_from_xy(x[valid], y[valid]), crs=4326).to_crs(5179)
    tree = cKDTree(np.c_[points.x, points.y])
    for slug, count in {"gabuk": 50, "yonghwa": 33, "gyeolseong": 29, "yeonghyeon": 24}.items():
        grid = gpd.read_file(ROOT / f"results/cases/{slug}/grid_results.gpkg")
        centers = grid.geometry.centroid
        xy = np.c_[centers.x, centers.y]
        distance, _ = tree.query(xy)
        counts = [len(indexes) for indexes in tree.query_ball_point(xy, 500)]
        assert np.allclose(distance, grid["nearest_stop_distance_m_recalculated"], atol=1e-7)
        assert np.array_equal(counts, grid["stops_within_500m"])
        assert grid["valid_population"].sum() == count
        assert grid["routes_within_500m"].notna().all()
        assert grid["route_data_available"].all()
        assert not grid.loc[grid["valid_population"], "route_coverage_unknown_flag"].any()
        assert grid.loc[grid["elderly_gt_total_flag"], "elderly_ratio_analysis_pct"].isna().all()
        assert grid.loc[~grid["population_record_present"], "total_population"].isna().all()
        # 분석 함수와 별개로 각 중심점의 반경에 드는 ID와 원 노선관계를 직접 대조함.
        stops = pd.read_csv(ROOT / f"data/inputs/{slug}_stops.csv", dtype={"nodeid": str})
        stop_points = gpd.GeoSeries(gpd.points_from_xy(stops["gpslong"], stops["gpslati"]), crs=4326).to_crs(5179)
        routes = pd.read_csv(ROOT / f"data/raw/tago_20261022/{slug}/route_stops.csv", dtype={"nodeid": str, "routeid": str})
        signature = routes.groupby(["routeid", "nodeid", "nodeord"])[["nodenm", "gpslong", "gpslati", "updowncd"]].nunique(dropna=False)
        bad_ids = set(signature.loc[signature.gt(1).any(axis=1)].index.get_level_values("nodeid"))
        routes = routes.loc[~routes["nodeid"].isin(bad_ids)]
        for index, origin in enumerate(centers):
            ids = stops.loc[stop_points.distance(origin).le(500), "nodeid"]
            expected = routes.loc[routes["nodeid"].isin(ids), "routeid"].nunique()
            assert grid.iloc[index]["routes_within_500m"] == expected
