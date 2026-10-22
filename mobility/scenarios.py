"""탄소·지원 이동량의 조건부 시나리오. 접근성 점수나 AI 입력에 반영하지 않는다."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .common import csv
from .local import MEDICAL_TARGET_TYPES


def calculate_scenario(target, assumptions, road_distance_factor, factors):
    """실제 관측값과 가정을 구분하며 차량-km와 승객-km를 변환한다.

    drt_deadhead_share = 공차 차량-km / 승객탑승 차량-km.
    전체 차량거리 중 공차 비율은 d/(1+d). 따라서 전체거리는 (1+d)를 곱함.
    drt_average_occupancy는 승객탑승 차량거리 기준 거리 가중 평균 탑승인원임.
    """
    for name in ["adoption_rate", "car_substitution_share"]:
        if not 0 <= assumptions[name] <= 1:
            raise ValueError(f"{name}: 비율은 0~1")
    if assumptions["drt_average_occupancy"] <= 0 or assumptions["drt_deadhead_share"] < 0 or road_distance_factor <= 0:
        raise ValueError("탑승인원·거리계수는 양수, 공차 추가비율은 0 이상이어야 함")
    if assumptions["medical_round_trips_per_person_year"] < 0 or factors["car_average_occupancy"] <= 0:
        raise ValueError("왕복횟수·승용차 탑승인원 오류")
    if factors["diesel_drt_km_per_l"] <= 0 or factors["electric_drt_kwh_per_km"] < 0:
        raise ValueError("DRT 에너지 소비 가정 오류")
    if target["elderly_gt_total_flag"].any() or target["elderly_population"].isna().any() or target["elderly_population"].lt(0).any():
        raise ValueError("시나리오 대상에 SGIS 인구 이상값 있음. 임의 보정·0명 대체 금지")
    eligible = float(target["elderly_population"].sum())
    expected_users = eligible * assumptions["adoption_rate"]
    # 인구값은 SGIS 비밀보호 처리값이고, 이용률·횟수는 관측 수요가 아닌 가정임.
    annual_round_trips = expected_users * assumptions["medical_round_trips_per_person_year"]
    annual_pkm = float((target["elderly_population"] * assumptions["adoption_rate"] * assumptions["medical_round_trips_per_person_year"] * 2 * target["nearest_primary_care_distance_m"] / 1000 * road_distance_factor).sum())
    replacement_pkm = annual_pkm * assumptions["car_substitution_share"]
    # 비자동차 대체분을 전부 신규 이동으로 보는 기존 가정. 실측 신규 의료이용이 아님.
    assumed_new_pkm = annual_pkm - replacement_pkm
    baseline_car = replacement_pkm / factors["car_average_occupancy"] * factors["car_gco2_per_vehicle_km"] / 1e6
    occupied_km = annual_pkm / assumptions["drt_average_occupancy"]
    deadhead_km = occupied_km * assumptions["drt_deadhead_share"]
    all_km = occupied_km + deadhead_km
    diesel_vehicle_g = factors["diesel_kgco2_per_l"] * 1000 / factors["diesel_drt_km_per_l"]
    electric_vehicle_g = factors["electric_drt_kwh_per_km"] * factors["electricity_gco2e_per_kwh"]
    diesel_emissions, electric_emissions = all_km * diesel_vehicle_g / 1e6, all_km * electric_vehicle_g / 1e6
    avoided_per_total_pkm = assumptions["car_substitution_share"] * factors["car_gco2_per_vehicle_km"] / factors["car_average_occupancy"]
    return {**assumptions, "road_distance_factor": road_distance_factor, "target_grid_count": len(target),
            "eligible_elderly_sgis_estimate": eligible, "expected_users_assumed": expected_users,
            "annual_round_trips_assumed": annual_round_trips,
            "newly_enabled_round_trips_assumed": annual_round_trips * (1 - assumptions["car_substitution_share"]),
            "annual_passenger_km": annual_pkm, "substituted_car_passenger_km": replacement_pkm,
            "newly_enabled_access_passenger_km": assumed_new_pkm,
            "occupied_vehicle_km": occupied_km, "deadhead_vehicle_km": deadhead_km,
            "total_drt_vehicle_km": all_km,
            "deadhead_fraction_of_total": assumptions["drt_deadhead_share"] / (1 + assumptions["drt_deadhead_share"]),
            "baseline_car_emissions_tco2_year": baseline_car,
            "diesel_drt_emissions_tco2_year": diesel_emissions,
            "diesel_drt_net_change_tco2_year": diesel_emissions - baseline_car,
            "electric_drt_emissions_tco2e_year": electric_emissions,
            "electric_drt_net_change_tco2e_year": electric_emissions - baseline_car,
            "diesel_break_even_occupancy_at_assumed_substitution": diesel_vehicle_g * (1 + assumptions["drt_deadhead_share"]) / avoided_per_total_pkm if avoided_per_total_pkm > 0 else np.nan,
            "electric_break_even_occupancy_at_assumed_substitution": electric_vehicle_g * (1 + assumptions["drt_deadhead_share"]) / avoided_per_total_pkm if avoided_per_total_pkm > 0 else np.nan,
            "interpretation": "가정 기반 운영에너지 비교. 건강·진료이용·시간단축 실측 아님. CO2/CO2e 범위 차이 존재; 전과정평가 아님"}


def run(completed, output, config):
    rows, operation_rows = [], []
    factors = config["carbon"]
    for slug, grid in completed.items():
        target = grid.loc[grid["valid_population"] & grid["recalibrated_priority_type"].isin(MEDICAL_TARGET_TYPES)].copy()
        if target.empty:
            continue
        # 전체 표와 유형별 표를 분리. 전체와 부분집단을 합쳐서 재집계하지 않는다.
        groups = [("의료지원 대상 전체", target)] + list(target.groupby("recalibrated_priority_type"))
        for name, subset in groups:
            for factor, assumptions in product(config["road_distance_factors"], factors["scenarios"]):
                rows.append({"case_slug": slug, "target_group": name, **calculate_scenario(subset, assumptions, factor, factors)})
        base = next(x for x in factors["scenarios"] if x["scenario"] == "기준")
        for occupancy, substitution, deadhead in product([2, 3, 4, 5], [.3, .6, .9], [.15, .25, .4]):
            assumptions = {**base, "drt_average_occupancy": occupancy, "car_substitution_share": substitution, "drt_deadhead_share": deadhead}
            operation_rows.append({"case_slug": slug, **calculate_scenario(target, assumptions, 1.3, factors)})
    result = pd.DataFrame(rows)
    csv(result, output / "carbon_distance_scenarios.csv")
    csv(pd.DataFrame(operation_rows), output / "carbon_operating_condition_sensitivity.csv")
    return result
