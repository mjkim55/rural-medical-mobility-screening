"""확인한 KTDB·행안부·HIRA 원본의 실제 열 이름을 사용하는 입력 어댑터.

기존 전국 모델의 복합키 피벗·이름결합 절차를 독립 모듈로 옮겼다.
KTDB의 시간비는 동일 시설유형 지표 간 비율이지 동일 병원 OD 여행시간비가 아니다.
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .common import require_columns, require_unique

KEY_COLUMNS = ["Year", "HDCD", "Region", "HDCD_Lev", "Faci_CD", "Time_Zone"]
DISPLAY_COLUMNS = ["HDCD_SD_NM", "HDCD_SGG_NM", "HDCD_EMD_NM", "Region_NM", "Faci_CA", "Faci_NM", "Time_Zone_NM"]
FIXED_COLUMNS = KEY_COLUMNS + ["Mode"] + DISPLAY_COLUMNS + ["Mode_NM"]
FACILITY_LABELS = {"21": "공공의료시설", "22": "병·의원", "23": "종합병원", "31": "대규모점포", "32": "전통시장", "41": "버스터미널", "42": "철도역"}
PRIMARY_TYPES = ["의원", "보건소", "보건지소", "보건진료소"]


def parse_censored(value):
    if pd.isna(value):
        return np.nan, False
    text = str(value).strip().replace(",", "")
    censored = text.startswith(">")
    try:
        return float(text[1:] if censored else text), censored
    except ValueError:
        return np.nan, False


def load_ktdb(path: Path, sheet: str, threshold=False):
    dtype = {name: "string" for name in KEY_COLUMNS + ["Mode"]}
    if threshold:
        frame = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=6,
                              names=FIXED_COLUMNS + ["15분 이내", "30분 이내", "45분 이내", "60분 이내"],
                              usecols="A:S", dtype=dtype)
        value_column = "30분 이내"
    else:
        frame = pd.read_excel(path, sheet_name=sheet, header=4,
                              usecols=FIXED_COLUMNS + ["평균접근시간(분)"], dtype=dtype)
        value_column = "평균접근시간(분)"
    frame = frame.loc[frame["HDCD_Lev"].eq("4") & frame["Time_Zone"].eq("0_AllDay")
                      & frame["Faci_CD"].isin(FACILITY_LABELS) & frame["Mode"].isin(["1_PC", "2_PT"])].copy()
    if set(frame["Year"].dropna().unique()) != {"2023"}:
        raise ValueError("원자료 Year가 2023과 다름. 연도·경계·인구를 검토한 뒤 설정해야 함")
    require_unique(frame, KEY_COLUMNS + ["Mode"], path.name)
    parsed = frame[value_column].map(parse_censored)
    frame["value"] = parsed.map(lambda x: x[0])
    frame["censored"] = parsed.map(lambda x: x[1])
    return frame


def pivot_modes(frame, value_name):
    require_unique(frame, KEY_COLUMNS + ["Mode"], value_name)
    result = frame[KEY_COLUMNS + DISPLAY_COLUMNS].drop_duplicates(KEY_COLUMNS)
    for raw_col, suffix in [("value", ""), ("censored", "_censored")]:
        wide = frame.pivot(index=KEY_COLUMNS, columns="Mode", values=raw_col).reset_index()
        wide = wide.rename(columns={"1_PC": f"car_{value_name}{suffix}", "2_PT": f"pt_{value_name}{suffix}"})
        result = result.merge(wide, on=KEY_COLUMNS, validate="one_to_one", how="left")
    return result


def build_accessibility_table(directory: Path):
    time = pivot_modes(load_ktdb(directory / "01_평균접근시간_2023.xlsx", "평균접근시간"), "time_min")
    reach = pivot_modes(load_ktdb(directory / "02_접근가능인구비율_2023.xlsx", "접근 가능 인구 비율", True), "reach30_pct")
    facility = pivot_modes(load_ktdb(directory / "03_접근가능시설수_2023.xlsx", "접근 가능 시설 수", True), "facility30_count")
    for other in [reach, facility]:
        extra = [column for column in other if column not in time]
        time = time.merge(other[KEY_COLUMNS + extra], on=KEY_COLUMNS, how="left", validate="one_to_one")
    time["time_gap_ratio"] = time["pt_time_min"].div(time["car_time_min"].where(time["car_time_min"].gt(0)))
    time["extra_time_min"] = time["pt_time_min"] - time["car_time_min"]
    time["reach30_gap_pct"] = time["car_reach30_pct"] - time["pt_reach30_pct"]
    time["facility30_gap_count"] = time["car_facility30_count"] - time["pt_facility30_count"]
    return time


def make_emd_wide(access):
    identity = ["HDCD", "HDCD_SD_NM", "HDCD_SGG_NM", "HDCD_EMD_NM", "Region", "Region_NM"]
    result = access[identity].drop_duplicates("HDCD")
    metrics = ["car_time_min", "pt_time_min", "time_gap_ratio", "extra_time_min", "car_reach30_pct", "pt_reach30_pct", "reach30_gap_pct", "car_facility30_count", "pt_facility30_count", "facility30_gap_count", "car_time_min_censored", "pt_time_min_censored"]
    for code in FACILITY_LABELS:
        subset = access.loc[access["Faci_CD"].eq(code), ["HDCD"] + metrics]
        require_unique(subset, ["HDCD"], f"시설 {code}")
        result = result.merge(subset.rename(columns={col: f"{code}_{col}" for col in metrics}), on="HDCD", how="left", validate="one_to_one")
    return result


def normalize_name(value, loose=False):
    text = "" if pd.isna(value) else str(value)
    text = text.replace("ㆍ", "·").replace("・", "·")
    text = re.sub(r"[\s\-·\.]", "", text).replace("세종특별자치시세종시", "세종특별자치시")
    text = re.sub(r"제제(?=\d)", "제", text)
    return re.sub(r"제(?=\d)", "", text) if loose else text


def join_population(emd, path):
    pop = pd.read_excel(path, sheet_name="연령별인구현황", header=3, dtype={"행정기관코드": "string", "행정기관": "string"})
    pop = pop.loc[~pop["행정기관코드"].str.zfill(10).str.endswith("00000")].copy()
    pop = pop.rename(columns={"총 인구수": "total_population", "연령구간인구수": "elderly_population"})
    for col in ["total_population", "elderly_population"]:
        pop[col] = pd.to_numeric(pop[col].astype("string").str.replace(",", "", regex=False), errors="coerce")
    wanted = ["행정기관코드", "total_population", "elderly_population"]
    for loose in [False, True]:
        pop["name_key"] = pop["행정기관"].map(lambda x: normalize_name(x, loose))
        lookup = pop.loc[~pop["name_key"].duplicated(keep=False)].set_index("name_key")
        names = emd[["HDCD_SD_NM", "HDCD_SGG_NM", "HDCD_EMD_NM"]].replace("-", pd.NA).apply(lambda row: " ".join(row.dropna().astype(str)), axis=1)
        key = names.map(lambda x: normalize_name(x, loose))
        missing = emd["total_population"].isna() if "total_population" in emd else pd.Series(True, index=emd.index)
        for col in wanted:
            if col not in emd:
                emd[col] = pd.Series(pd.NA, index=emd.index, dtype="string") if col == "행정기관코드" else np.nan
            mapped = key.loc[missing].map(lookup[col])
            emd.loc[missing, col] = mapped if col == "행정기관코드" else mapped.astype(float)
    return emd


def load_hira(path, sheet):
    columns = ["암호화요양기호", "요양기관명", "종별코드명", "시군구코드명", "좌표(X)", "좌표(Y)"]
    frame = pd.read_excel(path, sheet_name=sheet, usecols=columns)
    require_columns(frame, columns, "HIRA")
    for col in ["좌표(X)", "좌표(Y)"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    valid = frame["좌표(X)"].between(124, 132) & frame["좌표(Y)"].between(33, 39)
    dropped = frame.loc[~valid].copy()
    frame = frame.loc[valid].copy()
    require_unique(frame, ["암호화요양기호"], "HIRA 좌표")
    points = gpd.GeoDataFrame(frame, geometry=gpd.points_from_xy(frame["좌표(X)"], frame["좌표(Y)"]), crs=4326).to_crs(5179)
    return points, dropped
