"""공통 입출력과 자료 계약. 원자료와 기존 산출물은 덮어쓰지 않는다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not 0 <= config["population_weight"] <= 1:
        raise ValueError("population_weight는 0~1이어야 함")
    if not 0 < config["priority_quantile"] < config["focus_quantile"] < 1:
        raise ValueError("우선·집중검토 분위수 순서가 잘못됨")
    return config


def require_columns(frame, columns, name="자료"):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: 필수 열 없음 {sorted(missing)}")


def require_unique(frame, keys, name="자료"):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{name}: {keys} 결측 또는 중복. 임의 첫 행 선택 금지")


def csv(frame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def json_file(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON 표준상 NaN/Infinity를 허용하지 않는다. 결측은 null로 기록한다.
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_column(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.lower().isin(["true", "1", "yes"])
