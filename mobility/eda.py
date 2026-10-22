"""보고서에 재사용할 원변수 분포·품질·민감도 표와 정적 그림을 만든다."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from .common import bool_column, csv
from .definitions import variable_dictionary
from .national import MOBILITY_FEATURES, POPULATION_FEATURES


def configure_font():
    available = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((x for x in ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"] if x in available), "DejaVu Sans")
    plt.rcParams["axes.unicode_minus"] = False


def run(national, rural, inputs, output, validation, profiles, scenarios):
    configure_font()
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    columns = ["total_population", "elderly_population", "population_density_per_km2"] + POPULATION_FEATURES + MOBILITY_FEATURES
    descriptive = national[columns].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]).T.reset_index(names="variable")
    descriptive["missing_count"] = national[columns].isna().sum().to_numpy()
    csv(descriptive, output / "eda_raw_descriptive.csv")
    csv(national[columns].corr(method="spearman").reset_index(names="variable"), output / "eda_spearman_correlation.csv")
    # 정책선별 범위를 정당화할 비교자료. 차이가 있어도 비수도권 제한의 유일한 증거는 아님.
    group = np.where(national["is_noncapital"], "비수도권 ", "수도권 ") + np.where(national["is_eup_myeon"], "읍·면", "동·기타")
    comparison = national.assign(scope_group=group).groupby("scope_group")[columns].agg(["count", "median", "mean"])
    comparison.columns = [f"{a}__{b}" for a, b in comparison.columns]
    csv(comparison.reset_index(), output / "eda_capital_rural_comparison.csv")
    quality = []
    for column in ["population_data_invalid_flag", "incomplete_indicator_flag", "medical_count_boundary_assignment_diff_flag"]:
        quality.append({"check": column, "count": int(bool_column(national[column]).sum())})
    for col in ["22_pt_time_min_censored", "22_car_time_min_censored"]:
        quality.append({"check": col, "count": int(bool_column(national[col]).sum())})
    csv(pd.DataFrame(quality), output / "eda_quality_counts.csv")
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    definitions = variable_dictionary().set_index("variable")
    for ax, column in zip(axes.flat, ["elderly_ratio_pct", "population_density_per_km2", "inverse_log_population_density"] + MOBILITY_FEATURES):
        values = national[column].dropna()
        ax.hist(values, bins=35, color="#555555", edgecolor="white", linewidth=.3)
        ax.set_title(column, fontsize=9)
        ax.set_ylabel("읍면동 수")
        ax.set_xlabel(f"값 ({definitions.loc[column, 'unit']})", fontsize=8)
    fig.savefig(figures / "raw_indicator_distributions.png", dpi=180)
    plt.close(fig)
    valid = national.loc[national["model_included"].fillna(False)]
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for label, frame in valid.groupby("balanced_cluster_type"):
        ax.scatter(frame["population_vulnerability_score"], frame["location_vulnerability_score"], s=8, alpha=.45, label=label)
    ax.set(xlabel="인구취약성 (점)", ylabel="이동취약성 (점)", title="전국 두 취약점수의 K-means 유형화")
    ax.legend(fontsize=8)
    fig.savefig(figures / "national_two_axis_clusters.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(validation["k"], validation["silhouette_score"], marker="o", color="#333333")
    ax.set(xlabel="군집 수 K", ylabel="실루엣 계수", title="K=4는 분리도 최대가 아닌 해석 목적의 선택")
    fig.savefig(figures / "kmeans_k_comparison.png", dpi=180)
    plt.close(fig)
    sensitivity = pd.read_csv(output / "threshold_weight_sensitivity.csv")
    matrix = sensitivity.pivot(index="signal_quantile", columns="population_weight", values="selected_n")
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.imshow(matrix.to_numpy(), cmap="Greys")
    ax.set(xticks=range(len(matrix.columns)), xticklabels=matrix.columns, yticks=range(len(matrix.index)), yticklabels=matrix.index,
           xlabel="인구취약성 가중치", ylabel="신호 백분위 기준", title="기준 변경에 따른 정책검토 후보 수")
    for i, j in np.ndindex(matrix.shape):
        ax.text(j, i, str(matrix.iloc[i, j]), ha="center", va="center", color="#B23A3A")
    fig.savefig(figures / "screening_threshold_sensitivity.png", dpi=180)
    plt.close(fig)
    if len(scenarios):
        base = scenarios.loc[scenarios["scenario"].eq("기준") & scenarios["target_group"].eq("의료지원 대상 전체")]
        for slug, table in base.groupby("case_slug"):
            fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
            ax.plot(table["road_distance_factor"], table["diesel_drt_net_change_tco2_year"], marker="o", label="디젤: 직접 CO2")
            ax.plot(table["road_distance_factor"], table["electric_drt_net_change_tco2e_year"], marker="s", label="전기: 전력 CO2e 포함 근사")
            ax.axhline(0, color="#444444", lw=.8)
            ax.set(xlabel="직선거리 대비 가정 거리계수", ylabel="연간 순배출 변화 (톤)", title=f"{slug}: 조건부 탄소 시나리오, 실제 감축량 아님")
            ax.legend(fontsize=8)
            fig.savefig(figures / f"{slug}_carbon_distance_sensitivity.png", dpi=180)
            plt.close(fig)
