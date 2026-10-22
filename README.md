# 고령층 의료이동 취약지역 선별 분석

공공데이터로 전국 읍면동의 인구·이동 취약구조를 유형화하고, 비수도권 읍·면의 기초의료 이동지원 검토지역을 선별함. 익산시와 추가 네 사례에서 실제 1km 격자·인구·의료시설·정류장·노선을 결합해 상세 근거를 확인함. 분석판은 `2026-10-22-final-v3`임.

이 모델은 현장검토 대상을 선별하고 정책 후보와 운영조건을 비교하는 분석 도구임. 현재 운행을 보증하는 버스 안내, 정류장 설치 최적화, 예산 최적화, 의료이용·건강·시간단축의 인과효과 추정은 수행하지 않음.

## 최종 제공물

| 파일 | 용도 |
| --- | --- |
| `analysis.ipynb` | 출력·개인경로 없는 공개 코드용 노트북 |
| `analysis_executed.ipynb` | 실제 결과·코드 해설·품질검사가 포함된 로컬 검토용 실행 노트북 |
| `results/national_interactive_map.html` | 인터넷·API 키 없이 열리는 전국·5개 사례 클릭지도 |
| `docs/METHODS.md` | 개념·계산식·AI·신호·정책·탄소의 정의 |
| `docs/INPUTS.md`, `docs/SOURCES.md` | 실제 입력 구조·파일 배치·출처·연도·이용조건 |

`package_code.py`가 만드는 코드 전용 ZIP은 GitHub 업로드 준비물임. 실제 저장소 생성·업로드·공개 링크 발급을 수행하지 않음.

## 실행한 분석 범위

- 전국 경계 3,530곳 중 모델 포함 3,525곳. 비수도권 읍·면 1,247곳을 별도 선별하여 집중 19곳·우선 132곳, 합계 151곳의 기준모형 후보를 얻음.
- K-means는 인구취약성·이동취약성 두 축만 사용함. Overall과 세 신호는 별도 후보 선별에 사용함. K=4는 해석 목적의 선택이며 실루엣 최고 K=2와 구분함. Isolation Forest는 특이패턴 표시이지 고위험의 정답이 아님.
- 신호 기준·인구가중치 9조합, 의료시설 경계 배정 방식, 의료 절대거리 1·3·5km를 비교함.
- 익산 508격자 중 유효인구 488격자, SGIS 고령>총인구 표시 13격자, 의료지원 시나리오 대상 57격자를 재현함.
- 추가 사례 유효인구 격자는 가북면 50개·용화면 33개·결성면 29개·영현면 24개임. 전국 대표표본이나 네 군집의 대표가 아니라 서로 다른 신호조합의 목적표집임.
- 추가 네 군 TAGO 539개 노선·24,073개 경유 레코드를 실제 읽고 격자별 고유노선 합집합을 재계산함. 관계 레코드 수는 운행횟수가 아님.
- 거리계수 1.1·1.3·1.5와 자동차대체율·합승·추가 공차 36조합을 이용한 조건부 지원 이동량·탄소를 계산함. 관측된 서비스 효과와 구분함.

## 입력이 준비된 로컬 환경에서 실행

Python 3.11 이상을 대상으로 하며 실제 실행 검증은 Python 3.13에서 수행함. 운영체제 글꼴에 따라 그림의 한글 표시가 달라질 수 있음.

```bash
python -m pip install -r requirements.txt
python run_analysis.py --strict
python -m pytest -q
```

`config.json`의 경로는 설정파일 폴더를 기준으로 해석함. 같은 출력 폴더에 재실행하면 그 폴더의 생성 결과만 갱신함. `output_dir`를 바꾸면 다른 설정의 결과를 별도로 보존할 수 있음.

`strict`는 5개 사례의 격자·노선 계산이 미완료면 실패함. 성공해도 원자료 완전성·현재 운행·정책효과가 검증됐다는 뜻은 아님. `results/quality_checks.csv`의 경고와 오류를 함께 확인해야 함.

## 처음 자료를 준비하는 경우

코드 ZIP에는 원자료가 없음. 공식 자료를 별도로 확보하고 [입력 계약](docs/INPUTS.md)에 맞게 배치해야 함. 원래 프로젝트·다운로드 파일을 보유했다면 아래 준비 도구를 사용할 수 있음.

```bash
python prepare_inputs.py --project-root /path/to/original-project --downloads /path/to/downloads --mode raw
python prepare_additional_cases.py --grid-zips /path/to/grid_tile_1.zip /path/to/grid_tile_2.zip /path/to/grid_tile_3.zip /path/to/grid_tile_4.zip --stops-csv /path/to/national_bus_stops_20251031.csv
```

TAGO 다운로드 묶음의 내용을 `data/raw/tago_20261022/`에 배치함. 새로 API를 수집할 때만 별도 승인이 유효한 키 파일을 인자로 전달함. 키 파일은 저장소·공개 ZIP에 넣지 않음.

```bash
python collect_tago_routes.py --key-file /path/to/private_key_file --discover
python collect_tago_routes.py --key-file /path/to/private_key_file --probe-city 34360
python collect_tago_routes.py --key-file /path/to/private_key_file --city 38390:gabuk --city 33340:yonghwa --city 34360:gyeolseong --city 38340:yeonghyeon
```

홍성군은 당시 도시목록에서 누락됐지만 공식 정류장 파일의 코드 34360으로 직접 노선 조회가 성공했음. 위 `--probe-city`는 그 직접조회 증거를 남기며 이 옵션이 있으면 수집 대신 점검 후 종료함. 조회시점마다 서비스 범위·코드는 다시 확인해야 함. 기존 성공 응답은 캐시를 재사용하므로 날짜를 바꿔 새로 수집할 때는 별도 `--output`을 지정하고 설정·출처를 함께 갱신해야 함.

## 노트북과 지도

```bash
python build_notebook.py
jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis_executed.ipynb --output-dir . --ExecutePreprocessor.timeout=600
```

실행 노트북은 저장소 루트에 보관함. 지도는 `results/national_interactive_map.html`을 브라우저에서 열면 됨. 전국 지역 클릭 시 점수·원지표·세 신호·후보 정책이 표시됨. 분석 범위에서 익산·네 사례를 선택하거나 상세창 버튼으로 이동하면 1km 근거를 확인할 수 있음. 탄소 창은 해당 사례 의료지원 대상 전체의 가정 계산이며 선택한 한 격자의 효과가 아님.

## 핵심 출력과 품질관리

| 결과 | 내용 |
| --- | --- |
| `results/run_summary.json` | 분석판·후보 수·사례 상태·오류·경고 개수 |
| `results/national_results.csv`, `.gpkg` | 전국 원지표·두 점수·Overall·AI 유형·별도 신호 |
| `results/selected_policy_candidates.csv` | 기준모형의 151개 검토 후보 |
| `results/screening_thresholds.json` | 실제 신호·등급 분위수 기준 |
| `results/case_comparison.csv` | 동일 정의의 사례별 격자·인구·의료거리·정류장 집계 |
| `results/case_completion_status.csv` | 노선 계산 완료와 원 수집 완전성을 구분한 상태 |
| `results/cases/*/grid_results.csv`, `.gpkg` | 새 노선까지 반영한 격자별 최종 결과 |
| `results/cases/*/route_input_conflicts.csv`, `route_input_quarantine.csv` | 원 응답 충돌·분석 제외 레코드 |
| `results/cases/*/stop_route_link_quality.csv` | 정류장 ID·API 좌표 비교·연결 품질 |
| `results/carbon_distance_scenarios.csv` | 거리계수·시나리오·대상유형별 조건부 계산 |
| `results/quality_checks.csv` | 계산 오류와 남은 자료 검증 경고 |
| `results/input_checksums.json`, `config_used.json`, `runtime_versions.json` | 실제 입력 해시·설정·환경 |

홍성군 동일키 충돌 64그룹·48개 정류장 ID의 관계 128행은 격리함. 해당 ID는 현재 결성면 유효인구 격자 500m 정류장과 겹치지 않음. 익산은 기존 노선목록 일부의 경유자료와 정류장 통계 기준일이 미확인이고, 1개 유효 격자의 노선 연결이 불완전해 공급유형을 ‘노선 연결 추가확인’으로 표시함. 이 격자의 관측 노선 수를 0이나 노선 빈곤으로 바꾸지 않음. 계산 오류가 없어도 이 경고들은 남겨둠.

SGIS 고령>총인구 원값을 임의 보정하지 않고 비율만 결측 처리함. 인구표 미수록을 0명으로 채우지 않음. 사례별 Q75는 내부 상대 기준이며 지역 간 절대 위험 순위를 뜻하지 않음. 2023년 전국 지표·2024 격자인구·2025 의료·정류장·2026 노선을 같은 시점의 현황이라고 주장하지 않음.

## GitHub 업로드 준비

```bash
python package_code.py
python verify_release.py
```

`verify_release.py`는 로컬에서 전체 분석·실행 노트북·ZIP 생성까지 마친 뒤 사용하는 최종 검증임. 노트북 실행 오류, 지도와 결과의 지역 수 불일치, ZIP CRC·최신 코드 일치 여부, 개인 경로·원자료 혼입을 확인하고 `results/release_validation.json`에 기록함. 원자료 없는 공개 CI용 명령은 아님.

ZIP에는 코드·설정·문서·깨끗한 노트북·합성 단위검사만 포함함. 원자료·실행 노트북·결과·독립 지도·인증정보·개인 절대경로는 포함하지 않음. 데이터 없는 공개 환경에서는 합성 단위검사가 실행되고 실제자료 회귀검사는 건너뜀. 전체 분석 재현에는 별도 공식 입력 확보가 필요함.

SGIS의 출처 명시·연구 완료 후 결과물 사본 또는 URL 제출 조건을 지켜야 함. 지도·경계·파생자료의 공개 허용 범위를 별도로 확인한 뒤 공개해야 하며, 다운로드 동의를 원자료 GitHub 재배포 허가로 간주하지 않음. 보고서와 실제 GitHub 열람 링크 작성·게시 작업은 이 코드에 포함하지 않음.
