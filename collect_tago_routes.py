#!/usr/bin/env python3
"""승인된 공식 TAGO 노선 API를 수집한다. 인증키·요청 URL은 저장·출력하지 않는다.

원 응답 JSON, 노선목록 CSV, 경유정류소 CSV, 실패·수집 요약을 보존한다.
이 도구는 2026년 조회시점 자료를 받으며 2025년 정류장 스냅샷과 구분한다.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import unquote, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import xml.etree.ElementTree as ET

import pandas as pd

BASE = "https://apis.data.go.kr/1613000/BusRouteInfoInqireService"


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(rows, path, columns=None):
    frame = pd.DataFrame(rows)
    if frame.empty and columns:
        frame = pd.DataFrame(columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def parse_response(text):
    try:
        data = json.loads(text)
        if "OpenAPI_ServiceResponse" in data:
            header = data["OpenAPI_ServiceResponse"].get("cmmMsgHeader", {})
            raise ValueError(f"공식 API 게이트웨이 코드 {header.get('returnReasonCode', 'unknown')}")
        response = data["response"]
        code = str(response["header"]["resultCode"])
        if code not in {"00", "0"}:
            raise ValueError(f"공식 API 오류 코드 {code}")
        body = response["body"]
        container = body.get("items") or {}
        items = container.get("item", []) if isinstance(container, dict) else []
        if isinstance(items, dict):
            items = [items]
        return items or [], int(body.get("totalCount", len(items or [])))
    except json.JSONDecodeError:
        # 인증·게이트웨이 오류는 _type=json 요청에도 XML로 올 수 있다.
        try:
            xml = ET.fromstring(text)
            code = xml.findtext(".//resultCode") or xml.findtext(".//returnReasonCode") or "unknown"
            raise ValueError(f"공식 API XML 응답, 코드 {code}")
        except ET.ParseError:
            raise ValueError("공식 API 응답을 JSON/XML로 읽을 수 없음") from None


class Client:
    def __init__(self, key, root):
        self.key = unquote(key.strip())
        self.root = root
        self.lock = threading.Lock()
        self.previous = 0.0
        self.requests = 0
        self.retries = 0

    def page(self, endpoint, params, path):
        if path.exists():
            return parse_response(path.read_text(encoding="utf-8"))
        last_error = "수집 실패"
        for attempt in range(3):
            with self.lock:
                delay = max(0, .35 - (time.monotonic() - self.previous))
                if delay:
                    time.sleep(delay)
                self.previous = time.monotonic()
                self.requests += 1
                self.retries += int(attempt > 0)
            url = BASE + "/" + endpoint + "?" + urlencode({"serviceKey": self.key, "_type": "json", **params})
            try:
                with urlopen(Request(url, headers={"User-Agent": "Public-data-research/1.0"}), timeout=30) as stream:
                    text = stream.read().decode("utf-8-sig")
                # 키가 오류 메시지에 반사되는 경우에도 파일에 남기지 않는다.
                safe_text = text.replace(self.key, "[REDACTED]")
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    parsed = parse_response(safe_text)
                except (ValueError, KeyError, TypeError):
                    path.with_name("error_latest.json").write_text(safe_text, encoding="utf-8")
                    raise
                path.write_text(safe_text, encoding="utf-8")
                return parsed
            except HTTPError as error:
                last_error = f"HTTP {error.code}"
            except (URLError, TimeoutError, OSError) as error:
                last_error = type(error).__name__
            except (ValueError, KeyError, TypeError) as error:
                last_error = str(error) if isinstance(error, ValueError) else type(error).__name__
            if attempt < 2:
                time.sleep(1 + attempt)
        raise RuntimeError(last_error) from None

    def pages(self, endpoint, params, directory):
        first, total = self.page(endpoint, {**params, "pageNo": 1, "numOfRows": 1000}, directory / "page_1.json")
        rows = list(first)
        if total and not rows:
            raise ValueError("totalCount는 양수지만 첫 페이지가 비어 있음")
        page_size = len(first)
        for number in range(2, math.ceil(total / max(page_size, 1)) + 1):
            records, observed_total = self.page(endpoint, {**params, "pageNo": number, "numOfRows": 1000}, directory / f"page_{number}.json")
            if observed_total != total:
                raise ValueError("페이지별 totalCount 변동: 재수집 필요")
            rows.extend(records)
        if len(rows) != total:
            raise ValueError(f"페이지 합계 {len(rows)}와 totalCount {total} 불일치")
        return rows, total


def collect_city(client, code, slug, output):
    directory = output / slug
    routes, total = client.pages("getRouteNoList", {"cityCode": code}, directory / "raw/route_list")
    if not routes:
        raise ValueError(f"{slug}: 노선목록이 비어 있음")
    frame = pd.DataFrame(routes)
    if "routeid" not in frame or frame["routeid"].isna().any() or frame["routeid"].duplicated().any():
        raise ValueError(f"{slug}: 노선 ID 결측·중복")
    save_csv(routes, directory / "route_list.csv")
    records, failures, per_route = [], [], []

    def collect_one(route):
        route_id = str(route["routeid"])
        safe_name = hashlib.sha256(route_id.encode()).hexdigest()[:20]
        try:
            rows, count = client.pages("getRouteAcctoThrghSttnList", {"cityCode": code, "routeId": route_id}, directory / "raw/route_stops" / safe_name)
            for row in rows:
                if str(row.get("routeid")) != route_id:
                    raise ValueError("요청 노선과 응답 routeid 불일치")
                if not all(key in row for key in ["nodeid", "nodeord", "gpslati", "gpslong"]):
                    raise ValueError("경유정류소 필수 필드 누락")
            return route_id, rows, count, None
        except (ValueError, RuntimeError) as error:
            return route_id, [], None, str(error)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(collect_one, route) for route in routes]
        for index, future in enumerate(as_completed(futures), 1):
            route_id, rows, count, error = future.result()
            if error:
                failures.append({"cityCode": code, "routeid": route_id, "error": error})
            else:
                records.extend(rows)
            per_route.append({"routeid": route_id, "totalCount": count, "downloaded_rows": len(rows),
                              "request_succeeded": error is None, "empty_route_flag": error is None and len(rows) == 0,
                              "raw_folder": hashlib.sha256(route_id.encode()).hexdigest()[:20]})
            if index % 25 == 0 or index == len(futures):
                print(f"{slug}: {index}/{len(futures)} 노선 경유자료 확인, 실패 {len(failures)}", flush=True)
    stops = pd.DataFrame(records)
    if not stops.empty:
        stops = stops.sort_values(["routeid", "nodeord"], kind="stable")
    save_csv(stops, directory / "route_stops.csv", ["routeid", "nodeid", "nodeord", "nodenm", "gpslati", "gpslong"])
    save_csv(failures, directory / "failures.csv", ["cityCode", "routeid", "error"])
    save_csv(sorted(per_route, key=lambda row: row["routeid"]), directory / "route_collection_status.csv")
    summary = {"slug": slug, "cityCode": code, "route_list_totalCount": total,
               "downloaded_routes": len(routes), "routes_with_stop_records": len(set(stops["routeid"])) if len(stops) else 0,
               "stop_records": len(stops), "unique_stop_ids": int(stops["nodeid"].nunique()) if len(stops) else 0,
               "failed_routes": len(failures), "empty_routes": sum(row["empty_route_flag"] for row in per_route),
               "all_route_requests_succeeded": not failures, "collected_at_utc": datetime.now(timezone.utc).isoformat(),
               "reference_period": "2026-10-22 수집시점, 2025 정류장 CSV와 시점 다름",
               "official_source": "https://www.data.go.kr/data/15098529/openapi.do",
               "headway_or_run_count_estimated": False, "service_key_saved": False}
    save_json(summary, directory / "collection_summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data/raw/tago_20261022")
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--probe-city", help="다른 공식 파일에서 확인한 도시코드의 실제 노선 응답만 검사")
    parser.add_argument("--city", action="append", default=[], help="실제 조회한 도시코드:사례slug")
    args = parser.parse_args()
    client = Client(args.key_file.read_text(encoding="utf-8"), args.output)
    cities, _ = client.page("getCtyCodeList", {}, args.output / "city_codes_response.json")
    save_csv(cities, args.output / "city_codes.csv")
    targets = [row for row in cities if any(name in str(row.get("cityname", "")) for name in ["거창", "영동", "홍성", "고성"])]
    print(json.dumps(targets, ensure_ascii=False), flush=True)
    if args.discover:
        return
    if args.probe_city:
        rows, total = client.pages("getRouteNoList", {"cityCode": args.probe_city}, args.output / f"probe_{args.probe_city}/raw/route_list")
        save_csv(rows, args.output / f"probe_{args.probe_city}/route_list.csv")
        print(json.dumps({"cityCode": args.probe_city, "listed_in_city_api": any(str(row["citycode"]) == args.probe_city for row in cities),
                          "totalCount": total, "rows": len(rows), "fields": list(rows[0]) if rows else []}, ensure_ascii=False))
        return
    known_codes = {str(row["citycode"]) for row in cities}
    summaries = []
    for spec in args.city:
        code, slug = spec.split(":", 1)
        if code not in known_codes:
            probe = args.output / f"probe_{code}/raw/route_list/page_1.json"
            if not probe.exists() or parse_response(probe.read_text(encoding="utf-8"))[1] <= 0:
                raise ValueError("도시목록에도 없고 공식 코드 직접조회 성공 증거도 없음. 코드 추정 금지")
        result = collect_city(client, code, slug, args.output)
        result["listed_in_city_api"] = code in known_codes
        save_json(result, args.output / slug / "collection_summary.json")
        summaries.append(result)
        all_summaries = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.output.glob("*/collection_summary.json"))]
        save_csv(all_summaries, args.output / "collection_summary.csv")
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    run_name = "_".join(spec.split(":", 1)[1] for spec in args.city) or "discovery"
    save_json({"requests_this_run": client.requests, "retries_this_run": client.retries,
               "key_or_request_url_logged": False}, args.output / f"request_audit_{run_name}.json")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        # URL이 포함될 수 있는 HTTP 예외의 전체 traceback은 출력하지 않는다.
        print(f"수집 중단: {type(error).__name__}: {str(error)[:180]}")
        raise SystemExit(1) from None
