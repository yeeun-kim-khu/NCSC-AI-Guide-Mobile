"""
scripts/fetch_notices.py
GitHub Actions에서 실행되어 과학관 공지사항을 크롤링하고
data/notices_cache.json 에 저장합니다.

새 홈페이지(React SPA) 전환 후 REST API 방식으로 변경:
  API: https://www.sciencecenter.go.kr/csc-api/api/v1/notice/list
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

NOTICE_API_URL = "https://www.sciencecenter.go.kr/csc-api/api/v1/notice/list"
NOTICE_DETAIL_BASE = "https://www.sciencecenter.go.kr/csc/news/notice/d"
LIMIT = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ko",
    "Origin": "https://www.sciencecenter.go.kr",
    "Referer": "https://www.sciencecenter.go.kr/csc/news/notice",
}


def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_notices(limit: int = LIMIT) -> list[dict]:
    last_err = None
    for attempt in range(1, 4):
        try:
            session = _build_session()
            resp = session.get(
                NOTICE_API_URL,
                params={"page": 1, "size": limit, "searchType": "ALL"},
                headers=HEADERS,
                timeout=(15, 30),
                verify=False,
            )
            resp.raise_for_status()
            body = resp.json()
            if not body.get("success") or not body.get("data"):
                print(f"[fetch_notices] API 응답 이상: {body}")
                return []
            items = body["data"].get("content", [])
            results = []
            for item in items:
                notice_id = item.get("id") or item.get("no")
                title = (item.get("title") or "").strip()
                if not title or not notice_id:
                    continue
                href = f"{NOTICE_DETAIL_BASE}/{notice_id}"
                results.append({"title": title, "href": href, "pkid": str(notice_id)})
            return results
        except Exception as e:
            last_err = e
            print(f"[fetch_notices] attempt {attempt} failed: {e}")
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"fetch failed: {last_err}")


def main():
    out_path = Path(__file__).parent.parent / "data" / "notices_cache.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[fetch_notices] 크롤링 시작: {NOTICE_API_URL}")
    try:
        notices = fetch_notices()
        if not notices:
            print("[fetch_notices] 공지사항을 찾지 못했습니다. 캐시 업데이트 건너뜀.")
            return

        cache = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "notices": notices,
        }
        out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[fetch_notices] {len(notices)}건 저장 완료 → {out_path}")
    except Exception as e:
        print(f"[fetch_notices] 오류: {e}")
        raise


if __name__ == "__main__":
    main()
