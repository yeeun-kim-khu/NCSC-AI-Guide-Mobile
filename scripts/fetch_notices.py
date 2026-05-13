"""
scripts/fetch_notices.py
GitHub Actions에서 실행되어 과학관 공지사항을 크롤링하고
data/notices_cache.json 에 저장합니다.
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import urllib3
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MUSEUM_BASE_URL = "https://www.csc.go.kr"
NOTICE_LIST_URL = f"{MUSEUM_BASE_URL}/boardList.do?bbspkid=22"
LIMIT = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "identity",
    "Connection": "close",
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


def _fetch_html(url: str) -> bytes:
    last_err = None
    for attempt in range(1, 4):
        try:
            session = _build_session()
            resp = session.get(url, timeout=(15, 30), verify=False, headers=HEADERS, stream=True)
            resp.raise_for_status()
            data = resp.content
            if data:
                return data
        except Exception as e:
            last_err = e
            print(f"[fetch] attempt {attempt} failed: {e}")
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"fetch failed: {last_err}")


def fetch_notices(limit: int = LIMIT) -> list[dict]:
    html = _fetch_html(NOTICE_LIST_URL)
    soup = BeautifulSoup(html, "html.parser")

    notice_anchors = soup.select(
        "div.rbbs_list_sec a[onclick*='goView'], div.rbbs_list a[onclick*='goView'], table a[onclick*='goView']"
    )
    if not notice_anchors:
        notice_anchors = soup.select("a[onclick*='goView']")

    results = []
    seen = set()

    for a in notice_anchors:
        onclick = a.get("onclick", "")
        m_full = re.search(
            r"goView\(\s*'(?P<pkid>\d+)'\s*,\s*'(?P<num>\d+)'\s*,\s*'(?P<page>\d+)'\s*\)",
            onclick,
        )
        m = m_full or re.search(r"goView\(\s*'(?P<pkid>\d+)'", onclick)
        if not m:
            continue
        pkid = m.group("pkid")
        num = m.group("num") if m_full else "0"

        title_el = a.select_one("div.title_line div.title div.text")
        title = title_el.get_text(" ", strip=True) if title_el else a.get_text(" ", strip=True)
        if not title:
            continue

        skip_keywords = [
            "요청하신 페이지를 찾을 수 없습니다", "죄송합니다", "이전페이지",
            "대표번호", "Science Center Information", "국립어린이과학관 메인",
        ]
        if any(k in title for k in skip_keywords):
            continue
        if len(title) > 120:
            continue

        href = f"{MUSEUM_BASE_URL}/boardView.do?bbspkid=22&pkid={pkid}&num={num}"
        if href in seen:
            continue
        seen.add(href)
        results.append({"title": title, "href": href, "pkid": pkid})
        if len(results) >= limit:
            break

    return results


def main():
    out_path = Path(__file__).parent.parent / "data" / "notices_cache.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[fetch_notices] 크롤링 시작: {NOTICE_LIST_URL}")
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
