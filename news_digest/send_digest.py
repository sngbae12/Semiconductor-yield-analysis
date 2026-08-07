"""
반도체 QC/양산관리 뉴스 요약 -> 이메일 발송 스크립트

EE Times, SemiWiki, KIPOST에서 최신 헤드라인을 수집한 뒤 Claude API로
품질관리(QC)/양산관리 직무 관점의 요약을 생성하고, Gmail SMTP로 발송합니다.

필요한 환경변수 (.env 또는 실제 환경변수):
- ANTHROPIC_API_KEY   : Anthropic API 키
- GMAIL_ADDRESS        : 발신용 Gmail 주소
- GMAIL_APP_PASSWORD   : Gmail 앱 비밀번호 (일반 로그인 비밀번호 아님)
- RECIPIENT_EMAIL      : 수신 이메일 (기본값: GMAIL_ADDRESS와 동일)
- CLAUDE_MODEL         : 사용할 모델 ID (기본값 아래 DEFAULT_MODEL)

사용법:
    python send_digest.py
"""

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import anthropic
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

SITES = {
    "EE Times": "https://www.eetimes.com/",
    "SemiWiki": "https://semiwiki.com/",
    "KIPOST": "https://www.kipost.net/news/articleList.html?sc_sub_section_code=S2N9&view_type=sm",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NewsDigestBot/1.0)"}


def fetch_page_links(url: str, max_chars: int = 20000) -> str:
    """페이지에서 헤드라인으로 보이는 링크 텍스트를 긁어서 원본 덩어리로 반환."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = []
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        href = a["href"]
        if text and len(text) > 8:
            lines.append(f"- {text} ({href})")

    return "\n".join(lines[:200])[:max_chars]


def gather_raw_content() -> dict:
    raw = {}
    for name, url in SITES.items():
        try:
            raw[name] = fetch_page_links(url)
            if not raw[name]:
                raw[name] = "[수집된 링크 없음]"
        except Exception as exc:
            raw[name] = f"[수집 실패: {exc}]"
    return raw


def build_prompt(raw: dict) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    sections = [f"### {name}\n{content}" for name, content in raw.items()]
    joined = "\n\n".join(sections)

    return f"""당신은 반도체 산업 뉴스 요약 담당입니다. 아래는 오늘({today}) EE Times, SemiWiki, KIPOST 사이트에서 수집한 헤드라인/링크 원본 목록입니다. 이 중에서 반도체 품질관리(QC/QA) 및 양산관리(수율, 공정 안정화, 생산성, 불량 분석 등) 직무와 관련성이 높은 뉴스를 선별해서 요약해주세요.

요약 기준:
- 특정 기업 실적/주가 뉴스보다는 반도체 산업 전반의 기술/공정/장비/트렌드 관점
- 품질관리(QC/QA) 및 양산관리 직무 관련성이 높은 것을 우선 선별
- 애매하면 제외하고, 각 항목은 반드시 구체적 사실(수치, 날짜, 기관/기업명 등)을 포함할 것

출력 형식:
- 사이트별로 구분해서 2~5개 핵심 뉴스 항목 나열 (관련성 높은 게 부족하면 그만큼만 작성)
- 각 뉴스 항목마다 아래 구조를 지킬 것:
  1) 제목 (마크다운 굵게)
  2) "사실:" 로 시작하는 줄 — 기사에서 확인 가능한 구체적 사실을 명확히 적시
  3) "쉬운 설명:" 으로 시작하는 줄 — 2~3줄로 전문용어를 최소화해 쉽게 설명하고, 품질관리/양산관리 직무 관점에서 왜 볼만한지 포함
  4) 원문 링크
- 마지막에 "오늘의 핵심 시사점"을 3줄로 요약

원본 데이터:
{joined}

주의: 원본 목록에는 광고, 네비게이션 메뉴 등 뉴스가 아닌 링크도 섞여 있을 수 있습니다. 실제 뉴스 기사로 보이는 것만 선별하세요. 관련성 높은 기사를 찾기 어려우면 억지로 만들지 말고 해당 사이트는 "관련 기사를 찾지 못했습니다"라고 표시하세요."""


def summarize_with_claude(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)
    message = client.messages.create(
        model=model,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def send_email(subject: str, body: str) -> None:
    sender = os.environ["GMAIL_ADDRESS"]
    password = os.environ["GMAIL_APP_PASSWORD"]
    recipient = os.environ.get("RECIPIENT_EMAIL", sender)

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())


def main() -> None:
    raw = gather_raw_content()
    prompt = build_prompt(raw)
    digest = summarize_with_claude(prompt)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"[반도체 QC/양산관리 뉴스 요약] {now}"
    send_email(subject, digest)
    print(f"발송 완료: {subject}")


if __name__ == "__main__":
    main()
