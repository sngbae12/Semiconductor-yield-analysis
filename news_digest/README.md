# 반도체 QC/양산관리 뉴스 요약 이메일 봇

EE Times, SemiWiki, KIPOST에서 헤드라인을 수집하고, Claude API로 반도체
품질관리(QC)/양산관리 직무 관점 요약을 생성한 뒤, Gmail로 발송하는 스크립트입니다.
본인 컴퓨터(또는 항상 켜져 있는 서버)에서 스케줄러로 돌리는 걸 전제로 합니다.

## 1. 설치

```bash
cd news_digest
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 환경변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 아래 값을 채워주세요.

| 변수 | 설명 |
|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com)에서 발급. 결제 수단 등록 필요 (종량제) |
| `GMAIL_ADDRESS` | 발신용 Gmail 주소 |
| `GMAIL_APP_PASSWORD` | 일반 로그인 비밀번호 아님. 아래 3번 참고해서 발급 |
| `RECIPIENT_EMAIL` | 요약을 받을 이메일 주소 (기본값: `sngbae12@gmail.com`) |

`.env` 파일은 `.gitignore`에 이미 등록되어 있어 git에 커밋되지 않습니다.

## 3. Gmail 앱 비밀번호 발급

1. Google 계정에서 2단계 인증을 먼저 켜야 합니다.
2. https://myaccount.google.com/apppasswords 접속
3. 앱 이름을 아무거나 입력(예: "news-digest") 후 생성된 16자리 비밀번호를 `GMAIL_APP_PASSWORD`에 붙여넣기

## 4. 테스트 실행

```bash
python send_digest.py
```

정상적으로 실행되면 "발송 완료: ..." 메시지가 뜨고, `RECIPIENT_EMAIL`로 메일이 도착합니다.

## 5. 매일 자동 실행 (스케줄링)

### macOS / Linux (cron)

`crontab.example` 파일을 참고해서 `crontab -e`에 등록하세요. 경로는 실제 환경에 맞게 수정해야 합니다.

```bash
crontab -e
# crontab.example 내용을 참고해서 08:30 / 12:00 / 18:30 3줄 추가
```

### Windows (작업 스케줄러)

1. "작업 스케줄러" 실행 → "작업 만들기"
2. 트리거: 매일, 08:30 / 12:00 / 18:30 각각 별도 트리거로 3개 등록 (또는 작업 3개 생성)
3. 동작: 프로그램 시작
   - 프로그램: `venv\Scripts\python.exe`의 전체 경로
   - 인수: `send_digest.py`
   - 시작 위치: `news_digest` 폴더 경로

## 6. 비용

- Claude API: 사용한 토큰만큼 종량제 과금 (건당 대략 몇 센트 수준, 실제 비용은 [가격표](https://www.anthropic.com/pricing) 참고)
- Gmail SMTP 발송: 무료
- 뉴스 수집(requests): 무료

## 7. 참고 / 한계

- 사이트 구조(HTML)가 바뀌면 헤드라인 수집이 부정확해질 수 있습니다. 이 경우 `send_digest.py`의
  `SITES` 딕셔너리에 있는 URL이나 `fetch_page_links` 함수의 선택자 로직을 조정해야 합니다.
- 일부 사이트는 봇 차단(예: Cloudflare)을 걸어둘 수 있어 요청이 실패할 수 있습니다. 이 경우
  로그(`digest.log`)에서 실패 사유를 확인하세요.
- 컴퓨터가 꺼져 있거나 절전 모드면 예약된 시간에 실행되지 않습니다.
