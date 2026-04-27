# static_translations.py
# 핵심 FAQ 답변의 4개 언어 사전 번역 사전.
# 숫자/시간/금액/장소명을 100% 정확히 보존하기 위해 LLM 번역을 우회한다.
#
# 구조: STATIC_FAQ_ANSWERS[category][language][mode] = str
#   - category: classify_basic_category 가 반환하는 키 (floor_guide, parking, ...)
#   - language: "한국어" | "English" | "日本語" | "中文"
#   - mode: "어린이" | "청소년/성인"  (없으면 fallback "default" 사용)
#
# 누락된 (category, language, mode) 조합은 None 반환 → 호출측에서 LLM 번역 fallback.

from typing import Optional


def get_static_answer(category: str, language: str, mode: str) -> Optional[str]:
    """정적 사전 번역 답변 반환. 없으면 None (LLM 번역 fallback 신호)."""
    cat_dict = STATIC_FAQ_ANSWERS.get(category)
    if not cat_dict:
        return None
    lang_dict = cat_dict.get(language)
    if not lang_dict:
        return None
    return lang_dict.get(mode) or lang_dict.get("default")


# ============================================================================
# PARKING — 가장 짧고 critical (환각 사전 차단)
# ============================================================================
_PARKING_KR_KID = """🚗 주차 안내

### 핵심만 먼저!
- **국립어린이과학관 건물 안에는 주차장이 없어요.**
- 그래서 **지하철이나 버스**로 오는 게 제일 편해요!
- 차로 오게 되면 근처 유료 주차장을 써야 해요.

### 🚇 지하철로 오는 방법
- **4호선 혜화역 4번 출구** → 걸어서 약 10~15분
  - 혜화역에서 나오면 창경궁 방향(북쪽)으로 쭉 걸어오면 돼요!
- **1호선 종로5가역 2번 출구** → 걸어서 약 20분

### 🚌 버스로 오는 방법
- **창경궁 앞(홍화문)** 정류장에서 내려서 5분 정도 걸어오세요.
- 지나가는 버스 번호(예): 101, 102, 104, 106, 107, 108, 140, 150, 160 등

### 🅿️ 꼭 차로 와야 한다면
- **창경궁 주차장**(가장 가까워요) — 창경궁 홍화문 바로 옆
- **서울대학교병원 주차장** — 대학로 쪽
- **종로구청 주차장** — 종로3가 근처
- 주말이나 공휴일에는 주차장이 꽉 차 있을 수 있어요. 조금 일찍 오거나 대중교통을 추천해요!

### ♿ 도움이 필요하다면
- 혜화역 4번 출구에는 엘리베이터가 있어요.
- 과학관 1층 안내데스크에서 **유모차(5대)**, **휠체어(2대)**를 빌릴 수 있어요. (신분증 꼭 챙겨오기!)

### 📍 주소 & 연락처
- **주소**: 서울특별시 종로구 창경궁로 215 (와룡동 2-1)
- **전화**: 02-3668-3350

⚠️ 주차/교통 안내는 바뀔 수 있으니, 출발 전에 공식 홈페이지(www.csc.go.kr)에서 한 번 더 확인해줘!"""

_PARKING_KR_DEFAULT = """🚗 주차 안내

### 핵심 안내
- **국립어린이과학관은 전용 주차장이 없습니다.**
- 차량으로 오시는 경우 인근 유료 주차장을 이용하셔야 하며, **가능하면 대중교통 이용을 권장드립니다.**

### 🚇 지하철 (가장 빠르고 편리한 방법)
- **4호선 혜화역 4번 출구** → 도보 약 10~15분
  - 혜화역 4번 출구로 나와 창경궁로를 따라 북쪽(창경궁 방향)으로 직진하시면 됩니다.
- **1호선 종로5가역 2번 출구** → 도보 약 20분

### 🚌 버스
- **창경궁 정문(홍화문)** 정류장 하차 후 도보 약 5분
- 경유 노선(일부): 101, 102, 104, 106, 107, 108, 140, 150, 160 등
- 버스 도착 정보는 서울 버스 앱이나 정류장 전광판에서 확인 가능합니다.

### 🅿️ 차량 이용 시 (인근 유료 주차장)
- **창경궁 주차장** (가장 가까움) — 창경궁 홍화문 옆
  - 소액 단위 요금제 / **주말·공휴일에는 만차 가능성이 높습니다.**
- **서울대학교병원 주차장** — 대학로 방면
- **종로구청 주차장** — 종로3가 인근

※ 주차 요금·운영시간은 주차장별로 상이하니 방문 전 확인을 권장드립니다.

### ♿ 장애인·교통약자 편의
- **혜화역 4번 출구**에는 엘리베이터가 설치되어 있습니다.
- 창경궁 주차장에는 장애인 주차구역이 마련되어 있습니다.
- 과학관 1층 안내데스크에서 **유모차(5대)** 및 **휠체어(2대)** 대여 가능 (신분증 지참).
- 의무실은 1층에 있으며 일반의약품을 구비하고 있습니다.

### 📍 주소 및 문의
- **주소**: 서울특별시 종로구 창경궁로 215 (와룡동 2-1)
- **대표전화**: 02-3668-3350
- **공식 홈페이지**: https://www.csc.go.kr

⚠️ 주차장 및 교통편 관련 최신 안내는 방문 전 공식 홈페이지 '오시는 길' 페이지에서 꼭 확인해 주세요."""

_PARKING_EN_DEFAULT = """🚗 Parking & Transportation Guide

### Key Point
- **The National Children's Science Center (NCSC) does NOT have its own parking lot.**
- If you must drive, please use a nearby paid parking lot. **Public transportation is strongly recommended.**

### 🚇 Subway (fastest & most convenient)
- **Line 4, Hyehwa Station (Exit 4)** → ~10–15 min walk
  - From Exit 4, head north along Changgyeonggung-ro toward Changgyeonggung Palace.
- **Line 1, Jongno 5-ga Station (Exit 2)** → ~20 min walk

### � Bus
- Get off at the **"Changgyeonggung Palace / Honghwamun Gate"** bus stop, then walk ~5 minutes.
- Routes passing nearby (examples): 101, 102, 104, 106, 107, 108, 140, 150, 160.
- You can check real-time arrivals via the Seoul Bus app or the stop's LED board.

### 🅿️ If you must drive (nearby paid parking lots)
- **Changgyeonggung Palace Parking Lot** (closest) — next to Honghwamun Gate
  - Small-unit fees / **often full on weekends and holidays**.
- **Seoul National University Hospital Parking Lot** — toward Daehak-ro.
- **Jongno-gu Office Parking Lot** — near Jongno 3-ga.

Note: Rates and operating hours vary by parking lot. Please check in advance.

### ♿ Accessibility
- **Exit 4 of Hyehwa Station** has an elevator.
- The Changgyeonggung Palace Parking Lot has designated accessible parking spaces.
- Stroller rental (5 available) and wheelchair rental (2 available) are offered at the 1F Information Desk (ID required).
- A First Aid room with basic medicines is located on the 1st floor.

### 📍 Address & Contact
- **Address**: 215 Changgyeonggung-ro, Jongno-gu, Seoul
- **Phone**: +82-2-3668-3350
- **Official website**: https://www.csc.go.kr

⚠️ Parking and transit information can change. Please confirm on the official website's "Directions" page before your visit."""

_PARKING_EN_KID = """🚗 Parking & How to Get Here

### Quick answer!
- **There's NO parking lot inside the museum building.**
- The **subway or bus** is the easiest way to come!
- If you drive, you'll need a paid parking lot nearby.

### 🚇 By Subway
- **Line 4, Hyehwa Station, Exit 4** → about a 10–15 minute walk
  - From the exit, walk north toward Changgyeonggung Palace.
- **Line 1, Jongno 5-ga Station, Exit 2** → about a 20 minute walk

### 🚌 By Bus
- Get off at the **Changgyeonggung Palace (Honghwamun Gate)** stop and walk ~5 minutes.
- Example bus routes: 101, 102, 104, 106, 107, 108, 140, 150, 160.

### 🅿️ If You Must Drive
- **Changgyeonggung Palace Parking Lot** (closest) — right next to Honghwamun Gate.
- **Seoul National University Hospital Parking Lot** — near Daehak-ro.
- **Jongno-gu Office Parking Lot** — near Jongno 3-ga.
- On weekends and holidays lots can fill up — come early or take public transit!

### ♿ If You Need Help
- Exit 4 of Hyehwa Station has an elevator.
- At the 1F Information Desk you can borrow a **stroller (5 available)** or **wheelchair (2 available)**. (Bring an ID card!)

### 📍 Address & Contact
- **Address**: 215 Changgyeonggung-ro, Jongno-gu, Seoul
- **Phone**: +82-2-3668-3350

⚠️ Please check the official website (www.csc.go.kr) before leaving home, in case anything changes!"""

_PARKING_JP_DEFAULT = """🚗 駐車・交通のご案内

### 重要ポイント
- **国立こども科学館(NCSC)には専用駐車場がございません。**
- お車でお越しの場合は近隣の有料駐車場をご利用いただくことになりますので、**できるだけ公共交通機関のご利用をおすすめします。**

### 🚇 地下鉄(最も早くて便利)
- **4号線 恵化(ヘファ)駅 4番出口** → 徒歩約 10~15分
  - 4番出口を出て、昌慶宮路(チャンギョングンロ)を北(昌慶宮方面)へ直進してください。
- **1号線 鍾路5街駅 2番出口** → 徒歩約 20分

### 🚌 バス
- **「昌慶宮・弘化門(ホンファムン)」** バス停で下車、徒歩約5分。
- 通過路線(例):101、102、104、106、107、108、140、150、160 など
- バス到着情報はソウルバスアプリや停留所の電光掲示板で確認できます。

### 🅿️ お車の場合(近隣の有料駐車場)
- **昌慶宮駐車場**(最も近い) — 弘化門のすぐ横
  - 少額単位の料金制 / **週末・祝日は満車になる可能性が高いです。**
- **ソウル大学校病院駐車場** — 大学路(テハンノ)方面
- **鍾路区庁駐車場** — 鍾路3街付近

※ 駐車料金・営業時間は駐車場ごとに異なります。事前確認をおすすめします。

### ♿ バリアフリー
- **恵化駅 4番出口**にはエレベーターが設置されています。
- 昌慶宮駐車場には障がい者専用区画があります。
- 1階の総合案内でベビーカー(5台)・車いす(2台)の貸出が可能です(身分証が必要)。
- 医務室は1階にあり、一般用医薬品を常備しています。

### 📍 所在地・お問い合わせ
- **住所**:ソウル特別市 鍾路区 昌慶宮路 215 (臥龍洞 2-1)
- **代表電話**:+82-2-3668-3350
- **公式サイト**:https://www.csc.go.kr

⚠️ 駐車・交通情報は変更される場合があります。ご来館前に公式サイトの「オシヌンギル(交通案内)」ページを必ずご確認ください。"""

_PARKING_JP_KID = """🚗 ちゅうしゃ・アクセスのご案内

### まず大事なポイント!
- **建物の中にちゅうしゃ場はないよ。**
- だから **地下鉄かバス** で来るのがいちばんラク!
- 車で来るときは近くのゆうりょうちゅうしゃ場を使ってね。

### 🚇 地下鉄で来るには
- **4号線 恵化(ヘファ)駅 4番出口** → 歩いて約10~15分
  - 駅を出たら、昌慶宮の方(北)にまっすぐ歩いてね。
- **1号線 鍾路5街駅 2番出口** → 歩いて約20分

### 🚌 バスで来るには
- **「昌慶宮・弘化門」** のバス停で降りて、5分くらい歩いてね。
- 通るバスの例:101、102、104、106、107、108、140、150、160 など

### 🅿️ どうしても車で来るときは
- **昌慶宮ちゅうしゃ場**(いちばん近い) — 弘化門のすぐとなり
- **ソウル大学校病院ちゅうしゃ場** — 大学路の方
- **鍾路区庁ちゅうしゃ場** — 鍾路3街の近く
- 週末や祝日はいっぱいで入れないこともあるから、早めに来るか、電車・バスで来るのがおすすめ!

### ♿ こまった時は
- 恵化駅 4番出口にはエレベーターがあるよ。
- 1階の案内デスクでベビーカー(5台)や車いす(2台)を借りられるよ。(身分証をわすれずに!)

### 📍 住所・電話
- **住所**:ソウル特別市 鍾路区 昌慶宮路 215
- **電話**:+82-2-3668-3350

⚠️ 出発する前に、公式サイト(www.csc.go.kr)で一度チェックしてね!"""

_PARKING_ZH_DEFAULT = """🚗 停车与交通指南

### 核心信息
- **国立儿童科学馆(NCSC)没有专用停车场。**
- 如自驾前来,需使用附近的收费停车场,**建议尽量使用公共交通**。

### 🚇 地铁(最快最方便)
- **4号线 惠化(Hyehwa)站 4号出口** → 步行约 10~15 分钟
  - 从4号出口出站,沿昌庆宫路向北(昌庆宫方向)直走即可。
- **1号线 钟路5街站 2号出口** → 步行约 20 分钟

### 🚌 公交
- 在 **"昌庆宫・弘化门"** 站下车,步行约 5 分钟。
- 经过路线(部分):101、102、104、106、107、108、140、150、160 等
- 可通过"首尔巴士"APP 或站台电子屏查看实时到站信息。

### 🅿️ 自驾时(附近收费停车场)
- **昌庆宫停车场**(最近) — 弘化门旁
  - 小额单位计费 / **周末和节假日经常客满。**
- **首尔大学医院停车场** — 大学路方向
- **钟路区厅停车场** — 钟路3街附近

※ 各停车场的收费和运营时间不同,建议提前确认。

### ♿ 无障碍设施
- **惠化站 4号出口**设有电梯。
- 昌庆宫停车场设有残疾人专用车位。
- 科学馆1层咨询台可租借婴儿车(5辆)和轮椅(2辆)(需出示身份证)。
- 医务室位于1层,备有常用药品。

### 📍 地址与联系方式
- **地址**:首尔特别市 钟路区 昌庆宫路 215 (卧龙洞 2-1)
- **电话**:+82-2-3668-3350
- **官方网站**:https://www.csc.go.kr

⚠️ 停车与交通信息可能有变,来访前请务必在官网"交通指南"页面再次确认。"""

_PARKING_ZH_KID = """🚗 停车和怎么来

### 先说重点!
- **科学馆建筑内没有停车场哦。**
- 所以坐 **地铁或公交** 来最方便!
- 如果开车来,就要用附近的收费停车场啦。

### 🚇 坐地铁来
- **4号线 惠化站 4号出口** → 走路约 10~15 分钟
  - 从4号出口出来,朝昌庆宫方向(北边)一直走就好。
- **1号线 钟路5街站 2号出口** → 走路约 20 分钟

### 🚌 坐公交来
- 在 **"昌庆宫・弘化门"** 站下车,再走5分钟左右。
- 经过的公交车(例):101、102、104、106、107、108、140、150、160 等

### 🅿️ 实在要开车来
- **昌庆宫停车场**(最近) — 就在弘化门旁边
- **首尔大学医院停车场** — 大学路那边
- **钟路区厅停车场** — 钟路3街附近
- 周末或假日可能会客满,早点来或者坐公交地铁更好!

### ♿ 需要帮助时
- 惠化站 4号出口有电梯。
- 科学馆1层咨询台可以借婴儿车(5辆)和轮椅(2辆)哦。(记得带身份证!)

### 📍 地址和电话
- **地址**:首尔特别市 钟路区 昌庆宫路 215
- **电话**:+82-2-3668-3350

⚠️ 出发前再到官网(www.csc.go.kr)看一下最新信息哦!"""

_PARKING = {
    "한국어": {
        "어린이": _PARKING_KR_KID,
        "default": _PARKING_KR_DEFAULT,
    },
    "English": {
        "어린이": _PARKING_EN_KID,
        "default": _PARKING_EN_DEFAULT,
    },
    "日本語": {
        "어린이": _PARKING_JP_KID,
        "default": _PARKING_JP_DEFAULT,
    },
    "中文": {
        "어린이": _PARKING_ZH_KID,
        "default": _PARKING_ZH_DEFAULT,
    },
}

# ============================================================================
# ADMISSION FEE — 금액 critical
# ============================================================================
_ADMISSION = {
    "한국어": {
        "어린이": """관람료를 보기 쉽게 정리해드릴게요! 💸

#### 상설전시관(연나이 기준)

| 구분 | 개인 | 단체 | 대상 |
| --- | ---: | ---: | --- |
| 성인 | 2,000원 | 이용불가 | 19세 이상 |
| 청소년 | 1,000원 | 이용불가 | 13~18세 |
| 초등학생 | 1,000원 | 500원 | 7~12세 |
| 유아 | 무료 | 무료 | 6세 이하 |
| 우대고객 | 무료 | 이용불가 | 경로우대자, 장애인 등(증빙 필요) |


#### 천체투영관(연나이 기준)

| 구분 | 개인 | 단체 | 대상 |
| --- | ---: | ---: | --- |
| 성인 | 1,500원 | 이용불가 | 19세 이상 |
| 청소년 | 1,000원 | 이용불가 | 13~18세 |
| 초등학생 | 1,000원 | 1,000원 | 7~12세 |
| 유아 | 1,000원 | 1,000원 | 4~6세(성인 보호자 동반 및 결제 시) |
| 우대고객 | 1,000원 | 이용불가 | 경로우대자, 장애인 등(증빙 필요) |


#### 요금 할인/면제(요약)
- 우대고객은 **신분증/증명서 지참 필수**
- 중증장애(1~3급): 본인 + 동반 보호자 1인 혜택(상설전시관 무료 등)
- 경증장애(4급 이상): 본인 혜택
- 다자녀카드: 상설전시관 개인요금 **50% 할인**(신분증 + 카드 지참)
""",
    },
    "English": {
        "default": """Admission fees (based on Korean age system).

#### Permanent Exhibition Hall

| Category | Individual | Group | Eligibility |
| --- | ---: | ---: | --- |
| Adult | 2,000 KRW | N/A | Age 19+ |
| Teen | 1,000 KRW | N/A | Age 13–18 |
| Elementary | 1,000 KRW | 500 KRW | Age 7–12 |
| Preschool | Free | Free | Age 6 and under |
| Concession | Free | N/A | Seniors, persons with disabilities, etc. (proof required) |


#### Planetarium

| Category | Individual | Group | Eligibility |
| --- | ---: | ---: | --- |
| Adult | 1,500 KRW | N/A | Age 19+ |
| Teen | 1,000 KRW | N/A | Age 13–18 |
| Elementary | 1,000 KRW | 1,000 KRW | Age 7–12 |
| Preschool | 1,000 KRW | 1,000 KRW | Age 4–6 (must be accompanied and paid by an adult guardian) |
| Concession | 1,000 KRW | N/A | Seniors, persons with disabilities, etc. (proof required) |


#### Discounts / Exemptions
- Concession visitors must bring **ID / proof documents**.
- Severe disability (Grades 1–3): free admission to permanent exhibits for the visitor + 1 accompanying guardian.
- Mild disability (Grade 4+): visitor only.
- Multi-child family card: **50% off** the individual rate at the permanent exhibition (ID + card required).
""",
    },
    "日本語": {
        "default": """観覧料のご案内です(年年齢基準)。

#### 常設展示館

| 区分 | 個人 | 団体 | 対象 |
| --- | ---: | ---: | --- |
| 大人 | 2,000ウォン | 利用不可 | 19歳以上 |
| 青少年 | 1,000ウォン | 利用不可 | 13~18歳 |
| 小学生 | 1,000ウォン | 500ウォン | 7~12歳 |
| 幼児 | 無料 | 無料 | 6歳以下 |
| 優待 | 無料 | 利用不可 | シニア、障がいのある方など(証明書必要) |


#### プラネタリウム

| 区分 | 個人 | 団体 | 対象 |
| --- | ---: | ---: | --- |
| 大人 | 1,500ウォン | 利用不可 | 19歳以上 |
| 青少年 | 1,000ウォン | 利用不可 | 13~18歳 |
| 小学生 | 1,000ウォン | 1,000ウォン | 7~12歳 |
| 幼児 | 1,000ウォン | 1,000ウォン | 4~6歳(成人保護者の同伴・お支払いが必要) |
| 優待 | 1,000ウォン | 利用不可 | シニア、障がいのある方など(証明書必要) |


#### 割引/免除(要約)
- 優待のお客様は**身分証/証明書の持参が必須**です。
- 重度障害(1~3級):本人+同伴の保護者1名(常設展示館は無料 等)
- 軽度障害(4級以上):本人のみ
- 多子世帯カード:常設展示館の個人料金が **50%割引**(身分証+カード持参)
""",
    },
    "中文": {
        "default": """门票信息(以韩国年龄为准)。

#### 常设展览馆

| 区分 | 个人 | 团体 | 对象 |
| --- | ---: | ---: | --- |
| 成人 | 2,000韩元 | 不可使用 | 19岁以上 |
| 青少年 | 1,000韩元 | 不可使用 | 13~18岁 |
| 小学生 | 1,000韩元 | 500韩元 | 7~12岁 |
| 幼儿 | 免费 | 免费 | 6岁以下 |
| 优待客户 | 免费 | 不可使用 | 老年人、残障人士等(需出示证明) |


#### 天体投影馆

| 区分 | 个人 | 团体 | 对象 |
| --- | ---: | ---: | --- |
| 成人 | 1,500韩元 | 不可使用 | 19岁以上 |
| 青少年 | 1,000韩元 | 不可使用 | 13~18岁 |
| 小学生 | 1,000韩元 | 1,000韩元 | 7~12岁 |
| 幼儿 | 1,000韩元 | 1,000韩元 | 4~6岁(须有成人监护人同行及支付) |
| 优待客户 | 1,000韩元 | 不可使用 | 老年人、残障人士等(需出示证明) |


#### 折扣/免费(摘要)
- 优待客户**必须携带身份证/证明文件**。
- 重度残疾(1~3级):本人 + 同行监护人1名享受优惠(常设展览馆免费等)
- 轻度残疾(4级以上):仅限本人
- 多子女家庭卡:常设展览馆个人票价 **50%折扣**(需带身份证+卡)
""",
    },
}

# ============================================================================
# PLANETARIUM TIMETABLE — 시간 critical
# ============================================================================
_PLANETARIUM_TIMETABLE = {
    "한국어": {
        "default": """천체투영관 시간표를 정리해드릴게요! 🌙

## 오늘/상설 시간표(안내)

| 회차 | 시간 | 프로그램 | 정원 | 권장연령 |
| --- | --- | --- | ---: | --- |
| 1 | 10:00 ~ 10:40 | 별자리 해설 + 코코몽 우주탐험 | 65명 | 유아 이상 |
| 2 | 11:00 ~ 11:40 | 별자리 해설 + 길냥이 키츠 슈퍼문 대모험 | 65명 | 유아 이상 |
| 3 | 12:00 ~ 12:40 | 바니 앤 비니 | 65명 | 유아 이상 |
| 4 | 14:00 ~ 14:40 | 다이노소어 | 65명 | 초등학생 이상 |
| 5 | 15:00 ~ 15:40 | 별자리 해설 + 길냥이 키츠 우주정거장의 비밀 | 65명 | 유아 이상 |
| 6 | 16:00 ~ 16:40 | 바니 앤 비니 | 65명 | 유아 이상 |


## 유의사항
- **연나이 4세 이상** 어린이부터 입장 가능합니다.
- 미취학 아동은 **보호자 동반 필수**(유아만 입장 불가)
- 상영 시작 이후에는 안전상 **입장/퇴장 불가**
- 환불은 현장에서 **상영 시작 30분 전까지** 가능
- 내부 음식물 섭취 금지, 휴대전화는 진동으로 설정
- 천체투영관 예약은 **2주 전 0시 오픈**됩니다.(상설전시장 예약도 필수)
""",
    },
    "English": {
        "default": """Planetarium schedule. 🌙

## Daily Schedule

| # | Time | Program | Capacity | Suggested Age |
| --- | --- | --- | ---: | --- |
| 1 | 10:00 ~ 10:40 | Constellation talk + Cocomong Space Adventure | 65 | Preschool+ |
| 2 | 11:00 ~ 11:40 | Constellation talk + Alley-cat Kits: Super-moon Adventure | 65 | Preschool+ |
| 3 | 12:00 ~ 12:40 | Barney & Beanie | 65 | Preschool+ |
| 4 | 14:00 ~ 14:40 | Dinosaur | 65 | Elementary+ |
| 5 | 15:00 ~ 15:40 | Constellation talk + Alley-cat Kits: Secret of the Space Station | 65 | Preschool+ |
| 6 | 16:00 ~ 16:40 | Barney & Beanie | 65 | Preschool+ |


## Notes
- Open to children **age 4+ (Korean year-age system)**.
- Preschoolers **must be accompanied by a guardian** (children may not enter alone).
- For safety, **no entry or exit once the show begins**.
- On-site refunds available **up to 30 minutes before the show**.
- No food or drink inside; please set phones to silent.
- Planetarium reservations open **at midnight, 2 weeks in advance** (permanent-exhibit reservation also required).
""",
    },
    "日本語": {
        "default": """プラネタリウムの時間表です。🌙

## 本日/通常スケジュール

| 回 | 時間 | プログラム | 定員 | 推奨年齢 |
| --- | --- | --- | ---: | --- |
| 1 | 10:00 ~ 10:40 | 星座解説 + ココモン宇宙たんけん | 65名 | 幼児以上 |
| 2 | 11:00 ~ 11:40 | 星座解説 + のらねこキッツ スーパームーン大冒険 | 65名 | 幼児以上 |
| 3 | 12:00 ~ 12:40 | バーニー&ビニー | 65名 | 幼児以上 |
| 4 | 14:00 ~ 14:40 | ダイナソー | 65名 | 小学生以上 |
| 5 | 15:00 ~ 15:40 | 星座解説 + のらねこキッツ 宇宙ステーションの秘密 | 65名 | 幼児以上 |
| 6 | 16:00 ~ 16:40 | バーニー&ビニー | 65名 | 幼児以上 |


## ご注意
- **満4歳以上(韓国年年齢)** のお子様からご入場いただけます。
- 未就学のお子様は**保護者の同伴が必須**です(お子様だけのご入場はできません)。
- 上映開始後は安全のため**入退場不可**です。
- 払い戻しは現場で**上映開始30分前まで**可能です。
- 館内での飲食はご遠慮ください。携帯電話はマナーモードに設定してください。
- プラネタリウムの予約は**2週間前の0時にオープン**します(常設展示館の予約も必要)。
""",
    },
    "中文": {
        "default": """天体投影馆时间表。🌙

## 当日/常设排程

| 场次 | 时间 | 节目 | 定员 | 建议年龄 |
| --- | --- | --- | ---: | --- |
| 1 | 10:00 ~ 10:40 | 星座解说 + Cocomong 宇宙探险 | 65人 | 幼儿以上 |
| 2 | 11:00 ~ 11:40 | 星座解说 + 流浪猫Kits 超级月亮大冒险 | 65人 | 幼儿以上 |
| 3 | 12:00 ~ 12:40 | Barney & Beanie | 65人 | 幼儿以上 |
| 4 | 14:00 ~ 14:40 | 恐龙 | 65人 | 小学生以上 |
| 5 | 15:00 ~ 15:40 | 星座解说 + 流浪猫Kits 太空站的秘密 | 65人 | 幼儿以上 |
| 6 | 16:00 ~ 16:40 | Barney & Beanie | 65人 | 幼儿以上 |


## 注意事项
- **韩国年龄满4岁以上**的儿童可以入场。
- 学龄前儿童**必须有监护人陪同**(儿童单独不可入场)。
- 放映开始后,出于安全考虑**禁止入场或退场**。
- 现场退款可在**放映开始前30分钟**之前办理。
- 馆内禁止饮食,请将手机调为静音。
- 天体投影馆预约**于2周前的0点开放**(常设展览馆也需预约)。
""",
    },
}

# ============================================================================
# FLOOR GUIDE
# ============================================================================
_FLOOR_GUIDE = {
    "한국어": {
        "default": """층별 안내를 한눈에 보기 쉽게 정리해드릴게요! 😊

## 1층
- 매표소·안내데스크
- AI놀이터 / 생각놀이터 / 행동놀이터
- 천체투영관, 과학극장
- 어린이교실
- 수유실·유아휴게실, 의무실
- 물품보관함(락커)
- 유모차·휠체어 대여(신분증 제시)
- 꿈트리 동산(창경궁 방향)

## 2층
- 빛놀이터 / 탐구놀이터 / 관찰놀이터
- 창작교실
- 휴게실(영유아놀이터 포함)
- 물품보관함(락커)

## 3층(옥상)
- 하늘마당(옥상)
  - 과학관 퇴장 후 오른쪽으로 돌아 언덕을 따라 올라가면 돼요.

## (중요) 입구/출구 안내
- **입구: 2층 게이트**
- **출구: 1층 게이트**

## 입장 팁
- 1층 매표소에서 매표(또는 예약 확인) 후, 2층 입구 게이트로 들어오세요.
- 과학관 입구(2층)에서 예약한 입장권(QR코드) 확인 후 관람해주시기 바랍니다.""",
    },
    "English": {
        "default": """Floor guide for the National Children's Science Center. 😊

## 1F
- Tickets & Information Desk
- AI Zone / Thinking Zone / Activity Zone
- Planetarium, Science Theater
- Kids Classroom
- Baby Care / Lounge, First Aid
- Lockers
- Stroller & wheelchair rental (ID required)
- Little Library Garden (toward Changgyeonggung)

## 2F
- Light Zone / Discovery Zone / Observation Zone
- Creative Classroom
- Lounge (with baby play area)
- Lockers

## 3F (Rooftop)
- Sky Courtyard
  - After exiting the museum, turn right and walk up the hill.

## (Important) Entry / Exit
- **Entry: 2F gate**
- **Exit: 1F gate**

## Entry Tips
- After purchasing tickets (or confirming reservation) at the 1F ticket office, please enter through the 2F entry gate.
- At the entry gate (2F), please present your reserved ticket (QR code) before entering.""",
    },
    "日本語": {
        "default": """フロア案内です。😊

## 1階
- チケット売り場・案内デスク
- AIゾーン / 考えるゾーン / アクティブゾーン
- プラネタリウム、サイエンスシアター
- こども教室
- 授乳室・乳幼児休憩室、医務室
- ロッカー
- ベビーカー・車いすの貸出(身分証提示)
- ドリームツリーガーデン(昌慶宮方面)

## 2階
- ひかりゾーン / 探究ゾーン / かんさつゾーン
- 創作教室
- 休憩室(乳幼児プレイエリア併設)
- ロッカー

## 3階(屋上)
- 空のひろば
  - 退館後、右に進んで坂道を上ってください。

## (重要)入口/出口
- **入口:2階ゲート**
- **出口:1階ゲート**

## 入場のヒント
- 1階の窓口でチケット購入(または予約確認)後、2階の入口ゲートからお入りください。
- 入口(2階)で予約した入場券(QRコード)をご提示ください。""",
    },
    "中文": {
        "default": """楼层导览。😊

## 1层
- 售票处・咨询台
- AI区 / 思考区 / 行动区
- 天体投影馆、科学剧场
- 儿童教室
- 哺乳室・婴幼儿休息室、医务室
- 储物柜
- 婴儿车・轮椅租借(需出示身份证)
- 梦想树花园(昌庆宫方向)

## 2层
- 光区 / 探究区 / 观察区
- 创作教室
- 休息室(含婴幼儿游乐区)
- 储物柜

## 3层(屋顶)
- 天空广场
  - 离开馆后向右拐,沿坡道上去即可。

## (重要)入口/出口
- **入口:2层闸口**
- **出口:1层闸口**

## 入场提示
- 在1层售票处购票(或确认预约)后,请从2层入口闸口进入。
- 在入口(2层)请出示已预约的入场券(QR码)。""",
    },
}

# ============================================================================
# FACILITY AMENITIES
# ============================================================================
_FACILITY = {
    "한국어": {
        "default": """편의시설 안내를 한 번에 정리해서 알려드릴게요! 😊

## 1층
- **의무실(First Aid)**: 1층 / 일반의약품 구비
- **수유실·유아휴게실(Baby Care)**: 1층 / 싱크대, 전자레인지, 쇼파 등
- **물품보관함(Locker)**: 1층 매표소 인근
- **유모차·휠체어 대여**: 1층 매표소·안내데스크에서 신분증 제시 후 대여
  - 수량: 유모차 5대 / 휠체어 2대
  - 유모차 이용: 36개월 이하
- **매표소·안내데스크(Tickets & Information)**: 1층
- **꿈트리 동산(Little Library)**: 1층(창경궁 방향)

## 2층
- **휴게실(Lounge)**: 2층
- **영유아놀이터(Baby Lounge)**: 2층 휴게실 내
- **물품보관함(Locker)**: 2층 휴게실 내부

## 3층(옥상)
- **하늘마당(Courtyard)**: 3층 옥상
  - 안내: 과학관 퇴장 후 오른쪽으로 돌아 언덕을 따라 올라가면 돼요.

대표번호는 모두 동일해요: **02-3668-3350**""",
    },
    "English": {
        "default": """Facility & amenity guide. 😊

## 1F
- **First Aid**: 1F / over-the-counter medicines available
- **Baby Care / Family Lounge**: 1F / sink, microwave, sofas, etc.
- **Lockers**: 1F, near the ticket office
- **Stroller & wheelchair rental**: at the 1F ticket office / information desk (ID required)
  - Inventory: 5 strollers / 2 wheelchairs
  - Stroller use: for children 36 months and under
- **Tickets & Information Desk**: 1F
- **Little Library Garden**: 1F (toward Changgyeonggung)

## 2F
- **Lounge**: 2F
- **Baby Play Area**: inside the 2F Lounge
- **Lockers**: inside the 2F Lounge

## 3F (Rooftop)
- **Sky Courtyard**: 3F rooftop
  - After exiting the museum, turn right and walk up the hill.

Main number for all inquiries: **02-3668-3350**""",
    },
    "日本語": {
        "default": """施設・サービスのご案内です。😊

## 1階
- **医務室**:1階 / 一般用医薬品を常備
- **授乳室・乳幼児休憩室**:1階 / シンク、電子レンジ、ソファあり
- **ロッカー**:1階 チケット売り場付近
- **ベビーカー・車いすの貸出**:1階 チケット売り場・案内デスクにて身分証提示後にご利用ください
  - 台数:ベビーカー5台 / 車いす2台
  - ベビーカー利用:36か月以下
- **チケット売り場・案内デスク**:1階
- **ドリームツリーガーデン**:1階(昌慶宮方面)

## 2階
- **休憩室**:2階
- **乳幼児プレイエリア**:2階 休憩室内
- **ロッカー**:2階 休憩室内

## 3階(屋上)
- **空のひろば**:3階 屋上
  - ご案内:退館後、右に進んで坂道を上ってください。

代表電話番号は共通です:**02-3668-3350**""",
    },
    "中文": {
        "default": """便利设施一览。😊

## 1层
- **医务室**:1层 / 备有常用药品
- **哺乳室・婴幼儿休息室**:1层 / 配有水槽、微波炉、沙发等
- **储物柜**:1层 售票处附近
- **婴儿车・轮椅租借**:1层 售票处・咨询台凭身份证租借
  - 数量:婴儿车5辆 / 轮椅2辆
  - 婴儿车适用:36个月以下幼儿
- **售票处・咨询台**:1层
- **梦想树花园**:1层(昌庆宫方向)

## 2层
- **休息室**:2层
- **婴幼儿游乐区**:2层 休息室内
- **储物柜**:2层 休息室内

## 3层(屋顶)
- **天空广场**:3层 屋顶
  - 指引:离开馆后向右拐,沿坡道上去即可。

总机号码均相同:**02-3668-3350**""",
    },
}

# ============================================================================
# EXHIBIT GUIDE
# ============================================================================
_EXHIBIT_GUIDE = {
    "한국어": {
        "default": """전시관(놀이터)들을 짧게 소개해드릴게요! 😊

- **AI놀이터(1층)**: AI 미션을 해결하며 인공지능을 쉽고 재미있게 체험하는 공간이에요.
- **행동놀이터(1층)**: 몸을 움직이며 건강/운동 원리를 체험하는 활동형 전시관이에요.
- **생각놀이터(1층)**: 어린이들의 생각을 키우는 전시관(2026년 5월 개관 예정)입니다.
- **빛놀이터(2층)**: 빛/숲/생태 주제를 미디어 인터랙션으로 몰입 체험하는 공간이에요.
- **탐구놀이터(2층)**: 생활 속 도구·에너지·기계 원리를 직접 만지고 실험하며 탐구해요.
- **관찰놀이터(2층)**: 공룡/화석/표본 등을 관찰하며 과학적 사고력을 키우는 공간이에요.

원하시면 "AI놀이터 전시물 뭐가 있어?"처럼 **특정 놀이터 이름**을 말해주면 더 자세히도 찾아서 안내해줄게요!""",
    },
    "English": {
        "default": """A quick tour of our exhibition zones. 😊

- **AI Zone (1F)**: Solve AI missions and experience artificial intelligence in an easy, fun way.
- **Activity Zone (1F)**: An active exhibit where children move their bodies to learn about health and movement.
- **Thinking Zone (1F)**: A zone designed to grow children's thinking skills (opening May 2026).
- **Light Zone (2F)**: An immersive media-interactive space themed around light, forests, and ecology.
- **Discovery Zone (2F)**: Touch and experiment with everyday tools, energy, and mechanical principles.
- **Observation Zone (2F)**: Observe dinosaurs, fossils, and specimens to grow scientific thinking.

Tell me a **specific zone name** (e.g., "What's in the AI Zone?") and I can give you a more detailed guide!""",
    },
    "日本語": {
        "default": """展示館(ゾーン)を簡単にご紹介します。😊

- **AIゾーン(1階)**:AIミッションを解きながら、人工知能を楽しく体験できる空間です。
- **アクティブゾーン(1階)**:体を動かしながら健康・運動の原理を体験する活動型展示館です。
- **考えるゾーン(1階)**:こどもたちの思考力を育てる展示館(2026年5月オープン予定)です。
- **ひかりゾーン(2階)**:光・森・生態をテーマに、メディアインタラクションで没入体験できる空間です。
- **探究ゾーン(2階)**:身の回りの道具・エネルギー・機械の原理を、実際に触って実験しながら探究します。
- **かんさつゾーン(2階)**:恐竜・化石・標本などを観察しながら科学的思考を育てる空間です。

「AIゾーンには何がある?」のように **特定のゾーン名** を教えていただければ、もっと詳しくご案内します!""",
    },
    "中文": {
        "default": """简单介绍各展览区(游乐区)。😊

- **AI区(1层)**:通过解决AI任务,轻松有趣地体验人工智能。
- **行动区(1层)**:通过身体活动来体验健康与运动原理的活动型展馆。
- **思考区(1层)**:培养儿童思考力的展馆(预计 2026 年 5 月开馆)。
- **光区(2层)**:以光、森林、生态为主题,通过媒体互动进行沉浸式体验的空间。
- **探究区(2层)**:亲手触摸和实验,探究生活中的工具、能源与机械原理。
- **观察区(2层)**:观察恐龙、化石、标本等,培养科学思维的空间。

告诉我**具体展区名称**(例:"AI区有什么?"),我可以更详细地为你介绍!""",
    },
}

# ============================================================================
# ROUTE BY AGE
# ============================================================================
_ROUTE_BY_AGE = {
    "한국어": {
        "default": """연령별로 추천 동선을 알려드릴게요! 😊

- 4~7세(유아)
  - 짧게 집중할 수 있는 체험 위주로, '빛놀이터'나 몸으로 움직이는 전시를 먼저 추천해요.
  - 중간중간 쉬는 시간(휴게실/수유실)도 꼭 챙겨주세요.

- 초등 저학년
  - '탐구놀이터/관찰놀이터'에서 직접 만지고 해보는 체험을 먼저 하고,
  - 관심이 생기면 '천체투영관'으로 마무리하면 좋아요.

- 초등 고학년
  - 'AI놀이터'에서 미션형 체험을 하고,
  - '탐구놀이터'에서 원리 탐색을 한 뒤,
  - 시간이 되면 '전시해설/과학쇼' 같은 프로그램도 추천해요.

원하시면 아이 나이(예: 6살, 초2, 초5)랑 지금 위치(1층/2층)를 말해주면 더 딱 맞게 짜줄게요!""",
    },
    "English": {
        "default": """Age-based recommended routes. 😊

- Ages 4–7 (Preschool)
  - Start with short, hands-on experiences. The Light Zone and movement-based exhibits work great.
  - Don't forget breaks at the lounge or baby-care room between sessions.

- Lower Elementary
  - Begin with hands-on experiments at the Discovery Zone or Observation Zone.
  - If interest grows, finish with the Planetarium.

- Upper Elementary
  - Try the mission-style experiences at the AI Zone first.
  - Then explore science principles at the Discovery Zone.
  - If time allows, programs like the Exhibit Talk or Science Show are great additions.

Tell me your child's age (e.g., 6 years old, 2nd grade, 5th grade) and current location (1F/2F), and I'll tailor it more precisely!""",
    },
    "日本語": {
        "default": """年齢別おすすめ動線です。😊

- 4~7歳(幼児)
  - 短時間集中型の体験を中心に、「ひかりゾーン」や体を動かす展示をまずおすすめします。
  - 合間に休憩(休憩室/授乳室)も忘れずに。

- 小学生(低学年)
  - まずは「探究ゾーン/かんさつゾーン」で直接触れて体験し、
  - 興味が出たら「プラネタリウム」で締めくくると良いでしょう。

- 小学生(高学年)
  - 「AIゾーン」でミッション型の体験をしてから、
  - 「探究ゾーン」で原理を探究し、
  - 時間があれば「展示解説/サイエンスショー」もおすすめです。

お子様の年齢(例:6歳、小学2年、小学5年)と現在地(1階/2階)を教えていただければ、もっとぴったりのコースをご提案します!""",
    },
    "中文": {
        "default": """按年龄推荐的参观路线。😊

- 4~7岁(幼儿)
  - 以短时集中的体验为主,推荐先去"光区"或可以活动身体的展馆。
  - 中途别忘了去休息室/哺乳室休息一下。

- 小学低年级
  - 先在"探究区/观察区"进行亲手体验,
  - 如果兴趣浓厚,最后以"天体投影馆"收尾。

- 小学高年级
  - 先在"AI区"挑战任务式体验,
  - 然后在"探究区"探索原理,
  - 时间允许的话,推荐"展品讲解/科学秀"等节目。

告诉我孩子的年龄(如:6岁、小学二年级、小学五年级)和当前位置(1层/2层),我可以为你定制更精准的路线!""",
    },
}

# ============================================================================
# RESERVATION GUIDE
# ============================================================================
_RESERVATION = {
    "한국어": {
        "default": """예약안내를 친절하게 정리해서 알려드릴게요! 😊

## 예약 기본 안내
- 하루 입장 인원은 **최대 1,600명**으로 제한됩니다.

## (중요) 어린이 동반 없는 성인/청소년 관람객 안내
- **어린이(신체연령 초등학생 이하)를 동반하지 않은 성인 및 청소년**은 관람을 위해 사전 협의가 필요합니다.
- 방문을 원하실 경우, **방문 3일 전까지** 방문신청서를 담당자 메일로 보내주세요.
  - 담당자 메일: **proxima11@korea.kr**
  - 방문 신청서 양식은 '성인 및 청소년 관람객 입장안내' 게시글의 첨부파일을 확인해 주세요.

## 체험별 사전예약 비율(요약)
- **상설전시관**
  - 인터넷 예매: (3~6월/9~12월) 평일 50%, 주말 75% / (7~8월/1~2월) 평일 75%, 주말 75%
  - 현장 판매: (3~6월/9~12월) 평일 50%, 주말 25% / (7~8월/1~2월) 평일 25%, 주말 25%
  - 단체: **사전예약 필수**(단체 관람 이용안내 참조)
- **천체투영관**
  - 인터넷 예매: **100% (사전예약 필수)**

## 예약 가능 기간(개인/단체)
- 예약 판매를 우선으로 하며, **잔여석에 한해 현장판매**를 진행합니다.
- 인터넷 예매는 관람일 **2주(14일) 전부터** 가능합니다.
  - 예1) 오늘이 5월 8일이면, 5월 22일(14일 후)까지 예약 가능(하루씩 자동 연장)
  - 예2) 5월 22일 예약은 5월 8일 **00:00부터** 가능
- 예약 기간은 과학관 운영 사정에 따라 변경될 수 있습니다.(변동 시 별도 공지)
- **평일에 예약 가능**하며, **주말 및 공휴일에는 예약을 받지 않습니다.**

## 예약 시 유의사항
- 상설전시관/천체투영관은 **개인예약 선택 후** 예약할 수 있습니다.
- 예약 신청 시 받은 **문자 메시지(URL)**를 보관해 주세요. (신청 확인/취소에 사용)
- **5명 이상 어린이 기관**은 단체예약 페이지에서 예약해 주세요.
- 예약한 **입장시간을 지켜** 입장해 주세요.
- 예약 고객은 예약 완료 후 발권되는 **'모바일 QR입장권'**으로 입장할 수 있습니다.
- **결제까지 완료**되어야 최종 예약 완료입니다.

원하시면 지금 상황(개인/단체/교육, 관람 날짜, 인원, 어린이 동반 여부)을 말해주시면 딱 맞게 안내해드릴게요!""",
    },
    "English": {
        "default": """A friendly summary of the reservation guide. 😊

## General
- Daily admission is limited to **1,600 visitors**.

## (Important) Adult / teen visitors without children
- **Adults and teens not accompanied by a child (elementary age or younger)** must arrange their visit in advance.
- If you wish to visit, please send a visit application form to the staff email **at least 3 days before** your visit.
  - Email: **proxima11@korea.kr**
  - The visit-application form is attached to the "Adult & Teen Visitor Entry Guide" notice on the website.

## Reservation share by program (summary)
- **Permanent Exhibition Hall**
  - Online: (Mar–Jun / Sep–Dec) Weekdays 50%, Weekends 75% / (Jul–Aug / Jan–Feb) Weekdays 75%, Weekends 75%
  - On-site: (Mar–Jun / Sep–Dec) Weekdays 50%, Weekends 25% / (Jul–Aug / Jan–Feb) Weekdays 25%, Weekends 25%
  - Groups: **reservation required** (see Group Visit Guide)
- **Planetarium**
  - Online: **100% (reservation required)**

## Reservation Window (Individual / Group)
- Reservations are prioritized; **remaining seats are sold on-site**.
- Online reservations open **14 days before** your visit date.
  - Ex 1) If today is May 8, you can reserve up to May 22 (auto-extending day by day).
  - Ex 2) Reservation for May 22 opens at **00:00 on May 8**.
- The reservation window may change depending on operational circumstances (announced separately).
- Reservations are accepted **on weekdays only**, **not on weekends or holidays**.

## Important Notes
- For both the Permanent Exhibition Hall and Planetarium, please choose **"Individual Reservation"** to make your booking.
- Please keep the **text message (URL)** you receive after booking — it is used for confirming or cancelling your reservation.
- **Children's groups of 5 or more** must use the Group Reservation page.
- Please **arrive at your reserved entry time**.
- Reserved guests enter using the **mobile QR ticket** issued after booking.
- A reservation is finalized **only after payment is complete**.

Tell me your situation (individual / group / education, visit date, party size, whether children are with you) and I'll guide you precisely!""",
    },
    "日本語": {
        "default": """予約案内を分かりやすくまとめました。😊

## 基本案内
- 1日の入場人数は **最大 1,600名** に制限されています。

## (重要)こども同伴のない大人・青少年の方へ
- **こども(身体年齢が小学生以下)を同伴しない大人・青少年**の方は、観覧のため事前協議が必要です。
- ご訪問希望の場合、**訪問の3日前まで**に訪問申請書を担当者メールにお送りください。
  - 担当者メール:**proxima11@korea.kr**
  - 訪問申請書の様式は、「成人および青少年来館者入場案内」のお知らせの添付ファイルをご確認ください。

## 体験別の事前予約割合(要約)
- **常設展示館**
  - インターネット予約:(3~6月/9~12月)平日50%、週末75% /(7~8月/1~2月)平日75%、週末75%
  - 現場販売:(3~6月/9~12月)平日50%、週末25% /(7~8月/1~2月)平日25%、週末25%
  - 団体:**事前予約必須**(団体観覧ご利用案内をご参照)
- **プラネタリウム**
  - インターネット予約:**100%(事前予約必須)**

## 予約可能期間(個人/団体)
- 予約販売が優先され、**残席に限り現場販売**を行います。
- インターネット予約は観覧日の **2週間(14日)前から** 可能です。
  - 例1)本日が5月8日の場合、5月22日(14日後)まで予約可能(1日ずつ自動延長)
  - 例2)5月22日の予約は5月8日 **00:00から** 可能
- 予約期間は科学館の運営状況により変更される場合があります(変更時は別途告知)。
- **予約は平日のみ受付**で、**週末・祝日は受付けません**。

## 予約時のご注意
- 常設展示館/プラネタリウムは **「個人予約」を選択後** に予約できます。
- 予約申請時に届く **メッセージ(URL)** は保管してください。(申請確認/キャンセルに使用)
- **5名以上のこども団体** は団体予約ページからご予約ください。
- 予約された **入場時間を守って** ご入場ください。
- 予約完了後に発行される **「モバイルQR入場券」** でご入場できます。
- **決済完了** をもって最終予約完了となります。

状況(個人/団体/教育、観覧日、人数、こども同伴の有無)を教えていただければ、ぴったりご案内します!""",
    },
    "中文": {
        "default": """温馨整理预约信息。😊

## 基本信息
- 每日入场人数限制为 **最多 1,600 人**。

## (重要)未带儿童的成人/青少年访客
- **未带儿童(身体年龄在小学生及以下)的成人和青少年**需事先协商方可参观。
- 如希望来访,请**在来访前3天**将来访申请书发送至工作人员邮箱。
  - 工作人员邮箱:**proxima11@korea.kr**
  - 来访申请书样式请参考"成人及青少年访客入场指南"公告中的附件。

## 各项目预约比例(摘要)
- **常设展览馆**
  - 网上预约:(3~6月/9~12月)平日 50%,周末 75% /(7~8月/1~2月)平日 75%,周末 75%
  - 现场售票:(3~6月/9~12月)平日 50%,周末 25% /(7~8月/1~2月)平日 25%,周末 25%
  - 团体:**必须预约**(请参考团体参观使用指南)
- **天体投影馆**
  - 网上预约:**100%(必须预约)**

## 可预约期间(个人/团体)
- 优先预约售票,**剩余座位现场销售**。
- 网上预约可在参观日的 **2 周(14 天)前** 开始。
  - 例1)今天是 5 月 8 日,可预约至 5 月 22 日(14 天后,每天自动延长)。
  - 例2)5 月 22 日的预约可从 5 月 8 日 **00:00** 开始。
- 预约期间可能根据科学馆运营情况变更(变更时另行通知)。
- **仅平日可预约**,**周末和节假日不接受预约**。

## 预约注意事项
- 常设展览馆/天体投影馆需 **选择"个人预约"** 后方可预约。
- 请妥善保存预约时收到的 **短信(URL)**(用于确认/取消预约)。
- **5 人以上的儿童团体** 请使用团体预约页面。
- 请 **遵守预约的入场时间** 入场。
- 预约客户可使用预约完成后发放的 **"手机 QR 入场券"** 入场。
- **完成支付** 后预约才算最终成立。

告诉我你的情况(个人/团体/教育、参观日期、人数、是否带儿童),我可以精准地为你介绍!""",
    },
}

# ============================================================================
# OPERATING HOURS — 동적 status 부분은 별도 처리. 여기는 안내문 본문 템플릿.
# ============================================================================
# 사용법: get_operating_hours_text(language, mode, status_text)
# status_text 는 한국어로 들어오므로 별도 매핑 사용

OPERATING_STATUS_TRANSLATIONS = {
    # KO substring → {language: translated full sentence}
    "현재 정상 운영 중입니다": {
        "English": "We're open right now! Hours: 09:30–17:30. Last entry: 16:30.",
        "日本語": "現在通常通り開館中です!開館時間は09:30~17:30、最終入場は16:30です。",
        "中文": "现在正常开馆中!开放时间 09:30~17:30,最后入场 16:30。",
    },
    "아직 개관 전이에요": {
        "English": "We haven't opened yet today. Hours: 09:30–17:30. Last entry: 16:30.",
        "日本語": "本日はまだ開館前です。開館時間は09:30~17:30、最終入場は16:30です。",
        "中文": "今天还未开馆。开放时间 09:30~17:30,最后入场 16:30。",
    },
    "오늘 관람 시간은 종료됐어요": {
        "English": "Today's visiting hours have ended. Hours: 09:30–17:30. Last entry: 16:30.",
        "日本語": "本日の観覧時間は終了しました。開館時間は09:30~17:30、最終入場は16:30です。",
        "中文": "今天的参观时间已结束。开放时间 09:30~17:30,最后入场 16:30。",
    },
    "정기휴관일(월요일)": {
        "English": "today is the regular Monday closure.",
        "日本語": "本日は月曜日の定期休館日です。",
        "中文": "今天是周一定期休馆日。",
    },
    "휴관일(1월 1일)": {
        "English": "today is closed for the New Year holiday (Jan 1).",
        "日本語": "本日は1月1日の休館日です。",
        "中文": "今天因元旦(1月1日)休馆。",
    },
}

_OPERATING_HOURS_TEMPLATE = {
    "한국어": {
        "어린이": "오늘 어린이과학관은 어떨까요? 🚀\n\n{status}\n\n휴관일은 기본적으로 **매주 월요일**, **1월 1일**, **설날/추석 당일**이에요.\n(월요일이 공휴일이면 개관하고, 화요일에 대체 휴관할 수 있어요.)",
        "default": "운영 상태 안내입니다.\n\n{status}\n\n휴관일은 기본적으로 매주 월요일, 1월 1일, 설날/추석 당일입니다. (월요일 공휴일은 개관 후 화요일 대체 휴관 가능)",
    },
    "English": {
        "default": "Operating status:\n\n{status}\n\nWe are typically closed every **Monday**, **January 1**, and on the **day of Lunar New Year and Chuseok**. (If a Monday falls on a public holiday, we open that day and may close on Tuesday instead.)",
    },
    "日本語": {
        "default": "本日の運営状況:\n\n{status}\n\n休館日は原則として**毎週月曜日**、**1月1日**、**旧正月・秋夕の当日**です。(月曜日が祝日の場合は開館し、火曜日に振替休館することがあります。)",
    },
    "中文": {
        "default": "运营状态:\n\n{status}\n\n休馆日原则上为**每周一**、**1月1日**、**春节/中秋节当日**。(若周一为公休日则正常开馆,周二可能调休。)",
    },
}


def get_operating_hours_text(language: str, mode: str, ko_status: str) -> Optional[str]:
    """운영시간 답변. status는 한국어 동적 텍스트 → 언어별 매핑 후 템플릿에 주입.

    NOTE: 2026-04 이후 core.py 의 한국어 operating_hours 답변이 크게 풍부화되어,
    이 짧은 템플릿은 더 이상 동등한 번역을 제공하지 못한다. 따라서 비활성화하여
    answer_rule_based_localized 가 translate_answer_cached(LLM) 로 폴백하도록 한다.
    향후 4언어 풍부화 작업이 끝나면 아래 줄을 제거하고 새 템플릿으로 교체할 것.
    """
    return None  # ← LLM 번역 폴백 강제 (풍부화된 KO 동기화 보장)
    template_lang = _OPERATING_HOURS_TEMPLATE.get(language)
    if not template_lang:
        return None
    template = template_lang.get(mode) or template_lang.get("default")
    if not template:
        return None

    if language == "한국어":
        return template.format(status=ko_status)

    # 외국어: status 매핑 시도
    translated_status = None
    for ko_key, lang_map in OPERATING_STATUS_TRANSLATIONS.items():
        if ko_key in ko_status:
            translated_status = lang_map.get(language)
            break

    if not translated_status:
        # 매핑 없으면 fallback signal
        return None

    return template.format(status=translated_status)


# ============================================================================
# 메인 사전
# ============================================================================
STATIC_FAQ_ANSWERS = {
    "parking": _PARKING,
    "admission_fee": _ADMISSION,
    "planetarium_timetable": _PLANETARIUM_TIMETABLE,
    "floor_guide": _FLOOR_GUIDE,
    "facility_amenities": _FACILITY,
    # exhibit_guide / route_by_age 는 core.py 에서 풍부화되어 정적 번역과 길이가 크게 어긋남.
    # → 의도적으로 dict 에서 제외하여 translate_answer_cached(LLM, 24h 캐시) 로 폴백.
    # _EXHIBIT_GUIDE / _ROUTE_BY_AGE 는 보존하지만 사용되지 않음 (향후 4언어 동기화 시 재활성화).
    "reservation_guide": _RESERVATION,
}
