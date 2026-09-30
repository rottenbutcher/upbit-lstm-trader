# 설계와 공식 출처

확인일: 2026-09-30. 실제 거래소·클라우드 정책은 변경될 수 있습니다.

## 데이터

[Upbit 분 캔들](https://docs.upbit.com/kr/reference/list-candles-minutes)은 요청당 최대 200개, `to` 이전(exclusive) 데이터를 반환합니다. 거래가 없는 구간의 캔들은 생기지 않습니다. UTC로 통일하며, 현재 진행 중인 봉은 제외합니다. 역방향 페이지를 SQLite에 원자적으로 저장하고 cursor와 함께 커밋합니다. 완료 후 새 봉은 별도로 증분 갱신합니다. HTTP 오류는 빈 데이터로 바꾸지 않습니다. 장기간 누락된 데이터는 0 거래량으로 표현되지만 거래소 장애와 거래 없음의 완벽한 구분은 불가능합니다.

종목은 최초 실행 시점의 거래대금 상위 5개로 고정합니다. 과거 상장폐지 종목을 포함한 시점별 universe 복원은 하지 않습니다. 최소 역사 조건을 만족하지 않으면 다른 종목으로 자동 교체하지 않고 오류를 표시합니다.

## 학습·평가

`feature_frame()`은 과거/현재 값만 사용하는 유한 길이 rolling 연산으로 학습과 추론을 공유합니다. 초기 48봉은 워밍업이고, 모델은 그 뒤 96봉을 받습니다. 거래량은 48봉 로그 z-score이며 무한 과거 EMA를 사용하지 않아 서버의 짧은 조회 창과 학습 계산이 일치합니다. 정규화 평균·표준편차는 학습 기간만으로 구합니다.

신호 시점 t의 입력 마지막 값은 close[t]. 목표는 log(open[t+7]/open[t+1])이며 t+1 진입 후 30분 수익률입니다. 실제 시세가 5분 안에 크게 바뀌면 시장가 체결은 이 가정과 달라집니다. 학습 라벨의 마지막 가격 시점이 다음 구간에 들어가지 않도록 purge합니다. 검증/테스트에서 과거 입력창을 가져오는 것은 당시 사용 가능한 정보입니다.

큰 배열은 NumPy `.npy` memmap, 윈도우는 DataLoader에서 요청할 때 생성합니다. 8GB VRAM에 맞춰 64 hidden/2-layer LSTM을 사용합니다. CPU 검증과 서버 ONNX 추론은 학습 GPU와 독립적입니다. PyTorch→ONNX 변환 후 batch=1 및 batch=5에서 오차를 검증합니다. 금융 데이터의 방향성 예측이 쉬워진다는 보장은 없습니다.

[PyTorch 2.7 공식 발표](https://pytorch.org/blog/pytorch-2-7/)의 CUDA 12.8 / Blackwell 지원에 맞춰 RTX 5060 설정을 구성했습니다. 드라이버는 [NVIDIA 공식 다운로드](https://www.nvidia.com/en-us/drivers/)에서 준비합니다. [설치 안내](https://pytorch.org/get-started/locally/), [이전 버전](https://pytorch.org/get-started/previous-versions/)도 참고하세요. 사용자 GPU에서의 속도/최고 VRAM은 아직 측정하지 않았습니다.

백테스트는 현금과 종목별 원장을 통합해 전체 투자 한도와 주문금액 상한을 적용합니다. 현재 호가 스프레드/깊이는 과거 시가 자료만으로 재현하지 못하므로 실제 실행부가 더 적게 거래할 수 있습니다. 순서대로 매도 후 매수하며 다음 봉 시가와 비용으로 평가합니다. 손절은 종가에서 판단하며 장중 저가 체결을 가정하지 않습니다. 마지막 미청산 포지션은 비용을 차감한 평가가격으로 청산해 비교합니다. 현금과 동일 50% 노출 단순 보유를 기준선으로 제공합니다.

## 주문과 장애 처리

[Upbit 인증](https://docs.upbit.com/kr/reference/auth)은 HS512 JWT 및 nonce, URL 인코딩 전 query string의 SHA512 해시를 사용합니다. POST 본문은 JSON입니다. 조회 재시도와 주문 생성 재시도를 분리하며 POST는 자동 재시도하지 않습니다.

[주문 생성](https://docs.upbit.com/kr/reference/new-order)의 `identifier`는 주문 상태와 관계없이 재사용할 수 없습니다. 시장가 매수는 `ord_type=price`+KRW 금액, 매도는 `ord_type=market`+수량입니다. 접수 직전에 SQLite intent를 커밋하고 [주문 조회](https://docs.upbit.com/kr/reference/get-order)로 실제 체결을 대조합니다. timeout/404처럼 불확실한 상태는 정지하며 새 식별자로 동일 주문을 반복하지 않습니다.

[주문 가능 정보](https://docs.upbit.com/kr/reference/available-order-information)에서 계정 적용 수수료, 주문 지원 유형, 최소/최대 금액, 잔고를 확인합니다. 최소금액을 코드 상수만으로 판단하지 않습니다. 호가 스프레드와 매수 최우선 호가 잔량도 검사하지만 시장가 주문의 최종 슬리피지는 보장하지 못합니다.

[API 키/IP 설정](https://docs.upbit.com/kr/docs/api-key)은 사용자 직접 작업입니다. 이 프로젝트는 출금 API를 호출하거나 출금 권한을 요구하지 않습니다.

## Oracle

[Oracle Always Free 공식 문서](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)의 현재 A1 무료 안내는 월 1,500 OCPU 시간/9,000 GB 시간(2 OCPU/12GB 상당)이며, 오래된 4 OCPU/24GB 설명을 그대로 적용하지 않았습니다. 1 OCPU/6GB 한 대로 추론하기에 충분한 작은 모델을 구성합니다. 실제 계정의 Always Free 표시, 홈 리전, 다른 사용량을 함께 확인하세요. 공식 문서는 7일간 유휴 사용률 기준으로 회수될 수 있다고 설명합니다. 따라서 무료 VM에 무중단 운용을 보장할 수 없습니다.

systemd oneshot+timer를 사용하고 HTTP 수신 포트를 열지 않습니다. UTC 매 5분+15초에 실행하고 `Persistent=false`로 밀린 실행을 소급하지 않습니다. SQLite, 파일 잠금, 동일 bar primary key가 재부팅/중복 실행을 보호합니다. 멀티서버 동시 운용은 지원하지 않습니다.
