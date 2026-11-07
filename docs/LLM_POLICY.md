# LLM 일일 파라미터 정책

LSTM은 5분마다 30분 수익률을 예측하고, LLM은 **하루 한 번 거시 환경을 읽어 매매 파라미터를 조정**합니다. LLM은 주문을 내지 않으며 Upbit 키를 보지 못합니다. 숫자만 제안하고, 코드가 안전 범위로 검증한 값만 실거래에 반영됩니다.

```
[매일 00:20 UTC = 09:20 KST, upbit-lstm-policy.timer]
 거시지표(FRED) + 공식 발표(Fed·ECB·SEC·BLS) + 공포탐욕지수 + Upbit 일봉 추세
   → LLM (구조화된 도구 호출 강제)
   → policy.sanitize(): 범위 고정 + 하루 변경폭 제한
   → /var/lib/upbit-lstm/policy.json (원자적 저장, 이력 보관)
[5분마다, upbit-lstm.timer]
 policy.json 재검증 → 전략 파라미터에 적용 → LSTM 예측 → 주문 판단
```

## 수집 데이터

| 구분 | 출처 | 내용 |
|---|---|---|
| 금리·물가 | FRED 공개 CSV (키 불필요) | 연방기금금리, 미 2년/10년물, 장단기 금리차, 기대인플레이션 |
| 위험 지표 | FRED | VIX, 하이일드 스프레드, 달러지수, 원/달러, WTI, S&P 500, 나스닥 |
| 공식 문서 | RSS | Fed·ECB·SEC 보도자료, BLS 최신 통계 발표 (최근 7일) |
| 암호화폐 | alternative.me, CoinGecko | 공포·탐욕 지수, BTC 점유율, 시총 변화 |
| 거래 종목 | Upbit 공개 일봉 | 1/7/30일 수익률, 일간 변동성, 20일 평균 대비 위치 |

각 출처는 독립적으로 실패할 수 있습니다. 성공한 출처가 `llm.min_sources`(기본 3)보다 적으면 LLM을 호출하지 않고 기존 정책을 유지합니다. 입력 스냅샷은 `policy-inputs/`에 저장되어 사후 검증할 수 있습니다. 출처는 `config.toml`의 `[macro]`에서 바꿀 수 있습니다.

## LLM이 조정하는 값과 안전 범위

기준값은 **학습·백테스트로 검증된 모델 매니페스트의 값**입니다. 범위는 `config.toml`의 `[policy]`에서 정합니다.

| 파라미터 | 기본 범위 | 비고 |
|---|---|---|
| `allow_new_entries` | true/false | false면 당일 신규 매수 중단 (매도·손절은 계속) |
| `threshold_bps` | 기준 × 0.8 ~ 3.0 | 매수에 필요한 최소 예측 수익 |
| `position_fraction` | 기준 × 0 ~ **1.0** | 위험 축소만 가능 |
| `max_exposure` | 기준 × 0 ~ **1.0** | 위험 축소만 가능 |
| `stop_loss` | 기준 × 0.5 ~ **1.0** | 더 넓은 손절 불가 |
| `daily_loss_limit` | 기준 × 0.5 ~ **1.0** | 더 느슨하게 불가 |
| `take_profit` | 기준 × 0.5 ~ 1.5 | |
| `max_hold_bars` | 기준 × 0.5 ~ 2.0 | |
| `exit_bps` | -20 ~ 30 | |
| `uncertainty_penalty` | 0 ~ 2 | 신호 = 예측 − 계수 × 앙상블 불일치 |
| `blocked_markets` | 학습 종목 중 | 특정 종목 신규 매수 차단 |

강제되는 규칙:

- **위험 배수의 상한은 코드에서 1.0으로 고정됩니다.** 설정 파일에서 1보다 크게 쓰면 시작 시 오류가 납니다. LLM은 검증된 위험보다 더 큰 위험을 질 수 없습니다.
- `max_drawdown`, 수수료, 슬리피지, 주문 상한, 자본은 정책으로 바꿀 수 없습니다.
- 한 번의 갱신으로 각 값은 허용 범위의 `max_step`(기본 50%)까지만 움직입니다.
- LLM 신뢰도가 `min_confidence` 미만이면 크기 조정은 무시합니다. 다만 매수 중단 요청은 따릅니다.
- 정책은 `valid_hours`(36시간) 후 만료됩니다. 파일이 없거나, 손상되었거나, 만료되면 기준값으로 돌아갑니다. 보수적으로 운용하려면 `on_stale = "no_new_entries"`로 설정하세요.
- 런타임은 5분마다 파일을 **다시 검증**합니다. 사람이 파일을 직접 고쳐도 범위를 넘을 수 없습니다.
- 공식 문서 텍스트는 신뢰하지 않는 데이터로 표시해 전달합니다. 문서 속 지시문은 무시하도록 지시하지만, 그보다 중요한 방어는 위의 범위 제한입니다.

## Oracle 서버 설정

```bash
sudo bash /opt/upbit-lstm/current/deploy/configure-llm.sh
```

키를 입력하면 `/etc/upbit-lstm-llm.env`(root:upbitbot 640)에 저장합니다. 이 파일은 Upbit 키 파일과 분리되어 있습니다. 이어서 실제 데이터 수집과 LLM 호출을 **dry run**으로 시험하고, 성공하면 일일 타이머를 켭니다. 이 단계를 건너뛰면 봇은 기존과 똑같이 기준값으로 동작합니다.

기본 제공자는 Anthropic(`claude-sonnet-5-5`)입니다. OpenAI 호환 API를 쓰려면 `[llm] provider = "openai"`, `model`, 필요하면 `base_url`을 지정하세요. 모델 변경은 학습 PC에서 `config.toml`을 수정해 다시 배포합니다.

## 운영 명령

```bash
sudo bash /opt/upbit-lstm/current/deploy/botctl.sh policy-show                 # 다음 사이클에 적용될 값과 상태
sudo bash /opt/upbit-lstm/current/deploy/botctl.sh policy-update --dry-run     # 저장 없이 시험
sudo systemctl start upbit-lstm-policy.service                                  # 지금 즉시 갱신
sudo journalctl -u upbit-lstm-policy.service --since today --no-pager
# 수동 조정 (같은 범위 적용)
sudo bash /opt/upbit-lstm/current/deploy/botctl.sh policy-set --set allow_new_entries=false --confirm
sudo bash /opt/upbit-lstm/current/deploy/botctl.sh policy-set --set threshold_bps=60 --set position_fraction=0.05 --confirm
sudo bash /opt/upbit-lstm/current/deploy/botctl.sh policy-clear --confirm        # 기준값으로 복귀
sudo systemctl disable --now upbit-lstm-policy.timer                            # LLM 갱신 끄기
```

`health.json`의 `policy` 항목에서 매 사이클에 적용된 정책 출처, 생성 시각, 만료 여부를 볼 수 있습니다. 모든 정책 버전은 `policy-history/`에 남습니다.

## 한계

- LLM이 제안하는 파라미터 조합은 **백테스트로 검증된 것이 아닙니다.** 그래서 위험을 줄이는 방향으로만 허용합니다. 매수 기준을 낮추는 하한(0.8배)도 원치 않으면 `threshold_multiplier = [1.0, 3.0]`으로 설정하세요.
- 거시 지표 대부분은 하루 이상 늦게 갱신됩니다. 장중 급변에 대한 대응은 LSTM과 기존 손실 제한이 담당합니다.
- LLM 호출에는 API 비용이 듭니다. 기본 설정은 하루 1회 호출입니다.
- 모의매매로 충분히 관찰한 뒤 실거래에 적용하세요.
