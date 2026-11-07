# 운영과 장애 복구

## 정상 상태

```bash
sudo systemctl list-timers upbit-lstm.timer
sudo journalctl -u upbit-lstm.service --since '1 hour ago' --no-pager
sudo bash /opt/upbit-lstm/current/deploy/botctl.sh status
```

`health.json`은 마지막 결과, 예측, 봇 소유 수량과 시각을 기록합니다. `status`는 기록이 11분 이상 오래됐는지도 표시합니다. 별도 외부 알림 서비스가 없으므로 서버 회수/완전 정지는 외부에서 확인해야 합니다. 무료 VM은 항상 켜져 있다는 보장이 없습니다.

## LLM 일일 정책

`sudo bash /opt/upbit-lstm/current/deploy/configure-llm.sh`로 키를 등록하면 매일 09:20 KST에 파라미터가 갱신됩니다. 확인은 `botctl.sh policy-show`로 합니다. 즉시 신규 매수를 막으려면 `botctl.sh policy-set --set allow_new_entries=false --confirm`을 실행합니다(매도·손절은 계속). 자세한 내용은 [LLM_POLICY.md](LLM_POLICY.md)를 보세요. LLM이나 데이터 수집이 실패해도 매매는 멈추지 않고, 정책 만료 후 검증된 기준값으로 돌아갑니다.

## 즉시 중지 / 모의매매 복귀

```bash
sudo touch /var/lib/upbit-lstm/STOP
sudo systemctl stop upbit-lstm.timer
```

STOP 파일은 **매도도 포함해 모든 새로운 주문을 멈추며**, 이미 접수된 주문을 취소하거나 코인을 청산하지 않습니다. 실행 중인 주문의 상태는 Upbit에서 직접 확인하세요. 모의매매로 돌아가려면 `/etc/upbit-lstm.env`의 `MODE=paper`와 빈 `LIVE_ACK=`를 설정하세요. 키를 지워도 됩니다. 중지 원인을 해결한 다음 STOP 파일을 직접 제거하고 타이머를 다시 시작하세요. 활성 주문 처리 중 서비스를 강제 종료하는 대신 타이머부터 중지하세요.

## 연속 오류 / 잔고 불일치

연속 3회 실패하면 오류 차단기가 작동합니다. 원인을 고쳐도 자동 해제하지 않습니다. 로그를 확인한 후:

```bash
sudo bash /opt/upbit-lstm/current/deploy/botctl.sh reset-errors --mode live --confirm
```

모의매매는 `--mode paper`를 사용합니다. 오류 초기화는 미확정 주문을 삭제하거나 다시 제출하지 않습니다. 수동 매도나 다른 봇 때문에 계좌 잔고가 원장보다 적으면 봇을 중지하고 거래 내역과 원장을 먼저 대조하세요. **`live.sqlite` 삭제로 해결하지 마세요.** 봇이 무엇을 소유하고 어떤 주문을 보냈는지 잃게 됩니다.

## 주문 응답 유실 / 404

원장에 intent가 있는데 주문 조회가 404라면, 주문이 전송되지 않았거나 아직 조회되지 않을 수 있습니다. 코드는 두 경우를 추측하지 않고 정지합니다. 원장 `intents.id`와 Upbit 주문 내역을 대조하세요. 의도적으로 자동 재전송이나 자동 원장 삭제 기능을 제공하지 않습니다. 접수되지 않은 것이 확인된 경우에만 운영자가 백업 후 해당 intent를 `abandoned`로 변경할 수 있습니다. 이 상태 변경은 주문 접수 여부를 완전히 확인한 뒤 수행해야 하며, 동일 식별자를 재사용하지 않습니다. 불확실하면 거래소 확인을 기다리세요.

부분 체결은 최종 `done`/`cancel` 상태의 실제 체결 수량·금액·수수료만 원장에 반영합니다. `wait` 상태가 계속되면 신규 거래를 진행하지 않습니다. 봇은 임의의 기존 주문을 취소하지 않습니다.

## 손실 제한

일일 손실 기준은 UTC 날짜, 즉 한국 시간 09시 경계입니다. 일일 제한은 다음 UTC 날짜에 초기화됩니다. 최대 낙폭 제한은 자동으로 초기화하지 않습니다. 모든 봇 관리 포지션이 정리되고 미확정 주문이 없을 때만 다음 명령으로 기준을 재설정할 수 있습니다.

```bash
sudo bash /opt/upbit-lstm/current/deploy/botctl.sh resume-risk --mode live --confirm
```

작은 잔량이 최소 주문 미만이면 자동 청산하지 못할 수 있습니다. 잔량이 있으면 기준 재설정과 배포 교체를 막습니다. 잔량 처리도 계좌/원장 대조 후 운영자가 결정해야 합니다.

## 보유 코인 등록

실거래 키 설정 후 타이머를 잠시 중지하고 `adopt`를 실행할 수 있습니다. API는 잔고를 읽기만 하며, 등록 자체로 주문하지 않습니다. 등록 시점 시장가격으로 원장에 기록하고 그만큼 가상 현금을 줄입니다. **세금용 취득 원가를 복원하는 기능은 아닙니다.** 원래 평단가와 다를 수 있습니다. 등록 수량은 다음 실행부터 매도 대상입니다.

```bash
sudo systemctl stop upbit-lstm.timer
sudo bash /opt/upbit-lstm/current/deploy/botctl.sh adopt --market KRW-BTC --quantity 0.0001 --confirm
sudo systemctl start upbit-lstm.timer
```

종목별 한도와 나머지 KRW 예산 조건 때문에 전체 보유 코인을 한꺼번에 등록할 수 없습니다. 원화 없이 코인만 보유한 경우, 원하는 운용 범위에 맞는 자금 구성을 먼저 직접 정해야 합니다.

## 학습 이어하기 (체크포인트)

체크포인트는 모델 구조, 피처, 학습률·샘플링, 데이터 분할, 저장된 캔들에만 묶입니다. epoch 수, patience, 배치 크기, 앙상블 수, 전략·승인·자본·LLM 설정은 바꿔도 이어서 학습할 수 있습니다. 앙상블 수를 늘리면 완료된 모델은 재사용합니다.

이어서 할 수 없으면 GUI가 달라진 항목을 이전 값 → 현재 값으로 보여 주고, **체크포인트 설정으로 되돌리고 이어서 학습** 또는 **처음부터 학습(데이터 유지)**을 선택하게 합니다. 명령줄에서는 `train --resume`이 종료 코드 3과 함께 달라진 항목을 출력하고, `artifacts/resume-mismatch.json`에도 기록합니다. 새로 학습하면 이전 체크포인트는 삭제하지 않고 `artifacts/old-checkpoints/<시각>/`으로 옮깁니다. 이전 버전 프로그램으로 만든 체크포인트는 이어서 할 수 없으므로 새로 학습하세요.

## 재학습과 모델 교체

1. 기존 `artifacts` 결과를 다른 폴더에 보관하고 학습 PC에서 `TRAIN.cmd`를 실행합니다. 기존 데이터는 이어받고 새 봉을 갱신합니다. 종목 목록은 고정됩니다.
2. 새 결과를 검토합니다. 같은 테스트 기간에 맞춰 반복 튜닝하면 과적합될 수 있으니 새로 쌓인 데이터로 평가하세요.
3. 서버 타이머를 중지하고 봇 관리 포지션 및 미확정 주문을 정리합니다. installer는 이 항목들이 남아 있으면 교체를 거부합니다.
4. 새 ZIP을 **새 디렉터리**에 풀고 installer를 실행합니다. `/var/lib/upbit-lstm`와 `/etc/upbit-lstm.env`는 유지합니다. 이전 release 디렉터리는 자동 삭제하지 않습니다.

새로 선택한 상위 5종목으로 바꾸려면 학습 PC에서 `data`를 별도 보관하고 새로 수집합니다. 동일 모델에서 종목 번호/순서를 임의 변경하지 마세요.

## 백업과 키 보안

- 키: `/etc/upbit-lstm.env`, root 소유, upbitbot 그룹 읽기(640).
- 상태: `/var/lib/upbit-lstm`, upbitbot 소유(700). SQLite WAL까지 일관되게 백업하려면 타이머와 실행 서비스를 정지한 뒤 디렉터리 전체를 보관하거나 SQLite backup API를 사용하세요.
- GitHub와 배포 ZIP에는 키·상태·학습 데이터가 들어가지 않습니다.
- 서버를 복제한 뒤 **두 대가 같은 키/계정으로 동시에 거래하지 않도록** 이전 서버를 먼저 정지하세요. 파일 잠금은 단일 서버 범위이며 분산 잠금이 아닙니다.
- 타임스탬프 오류가 나면 NTP/시간 동기화를 확인하세요.
- 서버 로그는 `journalctl`로 보고 별도 로그 공개 웹서버는 두지 않습니다.
