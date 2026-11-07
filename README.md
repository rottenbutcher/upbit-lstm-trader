# Upbit LSTM Trader

Upbit 원화마켓 거래대금 상위 5종목의 5분봉을 수집하고, **RTX 5060 8GB에서 로컬 학습 → Oracle CPU에서 5분마다 판단**하는 현물 자동매매 프로젝트입니다. 기본 모드는 모의매매입니다. 실제 학습 전 모델이나 수익률을 제공하지 않으며, 수익·정확도는 보장할 수 없습니다.

## 구성 요약

- **LSTM 앙상블 (5분마다):** 일봉 수준 문맥을 포함한 27개 피처, 어텐션 풀링, 독립 학습 3개 모델 평균. 모델 간 불일치를 불확실성으로 함께 출력합니다. 상세 내용은 [docs/DESIGN.md](docs/DESIGN.md#lstm-v2)를 보세요.
- **LLM 일일 정책 (하루 1회):** 거시 지표와 Fed·ECB 등 공식 발표를 LLM이 읽고 다음 날 매매 파라미터를 정합니다. 검증된 위험 한도 안에서만 반영되며, 상세 내용은 [docs/LLM_POLICY.md](docs/LLM_POLICY.md)를 보세요.
- **학습 GUI:** 설정 편집, 프리셋, 학습 실행, 진행률, 보고서 열기를 한 창에서 처리합니다.

## 가장 쉬운 방법 — 학습 GUI

Windows는 **`GUI.cmd` 더블클릭**, Linux는 `bash scripts/train.sh gui`를 실행합니다. 처음 한 번 학습 환경을 자동 설치한 뒤 설정 창이 열립니다.

1. 위쪽에서 **프리셋**(빠른 점검 / 균형 / 고성능)을 고르고 `적용`을 누르거나, 탭별로 값을 직접 바꿉니다. 각 항목 옆에 설명이 있습니다.
2. **`설정 저장`**을 누릅니다. 학습과 같은 검증을 먼저 통과해야 저장되며, `config.toml`의 주석은 유지됩니다. 이전 설정은 `config.toml.bak`에 보관됩니다.
3. **`① 데이터 받기 + 학습 + 패키지`**를 누릅니다. 진행률 막대에 앙상블 모델 번호, epoch, 검증 오차가 표시됩니다. `중지` 후에는 `이어서 학습`으로 재개합니다.
4. 끝나면 **`결과 보고서 열기`**를 누릅니다. 서버에 올릴 파일은 `dist/oracle-upload.zip`입니다.

GPU가 없는 PC에서 시험만 하려면 `GPU 없이(CPU)`를 체크하세요. 기존 `TRAIN.cmd` / `RESUME.cmd` / `HMM.cmd`도 그대로 동작합니다.

## 가장 짧은 실행 방법 (명령 방식)

### 1. 학습 PC — RTX 5060 8GB, Windows 10/11

1. [이 저장소](https://github.com/rottenbutcher/upbit-lstm-trader)를 학습할 데스크톱에 다운로드하고 압축을 풉니다. NVIDIA 드라이버를 최신 정식 버전으로 설치하세요.
2. **`TRAIN.cmd` 더블클릭**. Python 3.12와 CUDA 12.8용 PyTorch 2.7.1을 프로젝트 안에 설치하고 데이터 수집, 학습, 검증, ONNX 변환을 실행합니다. 별도 CUDA Toolkit은 필요하지 않습니다. 인터넷과 충분한 디스크 공간이 필요합니다. OneDrive 밖 짧은 경로(예: `C:\trading\upbit-lstm-trader`)를 권장합니다.
3. 완료 시 `artifacts/report.html`이 열리고 **`dist/oracle-upload.zip`**이 생성됩니다. 이 ZIP만 서버에 올리세요. API 키와 학습 데이터는 포함하지 않습니다.

PowerShell로 실행하려면:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\train.ps1
```

앙상블은 모델별로 체크포인트(`checkpoint.pt`, `checkpoint-1.pt`, …)를 저장합니다. 중단 후에는 완료된 모델을 건너뛰고 이어서 학습합니다. 학습 시간은 대략 `ensemble` 값에 비례합니다(기본 3).

학습 중단 시 **`RESUME.cmd`**로 마지막 완료 epoch부터 이어갑니다. 다운로드 중단은 `TRAIN.cmd`를 다시 실행하세요. `RESUME.cmd`는 데이터를 갱신하지 않습니다. 설정이나 데이터가 바뀌면 이전 체크포인트를 재개할 수 없으므로 새 `TRAIN.cmd` 실행이 필요합니다. 재학습 전 기존 `artifacts` 폴더는 별도 보관하세요.

### Linux 학습 PC — RTX 5060 8GB

Ubuntu 22.04/24.04의 x86_64 데스크톱 기준입니다. RTX 5060을 지원하는 NVIDIA 드라이버를 설치하고 재부팅한 뒤 `nvidia-smi`에 GPU가 보이는지 확인하세요. [PyTorch 2.7의 Blackwell / CUDA 12.8 지원](https://pytorch.org/blog/pytorch-2-7/)을 이용하며 프로젝트는 PyTorch 2.7.1 + CUDA 12.8 빌드를 설치합니다. 시스템 CUDA Toolkit을 별도로 설치할 필요는 없습니다.

최신 저장소를 받거나 기존 복제본을 업데이트하고, 저장소 폴더 안에서 실행합니다. 저장소가 비공개이므로 GitHub 인증이 필요합니다. ZIP을 받아 압축을 풀어도 됩니다.

```bash
nvidia-smi
bash scripts/train.sh check   # 최초 환경 설치 + 실제 GPU LSTM 연산 검사
bash scripts/train.sh         # 데이터 수집 → GPU LSTM 학습 → 평가 → Oracle ZIP 생성
bash scripts/train.sh hmm     # CPU HMM 학습 + 저장된 LSTM과 정확도 비교
```

`check`는 선택 사항이며 기본 학습 명령도 다운로드 전에 같은 GPU 검사를 수행합니다. Python 3.12와 라이브러리는 프로젝트 안에 자동 설치됩니다. 일반 사용자로 실행하고 학습 스크립트에 `sudo`를 붙이지 마세요. `curl` 등이 없으면 `sudo apt-get install curl ca-certificates tar coreutils`로 설치합니다.

학습 중단 후 같은 설정·데이터로 이어가려면 `bash scripts/train.sh resume`을 실행하세요. 다운로드 중단은 기본 명령으로 재실행합니다. 이력/정확도는 `artifacts/report.html`, HMM 비교는 `artifacts/hmm/comparison.html`, Oracle 업로드 파일은 `dist/oracle-upload.zip`입니다. Linux에서는 HTML 파일을 브라우저로 직접 여세요. Windows에서 옮길 때는 `.venv`와 `.tools` 대신 **소스 + `data` + `artifacts`**를 복사합니다.

LSTM은 기존 8GB 설정(배치 256, 혼합 정밀도, 메모리 부족 시 배치 축소)을 사용합니다. HMM은 `hmmlearn`의 CPU 구현이며 GPU로 자동 전환되지 않습니다. GPU 검사 통과는 연산 호환성 확인이고 전체 학습 시간·실측 정확도 보장은 아닙니다.

### HMM 정확도 비교 — `HMM.cmd` 더블클릭

**LSTM 학습이 끝난 같은 폴더에서 `HMM.cmd`를 실행**하면, 종목별 Gaussian Hidden Markov Model(HMM)을 CPU로 학습하고 **`artifacts/hmm/comparison.html`**을 자동으로 엽니다. 추가 GPU 메모리는 사용하지 않습니다. 저장소를 새로 다운로드했다면 기존 `data`와 `artifacts` 폴더를 함께 가져오세요. 소스 업데이트 때 이 폴더를 삭제할 필요는 없습니다.

- LSTM과 **같은 종목·검증/테스트 날짜·예측 대상 봉**에서 방향 정확도, 균형 정확도, 평균 절대 오차(MAE)를 비교합니다. 항상 상승·학습 구간 다수 방향·수익률 0 예측 기준도 표시합니다.
- 같은 자금·수수료·슬리피지·위험 제한으로 수익률, 최대 낙폭, 거래 수를 비교합니다. 모델마다 매수 기준은 검증 구간에서만 선택합니다.
- LSTM 파일이 있으면 데이터를 갱신하거나 LSTM을 재학습하지 않고 저장된 날짜로 비교합니다. 앞으로 생성하는 LSTM에는 원본 봉 해시를 기록해 데이터 변경도 탐지합니다. 이전 모델은 해시가 없어 보고서에 그 한계를 표시합니다.
- LSTM 파일이 없으면 데이터를 수집해 HMM만 먼저 평가합니다. 이후 `TRAIN.cmd` 완료 후 `HMM.cmd`를 다시 실행하면 비교 보고서가 생성됩니다.
- 이 기능은 **HMM 연구·비교**용입니다. Oracle 배포 패키지는 기존 LSTM을 사용합니다. HMM 보고서가 실거래 승인이나 자동 모델 교체를 의미하지 않습니다.

이미 설치된 Python 환경에서 직접 실행하려면(Python 3.12 권장):

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[hmm]"
.\.venv\Scripts\python.exe -m upbit_bot.cli train-hmm --compare-lstm artifacts
```

HMM 단독 실행은 `--compare-lstm artifacts`를 생략합니다. 클릭 방식은 필요한 환경을 자동 설치합니다. 상세 결과는 `comparison.json`, 종목/봉별 예측은 `*_predictions.csv`, 거래 내역은 `*_trades.csv`에 저장합니다.

기본값은 **종목별 3개 잠재 상태, 대각 공분산, 초기화 2회, 최대 EM 40회**입니다. 초기값은 학습 구간 우도로 선택하고 테스트는 선택에 사용하지 않습니다. 속도를 위해 전체 학습 기간에서 연속된 5분봉 블록을 고르며, 종목별 최대 60,000봉으로 HMM 분포·전이 확률을 적합합니다. 정규화와 상태별 미래 수익률 대응은 학습 구간만 사용하고, 평가는 미래를 참조하지 않는 forward filtering으로 수행합니다. 상태는 자동으로 특정 의미의 상승/하락/횡보 정답이 되는 것이 아니므로 **상태 분류 정확도가 아닌 미래 수익 방향의 정확도**를 측정합니다.

`config.toml`의 `[hmm]`에서 조정할 수 있습니다. `max_fit_rows=0`이면 모든 유효 학습 봉 블록으로 적합하지만 더 오래 걸립니다. 더 큰 모델이 더 정확하다는 보장은 없으며 실제 비교 수치는 로컬 학습 후 확인할 수 있습니다. [HMM 설계와 검증](docs/HMM.md).

### 2. Oracle — Ubuntu 24.04 LTS

**홈 리전의 Always Free 표시가 있는 Ampere A1 1 OCPU / 6GB RAM** 인스턴스를 준비합니다. ARM64와 AMD64 모두 지원합니다. GPU는 필요 없습니다. 공인 IP, 인터넷 연결, 본인 IP에 대한 SSH 접속을 설정하세요. 별도 웹서비스 포트는 열지 않습니다.

학습 PC에서 **`DEPLOY.cmd` 더블클릭** 후 서버 IP와 Oracle SSH 키 파일 경로를 입력하면 ZIP 업로드와 설치를 이어서 실행합니다. 첫 SSH 접속에서는 서버 지문을 직접 확인합니다. Ubuntu 기본 사용자 `ubuntu`를 기준으로 합니다.

직접 업로드하는 경우 ZIP을 서버 홈 폴더에 올리고 SSH에서:

```bash
sudo apt-get update && sudo apt-get install -y unzip
mkdir -p ~/upbit-release
unzip oracle-upload.zip -d ~/upbit-release
cd ~/upbit-release
sudo bash deploy/install.sh
```

이후 **5분 경계 + 15초**마다 모의매매가 실행됩니다. 재부팅 후 타이머가 자동으로 복구되며 과거에 놓친 거래를 소급 실행하지 않습니다.

```bash
sudo journalctl -u upbit-lstm.service -f
sudo bash /opt/upbit-lstm/current/deploy/botctl.sh status
sudo systemctl list-timers upbit-lstm.timer
```

### 3. 실거래 — 최초 1회 직접 설정

업로드만으로 개인 계정 주문 권한이 생기지는 않습니다. 아래 설정은 계정 소유자가 직접 해야 합니다.

1. [Upbit Open API](https://upbit.com/mypage/open_api_management)에서 **자산조회·주문조회·주문하기** 권한의 키를 발급합니다. 출금 권한은 필요하지 않습니다. 서버의 공인 IPv4를 허용 목록에 등록합니다. 키는 채팅이나 GitHub에 올리지 마세요.
2. 기본 운용자금은 **100,000원**, 종목당 10%, 전체 투자 50%, 주문당 최대 20,000원입니다. 다른 금액은 학습 전에 `config.toml`에서 설정해 같은 조건으로 검증하세요. 실거래 계정에 해당 현금 또는 명시적으로 등록할 코인이 있어야 합니다.
3. `report.html`의 성과 기준을 통과한 경우 아래 명령을 실행하고, `LIVE` 입력 후 키를 입력합니다. 비밀번호처럼 화면에 표시되지 않습니다. API 키의 주문하기 권한은 첫 실제 주문 시에만 완전히 확인됩니다.

```bash
sudo bash /opt/upbit-lstm/current/deploy/configure-live.sh
```

**기존 코인은 자동으로 매도하지 않습니다.** 보유 코인을 운용에 포함하려면 상위 5개 모델 종목 중 정확한 수량을 지정해 등록할 수 있습니다. 등록 가액은 운용금액 중 일부를 대체하고, 종목별 한도를 초과할 수 없습니다. 나머지 운용금액에 해당하는 KRW가 있어야 합니다. 아래는 문법 예시이며 0.0001 BTC가 적절한 수량이라는 뜻은 아닙니다.

```bash
sudo bash /opt/upbit-lstm/current/deploy/botctl.sh adopt --market KRW-BTC --quantity 0.0001 --confirm
```

등록 후에는 해당 수량의 매도가 자동화됩니다. 봇 운용 중 수동 거래·다른 봇의 동일 자산 거래는 피하세요. 등록하지 않은 코인은 봇이 새로 매수한 수량과 별도로 유지됩니다.

### 4. LLM 일일 파라미터 정책 — 선택

Anthropic(기본) 또는 OpenAI 호환 API 키를 서버에 등록하면, 매일 09:20 KST에 거시 지표와 공식 발표를 분석해 다음 날 매매 파라미터를 조정합니다. 조정 범위는 검증된 값보다 위험을 줄이는 방향으로 제한됩니다. 설정하지 않으면 기존처럼 검증된 기준값으로 동작합니다.

```bash
sudo bash /opt/upbit-lstm/current/deploy/configure-llm.sh
sudo bash /opt/upbit-lstm/current/deploy/botctl.sh policy-show
```

범위, 수집 출처, 수동 조정, 한계는 [docs/LLM_POLICY.md](docs/LLM_POLICY.md)를 보세요.

## 데이터·학습 설계

| 항목 | 기본값 / 의미 |
|---|---|
| 종목 선정 | 최초 실행 당시 정상 원화마켓의 24시간 **원화 거래대금** 상위 5개. USDT 등도 순위에 들면 포함 |
| 종목 고정 | `data/universe.json`에 선정 시점·순위를 저장. 재개 때 순위를 바꾸지 않음 |
| 기간 | 2016-09-30부터 요청하되 Upbit/각 종목의 실제 제공 시작까지만 수집. 10년 데이터가 없는 종목에 역사를 생성하지 않음 |
| 데이터 부족 | 종목당 365일 미만이면 학습 중단. 그 종목을 몰래 교체하지 않음 |
| 입력 | 최근 96개 5분봉(8시간), 수익률·변동성·캔들·거래량·시간 특징 16개 + 종목 one-hot |
| 예측 | 다음 봉 시가 진입을 가정한 이후 6봉(30분) 로그수익률. 판단은 매 5분마다 |
| 모델 | 공통 2층 LSTM, hidden 64, dropout 0.15, Huber loss |
| GPU 효율 | 배치 256, CUDA mixed precision, 메모리 부족 시 절반으로 재시도, 메모리 매핑 배열, 윈도우는 배치 단위 생성 |
| 빠른 학습 | 최대 12 epochs, early stopping 3, 학습 stride 3, epoch당 최대 120,000개를 무작위 추출 |
| 전체 학습 옵션 | `samples_per_epoch=0`, `stride=1`이면 각 epoch에 모든 유효 표본 사용. 상당히 느려질 수 있음 |
| 검증 | 공통 시간축에서 마지막 90일 테스트, 그 앞 90일 검증, 나머지 학습. 라벨 미래구간은 경계에서 제거 |
| 전처리 | 스케일러는 학습 기간만으로 적합. 미래 값으로 결측을 채우지 않음 |
| 선택 | 검증 MAE로 체크포인트, 검증 거래성과로 매수 기준 선택. 테스트로 기준을 재조정하지 않음 |
| 거래비용 | 편도 수수료 0.05%, 편도 슬리피지 5bps 가정. 실거래 수수료가 더 높으면 신규 매수 차단 |
| 추론 | ONNX Runtime CPU, 모델 해시·특징 순서·변환 전후 수치 일치 검증 |

빠른 기본 설정은 전체 역사에서 표본을 고르지만, **모든 5분봉을 매 epoch 학습하는 설정은 아닙니다.** 학습 시간은 종목별 역사와 GPU/CPU/디스크에 따라 달라집니다. 각 epoch 실제 소요시간을 출력합니다. 5개 종목 각각 10년이면 약 526만 봉, API 200개씩 약 26,300회 요청이며 네트워크 대기만으로도 상당한 시간이 듭니다. SQLite에 페이지별로 저장하므로 중단 후 이어받을 수 있습니다.

## 성과 기준과 운용 제한

- 검증/테스트 각각 최소 30건 완료 거래, 순수익 양수, 일별 Sharpe 양수, profit factor ≥ 1.05, 최대 낙폭 ≤ 10%.
- 시간순 검증 3개 소구간 중 2개 이상 수익 양수. **같은 모델을 구간별로 평가하며 walk-forward 재학습 검증은 아닙니다.**
- 기준 미달이면 모델은 생성되지만 실거래 신규 매수를 허용하지 않습니다. 예측력이나 수익이 없는 모델을 억지로 거래시키지 않습니다.
- 새 모델의 최신 데이터가 30일보다 오래되면 실거래 신규 매수를 막습니다. 지속 운용하려면 로컬 재학습·검토·업로드가 필요합니다.
- 종목별 2% 손절·4% 익절·최대 12봉 보유, 일일 손실 3%, 최고 자산 대비 손실 10% 제한. **5분마다 확인하는 프로그램 조건이며 거래소 상주 스톱 주문이 아닙니다.** 급변·서버 장애·스프레드 제한 때문에 실제 손실은 한도를 넘을 수 있습니다.
- 손실 제한 발동 시 신규 매수를 막고 체결 가능한 보유분에 매도 시도. 일일 제한은 다음 UTC 날짜에 초기화, 누적 낙폭 제한은 수동 복구 필요.
- 오래된 봉/호가, 과도한 스프레드, 최소 주문 미달, 잔고 불일치, 모델 불일치 때 주문 중단. 잔액이 5천원 미만인 코인 잔량은 남을 수 있습니다.
- 동일 봉 중복 실행 방지, 프로세스 잠금, 주문 전 intent 저장, 고유 identifier. 주문 생성 응답이 불명확하면 재전송하지 않고 조회로 대조합니다.
- Oracle 무료 VM은 유휴 자원 회수·가용 용량 부족의 영향을 받습니다. 무중단 보장은 없고, 의도적 부하로 회수 정책을 우회하지 않습니다.

## 제한과 검증 상태

현재 상위 5종목의 과거를 사용하므로 생존 편향과 종목 선정 편향이 있습니다. 이는 당시마다 매매 가능했던 상위 5종목을 복원한 연구가 아닙니다. 과거 호가장·시장충격·실제 체결 지연도 완전히 재현하지 못합니다. 다음 봉 시가로 체결하는 백테스트는 실행 가능성의 근사입니다. 기준 통과 자체는 수익 보장이 아닙니다. 같은 테스트 구간을 보며 설정을 반복 변경하면 더 이상 미사용 테스트가 아닙니다.

제작 과정에서는 합성 데이터의 작은 CPU 학습, ONNX 동등성, 패키징, 주문 복구 등을 테스트합니다. 전체 역사 학습과 RTX 5060 실측은 사용자 데스크톱에서 수행하며, Oracle 실제 설치와 키를 이용한 주문은 아직 실행하지 않은 단계입니다. **현재 저장소에 미리 학습된 운영 모델은 없습니다.**

상세 문서: [운영·장애 복구](docs/OPERATIONS.md), [설계·공식 출처](docs/DESIGN.md).

## 개발자 검증

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-runtime.txt onnx==1.21.0 -e '.[dev,hmm]'
pytest -q
ruff check .
```

모델·데이터·키·계좌 상태는 Git에 포함하지 않습니다. 서버의 `/var/lib/upbit-lstm`는 주문 복구에 필수이므로 삭제하지 말고 별도 보안 백업하세요.
