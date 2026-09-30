# 제작 시 검증 기록

검증일: 2026-09-30.

- Windows 로컬 환경, Python 3.14, PyTorch 2.12 CPU, NumPy 2.4.0, pandas 2.3.3, ONNX Runtime 1.24.4에서 검증.
- 단위·통합 테스트 28개 통과: 시간순 분리, 라벨 경계 제거, 전처리의 미래 참조 방지, 데이터 다운로드 재개, 짧은 추론창과 과거 계산의 일치, 수수료/다음 시가 백테스트, 모의 주문, 실주문 모형의 timeout 후 재개, 부분 체결, 주문 중복 방지, 위험 한도, 모델 무결성.
- 합성 데이터 2종목의 작은 CPU 학습 1 epoch → 검증/테스트 → ONNX export → batch=1/5 수치 비교 → ZIP 패키징 완료. **수익성 평가 결과로 사용하지 않습니다.**
- Ruff 정적 검사 통과. PowerShell 학습 실행기와 Bash 배포 스크립트의 문법 검사 통과.
- PyTorch legacy ONNX exporter가 deprecated/tracer 경고를 출력하지만, 고정 길이 입력에 대해 실제 batch 1/5의 결과 일치를 확인합니다. 배포에 쓰는 PyTorch 2.7.1 버전 호환성을 위해 legacy export를 명시합니다.
- RTX 5060 8GB 데스크톱의 학습, 전체 실제 역사 수집/학습, Oracle VM 설치, 계정 키를 사용한 실제 주문은 제작 과정에서 실행하지 않았습니다.
- 제작 PC에서 Upbit 공개 API 연결을 시도했으나 TLS 연결 시간이 초과되었습니다. 실제 API 연결은 사용자 학습 PC/서버에서 확인해야 합니다. 네트워크 검증을 통과한 것으로 표기하지 않습니다.
- 저장소 CI는 Ubuntu 24.04/Python 3.12/PyTorch 2.7.1 CPU를 별도로 검증하도록 구성했습니다. CI 상태는 GitHub Actions에서 확인할 수 있습니다.
