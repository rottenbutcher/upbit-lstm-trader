import html
from pathlib import Path


def write_report(report, destination):
    approved = report["approved_for_live"]
    rows = "".join(
        "<tr><td>"
        + html.escape(key)
        + "</td><td>"
        + f"{value:.4f}"
        + "</td><td>"
        + f"{report['test'][key]:.4f}"
        + "</td></tr>"
        for key, value in report["validation"].items()
    )
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in report["limitations"])
    document = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>Upbit LSTM 학습 결과</title><style>
    body{{max-width:900px;margin:48px auto;padding:0 24px;background:#0c1626;color:#e9eef7;font:16px/1.6 system-ui}}
    h1{{font-size:32px}} .card{{background:#18253a;padding:24px;border-radius:16px;margin:24px 0}}
    .status{{color:{"#7ce1a2" if approved else "#ffcf80"};font-weight:bold;font-size:24px}}
    table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #344157}}
    small{{color:#b5c2d6}}code{{color:#8acaff}}</style>
    <h1>Upbit LSTM · 학습 결과</h1><small>5분봉 입력 · 30분 수익률 예측 · 거래비용 반영</small>
    <div class="card"><div class="status">{"실거래 성과 기준 통과" if approved else "모의매매 유지 — 실거래 기준 미달"}</div>
    <p>과거 검증을 통과해도 미래 수익을 보장하지 않습니다. 실거래 전 실제 호가·체결 차이를 모의 운용으로 확인하세요.</p>
    <p>선택된 매수 기준: {report["threshold_bps"]:.1f} bps. 100 bps = 1%.</p></div>
    <div class="card"><table><tr><th>지표</th><th>검증 구간</th><th>미사용 테스트 구간</th></tr>{rows}</table></div>
    <p>테스트 구간 동일 위험노출 단순 보유 수익률: {report["buy_hold_test"]["return_pct"]:.2f}%. 현금 유지: 0%.</p>
    <p>업로드 파일: <code>dist/oracle-upload.zip</code>. 상세 기록: <code>report.json</code>,
    <code>test_equity.csv</code>, <code>test_trades.csv</code>.</p><div class="card"><h2>해석의 한계</h2><ul>{notes}</ul></div>
    </html>"""
    Path(destination).write_text(document, encoding="utf-8")
