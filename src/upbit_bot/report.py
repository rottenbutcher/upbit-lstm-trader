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


def write_comparison(report, destination):
    def fmt(value, percent=False):
        if value is None:
            return "—"
        return f"{value * 100:.2f}%" if percent else f"{value:,.3f}"

    sections = []
    for split, title in (("val", "검증 구간"), ("test", "테스트 구간")):
        rows, coins = [], []
        for model, result in report["results"].items():
            f = result["forecast"][split]
            trade = result["validation" if split == "val" else "test"]
            rows.append(
                f"<tr><th>{html.escape(model)}</th><td>{fmt(f['direction_accuracy'], True)}</td>"
                f"<td>{fmt(f['balanced_accuracy'], True)}</td><td>{fmt(f['mae_bps'])}</td>"
                f"<td>{trade['return_pct']:.2f}%</td><td>{trade['max_drawdown_pct']:.2f}%</td>"
                f"<td>{trade['trades']}</td><td>{result['threshold_bps']:.1f}</td></tr>"
            )
            for market, m in f["per_market"].items():
                coins.append(
                    f"<tr><td>{html.escape(market)}</td><td>{html.escape(model)}</td>"
                    f"<td>{m['samples']:,}</td><td>{fmt(m['direction_accuracy'], True)}</td>"
                    f"<td>{fmt(m['balanced_accuracy'], True)}</td><td>{fmt(m['mae_bps'])}</td>"
                    f"<td>{fmt(m['train_majority_accuracy'], True)}</td></tr>"
                )
        baseline = next(iter(report["results"].values()))["forecast"][split]
        sections.append(f"""<section><h2>{title}</h2><div class="scroll"><table>
        <tr><th>모델</th><th>방향 정확도 ↑</th><th>균형 정확도 ↑</th><th>MAE bps ↓</th>
        <th>비용 차감 수익률</th><th>최대 낙폭</th><th>완료 거래</th><th>매수 기준 bps</th></tr>
        {"".join(rows)}</table></div>
        <p>공통 표본: {baseline["samples"]:,}개 · 항상 상승 예측: {fmt(baseline["always_up_accuracy"], True)} ·
        학습 구간 다수 방향 예측: {fmt(baseline["train_majority_accuracy"], True)} ·
        수익률 0 예측 MAE: {fmt(baseline["zero_baseline_mae_bps"])} bps ·
        실제 수익률 0인 표본: {fmt(baseline["flat_target_fraction"], True)}</p>
        <details><summary>종목별 결과</summary><div class="scroll"><table>
        <tr><th>종목</th><th>모델</th><th>표본 수</th><th>방향 정확도</th><th>균형 정확도</th>
        <th>MAE bps</th><th>다수 방향 기준</th></tr>{"".join(coins)}</table></div></details></section>""")
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in report["limitations"])
    html_text = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width"><title>HMM · LSTM 비교</title>
    <style>body{{max-width:1200px;margin:40px auto;padding:0 24px;background:#0c1626;color:#e9eef7;
    font:16px/1.6 system-ui}}section{{background:#18253a;padding:24px;border-radius:16px;margin:24px 0}}
    .scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}
    td,th{{padding:12px;text-align:left;border-bottom:1px solid #344157}}small{{color:#b5c2d6}}
    h1{{color:#8acaff}}summary{{cursor:pointer}}code{{color:#8acaff}}</style>
    <h1>HMM · LSTM 예측력 비교</h1><p>5분봉 → {report["horizon_bars"] * 5}분 로그수익률 예측</p>
    <small>검증 시작: {html.escape(report["train_end"])}<br>
    테스트 시작: {html.escape(report["validation_end"])}<br>
    데이터 끝: {html.escape(report["data_end"])} · UTC 기준<br>
    생성: {html.escape(report["created_at"])} · 총 소요: {report["elapsed_seconds"] / 60:.1f}분</small>
    {"".join(sections)}<section><h2>결과 읽는 법</h2>
    <p>방향 정확도는 상승/비상승 판단의 적중률입니다. 균형 정확도는 두 방향의 재현율을 동일하게 평균하여
    상승장이 대부분인 구간의 착시를 줄입니다. 한 방향만 있는 구간은 —로 표시합니다.
    MAE는 예측 수익률과 실제 수익률의 평균 절대 오차입니다. 100 bps = 1%입니다.</p>
    <p>정확도와 거래 수익률은 별개의 지표입니다. 현금 유지 수익률은 0%이며,
    매매 0건인 결과는 수익성을 입증하지 않습니다. 모델별 매수 기준은 같은 후보군에서 검증 구간만으로 선택합니다.</p>
    <p>상세 수치: <code>comparison.json</code> · 봉별 예측: <code>*_predictions.csv</code> ·
    거래/자산 추이: <code>*_trades.csv</code>, <code>*_equity.csv</code>.</p>
    <ul>{notes}</ul></section></html>"""
    Path(destination).write_text(html_text, encoding="utf-8")
