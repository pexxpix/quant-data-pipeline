"""Vẽ biểu đồ giá, RSI, MFI, CMF của từng mã từ bảng indicators.

Kết quả là một trang web duy nhất: site/index.html
- Bảng tổng quan phiên gần nhất (bấm vào một mã để xem biểu đồ của mã đó)
- Ô chọn mã và biểu đồ 4 tầng: giá đóng cửa, RSI, MFI, CMF

Chạy sau indicators.py. Trên GitHub Actions, thư mục site/ được đăng lên GitHub Pages.
"""
import html
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots
from sqlalchemy import text

# Dùng chung ngưỡng và cách kết nối với indicators.py để hai file luôn khớp nhau
from indicators import (
    CMF_THRESHOLD,
    MFI_OVERBOUGHT,
    MFI_OVERSOLD,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    make_engine,
)

OUTPUT_DIR = "site"
VN_TZ = "Asia/Ho_Chi_Minh"
# Vì chỉ lưu 90 phiên, RSI ở khoảng 45 phiên đầu cửa sổ chưa ổn định -> vẽ nét đứt mờ
RSI_RELIABLE_AFTER = 45

QUERY = """
SELECT ticker, trading_date, close, rsi_14, mfi_14, cmf_20, rsi_zone, mfi_zone, cmf_signal
FROM indicators
ORDER BY ticker, trading_date
"""

COLORS = {"price": "#2563eb", "rsi": "#7c3aed", "mfi": "#0891b2", "up": "#16a34a", "down": "#dc2626"}


def build_figure(df: pd.DataFrame) -> go.Figure:
    x = df["trading_date"]
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.37, 0.21, 0.21, 0.21],
        subplot_titles=(
            "Giá đóng cửa (nghìn đồng)",
            f"RSI 14 phiên (quá mua ≥ {RSI_OVERBOUGHT}, quá bán ≤ {RSI_OVERSOLD})",
            f"MFI 14 phiên (quá mua ≥ {MFI_OVERBOUGHT}, quá bán ≤ {MFI_OVERSOLD})",
            "CMF 20 phiên (dương = tiền vào, âm = tiền ra)",
        ),
    )

    # 1) Giá
    fig.add_trace(go.Scatter(x=x, y=df["close"], name="Giá", mode="lines",
                             line=dict(color=COLORS["price"], width=2)), row=1, col=1)

    # 2) RSI: phần đầu cửa sổ (chưa ổn định) vẽ nét đứt mờ, phần sau vẽ nét liền
    split = min(RSI_RELIABLE_AFTER, len(df) - 1)
    early, late = df.iloc[: split + 1], df.iloc[split:]
    fig.add_trace(go.Scatter(x=early["trading_date"], y=early["rsi_14"], name="RSI (chưa ổn định)",
                             mode="lines", opacity=0.45,
                             line=dict(color=COLORS["rsi"], width=1.5, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=late["trading_date"], y=late["rsi_14"], name="RSI", mode="lines",
                             line=dict(color=COLORS["rsi"], width=2)), row=2, col=1)
    add_zone_lines(fig, row=2, low=RSI_OVERSOLD, high=RSI_OVERBOUGHT)

    # 3) MFI
    fig.add_trace(go.Scatter(x=x, y=df["mfi_14"], name="MFI", mode="lines",
                             line=dict(color=COLORS["mfi"], width=2)), row=3, col=1)
    add_zone_lines(fig, row=3, low=MFI_OVERSOLD, high=MFI_OVERBOUGHT)

    # 4) CMF: cột xanh khi dương, đỏ khi âm
    bar_colors = [COLORS["up"] if v >= 0 else COLORS["down"] for v in df["cmf_20"].fillna(0)]
    fig.add_trace(go.Bar(x=x, y=df["cmf_20"], name="CMF", marker_color=bar_colors), row=4, col=1)
    fig.add_hline(y=0, line_color="#6b7280", line_width=1, row=4, col=1)
    for level in (CMF_THRESHOLD, -CMF_THRESHOLD):
        fig.add_hline(y=level, line_color="#9ca3af", line_width=1, line_dash="dot", row=4, col=1)

    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    # Ẩn khoảng trống cuối tuần và ngày nghỉ lễ để các phiên nằm liền nhau
    dates = pd.to_datetime(x)
    weekdays = pd.bdate_range(dates.min(), dates.max())
    holidays = [d.strftime("%Y-%m-%d") for d in weekdays.difference(dates)]
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=holidays)])

    fig.update_layout(
        height=860, showlegend=False, template="plotly_white", hovermode="x unified",
        margin=dict(l=55, r=20, t=40, b=30), bargap=0.15,
    )
    for ann in fig.layout.annotations:  # tiêu đề từng tầng: canh trái, chữ nhỏ
        ann.update(x=0, xanchor="left", font=dict(size=13))
    return fig


def add_zone_lines(fig, row, low, high):
    fig.add_hrect(y0=low, y1=high, fillcolor="#e5e7eb", opacity=0.35, line_width=0, row=row, col=1)
    for level in (low, high):
        fig.add_hline(y=level, line_color="#9ca3af", line_width=1, line_dash="dash", row=row, col=1)


def zone_class(label):
    return {"quá mua": "hot", "quá bán": "cold", "dòng tiền vào": "in", "dòng tiền ra": "out"}.get(label, "")


def fmt(value, digits):
    return "–" if pd.isna(value) else f"{value:.{digits}f}"


def summary_rows(latest: pd.DataFrame) -> str:
    rows = []
    for _, r in latest.iterrows():
        t = html.escape(r["ticker"])
        rows.append(
            f'<tr data-ticker="{t}"><td class="tk">{t}</td><td>{r["trading_date"]}</td>'
            f'<td class="num">{fmt(r["close"], 2)}</td>'
            f'<td class="num">{fmt(r["rsi_14"], 1)}</td><td><span class="tag {zone_class(r["rsi_zone"])}">{r["rsi_zone"] or "–"}</span></td>'
            f'<td class="num">{fmt(r["mfi_14"], 1)}</td><td><span class="tag {zone_class(r["mfi_zone"])}">{r["mfi_zone"] or "–"}</span></td>'
            f'<td class="num">{fmt(r["cmf_20"], 3)}</td><td><span class="tag {zone_class(r["cmf_signal"])}">{r["cmf_signal"] or "–"}</span></td></tr>'
        )
    return "\n".join(rows)


PAGE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chỉ báo cổ phiếu</title>
<script src="__PLOTLY_SRC__"></script>
<style>
  body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }
  main { max-width: 1100px; margin: 0 auto; padding: 20px 16px 40px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .sub { color: #64748b; font-size: 13px; margin-bottom: 18px; }
  .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; margin-bottom: 16px; }
  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  th, td { padding: 7px 10px; border-bottom: 1px solid #f1f5f9; text-align: left; white-space: nowrap; }
  th { color: #64748b; font-weight: 600; font-size: 12px; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  tr[data-ticker] { cursor: pointer; }
  tr[data-ticker]:hover, tr.active { background: #eff6ff; }
  .tk { font-weight: 700; }
  .tag { font-size: 12px; padding: 2px 8px; border-radius: 999px; background: #f1f5f9; color: #475569; }
  .tag.hot { background: #fee2e2; color: #b91c1c; } .tag.cold { background: #dbeafe; color: #1d4ed8; }
  .tag.in { background: #dcfce7; color: #15803d; } .tag.out { background: #fee2e2; color: #b91c1c; }
  .bar { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
  select { font-size: 15px; padding: 6px 10px; border-radius: 8px; border: 1px solid #cbd5e1; }
  .note { color: #64748b; font-size: 12px; line-height: 1.5; }
</style>
</head>
<body>
<main>
  <h1>Chỉ báo cổ phiếu</h1>
  <div class="sub">Cập nhật lúc __UPDATED__ · dữ liệu 90 phiên gần nhất</div>

  <div class="card scroll">
    <table>
      <thead><tr><th>Mã</th><th>Phiên</th><th>Giá</th><th>RSI</th><th></th><th>MFI</th><th></th><th>CMF</th><th></th></tr></thead>
      <tbody>__ROWS__</tbody>
    </table>
  </div>

  <div class="card">
    <div class="bar"><label for="ticker"><b>Mã:</b></label><select id="ticker">__OPTIONS__</select></div>
    <div id="chart"></div>
    <p class="note">RSI nét đứt mờ: khoảng 45 phiên đầu của cửa sổ 90 phiên, RSI chưa ổn định nên chưa đáng tin.
      Các chỉ báo là công cụ theo dõi, không phải tín hiệu mua bán.</p>
  </div>
</main>
<script>
const FIGS = __FIGS__;
const select = document.getElementById("ticker");
function show(ticker) {
  if (!FIGS[ticker]) ticker = Object.keys(FIGS)[0];
  const fig = FIGS[ticker];
  Plotly.react("chart", fig.data, fig.layout, {responsive: true, displaylogo: false});
  select.value = ticker;
  document.querySelectorAll("tr[data-ticker]").forEach(r => r.classList.toggle("active", r.dataset.ticker === ticker));
  history.replaceState(null, "", "#" + ticker);
}
select.addEventListener("change", e => show(e.target.value));
document.querySelectorAll("tr[data-ticker]").forEach(r =>
  r.addEventListener("click", () => { show(r.dataset.ticker); document.getElementById("chart").scrollIntoView({behavior: "smooth"}); }));
show(decodeURIComponent(location.hash.slice(1)));
</script>
</body>
</html>
"""


def build_page(data: pd.DataFrame) -> str:
    data = data.copy()
    data["trading_date"] = pd.to_datetime(data["trading_date"]).dt.strftime("%Y-%m-%d")
    figs = {}
    for ticker, df in data.groupby("ticker"):
        figs[ticker] = json.loads(build_figure(df.reset_index(drop=True)).to_json())
    latest = data.groupby("ticker").tail(1).sort_values("ticker")
    options = "".join(f'<option value="{html.escape(t)}">{html.escape(t)}</option>' for t in figs)
    updated = datetime.now(ZoneInfo(VN_TZ)).strftime("%H:%M ngày %d/%m/%Y")
    figs_json = json.dumps(figs, ensure_ascii=False).replace("</", "<\\/")
    return (
        PAGE.replace("__PLOTLY_SRC__", f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js")
        .replace("__UPDATED__", updated)
        .replace("__ROWS__", summary_rows(latest))
        .replace("__OPTIONS__", options)
        .replace("__FIGS__", figs_json)
    )


def run(engine):
    with engine.connect() as conn:
        data = pd.read_sql(text(QUERY), conn)
    if data.empty:
        print("LỖI: bảng indicators chưa có dữ liệu. Hãy chạy indicators.py trước.")
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_page(data))
    print(f"Đã vẽ biểu đồ cho {data['ticker'].nunique()} mã -> {path}")
    return path


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("LỖI: Chưa cấu hình DATABASE_URL.")
        sys.exit(1)
    try:
        run(make_engine(database_url))
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"LỖI khi vẽ biểu đồ: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
