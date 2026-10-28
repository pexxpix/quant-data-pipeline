"""Vẽ biểu đồ giá, RSI, MFI, CMF, ADX của từng mã và độ rộng thị trường từ bảng indicators.

Kết quả là một trang web duy nhất: site/index.html
- Độ rộng thị trường: số mã lập đỉnh / đáy 52 tuần mỗi phiên
- Bảng tổng quan phiên gần nhất (bấm vào một mã để xem biểu đồ của mã đó)
- Ô chọn mã và biểu đồ 5 tầng: giá đóng cửa, RSI, MFI, CMF, ADX (+DI / -DI)
Chỉ báo được tính trên toàn bộ lịch sử đã lưu, nhưng chỉ vẽ DISPLAY_SESSIONS phiên gần nhất.

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
    ADX_NO_TREND,
    ADX_TREND,
    CMF_THRESHOLD,
    HIGH_LOW_LOOKBACK,
    MFI_OVERBOUGHT,
    MFI_OVERSOLD,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    make_engine,
)

OUTPUT_DIR = "site"
VN_TZ = "Asia/Ho_Chi_Minh"
DISPLAY_SESSIONS = 90  # số phiên gần nhất được vẽ trên biểu đồ

# Số phiên lịch sử (tính từ phiên đầu tiên có dữ liệu của mã) trước khi chỉ báo ổn định.
# Thử trên 300 chuỗi giá mô phỏng: RSI và +DI/-DI khớp với giá trị tính từ lịch sử dài
# (sai lệch < 2 điểm, 95% trường hợp) sau ~45 phiên; ADX làm mượt hai lần nên cần ~70 phiên.
# Với lịch sử dài (scraper giữ 340 phiên) cả 90 phiên hiển thị đều đã ổn định; nét đứt mờ
# chỉ còn xuất hiện ở mã mới niêm yết, chưa đủ lịch sử.
RSI_RELIABLE_AFTER = 45
ADX_RELIABLE_AFTER = 70
DI_RELIABLE_AFTER = 45

QUERY = """
SELECT ticker, trading_date, close, rsi_14, mfi_14, cmf_20,
       adx_14, plus_di_14, minus_di_14,
       rsi_zone, mfi_zone, cmf_signal, adx_signal,
       new_high_52w, new_low_52w
FROM indicators
ORDER BY ticker, trading_date
"""

# Bộ lọc "Top biến động + thanh khoản": chỉnh các con số ở đây
SCREEN_SESSIONS = 3       # so giá hiện tại với giá đóng cửa N phiên trước + KL trung bình N phiên gần nhất
SCREEN_BASE_SESSIONS = 5  # khối lượng nền: trung bình M phiên ngay trước N phiên đó (không chồng lấn)
SCREEN_VOL_MULT = 1.5     # KL trung bình N phiên gần nhất >= hệ số này x KL nền
SCREEN_TOP = 20           # số mã hiển thị

# Lấy từ daily_prices (giá gốc), cùng logic với truy vấn chạy tay trên Neon SQL Editor
SCREEN_QUERY = """
WITH p AS (
    SELECT ticker, trading_date, close, volume,
           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trading_date DESC) AS rn
    FROM daily_prices
),
agg AS (
    SELECT ticker,
           MAX(trading_date) FILTER (WHERE rn = 1)       AS last_date,
           MAX(close)        FILTER (WHERE rn = 1)       AS close_now,
           MAX(close)        FILTER (WHERE rn = :n + 1)  AS close_before,
           AVG(volume)       FILTER (WHERE rn <= :n)     AS vol_recent,
           AVG(volume)       FILTER (WHERE rn >  :n)     AS vol_base,
           COUNT(*)                                      AS n_rows
    FROM p
    WHERE rn <= :n + :base
    GROUP BY ticker
)
SELECT ticker,
       last_date,
       close_before,
       close_now,
       100 * (close_now / close_before - 1)  AS pct,
       vol_recent,
       vol_base,
       vol_recent / vol_base                 AS vol_ratio,
       close_now * vol_recent / 1e6          AS value_bn
FROM agg
WHERE n_rows = :n + :base                -- chỉ xét mã có đủ dữ liệu
  AND vol_base > 0
  AND vol_recent >= :mult * vol_base     -- điều kiện thanh khoản
  AND close_before > 0
ORDER BY pct DESC                        -- tăng mạnh nhất lên đầu
LIMIT :top
"""

COLORS = {"price": "#2563eb", "rsi": "#7c3aed", "mfi": "#0891b2", "up": "#16a34a", "down": "#dc2626",
          "adx": "#1e293b"}


def build_figure(df: pd.DataFrame) -> go.Figure:
    x = df["trading_date"]
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.32, 0.17, 0.17, 0.17, 0.17],
        subplot_titles=(
            "Giá đóng cửa (nghìn đồng)",
            f"RSI 14 phiên (quá mua ≥ {RSI_OVERBOUGHT}, quá bán ≤ {RSI_OVERSOLD})",
            f"MFI 14 phiên (quá mua ≥ {MFI_OVERBOUGHT}, quá bán ≤ {MFI_OVERSOLD})",
            "CMF 20 phiên (dương = tiền vào, âm = tiền ra)",
            f"ADX 14 phiên (≥ {ADX_TREND}: có xu hướng) · "
            "<span style='color:#16a34a'>+DI</span>/<span style='color:#dc2626'>−DI</span>",
        ),
    )

    # 1) Giá
    fig.add_trace(go.Scatter(x=x, y=df["close"], name="Giá", mode="lines",
                             line=dict(color=COLORS["price"], width=2)), row=1, col=1)

    # 2) RSI: phần đầu cửa sổ (chưa ổn định) vẽ nét đứt mờ, phần sau vẽ nét liền
    add_warmup_line(fig, df, "rsi_14", "RSI", COLORS["rsi"], 2, RSI_RELIABLE_AFTER, row=2)
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

    # 5) ADX: +DI xanh, -DI đỏ, ADX đậm nằm trên cùng; dải xám 20-25 là vùng chuyển tiếp
    #    (vẽ dải sau khi có đường: Plotly bỏ qua đường ngang trên tầng chưa có dữ liệu)
    add_warmup_line(fig, df, "plus_di_14", "+DI", COLORS["up"], 1.5, DI_RELIABLE_AFTER, row=5)
    add_warmup_line(fig, df, "minus_di_14", "−DI", COLORS["down"], 1.5, DI_RELIABLE_AFTER, row=5)
    add_warmup_line(fig, df, "adx_14", "ADX", COLORS["adx"], 2.5, ADX_RELIABLE_AFTER, row=5)
    add_zone_lines(fig, row=5, low=ADX_NO_TREND, high=ADX_TREND)

    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    dmi_max = df[["adx_14", "plus_di_14", "minus_di_14"]].max().max()
    fig.update_yaxes(range=[0, max(50, (0 if pd.isna(dmi_max) else dmi_max) + 5)], row=5, col=1)

    # Ẩn khoảng trống cuối tuần và ngày nghỉ lễ để các phiên nằm liền nhau
    dates = pd.to_datetime(x)
    weekdays = pd.bdate_range(dates.min(), dates.max())
    holidays = [d.strftime("%Y-%m-%d") for d in weekdays.difference(dates)]
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=holidays)])

    fig.update_layout(
        height=1040, showlegend=False, template="plotly_white", hovermode="x unified",
        margin=dict(l=55, r=20, t=40, b=30), bargap=0.15,
    )
    for ann in fig.layout.annotations:  # tiêu đề từng tầng: canh trái, chữ nhỏ
        ann.update(x=0, xanchor="left", font=dict(size=13))
    return fig


def add_warmup_line(fig, df, col, name, color, width, reliable_after, row):
    """Vẽ một đường chỉ báo: các phiên mà mã chưa có đủ `reliable_after` phiên lịch sử
    (cột `pos`, đếm từ phiên đầu tiên của mã) vẽ nét đứt mờ, phần sau nét liền."""
    early = df[df["pos"] <= reliable_after]
    late = df[df["pos"] >= reliable_after]
    if early[col].notna().any():
        fig.add_trace(go.Scatter(x=early["trading_date"], y=early[col], name=f"{name} (chưa ổn định)",
                                 mode="lines", opacity=0.45,
                                 line=dict(color=color, width=max(width - 0.5, 1), dash="dot")), row=row, col=1)
    fig.add_trace(go.Scatter(x=late["trading_date"], y=late[col], name=name, mode="lines",
                             line=dict(color=color, width=width)), row=row, col=1)


def ticker_list(tickers, limit=12):
    tickers = sorted(tickers)
    if not tickers:
        return "–"
    shown = ", ".join(tickers[:limit])
    return shown + (f" +{len(tickers) - limit} mã" if len(tickers) > limit else "")


def breadth_table(data: pd.DataFrame) -> pd.DataFrame:
    """Mỗi phiên: số mã lập đỉnh / đáy 52 tuần, số mã đủ dữ liệu để xét, và tên các mã."""
    eligible = data[data["new_high_52w"].notna()]
    if eligible.empty:
        return pd.DataFrame()
    is_high = eligible["new_high_52w"].fillna(False).astype(bool)
    is_low = eligible["new_low_52w"].fillna(False).astype(bool)
    g = eligible.groupby("trading_date")
    out = pd.DataFrame({
        "highs": is_high.groupby(eligible["trading_date"]).sum(),
        "lows": is_low.groupby(eligible["trading_date"]).sum(),
        "eligible": g.size(),
        "high_names": eligible[is_high].groupby("trading_date")["ticker"].apply(list),
        "low_names": eligible[is_low].groupby("trading_date")["ticker"].apply(list),
    })
    for col in ("high_names", "low_names"):
        out[col] = out[col].apply(lambda v: v if isinstance(v, list) else [])
    return out.sort_index().tail(DISPLAY_SESSIONS).reset_index()


def build_breadth_figure(b: pd.DataFrame) -> go.Figure:
    x = b["trading_date"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=b["highs"], name="Lập đỉnh 52 tuần", marker_color=COLORS["up"],
        customdata=list(zip(b["eligible"], b["high_names"].apply(ticker_list))),
        hovertemplate="%{y} mã lập đỉnh / %{customdata[0]} mã<br>%{customdata[1]}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=x, y=-b["lows"], name="Lập đáy 52 tuần", marker_color=COLORS["down"],
        customdata=list(zip(b["eligible"], b["low_names"].apply(ticker_list), b["lows"])),
        hovertemplate="%{customdata[2]} mã lập đáy / %{customdata[0]} mã<br>%{customdata[1]}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#6b7280", line_width=1)
    dates = pd.to_datetime(x)
    weekdays = pd.bdate_range(dates.min(), dates.max())
    holidays = [d.strftime("%Y-%m-%d") for d in weekdays.difference(dates)]
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=holidays)])
    peak = max(5, int(b["highs"].max()), int(b["lows"].max()))
    fig.update_yaxes(range=[-peak * 1.15, peak * 1.15], tickformat="d",
                     title=dict(text="đáy  ←  số mã  →  đỉnh", font=dict(size=11)))
    fig.update_layout(
        height=280, barmode="relative", showlegend=False, template="plotly_white",
        hovermode="x unified", margin=dict(l=55, r=20, t=10, b=30), bargap=0.15,
    )
    return fig


def breadth_headline(b: pd.DataFrame) -> str:
    if b.empty:
        return ("Chưa đủ dữ liệu: cần ít nhất "
                f"{HIGH_LOW_LOOKBACK + 1} phiên lịch sử mỗi mã để xét đỉnh/đáy 52 tuần.")
    last = b.iloc[-1]
    return (
        f'Phiên {html.escape(str(last["trading_date"]))}: '
        f'<span class="tag in">{int(last["highs"])} mã lập đỉnh</span> '
        f'<span class="tag out">{int(last["lows"])} mã lập đáy</span> '
        f'<span class="muted">trên {int(last["eligible"])} mã đủ 1 năm dữ liệu</span>'
    )


def add_zone_lines(fig, row, low, high):
    fig.add_hrect(y0=low, y1=high, fillcolor="#e5e7eb", opacity=0.35, line_width=0, row=row, col=1)
    for level in (low, high):
        fig.add_hline(y=level, line_color="#9ca3af", line_width=1, line_dash="dash", row=row, col=1)


def zone_class(label):
    return {"quá mua": "hot", "quá bán": "cold", "dòng tiền vào": "in", "dòng tiền ra": "out",
            "xu hướng tăng": "in", "xu hướng giảm": "out"}.get(label, "")


def fmt(value, digits):
    return "–" if pd.isna(value) else f"{value:.{digits}f}"


def adx_trend_arrow(now, before):
    """Mũi tên cho biết ADX đang mạnh lên hay yếu đi so với 3 phiên trước."""
    if pd.isna(now) or pd.isna(before) or abs(now - before) < 0.5:
        return ""
    if now > before:
        return ' <span class="arrow" title="ADX tăng so với 3 phiên trước: xu hướng mạnh lên">↑</span>'
    return ' <span class="arrow" title="ADX giảm so với 3 phiên trước: xu hướng yếu đi">↓</span>'


def is_true(value) -> bool:
    """True chỉ khi giá trị đúng là True (cột BOOLEAN có thể là None / NA)."""
    return not pd.isna(value) and bool(value)


def high_low_tag(r) -> str:
    if is_true(r["new_high_52w"]):
        return '<span class="tag in">đỉnh 52T</span>'
    if is_true(r["new_low_52w"]):
        return '<span class="tag out">đáy 52T</span>'
    return ""


def summary_rows(latest: pd.DataFrame) -> str:
    rows = []
    for _, r in latest.iterrows():
        t = html.escape(r["ticker"])
        rows.append(
            f'<tr data-ticker="{t}"><td class="tk">{t}</td><td>{r["trading_date"]}</td>'
            f'<td class="num">{fmt(r["close"], 2)}</td><td>{high_low_tag(r)}</td>'
            f'<td class="num">{fmt(r["rsi_14"], 1)}</td><td><span class="tag {zone_class(r["rsi_zone"])}">{r["rsi_zone"] or "–"}</span></td>'
            f'<td class="num">{fmt(r["mfi_14"], 1)}</td><td><span class="tag {zone_class(r["mfi_zone"])}">{r["mfi_zone"] or "–"}</span></td>'
            f'<td class="num">{fmt(r["cmf_20"], 3)}</td><td><span class="tag {zone_class(r["cmf_signal"])}">{r["cmf_signal"] or "–"}</span></td>'
            f'<td class="num">{fmt(r["adx_14"], 1)}{adx_trend_arrow(r["adx_14"], r["adx_3_ago"])}</td>'
            f'<td><span class="tag {zone_class(r["adx_signal"])}">{r["adx_signal"] or "–"}</span></td></tr>'
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
  /* ~100 mã: bảng cuộn trong khung riêng để biểu đồ không bị đẩy xuống quá xa */
  .scroll { overflow: auto; max-height: 440px; position: relative; padding-top: 0; }
  thead th { position: sticky; top: 0; z-index: 1; background: #fff; box-shadow: inset 0 -1px 0 #e2e8f0; }
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
  .arrow { color: #64748b; font-weight: 700; }
  h2 { font-size: 15px; margin: 0 0 6px; }
  .muted { color: #64748b; font-size: 12px; }
  .hscroll { overflow-x: auto; }
  th.r { text-align: right; }
  .pos { color: #15803d; font-weight: 600; } .neg { color: #b91c1c; font-weight: 600; }
  .empty { color: #64748b; font-size: 13px; padding: 10px; white-space: normal; }
  .headline { font-size: 14px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 4px; }
</style>
</head>
<body>
<main>
  <h1>Chỉ báo cổ phiếu</h1>
  <div class="sub">Cập nhật lúc __UPDATED__ · biểu đồ __DISPLAY__ phiên gần nhất, chỉ báo tính trên toàn bộ lịch sử đã lưu</div>

  <div class="card">
    <h2>Độ rộng thị trường: số mã lập đỉnh / đáy 52 tuần</h2>
    <div class="headline">__BREADTH_HEADLINE__</div>
    <div id="breadth"></div>
    <p class="note">Cột xanh: số mã có giá cao nhất phiên vượt đỉnh của 250 phiên trước đó; cột đỏ: số mã thủng đáy 250 phiên.
      Rê chuột vào cột để xem tên mã. VN-Index tăng mà số mã lập đỉnh giảm dần là dấu hiệu đà tăng đang hẹp lại.</p>
  </div>

  <div class="card" id="screen">
    <h2>Top biến động giá __SCREEN_N__ phiên kèm thanh khoản tăng</h2>
    <div class="muted" style="margin-bottom:6px">% giá: giá hiện tại so với giá đóng cửa __SCREEN_N__ phiên trước ·
      khối lượng trung bình __SCREEN_N__ phiên gần nhất ≥ __SCREEN_MULT__ lần trung bình __SCREEN_BASE__ phiên trước đó ·
      xếp theo % thay đổi giá, tăng mạnh nhất lên đầu · bấm vào mã để xem biểu đồ</div>
    <div class="hscroll">
    <table>
      <thead><tr><th>Mã</th><th>Phiên</th><th class="r">Giá __SCREEN_N__ phiên trước</th><th class="r">Giá hiện tại</th>
        <th class="r">% __SCREEN_N__ phiên</th><th class="r">KL TB __SCREEN_N__ phiên</th><th class="r">KL TB __SCREEN_BASE__ phiên trước</th>
        <th class="r">Tỷ lệ KL</th><th class="r">GTGD TB (tỷ đồng)</th></tr></thead>
      <tbody>__SCREEN_ROWS__</tbody>
    </table>
    </div>
  </div>

  <div class="card scroll">
    <table>
      <thead><tr><th>Mã</th><th>Phiên</th><th>Giá</th><th>52T</th><th>RSI</th><th></th><th>MFI</th><th></th><th>CMF</th><th></th><th>ADX</th><th></th></tr></thead>
      <tbody>__ROWS__</tbody>
    </table>
  </div>

  <div class="card">
    <div class="bar"><label for="ticker"><b>Mã:</b></label><select id="ticker">__OPTIONS__</select></div>
    <div id="chart"></div>
    <p class="note">Nét đứt mờ: mã mới niêm yết chưa đủ lịch sử nên RSI/ADX ở đoạn đó chưa ổn định, chưa đáng tin.
      ADX chỉ đo độ mạnh của xu hướng (≥ 25: có xu hướng, &lt; 20: đi ngang, dải xám ở giữa là vùng chuyển tiếp);
      hướng xem +DI (xanh) so với −DI (đỏ). Mũi tên ↑/↓: ADX so với 3 phiên trước.
      Các chỉ báo là công cụ theo dõi, không phải tín hiệu mua bán.</p>
  </div>
</main>
<script>
const FIGS = __FIGS__;
const BREADTH = __BREADTH__;
if (BREADTH) Plotly.newPlot("breadth", BREADTH.data, BREADTH.layout, {responsive: true, displaylogo: false});
const select = document.getElementById("ticker");
function show(ticker) {
  if (!FIGS[ticker]) ticker = Object.keys(FIGS)[0];
  const fig = FIGS[ticker];
  Plotly.react("chart", fig.data, fig.layout, {responsive: true, displaylogo: false});
  select.value = ticker;
  document.querySelectorAll("tr[data-ticker]").forEach(r => r.classList.toggle("active", r.dataset.ticker === ticker));
  // cuộn khung bảng tổng quan (không cuộn cả trang) để dòng đang chọn luôn nhìn thấy
  const row = document.querySelector('.scroll tr[data-ticker="' + ticker + '"]');
  const box = row && row.closest(".scroll");
  if (box && (row.offsetTop < box.scrollTop + 40 || row.offsetTop > box.scrollTop + box.clientHeight - 40))
    box.scrollTop = row.offsetTop - box.clientHeight / 2;
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


def screen_rows(screen) -> str:
    if screen is None or screen.empty:
        return ('<tr><td colspan="9" class="empty">Hôm nay không có mã nào thỏa điều kiện '
                '(thanh khoản chưa tăng đủ mạnh).</td></tr>')
    rows = []
    for _, r in screen.iterrows():
        t = html.escape(str(r["ticker"]))
        pct = r["pct"]
        cls = "pos" if pct > 0 else "neg" if pct < 0 else ""
        rows.append(
            f'<tr data-ticker="{t}"><td class="tk">{t}</td><td>{r["last_date"]}</td>'
            f'<td class="num">{fmt(r["close_before"], 2)}</td><td class="num">{fmt(r["close_now"], 2)}</td>'
            f'<td class="num {cls}">{pct:+.2f}%</td>'
            f'<td class="num">{r["vol_recent"]:,.0f}</td><td class="num">{r["vol_base"]:,.0f}</td>'
            f'<td class="num">{fmt(r["vol_ratio"], 2)}×</td><td class="num">{fmt(r["value_bn"], 1)}</td></tr>'
        )
    return "\n".join(rows)


def build_page(data: pd.DataFrame, screen: pd.DataFrame = None) -> str:
    data = data.sort_values(["ticker", "trading_date"]).reset_index(drop=True)
    data["trading_date"] = pd.to_datetime(data["trading_date"]).dt.strftime("%Y-%m-%d")
    data["pos"] = data.groupby("ticker").cumcount()  # phiên thứ mấy trong lịch sử của mã
    data["adx_3_ago"] = data.groupby("ticker")["adx_14"].shift(3)

    figs = {}
    for ticker, df in data.groupby("ticker"):
        shown = df.tail(DISPLAY_SESSIONS).reset_index(drop=True)
        figs[ticker] = json.loads(build_figure(shown).to_json())
    latest = data.groupby("ticker").tail(1).sort_values("ticker")

    breadth = breadth_table(data)
    breadth_fig = json.loads(build_breadth_figure(breadth).to_json()) if not breadth.empty else None

    options = "".join(f'<option value="{html.escape(t)}">{html.escape(t)}</option>' for t in figs)
    updated = datetime.now(ZoneInfo(VN_TZ)).strftime("%H:%M ngày %d/%m/%Y")

    def to_js(obj):
        return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")

    return (
        PAGE.replace("__PLOTLY_SRC__", f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js")
        .replace("__UPDATED__", updated)
        .replace("__DISPLAY__", str(DISPLAY_SESSIONS))
        .replace("__BREADTH_HEADLINE__", breadth_headline(breadth))
        .replace("__SCREEN_ROWS__", screen_rows(screen))
        .replace("__SCREEN_N__", str(SCREEN_SESSIONS))
        .replace("__SCREEN_BASE__", str(SCREEN_BASE_SESSIONS))
        .replace("__SCREEN_MULT__", f"{SCREEN_VOL_MULT:g}")
        .replace("__ROWS__", summary_rows(latest))
        .replace("__OPTIONS__", options)
        .replace("__BREADTH__", to_js(breadth_fig))
        .replace("__FIGS__", to_js(figs))
    )


def run(engine):
    with engine.connect() as conn:
        data = pd.read_sql(text(QUERY), conn)
        screen = pd.read_sql(text(SCREEN_QUERY), conn, params={
            "n": SCREEN_SESSIONS, "base": SCREEN_BASE_SESSIONS,
            "mult": SCREEN_VOL_MULT, "top": SCREEN_TOP,
        })
    if data.empty:
        print("LỖI: bảng indicators chưa có dữ liệu. Hãy chạy indicators.py trước.")
        sys.exit(1)
    print(f"Top biến động {SCREEN_SESSIONS} phiên kèm thanh khoản: {len(screen)} mã"
          + (f" ({', '.join(screen['ticker'])})" if not screen.empty else ""))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_page(data, screen))
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
