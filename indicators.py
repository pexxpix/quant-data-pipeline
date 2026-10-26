"""Tính RSI, MFI, CMF cho mọi mã trong bảng daily_prices và lưu vào bảng indicators.

Chạy sau scraper.py. Mỗi lần chạy tính lại toàn bộ từ dữ liệu mới nhất,
nên chạy bao nhiêu lần cũng cho cùng kết quả (không nhân đôi dòng).
"""
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# THIẾT LẬP: chỉnh các con số ở đây
# ---------------------------------------------------------------------------
RSI_PERIOD = 14          # số phiên tính RSI (nhỏ hơn = nhạy hơn nhưng nhiễu hơn)
RSI_OVERBOUGHT = 70      # RSI >= mức này: vùng quá mua
RSI_OVERSOLD = 30        # RSI <= mức này: vùng quá bán

MFI_PERIOD = 14
MFI_OVERBOUGHT = 80
MFI_OVERSOLD = 20

CMF_PERIOD = 20          # số phiên cộng dồn dòng tiền
CMF_THRESHOLD = 0.05     # |CMF| nhỏ hơn mức này coi là trung tính (lọc nhiễu)

SOURCE_TABLE = "daily_prices"
TABLE = "indicators"
VN_TZ = "Asia/Ho_Chi_Minh"

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    ticker        TEXT             NOT NULL,
    trading_date  DATE             NOT NULL,
    close         DOUBLE PRECISION,
    volume        BIGINT,
    rsi_14        DOUBLE PRECISION,
    mfi_14        DOUBLE PRECISION,
    cmf_20        DOUBLE PRECISION,
    rsi_zone      TEXT,
    mfi_zone      TEXT,
    cmf_signal    TEXT,
    snapshot_date DATE,
    computed_at   TIMESTAMP,
    PRIMARY KEY (ticker, trading_date)
)
"""

# Chỉ lấy snapshot mới nhất của mỗi mã: snapshot đó đã chứa đủ ~1 năm giá,
# và toàn bộ chuỗi được lấy cùng một lúc nên giá điều chỉnh (cổ tức, chia tách)
# nhất quán với nhau. Trộn nhiều snapshot có thể tạo "bước nhảy giá" giả.
PRICE_QUERY = f"""
SELECT p.ticker, p.trading_date, p.high, p.low, p.close, p.volume, p.snapshot_date
FROM {SOURCE_TABLE} p
JOIN (
    SELECT ticker, MAX(snapshot_date) AS snap
    FROM {SOURCE_TABLE}
    GROUP BY ticker
) m ON p.ticker = m.ticker AND p.snapshot_date = m.snap
"""


# ---------------------------------------------------------------------------
# Công thức chỉ báo
# ---------------------------------------------------------------------------
def wilder_average(values: pd.Series, period: int) -> pd.Series:
    """Trung bình làm mượt kiểu Wilder: giá trị đầu là trung bình cộng của
    `period` giá trị đầu tiên, sau đó mỗi phiên mới: avg = (avg_trước*(n-1) + x)/n.
    Cách này cho kết quả khớp với TradingView, FireAnt, StockCharts..."""
    result = pd.Series(np.nan, index=values.index)
    if len(values) <= period:
        return result
    seeded = values.copy()
    seeded.iloc[: period + 1] = np.nan
    seeded.iloc[period] = values.iloc[1 : period + 1].mean()
    return seeded.ewm(alpha=1 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI: so sánh độ lớn các phiên tăng với các phiên giảm. Thang 0-100."""
    delta = close.diff()
    avg_gain = wilder_average(delta.clip(lower=0), period)
    avg_loss = wilder_average(-delta.clip(upper=0), period)
    total = avg_gain + avg_loss
    # 100 * gain/(gain+loss) tương đương 100 - 100/(1+RS) nhưng không chia cho 0
    return (100 * avg_gain / total.replace(0, np.nan)).where(total != 0, 50.0).where(avg_gain.notna())


def mfi(df: pd.DataFrame, period: int = MFI_PERIOD) -> pd.Series:
    """MFI: giống RSI nhưng dùng 'giá điển hình x khối lượng' (dòng tiền). Thang 0-100."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical * df["volume"]
    direction = typical.diff()
    positive = money_flow.where(direction > 0, 0.0)
    negative = money_flow.where(direction < 0, 0.0)
    pos_sum = positive.rolling(period).sum()
    neg_sum = negative.rolling(period).sum()
    total = pos_sum + neg_sum
    out = 100 * pos_sum / total.replace(0, np.nan)
    out.iloc[:period] = np.nan  # phiên đầu tiên không có hướng so sánh
    return out


def cmf(df: pd.DataFrame, period: int = CMF_PERIOD) -> pd.Series:
    """CMF: giá đóng cửa nằm gần đỉnh hay đáy của phiên, nhân khối lượng,
    cộng dồn `period` phiên. Thang -1 đến +1."""
    day_range = (df["high"] - df["low"]).replace(0, np.nan)
    multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / day_range
    flow_volume = multiplier.fillna(0) * df["volume"]
    return flow_volume.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


def zone(value, low, high):
    if pd.isna(value):
        return None
    if value >= high:
        return "quá mua"
    if value <= low:
        return "quá bán"
    return "trung tính"


def cmf_label(value):
    if pd.isna(value):
        return None
    if value > CMF_THRESHOLD:
        return "dòng tiền vào"
    if value < -CMF_THRESHOLD:
        return "dòng tiền ra"
    return "trung tính"


def compute(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["trading_date"] = pd.to_datetime(prices["trading_date"]).dt.date
    for col in ["high", "low", "close", "volume"]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    frames = []
    for ticker, df in prices.groupby("ticker"):
        df = (
            df.dropna(subset=["close"])
            .drop_duplicates("trading_date", keep="last")
            .sort_values("trading_date")
            .reset_index(drop=True)
        )
        df["rsi_14"] = rsi(df["close"])
        df["mfi_14"] = mfi(df)
        df["cmf_20"] = cmf(df)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["rsi_zone"] = out["rsi_14"].apply(zone, args=(RSI_OVERSOLD, RSI_OVERBOUGHT))
    out["mfi_zone"] = out["mfi_14"].apply(zone, args=(MFI_OVERSOLD, MFI_OVERBOUGHT))
    out["cmf_signal"] = out["cmf_20"].apply(cmf_label)
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"]).dt.date
    out["computed_at"] = datetime.now(ZoneInfo(VN_TZ)).replace(tzinfo=None)
    for col in ["rsi_14", "mfi_14"]:
        out[col] = out[col].round(2)
    out["cmf_20"] = out["cmf_20"].round(4)
    return out[
        ["ticker", "trading_date", "close", "volume", "rsi_14", "mfi_14", "cmf_20",
         "rsi_zone", "mfi_zone", "cmf_signal", "snapshot_date", "computed_at"]
    ]


# ---------------------------------------------------------------------------
# Đọc / ghi database
# ---------------------------------------------------------------------------
def make_engine(database_url: str):
    url = database_url.strip().strip("'\"")
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return create_engine(url, pool_pre_ping=True)


def run(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        prices = pd.read_sql(text(PRICE_QUERY), conn)
    if prices.empty:
        print(f"LỖI: bảng {SOURCE_TABLE} chưa có dữ liệu. Hãy chạy scraper.py trước.")
        sys.exit(1)

    result = compute(prices)

    with engine.begin() as conn:
        conn.execute(text(DDL))
        conn.execute(text(f"DELETE FROM {TABLE}"))  # tính lại toàn bộ mỗi lần
        result.to_sql(TABLE, conn, if_exists="append", index=False, method="multi", chunksize=500)

    print(f"Đã lưu {len(result)} dòng vào bảng {TABLE}.\n")
    print_summary(result)
    return result


def print_summary(result: pd.DataFrame):
    latest = result.sort_values("trading_date").groupby("ticker").tail(1)
    print("TỔNG QUAN PHIÊN GẦN NHẤT")
    print(f"{'Mã':<6}{'Ngày':<12}{'Giá':>9}{'RSI':>8}  {'':<11}{'MFI':>7}  {'':<11}{'CMF':>8}  ")
    for _, r in latest.sort_values("ticker").iterrows():
        print(
            f"{r['ticker']:<6}{str(r['trading_date']):<12}{r['close']:>9.2f}"
            f"{r['rsi_14']:>8.1f}  {str(r['rsi_zone']):<11}"
            f"{r['mfi_14']:>7.1f}  {str(r['mfi_zone']):<11}"
            f"{r['cmf_20']:>8.3f}  {r['cmf_signal']}"
        )


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
        print(f"LỖI khi tính chỉ báo: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
