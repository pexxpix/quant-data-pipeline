"""Tính RSI, MFI, CMF, ADX/+DI/-DI, MA50 và cờ lập đỉnh/đáy 52 tuần cho mọi mã trong bảng
daily_prices, lưu vào bảng indicators.

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

ADX_PERIOD = 14
ADX_TREND = 25           # ADX >= mức này: có xu hướng rõ (hướng do +DI / -DI quyết định)
ADX_NO_TREND = 20        # ADX < mức này: thị trường đi ngang

HIGH_LOW_LOOKBACK = 250  # 52 tuần ~ 250 phiên: so đỉnh/đáy phiên hôm nay với 250 phiên trước đó

MA_PERIOD = 50           # đường trung bình giản đơn (SMA) của giá đóng cửa

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
    adx_14        DOUBLE PRECISION,
    plus_di_14    DOUBLE PRECISION,
    minus_di_14   DOUBLE PRECISION,
    ma_50         DOUBLE PRECISION,
    above_ma50    BOOLEAN,
    rsi_zone      TEXT,
    mfi_zone      TEXT,
    cmf_signal    TEXT,
    adx_signal    TEXT,
    new_high_52w  BOOLEAN,
    new_low_52w   BOOLEAN,
    snapshot_date DATE,
    computed_at   TIMESTAMP,
    PRIMARY KEY (ticker, trading_date)
)
"""

# Bảng indicators tạo từ phiên bản trước chưa có các cột mới: thêm vào nếu thiếu
# (chạy nhiều lần không sao, cột đã có thì Postgres bỏ qua).
MIGRATIONS = [
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS adx_14 DOUBLE PRECISION",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS plus_di_14 DOUBLE PRECISION",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS minus_di_14 DOUBLE PRECISION",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS adx_signal TEXT",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS new_high_52w BOOLEAN",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS new_low_52w BOOLEAN",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS ma_50 DOUBLE PRECISION",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS above_ma50 BOOLEAN",
]

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
    `period` giá trị hợp lệ đầu tiên, sau đó mỗi phiên mới: avg = (avg_trước*(n-1) + x)/n.
    Cách này cho kết quả khớp với TradingView, FireAnt, StockCharts..."""
    result = pd.Series(np.nan, index=values.index)
    valid = np.flatnonzero(values.notna().to_numpy())
    if len(valid) == 0:
        return result
    start = valid[0]                 # bỏ qua các phiên đầu chưa có giá trị (vd: phiên đầu của diff)
    seed_pos = start + period - 1    # phiên có đủ `period` giá trị để lấy trung bình đầu tiên
    if seed_pos >= len(values):
        return result
    seeded = values.copy()
    seeded.iloc[: seed_pos + 1] = np.nan
    seeded.iloc[seed_pos] = values.iloc[start : seed_pos + 1].mean()
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


def adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """ADX / +DI / -DI của Wilder. Thang 0-100.

    +DI: sức mạnh phía tăng (đỉnh hôm nay vượt đỉnh hôm qua bao nhiêu).
    -DI: sức mạnh phía giảm (đáy hôm nay thủng đáy hôm qua bao nhiêu).
    ADX: độ mạnh của xu hướng, KHÔNG cho biết hướng - phải đọc kèm +DI / -DI.
    Giá trị ADX đầu tiên có ở phiên thứ 2*period (làm mượt 2 lần)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    # Biên độ thực: lấy cả khoảng trống giá so với phiên trước (gap)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=False)

    up_move = high.diff()        # đỉnh nâng lên bao nhiêu
    down_move = -low.diff()      # đáy hạ xuống bao nhiêu
    # Mỗi phiên chỉ tính một phía: phía nào lớn hơn (và dương) mới được ghi nhận
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0).where(up_move.notna())
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0).where(down_move.notna())

    atr = wilder_average(true_range, period)
    atr_safe = atr.replace(0, np.nan)
    plus_di = (100 * wilder_average(plus_dm, period) / atr_safe).where(atr != 0, 0.0).where(atr.notna())
    minus_di = (100 * wilder_average(minus_dm, period) / atr_safe).where(atr != 0, 0.0).where(atr.notna())

    di_sum = plus_di + minus_di
    dx = (100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)).where(di_sum != 0, 0.0)
    dx = dx.where(di_sum.notna())
    return pd.DataFrame(
        {"adx": wilder_average(dx, period), "plus_di": plus_di, "minus_di": minus_di},
        index=df.index,
    )


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


def new_high_low(df: pd.DataFrame, lookback: int = HIGH_LOW_LOOKBACK):
    """Cờ lập đỉnh / đáy `lookback` phiên (mặc định 52 tuần) của từng phiên.

    Lập đỉnh: giá cao nhất hôm nay VƯỢT giá cao nhất của `lookback` phiên trước đó.
    Lập đáy:  giá thấp nhất hôm nay THỦNG giá thấp nhất của `lookback` phiên trước đó.
    Phiên chưa có đủ `lookback` phiên lịch sử (vd: mã mới niêm yết) -> để trống (NULL),
    không tính vào thống kê độ rộng thị trường."""
    prior_high = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    prior_low = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    new_high = (df["high"] > prior_high).astype("boolean").mask(prior_high.isna())
    new_low = (df["low"] < prior_low).astype("boolean").mask(prior_low.isna())
    return new_high, new_low


def moving_average(close: pd.Series, period: int = MA_PERIOD):
    """MA giản đơn: trung bình cộng giá đóng cửa `period` phiên gần nhất (tính cả phiên hôm nay).
    Trả về (ma, above): above = True khi giá đóng cửa hôm đó NẰM TRÊN MA.
    Mã chưa đủ `period` phiên -> để trống (NULL)."""
    ma = close.rolling(period, min_periods=period).mean()
    # làm tròn trước khi so: giá bằng đúng MA (sai số số thực ~1e-15) luôn tính là "không trên MA"
    above = (close.round(6) > ma.round(6)).astype("boolean").mask(ma.isna())
    return ma, above


def adx_label(adx_value, plus_di, minus_di):
    if pd.isna(adx_value):
        return None
    if adx_value >= ADX_TREND:
        return "xu hướng tăng" if plus_di >= minus_di else "xu hướng giảm"
    if adx_value >= ADX_NO_TREND:
        return "xu hướng yếu"
    return "đi ngang"


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
        dmi = adx(df)
        df["adx_14"] = dmi["adx"]
        df["plus_di_14"] = dmi["plus_di"]
        df["minus_di_14"] = dmi["minus_di"]
        df["new_high_52w"], df["new_low_52w"] = new_high_low(df)
        df["ma_50"], df["above_ma50"] = moving_average(df["close"])
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["rsi_zone"] = out["rsi_14"].apply(zone, args=(RSI_OVERSOLD, RSI_OVERBOUGHT))
    out["mfi_zone"] = out["mfi_14"].apply(zone, args=(MFI_OVERSOLD, MFI_OVERBOUGHT))
    out["cmf_signal"] = out["cmf_20"].apply(cmf_label)
    out["adx_signal"] = [
        adx_label(a, p, m) for a, p, m in zip(out["adx_14"], out["plus_di_14"], out["minus_di_14"])
    ]
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"]).dt.date
    out["computed_at"] = datetime.now(ZoneInfo(VN_TZ)).replace(tzinfo=None)
    for col in ["rsi_14", "mfi_14", "adx_14", "plus_di_14", "minus_di_14"]:
        out[col] = out[col].round(2)
    out["cmf_20"] = out["cmf_20"].round(4)
    out["ma_50"] = out["ma_50"].round(4)  # cờ above_ma50 đã tính trên giá trị chưa làm tròn
    return out[
        ["ticker", "trading_date", "close", "volume", "rsi_14", "mfi_14", "cmf_20",
         "adx_14", "plus_di_14", "minus_di_14", "ma_50", "above_ma50",
         "rsi_zone", "mfi_zone", "cmf_signal", "adx_signal", "new_high_52w", "new_low_52w",
         "snapshot_date", "computed_at"]
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
        for stmt in MIGRATIONS:
            conn.execute(text(stmt))
        conn.execute(text(f"DELETE FROM {TABLE}"))  # tính lại toàn bộ mỗi lần
        result.to_sql(TABLE, conn, if_exists="append", index=False, method="multi", chunksize=500)

    print(f"Đã lưu {len(result)} dòng vào bảng {TABLE}.\n")
    print_summary(result)
    return result


def print_summary(result: pd.DataFrame):
    latest = result.sort_values("trading_date").groupby("ticker").tail(1)
    print("TỔNG QUAN PHIÊN GẦN NHẤT")
    print(f"{'Mã':<6}{'Ngày':<12}{'Giá':>9}{'RSI':>8}  {'':<11}{'MFI':>7}  {'':<11}{'CMF':>8}  {'':<14}{'ADX':>6}  ")
    for _, r in latest.sort_values("ticker").iterrows():
        print(
            f"{r['ticker']:<6}{str(r['trading_date']):<12}{r['close']:>9.2f}"
            f"{r['rsi_14']:>8.1f}  {str(r['rsi_zone']):<11}"
            f"{r['mfi_14']:>7.1f}  {str(r['mfi_zone']):<11}"
            f"{r['cmf_20']:>8.3f}  {str(r['cmf_signal']):<14}"
            f"{r['adx_14']:>6.1f}  {r['adx_signal']}"
        )

    last_day = result["trading_date"].max()
    day = result[result["trading_date"] == last_day]
    eligible = day["new_high_52w"].notna().sum()
    print(
        f"\nĐỘ RỘNG THỊ TRƯỜNG phiên {last_day}: "
        f"{int(day['new_high_52w'].sum())} mã lập đỉnh 52 tuần, "
        f"{int(day['new_low_52w'].sum())} mã lập đáy 52 tuần "
        f"(trên {eligible} mã đủ 1 năm dữ liệu)"
    )
    with_ma = day["above_ma50"].notna().sum()
    above = int(day["above_ma50"].sum())
    if with_ma:
        print(f"{above}/{with_ma} mã ({100 * above / with_ma:.0f}%) đóng cửa trên MA{MA_PERIOD}")


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
