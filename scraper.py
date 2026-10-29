import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from sqlalchemy import create_engine, inspect, text

TABLE = "daily_prices"
# Top 100 cổ phiếu Việt Nam theo https://stag.vn/stocks (lấy ngày 27/10/2026, giữ đúng thứ tự trên trang)
TOP_100 = [
    "VIC", "VCB", "VHM", "TCB", "CTG", "BID", "VGI", "GAS", "VPB", "MCH",
    "HPG", "MBB", "ACV", "GVR", "STB", "VNM", "LPB", "MWG", "HDB", "ACB",
    "MSN", "FPT", "DMX", "TCX", "VJC", "MVN", "MSR", "VCK", "KSF", "HVN",
    "VRE", "SAB", "VEA", "BVH", "SSI", "SHB", "MSB", "VIB", "PLX", "SSB",
    "VPX", "BCM", "FOX", "NVB", "THD", "TPB", "POW", "KSV", "EIB", "OCB",
    "ABB", "HCM", "GEL", "FRT", "PGV", "KBC", "REE", "LPS", "NVL", "GMD",
    "SSH", "DNH", "GEX", "VGC", "VND", "VIX", "DCM", "HAG", "QNS", "SBT",
    "KDH", "PVS", "VCI", "SJS", "CC1", "IDP", "VNZ", "PVI", "CRV", "MBS",
    "PHP", "VBB", "DPM", "VSF", "KDC", "LGC", "OIL", "HUT", "VEF", "DGC",
    "VCR", "BAB", "F88", "PDR", "VHC", "DHG", "BMP", "DXG", "NLG", "TAL",
]
# Mã theo dõi riêng (ngoài top 100). Các mã cũ HPG, POW, PLX, HDB, EIB, TCX, VIX, GMD, MSN đã nằm trong TOP_100.
EXTRA = ["PC1", "TCH", "VPL", "DCL", "VPI", "PET", "VTP", "CTR", "NAB", "VSC", "BSR", "GEE", "PVD", "DGW", "PVT", "PVP"]
WATCH_LIST = TOP_100 + [t for t in EXTRA if t not in TOP_100]
# 340 phiên = 250 phiên (52 tuần) để so đỉnh/đáy + 90 phiên hiển thị trên biểu đồ.
# Nhờ có lịch sử dài, RSI/ADX trên cả 90 phiên hiển thị đều đã ổn định.
KEEP_SESSIONS = 340
# 340 phiên ~ 16-17 tháng lịch; xin dư 540 ngày lịch để chắc chắn đủ cả khi có nghỉ Tết/lễ,
# sau đó cắt lại đúng KEEP_SESSIONS phiên.
FETCH_CALENDAR_DAYS = 540
VN_TZ = "Asia/Ho_Chi_Minh"
COLUMNS = ["trading_date", "open", "high", "low", "close", "volume"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}


# --------------------------------------------------------------------------
# Parsers: mỗi nguồn trả về JSON một kiểu khác nhau -> chuẩn hoá về COLUMNS
# --------------------------------------------------------------------------
def parse_arrays(payload):
    """Dạng {"t":[...], "o":[...], "h":[...], "l":[...], "c":[...], "v":[...]}
    (có thể bọc trong khoá "data")."""
    if isinstance(payload.get("data"), dict):
        payload = payload["data"]
    if not payload.get("t"):
        return pd.DataFrame(columns=COLUMNS)
    dates = (
        pd.to_datetime(payload["t"], unit="s", utc=True)
        .tz_convert(VN_TZ)
        .strftime("%Y-%m-%d")
    )
    return pd.DataFrame(
        {
            "trading_date": list(dates),
            "open": payload["o"],
            "high": payload["h"],
            "low": payload["l"],
            "close": payload["c"],
            "volume": payload["v"],
        }
    )


def parse_tcbs(payload):
    """Dạng {"data": [{"tradingDate": "...", "open": ..., ...}]}."""
    rows = payload.get("data") or []
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows)
    df["trading_date"] = df["tradingDate"].astype(str).str.slice(0, 10)
    return df[COLUMNS]


# (tên nguồn, url, hàm tạo params, parser) - thử lần lượt, nguồn nào có dữ liệu thì dùng.
# VNDirect đứng đầu vì đang là nguồn chạy được; TCBS trả 404 "Service not found" từ 10/2026
# nên để cuối cùng (chỉ dùng nếu hai nguồn trên cùng lỗi).
PROVIDERS = [
    (
        "vndirect",
        "https://dchart-api.vndirect.com.vn/dchart/history",
        lambda t, a, b: {"resolution": "D", "symbol": t, "from": a, "to": b},
        parse_arrays,
    ),
    (
        "ssi",
        "https://iboard-api.ssi.com.vn/statistics/charts/history",
        lambda t, a, b: {"symbol": t, "resolution": "1D", "from": a, "to": b},
        parse_arrays,
    ),
    (
        "tcbs",
        "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term",
        lambda t, a, b: {"ticker": t, "type": "stock", "resolution": "D", "from": a, "to": b},
        parse_tcbs,
    ),
]


def fetch_from_provider(name, url, make_params, parser, ticker, start_ts, end_ts):
    """Gọi 1 nguồn. Luôn in ra lý do thật nếu thất bại (mã HTTP + đoạn đầu nội dung)."""
    try:
        resp = requests.get(
            url, params=make_params(ticker, start_ts, end_ts), headers=HEADERS, timeout=30
        )
    except requests.RequestException as e:
        print(f"   [{name}] {ticker}: lỗi mạng: {type(e).__name__}: {e}")
        return pd.DataFrame()

    if resp.status_code != 200:
        print(f"   [{name}] {ticker}: HTTP {resp.status_code} - {resp.text[:150]!r}")
        return pd.DataFrame()

    try:
        df = parser(resp.json())
    except Exception as e:  # noqa: BLE001 - cần in mọi lỗi parse
        print(f"   [{name}] {ticker}: không đọc được dữ liệu ({type(e).__name__}: {e}) - {resp.text[:150]!r}")
        return pd.DataFrame()

    if df.empty:
        print(f"   [{name}] {ticker}: HTTP 200 nhưng không có dòng nào - {resp.text[:150]!r}")
    return df


def get_stock_data(ticker, snapshot_date):
    """Lấy KEEP_SESSIONS phiên gần nhất của một mã."""
    print(f"Đang kéo dữ liệu cho mã {ticker}...")

    end_ts = int(time.time())
    start_ts = int((datetime.now() - timedelta(days=FETCH_CALENDAR_DAYS)).timestamp())

    for name, url, make_params, parser in PROVIDERS:
        df = fetch_from_provider(name, url, make_params, parser, ticker, start_ts, end_ts)
        if df.empty:
            continue

        for col in COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = (
            df.dropna(subset=["close"])
            .drop_duplicates("trading_date", keep="last")
            .sort_values("trading_date")
            .tail(KEEP_SESSIONS)  # đúng KEEP_SESSIONS phiên gần nhất
            .reset_index(drop=True)
        )
        if df.empty:
            continue
        df["volume"] = df["volume"].round().astype("Int64")

        df["ticker"] = ticker
        df["source"] = name
        df["snapshot_date"] = snapshot_date
        last = df.iloc[-1]
        print(
            f"-> Thành công ({name}): {len(df)} dòng, "
            f"phiên cuối {last['trading_date']} close={last['close']}"
        )
        return df

    print(f"-> THẤT BẠI: không nguồn nào trả về dữ liệu cho {ticker}")
    return pd.DataFrame()


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
def make_engine(database_url):
    url = database_url.strip().strip("'\"")
    if url.startswith("postgres://"):  # SQLAlchemy 2.x không nhận "postgres://"
        url = "postgresql://" + url[len("postgres://"):]
    return create_engine(url, connect_args={"connect_timeout": 30}, pool_pre_ping=True)


def describe_connection(engine):
    """In ra role/database đang kết nối (KHÔNG in mật khẩu) để dễ chẩn đoán lỗi quyền."""
    print(f"Kết nối tới: {engine.url.render_as_string(hide_password=True)}")
    with engine.connect() as conn:
        role, db, can_create = conn.execute(
            text(
                "SELECT current_user, current_database(), "
                "has_schema_privilege(current_user, 'public', 'CREATE')"
            )
        ).one()
    print(f"Role: {role} | Database: {db} | Có quyền CREATE trong schema public: {can_create}")


DDL = f"""
CREATE TABLE {TABLE} (
    trading_date  DATE             NOT NULL,
    open          DOUBLE PRECISION,
    high          DOUBLE PRECISION,
    low           DOUBLE PRECISION,
    close         DOUBLE PRECISION,
    volume        BIGINT,
    ticker        TEXT             NOT NULL,
    source        TEXT,
    snapshot_date DATE,
    PRIMARY KEY (ticker, trading_date)
)
"""


def load_to_database(df, engine):
    """Mỗi mã chỉ có 1 dòng cho mỗi phiên, tối đa KEEP_SESSIONS phiên gần nhất.

    - Mã lấy được dữ liệu: xoá toàn bộ dòng cũ của mã đó rồi ghi bản mới
      (giá điều chỉnh luôn nhất quán, chạy lại bao nhiêu lần cũng không nhân đôi).
    - Mã lấy lỗi hôm nay: giữ nguyên dữ liệu cũ, chỉ cắt cho còn tối đa KEEP_SESSIONS phiên.
    - Mã không còn trong WATCH_LIST: bị xoá khỏi bảng (thêm lại thì lần chạy sau
      sẽ tải lại đủ KEEP_SESSIONS phiên, nên không mất gì).
    """
    tickers = sorted(df["ticker"].unique())
    with engine.begin() as conn:
        if not inspect(conn).has_table(TABLE):
            conn.execute(text(DDL))
            print(f"Đã tạo bảng {TABLE}.")

        replaced = 0
        for t in tickers:
            replaced += conn.execute(
                text(f"DELETE FROM {TABLE} WHERE ticker = :t"), {"t": t}
            ).rowcount
        df.to_sql(TABLE, conn, if_exists="append", index=False, method="multi", chunksize=500)

        removed, trimmed = 0, 0
        existing = [r[0] for r in conn.execute(text(f"SELECT DISTINCT ticker FROM {TABLE}"))]
        for t in existing:
            if t not in WATCH_LIST:
                removed += conn.execute(
                    text(f"DELETE FROM {TABLE} WHERE ticker = :t"), {"t": t}
                ).rowcount
                continue
            # bảng kiểu cũ có thể còn nhiều bản sao (nhiều snapshot): chỉ giữ bản mới nhất
            trimmed += conn.execute(
                text(
                    f"DELETE FROM {TABLE} WHERE ticker = :t AND snapshot_date < "
                    f"(SELECT MAX(snapshot_date) FROM {TABLE} WHERE ticker = :t)"
                ),
                {"t": t},
            ).rowcount
            # ngày của phiên thứ KEEP_SESSIONS tính từ phiên mới nhất; xoá mọi phiên cũ hơn nó
            oldest_kept = conn.execute(
                text(
                    f"SELECT DISTINCT trading_date FROM {TABLE} WHERE ticker = :t "
                    f"ORDER BY trading_date DESC LIMIT 1 OFFSET :k"
                ),
                {"t": t, "k": KEEP_SESSIONS - 1},
            ).scalar()
            if oldest_kept is not None:
                trimmed += conn.execute(
                    text(f"DELETE FROM {TABLE} WHERE ticker = :t AND trading_date < :d"),
                    {"t": t, "d": oldest_kept},
                ).rowcount

        summary = conn.execute(
            text(
                f"SELECT ticker, COUNT(*), MIN(trading_date), MAX(trading_date) "
                f"FROM {TABLE} GROUP BY ticker ORDER BY ticker"
            )
        ).fetchall()

    print(
        f"Đã thay {replaced} dòng cũ bằng {len(df)} dòng mới; cắt {trimmed} dòng thừa "
        f"(trùng hoặc quá {KEEP_SESSIONS} phiên); xoá {removed} dòng của mã không còn trong WATCH_LIST."
    )
    print(f"Đã lưu thành công! Bảng {TABLE} hiện có:")
    for ticker, count, first, last in summary:
        print(f"   {ticker:<6} {count:>4} phiên, từ {first} đến {last}")


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("LỖI: Chưa cấu hình DATABASE_URL.")
        sys.exit(1)

    snapshot_date = datetime.now(ZoneInfo(VN_TZ)).date()  # ngày lấy dữ liệu
    print(f"Giữ {KEEP_SESSIONS} phiên gần nhất cho mỗi mã: {', '.join(WATCH_LIST)}")

    frames, failed = [], []
    for ticker in WATCH_LIST:
        df = get_stock_data(ticker, snapshot_date)
        if df.empty:
            failed.append(ticker)
        else:
            frames.append(df)
        time.sleep(1)

    if not frames:
        print("LỖI NGHIÊM TRỌNG: không lấy được dữ liệu cho bất kỳ mã nào.")
        sys.exit(1)

    all_data = pd.concat(frames, ignore_index=True)
    print("Đang kết nối Database...")
    try:
        engine = make_engine(database_url)
        describe_connection(engine)
        load_to_database(all_data, engine)
    except Exception as e:  # noqa: BLE001
        print(f"LỖI khi lưu vào Database: {type(e).__name__}: {e}")
        sys.exit(1)

    if failed:
        print(f"CẢNH BÁO: đã lưu các mã khác nhưng thiếu dữ liệu của: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
