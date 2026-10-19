import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from sqlalchemy import create_engine, inspect, text

TABLE = "daily_prices"
WATCH_LIST = ["HPG", "POW", "PC1", "PLX"]
LOOKBACK_DAYS = 365
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


# (tên nguồn, url, hàm tạo params, parser) - thử lần lượt, nguồn nào có dữ liệu thì dùng
PROVIDERS = [
    (
        "tcbs",
        "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term",
        lambda t, a, b: {"ticker": t, "type": "stock", "resolution": "D", "from": a, "to": b},
        parse_tcbs,
    ),
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
    print(f"Đang kéo dữ liệu cho mã {ticker}...")

    end_ts = int(time.time())
    start_ts = int((datetime.now() - timedelta(days=LOOKBACK_DAYS)).timestamp())

    for name, url, make_params, parser in PROVIDERS:
        df = fetch_from_provider(name, url, make_params, parser, ticker, start_ts, end_ts)
        if df.empty:
            continue

        for col in COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"]).sort_values("trading_date").reset_index(drop=True)
        if df.empty:
            continue

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


def load_to_database(df, snapshot_date, engine):
    """Ghi 1 snapshot. Chạy lại cùng ngày sẽ thay snapshot cũ thay vì nhân đôi dòng."""
    with engine.begin() as conn:
        if inspect(conn).has_table(TABLE):
            res = conn.execute(
                text(f"DELETE FROM {TABLE} WHERE snapshot_date = :d"), {"d": snapshot_date}
            )
            print(f"Đã xoá {res.rowcount} dòng cũ của snapshot {snapshot_date} (nếu có).")
        df.to_sql(TABLE, conn, if_exists="append", index=False, method="multi", chunksize=500)
        count = conn.execute(
            text(f"SELECT COUNT(*) FROM {TABLE} WHERE snapshot_date = :d"), {"d": snapshot_date}
        ).scalar()
    print(f"Đã lưu thành công! Bảng {TABLE} hiện có {count} dòng cho snapshot {snapshot_date}.")


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("LỖI: Chưa cấu hình DATABASE_URL.")
        sys.exit(1)

    snapshot_date = datetime.now(ZoneInfo(VN_TZ)).date()

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
        load_to_database(all_data, snapshot_date, engine)
    except Exception as e:  # noqa: BLE001
        print(f"LỖI khi lưu vào Database: {type(e).__name__}: {e}")
        sys.exit(1)

    if failed:
        print(f"CẢNH BÁO: đã lưu các mã khác nhưng thiếu dữ liệu của: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
