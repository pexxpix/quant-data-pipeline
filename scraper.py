import os
import sys
import time
import traceback
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine
from vnstock import Vnstock, __version__ as VNSTOCK_VERSION

DATABASE_URL = os.environ.get("DATABASE_URL")
WATCH_LIST = ["HPG", "POW", "PC1", "PLX"]


def get_stock_data(ticker: str) -> pd.DataFrame:
    print(f"Đang kéo dữ liệu cho mã {ticker}...")

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)

    try:
        stock = Vnstock().stock(symbol=ticker, source="TCBS")
        df = stock.quote.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        print(f"-> LỖI THẬT SỰ khi lấy {ticker}: [{type(e).__name__}] {e}")
        traceback.print_exc()
        return pd.DataFrame()

    if df is None or df.empty:
        print(f"-> Cảnh báo: không có dữ liệu trả về cho {ticker}.")
        return pd.DataFrame()

    # Chuẩn hoá cột theo định dạng bảng daily_prices
    df = df.rename(columns={"time": "trading_date"})
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.strftime("%Y-%m-%d")
    df["ticker"] = ticker
    df["snapshot_date"] = datetime.now().date()
    df = df[["trading_date", "open", "high", "low", "close", "volume", "ticker", "snapshot_date"]]

    print(f"-> Thành công: Lấy được {len(df)} dòng dữ liệu cho {ticker}.")
    return df


def load_to_database(df: pd.DataFrame) -> bool:
    if df.empty:
        print("Không có dữ liệu để load.")
        return False

    print("Đang kết nối Database...")
    engine = create_engine(DATABASE_URL)

    try:
        df.to_sql("daily_prices", engine, if_exists="append", index=False)
        print(f"Đã lưu thành công {len(df)} dòng vào database!")
        return True
    except Exception as e:
        print(f"Lỗi khi lưu vào Database: {e}")
        return False


def main():
    print(f"[debug] vnstock version: {VNSTOCK_VERSION}")

    if not DATABASE_URL:
        print("LỖI: Chưa cấu hình DATABASE_URL.")
        sys.exit(1)

    all_data = pd.DataFrame()
    for ticker in WATCH_LIST:
        df_ticker = get_stock_data(ticker)
        all_data = pd.concat([all_data, df_ticker], ignore_index=True)
        time.sleep(1)  # tránh gửi request quá dồn dập

    if all_data.empty:
        print("LỖI NGHIÊM TRỌNG: Không lấy được dữ liệu cho bất kỳ mã nào trong watch_list.")
        sys.exit(1)  # để GitHub Actions báo đỏ thay vì im lặng "thành công"

    if not load_to_database(all_data):
        sys.exit(1)


if __name__ == "__main__":
    main()

