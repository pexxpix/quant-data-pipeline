import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_stock_data(ticker):
    print(f"Đang kéo dữ liệu cho mã {ticker}...")
    
    # Lấy mốc thời gian: 1 năm trước đến hiện tại (định dạng Unix timestamp)
    end_time = int(time.time())
    start_time = int((datetime.now() - timedelta(days=365)).timestamp())
    
    # API thực tế của TCBS
    url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
    params = {
        "ticker": ticker,
        "type": "stock",
        "resolution": "D",
        "from": start_time,
        "to": end_time
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                df = pd.DataFrame(data['data'])
                
                # Sắp xếp và đổi tên cột theo chuẩn PIT
                df = df[['tradingDate', 'open', 'high', 'low', 'close', 'volume']]
                df.rename(columns={'tradingDate': 'trading_date'}, inplace=True)
                
                df['ticker'] = ticker
                df['snapshot_date'] = datetime.now().date()
                
                # Cắt chuỗi ngày tháng cho gọn (chỉ lấy yyyy-mm-dd)
                df['trading_date'] = df['trading_date'].str.slice(0, 10)
                
                print(f"-> Thành công: Lấy được {len(df)} dòng dữ liệu.")
                return df
        print(f"-> Lỗi: API không trả về dữ liệu (Status code: {response.status_code})")
    except Exception as e:
        print(f"-> Lỗi kết nối mạng: {e}")
        
    return pd.DataFrame()

def load_to_database(df):
    if df.empty:
        print("Không có dữ liệu để load.")
        return
        
    print("Đang kết nối Database...")
    engine = create_engine(DATABASE_URL)
    
    try:
        df.to_sql('daily_prices', engine, if_exists='append', index=False)
        print(f"Đã lưu thành công {len(df)} dòng vào database!")
    except Exception as e:
        print(f"Lỗi khi lưu vào Database: {e}")

def main():
    if not DATABASE_URL:
        print("LỖI: Chưa cấu hình DATABASE_URL.")
        return
        
    watch_list = ["HPG", "POW", "PC1", "PLX"]
    all_data = pd.DataFrame()
    
    for ticker in watch_list:
        df_ticker = get_stock_data(ticker)
        all_data = pd.concat([all_data, df_ticker], ignore_index=True)
        
    load_to_database(all_data)

if __name__ == "__main__":
    main()