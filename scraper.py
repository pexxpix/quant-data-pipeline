import os
import requests
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

# Lấy chìa khóa database từ kho bảo mật của GitHub
# Tuyệt đối không ghi cứng chuỗi kết nối vào đây!
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_stock_data(ticker):
    print(f"Đang kéo dữ liệu cho mã {ticker}...")
    
    # API giả định của TCBS (chỉ là ví dụ cấu trúc, bạn cần tìm API endpoint thực tế)
    # Thông thường các API này trả về định dạng JSON
    url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/historical/{ticker}"
    
    # Cào dữ liệu trong 1 năm gần nhất
    params = {
        "resolution": "D", # D là daily (hằng ngày)
        "limit": 365
    }
    
    # Thêm User-Agent để API không nghĩ mình là bot tấn công mạng
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    # Biến đổi JSON thành bảng Pandas DataFrame
    if response.status_code == 200:
        data = response.json()
        if 'data' in data:
            df = pd.DataFrame(data['data'])
            
            # Sắp xếp lại cột cho chuẩn PIT
            # (Lưu ý: tên cột phụ thuộc vào cấu trúc trả về thực tế của API)
            df = df[['tradingDate', 'open', 'high', 'low', 'close', 'volume']]
            df.rename(columns={'tradingDate': 'trading_date'}, inplace=True)
            
            # Thêm cột ticker và Snapshot Date (thời điểm cào)
            df['ticker'] = ticker
            df['snapshot_date'] = datetime.now().date()
            
            return df
    
    print(f"Lỗi khi cào mã {ticker}")
    return pd.DataFrame() # Trả về bảng rỗng nếu lỗi

def load_to_database(df):
    if df.empty:
        print("Không có dữ liệu để load.")
        return
        
    print("Đang kết nối Database...")
    # Tạo động cơ kết nối
    engine = create_engine(DATABASE_URL)
    
    try:
        # if_exists='append' nghĩa là thêm dòng mới, KHÔNG ghi đè dòng cũ
        # index=False là không lưu cột số thứ tự mặc định của Pandas
        df.to_sql('daily_prices', engine, if_exists='append', index=False)
        print(f"Đã lưu thành công {len(df)} dòng vào database!")
    except Exception as e:
        print(f"Lỗi khi lưu vào Database: {e}")

def main():
    if not DATABASE_URL:
        print("LỖI: Chưa cấu hình DATABASE_URL. Thoát chương trình.")
        return
        
    # Danh sách các mã bạn quan tâm
    watch_list = ["HPG", "POW", "PC1", "PLX"]
    
    all_data = pd.DataFrame()
    
    for ticker in watch_list:
        df_ticker = get_stock_data(ticker)
        # Gộp dữ liệu của từng mã thành một bảng lớn
        all_data = pd.concat([all_data, df_ticker], ignore_index=True)
        
    # Đẩy nguyên bảng lớn lên Database trong 1 lần
    load_to_database(all_data)

if __name__ == "__main__":
    main()