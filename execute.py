import os
import sys
import pandas as pd
import numpy as np

# Nhập các hàm dự đoán từ các tệp hiện tại
try:
    from ensemble import predict_gbdt_for_date, predict_lstm_for_date
    from markov_predict import predict_markov_for_date
except ImportError as e:
    print(f"Lỗi: Không tìm thấy các file script cần thiết (ensemble.py, markov_predict.py).")
    print(f"Chi tiết: {e}")
    sys.exit(1)

def main():
    csv_file_path = os.path.join("csv", "xxmb - xuly.csv")
    if not os.path.exists(csv_file_path):
        print(f"Error: csv file not found at {csv_file_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_file_path)
    target_col = 'g1-extract'
    
    if target_col not in df.columns:
        print(f"Error: column {target_col} not found in csv")
        sys.exit(1)
        
    df_clean = df.dropna(subset=[target_col]).copy()
    df_clean[target_col] = df_clean[target_col].astype(int)
    try:
        df_clean['date_parsed'] = pd.to_datetime(df_clean['day'], format='%d-%m-%Y')
    except Exception:
        df_clean['date_parsed'] = pd.to_datetime(df_clean['day'])
        
    df_clean = df_clean.sort_values('date_parsed').reset_index(drop=True)
    
    last_date = df_clean['date_parsed'].max()
    next_date = last_date + pd.Timedelta(days=1)
    
    # Tính toán dự đoán từ các mô hình độc lập
    p_gbdt = predict_gbdt_for_date(df_clean, next_date)
    p_lstm = predict_lstm_for_date(df_clean, next_date)
    p_markov = predict_markov_for_date(df_clean, next_date, alpha=0.15)
    
    # Tính toán đồng thuận (ensemble)
    p_ensemble = (p_gbdt + p_lstm + p_markov) / 3.0
    
    # Lấy top 5 của từng mô hình
    top5_lstm = np.argsort(p_lstm)[::-1][:5]
    top5_gbdt = np.argsort(p_gbdt)[::-1][:5]
    top5_markov = np.argsort(p_markov)[::-1][:5]
    top5_ensemble = np.argsort(p_ensemble)[::-1][:5]
    
    # 1. In kết quả LSTM
    print("lstm ")
    for i, num in enumerate(top5_lstm, 1):
        print(f"Gợi ý {i}: Cặp số [{num:02d}] với xác suất dự báo: {p_lstm[num]*100:.2f}%")
        
    # 2. In kết quả LGBM
    print("_________")
    print("lgbm")
    for i, num in enumerate(top5_gbdt, 1):
        print(f"Gợi ý {i}: Cặp số [{num:02d}] với xác suất dự báo: {p_gbdt[num]*100:.2f}%")
        
    # 3. In kết quả Markov
    print("_________")
    print("markov")
    for i, num in enumerate(top5_markov, 1):
        print(f"Gợi ý {i}: Cặp số [{num:02d}] với xác suất: {p_markov[num]*100:.2f}%")
        
    # 4. In kết quả Ensemble (Consensus)
    print("____________")
    print("esemble")
    for i, num in enumerate(top5_ensemble, 1):
        print(f"  Gợi ý {i}: Cặp số [{num:02d}] với độ tin cậy đồng thuận: {p_ensemble[num]*100:.2f}%")

if __name__ == "__main__":
    main()
