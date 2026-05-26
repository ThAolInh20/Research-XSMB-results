import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Thiết lập seed để đảm bảo kết quả nhất quán
np.random.seed(42)
torch.manual_seed(42)

# Màu sắc để in ra terminal đẹp mắt
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# =====================================================================
# CÁC MÔ HÌNH DỰ ĐOÁN THÀNH PHẦN
# =====================================================================

def predict_gbdt_for_date(df_clean, target_date):
    """
    Dự báo bằng mô hình Cây Quyết định trên dữ liệu lịch sử trước target_date
    """
    # Lọc dữ liệu lịch sử trước ngày mục tiêu
    df_hist = df_clean[df_clean['date_parsed'] < target_date].copy()
    
    # Trích xuất đặc trưng bảng cho dữ liệu lịch sử
    df_hist['day_of_week'] = df_hist['date_parsed'].dt.dayofweek
    for i in range(1, 8):
        df_hist[f'lag_{i}'] = df_hist['g1-extract'].shift(i)
        df_hist[f'lag_{i}_tens'] = df_hist[f'lag_{i}'] // 10
        df_hist[f'lag_{i}_units'] = df_hist[f'lag_{i}'] % 10
        df_hist[f'lag_{i}_sum'] = (df_hist[f'lag_{i}_tens'] + df_hist[f'lag_{i}_units']) % 10
        
    df_train = df_hist.dropna().copy()
    
    feature_cols = ['day_of_week']
    for i in range(1, 8):
        feature_cols.extend([f'lag_{i}', f'lag_{i}_tens', f'lag_{i}_units', f'lag_{i}_sum'])
        
    X_train = df_train[feature_cols].values
    y_train = df_train['g1-extract'].values
    
    # Khởi tạo mô hình
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=45, 
            learning_rate=0.05, 
            max_depth=4, 
            num_leaves=15, 
            random_state=42, 
            verbosity=-1,
            n_jobs=-1
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=50, 
            max_depth=5, 
            random_state=42, 
            n_jobs=-1
        )
        
    model.fit(X_train, y_train)
    
    # Chuẩn bị input của target_date dựa trên ngày cuối cùng của lịch sử
    last_row = df_hist.iloc[-1]
    input_dict = {'day_of_week': target_date.dayofweek}
    input_dict['lag_1'] = last_row['g1-extract']
    input_dict['lag_1_tens'] = input_dict['lag_1'] // 10
    input_dict['lag_1_units'] = input_dict['lag_1'] % 10
    input_dict['lag_1_sum'] = (input_dict['lag_1_tens'] + input_dict['lag_1_units']) % 10
    
    for i in range(2, 8):
        input_dict[f'lag_{i}'] = last_row[f'lag_{i-1}']
        input_dict[f'lag_{i}_tens'] = last_row[f'lag_{i-1}_tens']
        input_dict[f'lag_{i}_units'] = last_row[f'lag_{i-1}_units']
        input_dict[f'lag_{i}_sum'] = last_row[f'lag_{i-1}_sum']
        
    df_input = pd.DataFrame([input_dict])
    X_input = df_input[feature_cols].values
    
    # Dự báo
    proba_raw = model.predict_proba(X_input)[0]
    classes_seen = model.classes_
    
    probs_full = np.zeros(100)
    for idx, c in enumerate(classes_seen):
        probs_full[c] = proba_raw[idx]
        
    return probs_full


def predict_lstm_for_date(df_clean, target_date):
    """
    Huấn luyện nhanh và dự báo bằng mô hình LSTM đa nhiệm trên dữ liệu trước target_date
    """
    df_hist = df_clean[df_clean['date_parsed'] < target_date].copy()
    raw_seq = df_hist['g1-extract'].values
    
    # Tiền xử lý đa đặc trưng
    tens_seq = raw_seq // 10
    units_seq = raw_seq % 10
    sum_seq = (tens_seq + units_seq) % 10
    eo_seq = (tens_seq % 2) * 2 + (units_seq % 2)
    bs_seq = (np.where(tens_seq >= 5, 1, 0)) * 2 + (np.where(units_seq >= 5, 1, 0))
    
    data_list = [raw_seq, tens_seq, units_seq, sum_seq, eo_seq, bs_seq]
    
    # Nhập các thành phần Dataset và Model từ lstm.py để đảm bảo tính đồng bộ
    try:
        from lstm import LotteryFeatureDataset, LotteryMultiTaskLSTM
    except ImportError:
        print(f"{Colors.FAIL}❌ Lỗi: Không thể import cấu hình Dataset/LSTM từ lstm.py!{Colors.ENDC}")
        sys.exit(1)
        
    seq_len = 7
    dataset = LotteryFeatureDataset(data_list, seq_len)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = LotteryMultiTaskLSTM(hidden_dim=32, num_layers=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    
    # Huấn luyện nhanh 20 epochs (chỉ mất ~0.6s trên CPU do tập nhỏ)
    model.train()
    for epoch in range(20):
        for x_b, y_tens_b, y_units_b, _ in loader:
            optimizer.zero_grad()
            logits_t, logits_u = model(x_b)
            loss = criterion(logits_t, y_tens_b) + criterion(logits_u, y_units_b)
            loss.backward()
            optimizer.step()
            
    # Dự báo cho ngày mục tiêu
    model.eval()
    x_raw = raw_seq[-seq_len:]
    x_tens = tens_seq[-seq_len:]
    x_units = units_seq[-seq_len:]
    x_sum = sum_seq[-seq_len:]
    x_eo = eo_seq[-seq_len:]
    x_bs = bs_seq[-seq_len:]
    
    t_raw = torch.tensor(x_raw, dtype=torch.long)
    t_tens = torch.tensor(x_tens, dtype=torch.long)
    t_units = torch.tensor(x_units, dtype=torch.long)
    t_sum = torch.tensor(x_sum, dtype=torch.long)
    t_eo = torch.tensor(x_eo, dtype=torch.long)
    t_bs = torch.tensor(x_bs, dtype=torch.long)
    
    x = torch.stack([t_raw, t_tens, t_units, t_sum, t_eo, t_bs], dim=1)
    input_tensor = x.unsqueeze(0) # [1, seq_len, 6]
    
    with torch.no_grad():
        logits_tens, logits_units = model(input_tensor)
        prob_tens = torch.softmax(logits_tens, dim=1).numpy()[0]
        prob_units = torch.softmax(logits_units, dim=1).numpy()[0]
        
    probs_full = np.outer(prob_tens, prob_units).flatten()
    return probs_full


# =====================================================================
# MÔ HÌNH ĐỒNG THUẬN / ENSEMBLE CONSENSUS
# =====================================================================

def predict_ensemble_for_date(df_clean, target_date, alpha=0.15):
    """
    Mô hình Đồng thuận: Lấy trung bình xác suất dự báo của cả ba phương pháp:
    LSTM, Markov Chain và LightGBM (GBDT).
    """
    try:
        from markov_predict import predict_markov_for_date
    except ImportError:
        print(f"{Colors.FAIL}❌ Lỗi: Không thể import predict_markov_for_date từ markov_predict.py!{Colors.ENDC}")
        sys.exit(1)

    p_gbdt = predict_gbdt_for_date(df_clean, target_date)
    p_lstm = predict_lstm_for_date(df_clean, target_date)
    p_markov = predict_markov_for_date(df_clean, target_date, alpha=alpha)

    # Trung bình cộng xác suất dự báo của 3 mô hình độc lập
    probs_ensemble = (p_gbdt + p_lstm + p_markov) / 3.0
    return probs_ensemble


# =====================================================================
# CLI CHẠY ĐỘC LẬP
# =====================================================================

def main():
    print(f"{Colors.HEADER}{Colors.BOLD}============================================================= {Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}    🔮 HỆ THỐNG DỰ BÁO ĐỒNG THUẬN AI - XSMB (ENSEMBLE CLI) 🔮{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}============================================================= {Colors.ENDC}")

    csv_file_path = os.path.join("csv", "xxmb - xuly.csv")
    if not os.path.exists(csv_file_path):
        print(f"{Colors.FAIL}❌ Lỗi: Không tìm thấy tệp dữ liệu tại '{csv_file_path}'. Vui lòng crawl trước!{Colors.ENDC}")
        sys.exit(1)

    print(f"📖 Đang tải tập dữ liệu lịch sử...")
    df = pd.read_csv(csv_file_path)
    
    target_col = 'g1-extract'
    if target_col not in df.columns:
        print(f"{Colors.FAIL}❌ Lỗi: Cột '{target_col}' không tồn tại trong tệp CSV!{Colors.ENDC}")
        sys.exit(1)

    # Làm sạch dữ liệu ban đầu
    df_clean = df.dropna(subset=[target_col]).copy()
    df_clean[target_col] = df_clean[target_col].astype(int)
    try:
        df_clean['date_parsed'] = pd.to_datetime(df_clean['day'], format='%d-%m-%Y')
    except Exception:
        df_clean['date_parsed'] = pd.to_datetime(df_clean['day'])
        
    df_clean = df_clean.sort_values('date_parsed').reset_index(drop=True)
    
    # Xác định ngày dự đoán kế tiếp
    last_date = df_clean['date_parsed'].max()
    next_date = last_date + pd.Timedelta(days=1)
    
    days_vi = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    next_day_str = next_date.strftime('%d-%m-%Y')
    next_day_of_week = days_vi[next_date.dayofweek]

    print(f"📊 Dữ liệu mới nhất ngày: {Colors.CYAN}{last_date.strftime('%d-%m-%Y')}{Colors.ENDC}")
    print(f"🔮 Tiến hành chạy dự báo cho ngày tiếp theo: {Colors.GREEN}{Colors.BOLD}{next_day_str} ({next_day_of_week}){Colors.ENDC}")
    print("-" * 60)

    # 1. Dự đoán bằng LightGBM/GBDT
    print(f"🤖 1/3 Đang chạy dự đoán Cây Quyết định (LightGBM/GBDT)...", end="", flush=True)
    p_gbdt = predict_gbdt_for_date(df_clean, next_date)
    print(f" {Colors.GREEN}✓ Hoàn tất.{Colors.ENDC}")

    # 2. Dự đoán bằng LSTM
    print(f"🧠 2/3 Đang huấn luyện và chạy Mạng học sâu (PyTorch LSTM)...", end="", flush=True)
    p_lstm = predict_lstm_for_date(df_clean, next_date)
    print(f" {Colors.GREEN}✓ Hoàn tất.{Colors.ENDC}")

    # 3. Dự đoán bằng Markov
    print(f"📈 3/3 Đang tính toán ma trận chuyển tiếp (Bayes Markov)...", end="", flush=True)
    try:
        from markov_predict import predict_markov_for_date
        p_markov = predict_markov_for_date(df_clean, next_date, alpha=0.15)
        print(f" {Colors.GREEN}✓ Hoàn tất.{Colors.ENDC}")
    except ImportError:
        print(f"\n{Colors.FAIL}❌ Lỗi: Không thể chạy Markov!{Colors.ENDC}")
        sys.exit(1)

    # 4. Tính toán Đồng Thuận (Ensemble)
    print(f"🔮 Đang tổng hợp kết quả đồng thuận (Ensemble)...", end="", flush=True)
    p_ensemble = (p_gbdt + p_lstm + p_markov) / 3.0
    print(f" {Colors.GREEN}✓ Hoàn tất.{Colors.ENDC}")

    # =====================================================================
    # IN KẾT QUẢ DỰ BÁO CHI TIẾT
    # =====================================================================
    
    # Lấy Top 5 cho từng mô hình
    top5_gbdt = np.argsort(p_gbdt)[::-1][:5]
    top5_lstm = np.argsort(p_lstm)[::-1][:5]
    top5_markov = np.argsort(p_markov)[::-1][:5]
    top5_ensemble = np.argsort(p_ensemble)[::-1][:5]

    print("\n" + "=" * 60)
    print(f"          {Colors.BOLD}BẢNG SO SÁNH GỢI Ý TOP 5 CỦA CÁC MÔ HÌNH{Colors.ENDC}")
    print("=" * 60)
    print(f"{Colors.BOLD}{'Hạng':<6} | {'LightGBM (GBDT)':<18} | {'PyTorch LSTM':<18} | {'Bayes Markov':<18}{Colors.ENDC}")
    print("-" * 65)
    for i in range(5):
        num_gbdt, val_gbdt = top5_gbdt[i], p_gbdt[top5_gbdt[i]] * 100
        num_lstm, val_lstm = top5_lstm[i], p_lstm[top5_lstm[i]] * 100
        num_markov, val_markov = top5_markov[i], p_markov[top5_markov[i]] * 100
        print(f"Top {i+1:<2} | {num_gbdt:02d} ({val_gbdt:>5.2f}%)       | {num_lstm:02d} ({val_lstm:>5.2f}%)       | {num_markov:02d} ({val_markov:>5.2f}%)")
    print("-" * 65)

    print("\n" + "=" * 60)
    print(f"       🔮 {Colors.GREEN}{Colors.BOLD}KẾT QUẢ ĐỒNG THUẬN / ENSEMBLE KHUYẾN NGHỊ CUỐI CÙNG{Colors.ENDC} 🔮")
    print("=" * 60)
    for i in range(5):
        num_ens, val_ens = top5_ensemble[i], p_ensemble[top5_ensemble[i]] * 100
        print(f"  {Colors.BOLD}Gợi ý {i+1}: Cặp số [{Colors.BLUE}{num_ens:02d}{Colors.ENDC}] với độ tin cậy đồng thuận: {Colors.GREEN}{val_ens:.2f}%{Colors.ENDC}")
    print("=" * 60)

    print(f"\n{Colors.CYAN}💡 Lời khuyên khoa học:{Colors.ENDC}")
    print("  - Mô hình Đồng Thuận giúp triệt tiêu các sai số cá biệt của từng mô hình đơn lẻ.")
    print("  - Bản chất kết quả XSMB có entropy cực lớn; sự kết hợp này mang tính thực nghiệm tối ưu")
    print("    xác suất dựa trên dữ liệu lịch sử.")
    print(f"{Colors.HEADER}============================================================= {Colors.ENDC}\n")

if __name__ == "__main__":
    main()
