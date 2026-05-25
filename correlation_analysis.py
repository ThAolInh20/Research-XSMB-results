import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập seed để đảm bảo kết quả nhất quán
np.random.seed(42)
torch.manual_seed(42)

# Nhập các thành phần mô hình từ các tệp hiện tại
try:
    from lstm import LotteryFeatureDataset, LotteryMultiTaskLSTM
    from markov_predict import predict_markov_for_date
    from lgbm_predict import preprocess_and_feature_engineering
except ImportError as e:
    print(f"❌ Lỗi import: Không tìm thấy các file script cần thiết (lstm.py, markov_predict.py, lgbm_predict.py).")
    print(f"   Chi tiết: {e}")
    sys.exit(1)

# Phòng thủ: Kiểm tra thư viện LightGBM
try:
    import lightgbm as lgb
    USE_LGB = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    USE_LGB = False

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

# Các hàm tính toán toán học độc lập
def calculate_pearson(x, y):
    mean_x, mean_y = np.mean(x), np.mean(y)
    diff_x, diff_y = x - mean_x, y - mean_y
    num = np.sum(diff_x * diff_y)
    den = np.sqrt(np.sum(diff_x**2) * np.sum(diff_y**2))
    if den == 0:
        return 0.0
    return num / den

def calculate_spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return calculate_pearson(rx, ry)

def evaluate_accuracy_metrics(probs, y_true):
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    for i in range(len(y_true)):
        target = y_true[i]
        top_10 = np.argsort(probs[i])[::-1][:10]
        top_5 = top_10[:5]
        if target == top_10[0]:
            correct_top1 += 1
        if target in top_5:
            correct_top5 += 1
        if target in top_10:
            correct_top10 += 1
    n = len(y_true)
    return (correct_top1 / n) * 100, (correct_top5 / n) * 100, (correct_top10 / n) * 100

def main():
    print(f"{Colors.HEADER}{Colors.BOLD}========================================================================={Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}      HỆ THỐNG PHÂN TÍCH TƯƠNG QUAN & BACKTEST MÔ HÌNH ĐỒNG THUẬN (CLI){Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}========================================================================={Colors.ENDC}")
    
    csv_file_path = os.path.join("csv", "xxmb - xuly.csv")
    if not os.path.exists(csv_file_path):
        print(f"{Colors.FAIL}❌ Không tìm thấy file dữ liệu '{csv_file_path}'. Vui lòng crawl dữ liệu trước!{Colors.ENDC}")
        sys.exit(1)
        
    print(f"📖 Đang đọc dữ liệu từ: {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    
    target_col = 'g1-extract'
    # Tiền xử lý dữ liệu và tạo đặc trưng GBDT
    df_features, feature_cols, df_clean = preprocess_and_feature_engineering(df, target_col)
    
    # Chia dữ liệu theo thứ tự thời gian (80% Train, 20% Test)
    total_samples = len(df_features)
    train_size = int(total_samples * 0.80)
    test_size = total_samples - train_size
    
    df_train = df_features.iloc[:train_size].copy()
    df_test = df_features.iloc[train_size:].copy()
    
    print(f"\n{Colors.CYAN}📊 KẾT QUẢ PHÂN CHIA DỮ LIỆU:{Colors.ENDC}")
    print(f"  - Tổng số ngày quay: {total_samples}")
    print(f"  - Tập huấn luyện (Train - 80%): {train_size} ngày")
    print(f"  - Tập kiểm thử (Test - 20%): {test_size} ngày (Từ {df_test['day'].iloc[0]} đến {df_test['day'].iloc[-1]})")
    print("-" * 50)
    
    # -------------------------------------------------------------------------
    # 1. HUẤN LUYỆN & DỰ BÁO BATCH MÔ HÌNH GBDT
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}1. Đang huấn luyện mô hình GBDT Classifier...{Colors.ENDC}")
    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values
    
    if USE_LGB:
        gbdt_model = lgb.LGBMClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
    else:
        gbdt_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
    gbdt_model.fit(X_train, y_train)
    proba_gbdt_raw = gbdt_model.predict_proba(X_test)
    
    # Tái tạo xác suất đầy đủ cho 100 lớp
    gbdt_probs = np.zeros((len(X_test), 100))
    classes_seen = gbdt_model.classes_
    for idx, c in enumerate(classes_seen):
        gbdt_probs[:, c] = proba_gbdt_raw[:, idx]
    print(f"  {Colors.GREEN}✓ Mô hình GBDT đã hoàn thành batch prediction trên tập Test.{Colors.ENDC}")
    
    # -------------------------------------------------------------------------
    # 2. HUẤN LUYỆN & DỰ BÁO BATCH MÔ HÌNH LSTM
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}2. Đang huấn luyện mô hình PyTorch LSTM (35 Epochs)...{Colors.ENDC}")
    raw_seq = df_clean[target_col].values
    tens_seq = raw_seq // 10
    units_seq = raw_seq % 10
    sum_seq = (tens_seq + units_seq) % 10
    eo_seq = (tens_seq % 2) * 2 + (units_seq % 2)
    bs_seq = (np.where(tens_seq >= 5, 1, 0)) * 2 + (np.where(units_seq >= 5, 1, 0))
    data_list = [raw_seq, tens_seq, units_seq, sum_seq, eo_seq, bs_seq]
    
    # Ở đây chúng ta sử dụng cùng bộ dữ liệu đã được xử lý qua lùi lag tương thích
    # Vì sequence_length = 7 nên độ dài là len(raw_seq) - 7, tương đương len(df_features)
    dataset = LotteryFeatureDataset(data_list, sequence_length=7)
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    lstm_model = LotteryMultiTaskLSTM(hidden_dim=32, num_layers=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.003, weight_decay=1e-3)
    
    lstm_model.train()
    for epoch in range(1, 36):
        epoch_loss = 0.0
        for x_b, y_tens_b, y_units_b, _ in train_loader:
            optimizer.zero_grad()
            logits_tens, logits_units = lstm_model(x_b)
            loss = criterion(logits_tens, y_tens_b) + criterion(logits_units, y_units_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_b.size(0)
        
        if epoch % 10 == 0 or epoch == 1:
            avg_loss = epoch_loss / len(train_dataset)
            print(f"   - Epoch {epoch:02d}/35 | Train Loss: {avg_loss:.4f}")
            
    # Dự đoán batch trên tập test
    lstm_model.eval()
    lstm_probs = []
    with torch.no_grad():
        for x_batch, _, _, _ in test_loader:
            logits_tens, logits_units = lstm_model(x_batch)
            prob_tens = torch.softmax(logits_tens, dim=1)
            prob_units = torch.softmax(logits_units, dim=1)
            joint_prob = torch.bmm(prob_tens.unsqueeze(2), prob_units.unsqueeze(1))
            joint_prob = joint_prob.view(-1, 100).numpy()
            lstm_probs.append(joint_prob)
            
    lstm_probs = np.concatenate(lstm_probs, axis=0)
    print(f"  {Colors.GREEN}✓ Mô hình LSTM đã hoàn thành batch prediction trên tập Test.{Colors.ENDC}")
    
    # -------------------------------------------------------------------------
    # 3. DỰ BÁO BATCH MÔ HÌNH BAYES MARKOV
    # -------------------------------------------------------------------------
    print(f"\n{Colors.BOLD}3. Đang tính toán dự báo cuốn chiếu mô hình Bayes Markov...{Colors.ENDC}")
    markov_probs = []
    for idx in range(len(df_test)):
        test_date = df_test.iloc[idx]['date_parsed']
        probs = predict_markov_for_date(df_clean, test_date, target_col, alpha=0.15)
        markov_probs.append(probs)
    markov_probs = np.array(markov_probs)
    print(f"  {Colors.GREEN}✓ Mô hình Bayes Markov đã hoàn thành prediction trên tập Test.{Colors.ENDC}")
    
    # -------------------------------------------------------------------------
    # 4. KẾT HỢP DỰ ĐOÁN ĐỒNG THUẬN (ENSEMBLE)
    # -------------------------------------------------------------------------
    ensemble_probs = (gbdt_probs + lstm_probs + markov_probs) / 3.0
    print(f"\n{Colors.GREEN}✓ Đã tạo thành công mô hình Đồng Thuận / Ensemble.{Colors.ENDC}")
    
    # -------------------------------------------------------------------------
    # 5. TÍNH TOÁN SAI SỐ VÀ TƯƠNG QUAN
    # -------------------------------------------------------------------------
    y_pred = np.argmax(ensemble_probs, axis=1)
    
    # Tính tương quan Pearson & Spearman
    pearson_r = calculate_pearson(y_test, y_pred)
    spearman_rho = calculate_spearman(y_test, y_pred)
    
    # Tính sai số số học
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    
    # Tính toán Accuracy
    acc_gbdt1, acc_gbdt5, acc_gbdt10 = evaluate_accuracy_metrics(gbdt_probs, y_test)
    acc_lstm1, acc_lstm5, acc_lstm10 = evaluate_accuracy_metrics(lstm_probs, y_test)
    acc_markov1, acc_markov5, acc_markov10 = evaluate_accuracy_metrics(markov_probs, y_test)
    acc_ens1, acc_ens5, acc_ens10 = evaluate_accuracy_metrics(ensemble_probs, y_test)
    
    # -------------------------------------------------------------------------
    # IN BÁO CÁO KẾT QUẢ RA TERMINAL
    # -------------------------------------------------------------------------
    print(f"\n{Colors.CYAN}{Colors.BOLD}========================================================================={Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}                    BÁO CÁO PHÂN TÍCH HIỆU SUẤT MÔ HÌNH{Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}========================================================================={Colors.ENDC}")
    
    # In bảng so sánh độ chính xác
    print(f"{Colors.BOLD}{'Thuật toán / Mô hình':<30} | {'Top-1 Acc':<10} | {'Top-5 Acc':<10} | {'Top-10 Acc':<10}{Colors.ENDC}")
    print("-" * 75)
    print(f"{'1. GBDT (LightGBM/RF)':<30} | {acc_gbdt1:>8.2f}% | {acc_gbdt5:>8.2f}% | {acc_gbdt10:>8.2f}%")
    print(f"{'2. PyTorch LSTM':<30} | {acc_lstm1:>8.2f}% | {acc_lstm5:>8.2f}% | {acc_lstm10:>8.2f}%")
    print(f"{'3. Bayes Markov Chain':<30} | {acc_markov1:>8.2f}% | {acc_markov5:>8.2f}% | {acc_markov10:>8.2f}%")
    print(f"{Colors.GREEN}{Colors.BOLD}{'4. MÔ HÌNH ĐỒNG THUẬN (ENSEMBLE)':<30} | {acc_ens1:>8.2f}% | {acc_ens5:>8.2f}% | {acc_ens10:>8.2f}%{Colors.ENDC}")
    print(f"{Colors.WARNING}{'5. Đoán mò Ngẫu nhiên (Random)':<30} | {1.00:>8.2f}% | {5.00:>8.2f}% | {10.00:>8.2f}%{Colors.ENDC}")
    print("-" * 75)
    
    # In thông tin tương quan và sai số của mô hình đồng thuận
    print(f"\n{Colors.CYAN}{Colors.BOLD}📊 CHỈ SỐ TƯƠNG QUAN & SAI SỐ CỦA MÔ HÌNH ĐỒNG THUẬN (ENSEMBLE):{Colors.ENDC}")
    
    # Quy đổi màu sắc dựa trên chất lượng hệ số tương quan
    color_p = Colors.GREEN if abs(pearson_r) > 0.05 else Colors.WARNING
    color_s = Colors.GREEN if abs(spearman_rho) > 0.05 else Colors.WARNING
    
    print(f"  - {Colors.BOLD}Hệ số Tương quan Pearson (R):{Colors.ENDC} {color_p}{pearson_r:+.4f}{Colors.ENDC}")
    print(f"  - {Colors.BOLD}Hệ số Tương quan Thứ bậc Spearman (rho):{Colors.ENDC} {color_s}{spearman_rho:+.4f}{Colors.ENDC}")
    print(f"  - {Colors.BOLD}Sai số Trung bình Tuyệt đối (MAE):{Colors.ENDC} {Colors.WARNING}{mae:.2f} số{Colors.ENDC} (Độ lệch trung bình từ 0-99)")
    print(f"  - {Colors.BOLD}Sai số Bình phương Trung bình (RMSE):{Colors.ENDC} {Colors.WARNING}{rmse:.2f} số{Colors.ENDC}")
    
    # -------------------------------------------------------------------------
    # PHÂN TÍCH ĐỘ CALIBRATION (TƯƠNG QUAN ĐỘ TIN CẬY - ĐỘ CHÍNH XÁC)
    # -------------------------------------------------------------------------
    print(f"\n{Colors.CYAN}{Colors.BOLD}📊 PHÂN TÍCH ĐỘ HIỆU CHỈNH ĐỘ TIN CẬY (PROBABILITY-ACCURACY CALIBRATION):{Colors.ENDC}")
    confidences = np.max(ensemble_probs, axis=1)
    
    # Chia test set thành 5 nhóm có xác suất từ thấp đến cao
    sorted_indices = np.argsort(confidences)
    bins = np.array_split(sorted_indices, 5)
    
    bin_confs = []
    bin_hit_rates = []
    
    print(f"  {Colors.BOLD}{'Nhóm Độ Tin Cậy':<18} | {'Khoảng Xác suất (%)':<22} | {'Tỷ lệ Trúng Top-10':<20}{Colors.ENDC}")
    print("  " + "-" * 70)
    
    for i, bin_idx in enumerate(bins):
        b_conf = confidences[bin_idx]
        b_y_true = y_test[bin_idx]
        b_probs = ensemble_probs[bin_idx]
        
        avg_c = np.mean(b_conf) * 100
        min_c = np.min(b_conf) * 100
        max_c = np.max(b_conf) * 100
        
        # Tính tỷ lệ trúng Top-10
        hits = 0
        for j in range(len(b_y_true)):
            target = b_y_true[j]
            top_10 = np.argsort(b_probs[j])[::-1][:10]
            if target in top_10:
                hits += 1
        hit_rate = (hits / len(b_y_true)) * 100
        
        bin_confs.append(avg_c)
        bin_hit_rates.append(hit_rate)
        
        print(f"  Nhóm {i+1:<13} | {min_c:>5.2f}% - {max_c:>5.2f}% (tb: {avg_c:>5.2f}%) | {Colors.GREEN}{hit_rate:>8.2f}%{Colors.ENDC}")
        
    calib_correlation = calculate_pearson(bin_confs, bin_hit_rates)
    color_cal = Colors.GREEN if calib_correlation > 0.3 else (Colors.WARNING if calib_correlation > 0.0 else Colors.FAIL)
    print("  " + "-" * 70)
    print(f"  👉 {Colors.BOLD}Tương quan giữa Độ tin cậy (Confidence) và Tỷ lệ trúng thực tế:{Colors.ENDC} {color_cal}{calib_correlation:+.4f}{Colors.ENDC}")
    
    # -------------------------------------------------------------------------
    # GIẢI THÍCH Ý NGHĨA KHOA HỌC CHO NGƯỜI DÙNG
    # -------------------------------------------------------------------------
    print(f"\n{Colors.CYAN}{Colors.BOLD}💡 ĐÁNH GIÁ VÀ GIẢI THÍCH KHOA HỌC:{Colors.ENDC}")
    if calib_correlation > 0.4:
        print(f"  {Colors.GREEN}✓ HỆ THỐNG PHÁT TRIỂN CỰC KỲ KHỎE MẠNH!{Colors.ENDC}")
        print(f"    Mối tương quan dương rất mạnh ({calib_correlation:+.4f}) chứng tỏ mô hình đồng thuận có khả năng tự nhận biết")
        print(f"    vực sâu hoặc cơ hội. Khi mô hình dự đoán có xác suất cao, tỉ lệ trúng thực tế tăng vọt!")
    elif calib_correlation > 0.0:
        print(f"  {Colors.WARNING}⚠ HỆ THỐNG HOẠT ĐỘNG KHÁ ỔN ĐỊNH.{Colors.ENDC}")
        print(f"    Có mối tương quan dương nhẹ ({calib_correlation:+.4f}) giữa xác suất dự báo và tỷ lệ trúng thực tế.")
        print(f"    Mô hình có sự hiệu chuẩn khá tốt nhưng vẫn còn bị ảnh hưởng bởi nhiễu ngẫu nhiên.")
    else:
        print(f"  {Colors.FAIL}⚠ CẢNH BÁO TÍN HIỆU NHIỄU CAO.{Colors.ENDC}")
        print(f"    Hệ số tương quan âm hoặc gần bằng 0 ({calib_correlation:+.4f}) cho thấy độ tin cậy được tính")
        print(f"    chưa khớp chuẩn với xác suất thực tế. Đây là bản chất của dữ liệu xổ số có tính hỗn loạn cao.")
        
    print(f"\n  - Tương quan tuyến tính Pearson của giải đề đạt {pearson_r:+.4f}.")
    print(f"  - Sai số MAE {mae:.2f} số nghĩa là trung bình dự báo trượt khoảng {mae:.1f} đơn vị số học so với kết quả đúng.")
    
    # -------------------------------------------------------------------------
    # VẼ VÀ LƯU BIỂU ĐỒ BÁO CÁO ĐẸP MẮT
    # -------------------------------------------------------------------------
    try:
        print(f"\n🎨 Đang vẽ biểu đồ phân tích và lưu báo cáo đồ họa...")
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle('BÁO CÁO PHÂN TÍCH TƯƠNG QUAN & BACKTEST MÔ HÌNH ĐỒNG THUẬN XSMB', fontsize=16, fontweight='bold', color='#1f85de')
        
        # Biểu đồ 1: So sánh Độ chính xác các mô hình
        ax1 = axes[0, 0]
        models = ['GBDT', 'LSTM', 'Markov', 'Consensus (Ensemble)', 'Random']
        top1_accs = [acc_gbdt1, acc_lstm1, acc_markov1, acc_ens1, 1.0]
        top10_accs = [acc_gbdt10, acc_lstm10, acc_markov10, acc_ens10, 10.0]
        
        x = np.arange(len(models))
        ax1.bar(x - 0.2, top1_accs, 0.4, label='Top-1 Accuracy', color='#1f85de', alpha=0.85)
        ax1.bar(x + 0.2, top10_accs, 0.4, label='Top-10 Accuracy', color='#2ca02c', alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=15)
        ax1.set_ylabel('Độ chính xác (%)')
        ax1.set_title('So sánh độ chính xác giữa các mô hình (%)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Biểu đồ 2: Scatter plot thực tế vs dự đoán kèm đường hồi quy xu hướng
        ax2 = axes[0, 1]
        sns.regplot(x=y_test, y=y_pred, ax=ax2, 
                    scatter_kws={'alpha': 0.6, 'color': '#ff7f0e'}, 
                    line_kws={'color': '#1f85de', 'linewidth': 2})
        ax2.set_xlabel('Giá trị Thực tế (Actual)')
        ax2.set_ylabel('Giá trị Dự đoán (Predicted)')
        ax2.set_xlim(-5, 105)
        ax2.set_ylim(-5, 105)
        ax2.set_title(f'Tương quan Số học (Pearson R = {pearson_r:+.4f})', fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Biểu đồ 3: So sánh xu hướng 30 ngày gần đây
        ax3 = axes[1, 0]
        n_days_plot = min(30, len(y_test))
        ax3.plot(y_test[-n_days_plot:], label='Thực tế (Actual)', color='#e57373', marker='o', linewidth=2)
        ax3.plot(y_pred[-n_days_plot:], label='Dự đoán (Predicted)', color='#1f85de', marker='x', linestyle='--', linewidth=1.5)
        ax3.set_xlabel(f'{n_days_plot} kỳ quay kiểm thử gần nhất')
        ax3.set_ylabel('Giá trị số đề (00-99)')
        ax3.set_title(f'Đường xu hướng Actual vs Predicted ({n_days_plot} ngày cuối)', fontweight='bold')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.3)
        
        # Biểu đồ 4: Hiệu chuẩn độ tin cậy (Calibration Curve)
        ax4 = axes[1, 1]
        ax4.plot(bin_confs, bin_hit_rates, marker='o', color='#2ca02c', linewidth=2.5, markersize=8, label='Calibration')
        ax4.plot([np.min(bin_confs), np.max(bin_confs)], [np.min(bin_hit_rates), np.max(bin_hit_rates)], 
                 color='gray', linestyle='--', alpha=0.7, label='Lý tưởng (Perfect)')
        ax4.set_xlabel('Độ tin cậy trung bình của dự báo (%)')
        ax4.set_ylabel('Tỉ lệ trúng thực tế Top-10 (%)')
        ax4.set_title(f'Độ tin cậy vs Tỉ lệ trúng (Calibration R = {calib_correlation:+.4f})', fontweight='bold')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        os.makedirs("csv", exist_ok=True)
        report_image_path = os.path.join("csv", "ensemble_correlation_report.png")
        plt.savefig(report_image_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{Colors.GREEN}✓ Đã vẽ và lưu biểu đồ báo cáo trực quan tại: {Colors.BOLD}{report_image_path}{Colors.ENDC}")
    except Exception as ex:
        print(f"{Colors.WARNING}⚠ Lỗi khi vẽ biểu đồ báo cáo: {ex}{Colors.ENDC}")
        
    print(f"\n{Colors.HEADER}{Colors.BOLD}========================================================================={Colors.ENDC}")
    print(f"{Colors.GREEN}{Colors.BOLD}          CHƯƠNG TRÌNH PHÂN TÍCH TƯƠNG QUAN ĐÃ HOÀN THÀNH MỸ MÃN!{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}========================================================================={Colors.ENDC}\n")

if __name__ == "__main__":
    main()
