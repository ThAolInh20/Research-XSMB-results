import os
import sys
import pandas as pd
import numpy as np

# Thử import streamlit. Nếu chưa có, hướng dẫn người dùng cài đặt
try:
    import streamlit as st
except ImportError:
    print("=" * 80)
    print("THIẾU THƯ VIỆN STREAMLIT!")
    print("Giao diện người dùng yêu cầu thư viện 'streamlit'.")
    print("Vui lòng cài đặt và chạy bằng hai lệnh sau:")
    print("  pip install streamlit")
    print("  streamlit run main.py")
    print("=" * 80)
    sys.exit(1)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Import các hàm dự đoán từ module đồng thuận (ensemble)
from ensemble import predict_gbdt_for_date, predict_lstm_for_date, predict_ensemble_for_date

# Thiết lập giao diện Streamlit sang trọng
st.set_page_config(
    page_title="AI Lottery Prediction System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy chỉnh CSS tạo cảm giác Premium, hiện đại (Dark mode glassmorphism)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1f85de 0%, #0d47a1 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(31, 133, 222, 0.4);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .result-badge-win-top1 {
        background-color: #2e7d32;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        box-shadow: 0 0 10px rgba(46, 125, 50, 0.5);
    }
    .result-badge-win-top5 {
        background-color: #1b5e20;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .result-badge-win-top10 {
        background-color: #f57f17;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .result-badge-lose {
        background-color: #c62828;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# CÁC HÀM DỰ ĐOÁN CACHED & OPTIMIZED
# =====================================================================
def predict_autoregressive(df_clean, target_date, model_choice):
    """
    Dự báo tự hồi quy cuốn chiếu (Autoregressive Forecasting) cho các ngày trong tương lai.
    """
    latest_date = df_clean['date_parsed'].max()
    
    # Import mô hình Markov động để tránh lỗi import vòng lặp
    try:
        from markov_predict import predict_markov_for_date
    except ImportError:
        st.error("Không tìm thấy tệp 'markov_predict.py' trong thư mục gốc.")
        st.stop()
    
    # Nếu ngày mục tiêu là ngày tiếp theo (1 ngày trong tương lai) hoặc trong quá khứ
    if target_date <= latest_date + pd.Timedelta(days=1):
        if "LSTM" in model_choice:
            return predict_lstm_for_date(df_clean, target_date)
        elif "Markov" in model_choice:
            return predict_markov_for_date(df_clean, target_date, alpha=0.15)
        elif "Ensemble" in model_choice:
            return predict_ensemble_for_date(df_clean, target_date, alpha=0.15)
        else:
            return predict_gbdt_for_date(df_clean, target_date)
            
    # Nếu xa hơn 1 ngày trong tương lai (K > 1)
    df_temp = df_clean.copy()
    current_date = latest_date + pd.Timedelta(days=1)
    
    probs = None
    while current_date <= target_date:
        if "LSTM" in model_choice:
            probs = predict_lstm_for_date(df_temp, current_date)
        elif "Markov" in model_choice:
            probs = predict_markov_for_date(df_temp, current_date, alpha=0.15)
        elif "Ensemble" in model_choice:
            probs = predict_ensemble_for_date(df_temp, current_date, alpha=0.15)
        else:
            probs = predict_gbdt_for_date(df_temp, current_date)
            
        # Tìm số có xác suất cao nhất (Top-1) để làm kết quả giả lập cho ngày này
        top1_num = np.argmax(probs)
        
        # Thêm dòng giả lập mới vào df_temp
        new_row = {
            'day': current_date.strftime('%d-%m-%Y'),
            'g1-extract': top1_num,
            'date_parsed': current_date
        }
        df_temp = pd.concat([df_temp, pd.DataFrame([new_row])], ignore_index=True)
        current_date += pd.Timedelta(days=1)
        
    return probs

# =====================================================================
# GIAO DIỆN CHÍNH (STREAMLIT APP)
# =====================================================================

def main():
    # Tiêu đề giao diện
    st.markdown("<h1 style='text-align: center; color: #1f85de; margin-bottom: 2rem;'>🔮 HỆ THỐNG TRỰC QUAN AI XSMB</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #b0bec5; font-weight: normal; margin-top: -1.5rem;'>Phân tích định lượng & So sánh mô hình dự báo thời gian thực</h4>", unsafe_allow_html=True)
    st.write("---")

    # Đọc dữ liệu từ file CSV
    csv_file_path = os.path.join("csv", "xxmb - xuly.csv")
    if not os.path.exists(csv_file_path):
        st.error(f"Không tìm thấy file dữ liệu '{csv_file_path}'!")
        st.info("Vui lòng chạy file crawl dữ liệu trước để tạo file CSV.")
        st.stop()
        
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        st.error(f"Lỗi khi đọc file CSV: {e}")
        st.stop()
        
    target_col = 'g1-extract'
    if target_col not in df.columns:
        st.error(f"Cột dữ liệu '{target_col}' không tồn tại trong file CSV!")
        st.stop()
        
    # Làm sạch dữ liệu ban đầu
    df_clean = df.dropna(subset=[target_col]).copy()
    df_clean[target_col] = df_clean[target_col].astype(int)
    try:
        df_clean['date_parsed'] = pd.to_datetime(df_clean['day'], format='%d-%m-%Y')
    except Exception:
        df_clean['date_parsed'] = pd.to_datetime(df_clean['day'])
        
    df_clean = df_clean.sort_values('date_parsed').reset_index(drop=True)
    
    # -----------------------------------------------------------------
    # SIDEBAR: ĐIỀU KHIỂN & LỰA CHỌN
    # -----------------------------------------------------------------
    st.sidebar.markdown("<h2 style='color: #1f85de;'>Cấu hình hệ thống</h2>", unsafe_allow_html=True)
    st.sidebar.write("---")
    
    # 1. Chọn Mô hình
    model_choice = st.sidebar.selectbox(
        "🧠 Lựa chọn Mô hình dự báo",
        [
            "LightGBM / GBDT Classifier (Cây Quyết Định)",
            "PyTorch LSTM Neural Network (Mạng học sâu)",
            "Markov Chain Transition Model (Chuỗi Markov Chu kỳ)",
            "🔮 Mô hình Đồng Thuận / Ensemble (Kết hợp cả 3)"
        ]
    )
    
    # 2. Chọn/Nhập Ngày cần dự đoán (Hỗ trợ nhập ngày trong tương lai hoặc bất kỳ ngày nào trong quá khứ!)
    min_date = df_clean['date_parsed'].min() + pd.Timedelta(days=40)
    max_date = df_clean['date_parsed'].max() + pd.Timedelta(days=30) # Cho phép chọn tương lai tối đa 30 ngày
    
    selected_date_raw = st.sidebar.date_input(
        "📅 Chọn hoặc Nhập ngày cần dự đoán",
        value=df_clean['date_parsed'].max() + pd.Timedelta(days=1), # Mặc định chọn ngày tiếp theo (tương lai gần nhất)
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    
    selected_date = pd.to_datetime(selected_date_raw)
    selected_date_str = selected_date.strftime('%d-%m-%Y')
    
    latest_date = df_clean['date_parsed'].max()
    if selected_date in df_clean['date_parsed'].values:
        actual_val = df_clean[df_clean['date_parsed'] == selected_date][target_col].values[0]
    else:
        actual_val = None
    
    st.sidebar.write("---")
    st.sidebar.markdown("""
    ### 💡 Hướng dẫn sử dụng:
    1. Chọn mô hình bạn muốn dùng ở bên trên.
    2. Chọn hoặc **nhập bất kỳ ngày nào** bằng lịch tương tác.
    3. Nếu chọn ngày trong quá khứ, hệ thống sẽ tự động thực hiện **Backtest** (chỉ dùng dữ liệu trước ngày đó để học và đối chiếu kết quả thực tế).
    4. Nếu chọn ngày ở tương lai, hệ thống sẽ tự động kích hoạt chế độ **Dự báo tự hồi quy đa bước (Multi-step autoregressive forecasting)**!
    """)
    
    # -----------------------------------------------------------------
    # MAIN PANEL: HIỂN THỊ KẾT QUẢ DỰ BÁO
    # -----------------------------------------------------------------
    
    # Thứ trong tiếng Việt
    days_vi = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
    selected_day_of_week = selected_date.dayofweek
    
    # 3 Cột thống kê nhanh ở trên cùng
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h5 style='color: #b0bec5; margin: 0;'>📅 NGÀY ĐANG CHỌN DỰ ĐOÁN</h5>
            <h2 style='color: #1f85de; margin-top: 0.5rem;'>{selected_date_str}</h2>
            <p style='color: #81c784; margin: 0; font-weight: bold;'>({days_vi[selected_day_of_week]})</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Xác định thông tin hiển thị mô hình rút gọn
    if "LSTM" in model_choice:
        model_name_short = "LSTM (Deep Learning)"
        model_desc = "Tiền xử lý đa đặc trưng"
    elif "Markov" in model_choice:
        model_name_short = "Markov Chain Model"
        model_desc = "Chuỗi chuyển dịch đa chu kỳ"
    elif "Ensemble" in model_choice:
        model_name_short = "Ensemble Model"
        model_desc = "Trung bình xác suất 3 thuật toán"
    else:
        model_name_short = "GBDT (LightGBM)"
        model_desc = "Tối ưu chu kỳ lồng quay"

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h5 style='color: #b0bec5; margin: 0;'>🤖 MÔ HÌNH ĐANG SỬ DỤNG</h5>
            <h3 style='color: #ffb74d; margin-top: 0.8rem;'>{model_name_short}</h3>
            <p style='color: #90a4ae; margin: 0;'>{model_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        if actual_val is None:
            st.markdown(f"""
            <div class='metric-card'>
                <h5 style='color: #b0bec5; margin: 0;'>🎯 KẾT QUẢ THỰC TẾ GIẢI NHẤT</h5>
                <h2 style='color: #90a4ae; margin-top: 0.5rem;'>Chưa có</h2>
                <p style='color: #ffb74d; margin: 0; font-weight: bold;'>Đang chờ mở thưởng...</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='metric-card'>
                <h5 style='color: #b0bec5; margin: 0;'>🎯 KẾT QUẢ THỰC TẾ GIẢI NHẤT</h5>
                <h2 style='color: #e57373; margin-top: 0.5rem;'>{actual_val:02d}</h2>
                <p style='color: #90a4ae; margin: 0;'>Hai số cuối giải nhất</p>
            </div>
            """, unsafe_allow_html=True)
        
    st.write(" ")
    
    # Tiến hành tính toán dự đoán khi giao diện tải hoặc khi bấm
    with st.spinner("🧠 Đang tiền xử lý dữ liệu và huấn luyện mô hình thời gian thực..."):
        probs = predict_autoregressive(df_clean, selected_date, model_choice)
            
    # Lấy Top 10 số có xác suất cao nhất
    top_10_indices = np.argsort(probs)[::-1][:10]
    top_5_indices = top_10_indices[:5]
    
    # -----------------------------------------------------------------
    # PHẦN 2: HIỂN THỊ KẾT QUẢ CHI TIẾT
    # -----------------------------------------------------------------
    st.markdown("<h3 style='color: #1f85de;'>🔮 Kết quả phân tích và Gợi ý của AI</h3>", unsafe_allow_html=True)
    
    subcol1, subcol2 = st.columns([1, 1])
    
    with subcol1:
        st.subheader("⭐ Top 5 Cặp số gợi ý tốt nhất:")
        for idx, num in enumerate(top_5_indices, 1):
            prob_percent = probs[num] * 100
            
            # Đánh dấu nếu số này chính là số thực tế đã về
            is_actual = (actual_val is not None and num == actual_val)
            border_style = "border: 2px solid #2e7d32;" if is_actual else ""
            bg_color = "rgba(46, 125, 50, 0.15)" if is_actual else "rgba(255, 255, 255, 0.02)"
            
            st.markdown(f"""
            <div style='background: {bg_color}; padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; {border_style} display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <span style='color: #ffb74d; font-weight: bold; margin-right: 1rem;'>Gợi ý số {idx}:</span>
                    <span style='font-size: 1.5rem; font-weight: bold; color: #ffffff;'>Cặp số: <span style='color: #1f85de;'>{num:02d}</span></span>
                    {" <span style='color: #81c784; font-weight: bold; margin-left: 1rem;'>[KẾT QUẢ ĐÚNG]</span>" if is_actual else ""}
                </div>
                <div style='text-align: right;'>
                    <span style='font-size: 1.1rem; color: #81c784; font-weight: bold;'>Xác suất: {prob_percent:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Kiểm tra trạng thái trúng giải
        st.write(" ")
        st.subheader("🎯 Đánh giá hiệu suất:")
        
        if actual_val is None:
            k_days = (selected_date - latest_date).days
            if k_days > 1:
                st.markdown(f"<div class='result-badge-win-top1' style='background-color: #0288d1; box-shadow: 0 0 10px rgba(2, 136, 209, 0.5);'>⏳ DỰ BÁO TƯƠNG LAI XA: Dự báo tự hồi quy cuốn chiếu {k_days} ngày tiếp theo từ ngày mới nhất!</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-badge-win-top1' style='background-color: #0288d1; box-shadow: 0 0 10px rgba(2, 136, 209, 0.5);'>⏳ DỰ BÁO TƯƠNG LAI GẦN: Kết quả của kỳ quay tiếp theo sẽ được đối chiếu khi có dữ liệu thực tế!</div>", unsafe_allow_html=True)
        elif actual_val == top_10_indices[0]:
            st.markdown("<div class='result-badge-win-top1'>🎉 XUẤT SẮC: Đã đoán trúng chính xác tuyệt đối ngay tại gợi ý số 1! (Top-1 Match)</div>", unsafe_allow_html=True)
        elif actual_val in top_5_indices:
            st.markdown("<div class='result-badge-win-top5'>✅ THÀNH CÔNG: Số thực tế nằm trong Top-5 Cặp số gợi ý tốt nhất! (Top-5 Match)</div>", unsafe_allow_html=True)
        elif actual_val in top_10_indices:
            st.markdown("<div class='result-badge-win-top10'>⚠️ KHÁ TỐT: Số thực tế nằm trong Top-10 Cặp số gợi ý tốt nhất! (Top-10 Match)</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-badge-lose'>❌ TRƯỢT: Kỳ này mô hình chưa đưa ra kết quả trúng trong nhóm Top-10.</div>", unsafe_allow_html=True)
            
    with subcol2:
        st.subheader("📊 Biểu đồ xác suất của Top 10 Cặp số:")
        
        # Vẽ biểu đồ ngang phân bố xác suất
        fig, ax = plt.subplots(figsize=(8, 5.5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1b212c')
        
        labels = [f"{num:02d}" for num in top_10_indices]
        values = [probs[num] * 100 for num in top_10_indices]
        
        # Đánh dấu cột đúng bằng màu xanh lá, cột thường bằng màu xanh dương
        colors = ['#2ca02c' if (actual_val is not None and idx == actual_val) else '#1f85de' for idx in top_10_indices]
        
        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor='black', alpha=0.8)
        
        # Thêm text giá trị % xác suất lên đầu các cột
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                    va='center', ha='left', color='#ffffff', fontsize=10, fontweight='bold')
            
        ax.set_title("Xác suất dự báo (%)", color='#ffffff', fontsize=12, fontweight='bold')
        ax.tick_params(colors='#ffffff', labelsize=10)
        ax.grid(True, axis='x', color='gray', linestyle='--', alpha=0.3)
        
        # Bỏ viền xung quanh biểu đồ
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        st.pyplot(fig)

    # -----------------------------------------------------------------
    # PHẦN 3: PHÂN TÍCH THỐNG KÊ LỊCH SỬ TÍNH ĐẾN THỜI ĐIỂM ĐÓ
    # -----------------------------------------------------------------
    st.write("---")
    st.markdown(f"<h3 style='color: #1f85de;'>📊 Phân tích Thống kê lịch sử (Tính đến trước ngày {selected_date_str})</h3>", unsafe_allow_html=True)
    
    # Hàm thống kê lịch sử
    def analyze_statistics(df_hist, target_col):
        series = df_hist[target_col].values
        total_draws = len(series)
        
        freqs = np.zeros(100, dtype=int)
        for num in series:
            if 0 <= num < 100:
                freqs[num] += 1
                
        gaps = np.zeros(100, dtype=int)
        for num in range(100):
            indices = np.where(series == num)[0]
            if len(indices) == 0:
                gaps[num] = total_draws
            else:
                gaps[num] = total_draws - 1 - indices[-1]
                
        return freqs, gaps

    # Lấy dữ liệu trước ngày mục tiêu để thống kê
    df_hist_stats = df_clean[df_clean['date_parsed'] < selected_date].copy()
    freqs, gaps = analyze_statistics(df_hist_stats, target_col)
    
    stat_col1, stat_col2 = st.columns(2)
    
    with stat_col1:
        st.markdown("<p style='font-size: 1.1rem; font-weight: bold; color: #ffb74d;'>🔥 Cặp số về nhiều nhất (Hottest):</p>", unsafe_allow_html=True)
        freq_series = pd.Series(freqs)
        top_freq = freq_series.nlargest(5)
        
        data_freq = []
        for num, count in top_freq.items():
            percentage = (count / len(df_hist_stats)) * 100
            data_freq.append({"Cặp số": f"{num:02d}", "Tần suất (lần)": count, "Tỷ lệ (%)": f"{percentage:.2f}%"})
        st.table(pd.DataFrame(data_freq))
        
    with stat_col2:
        st.markdown("<p style='font-size: 1.1rem; font-weight: bold; color: #81c784;'>⏳ Cặp số đang khan lâu nhất (Gan hiện tại):</p>", unsafe_allow_html=True)
        gap_series = pd.Series(gaps)
        top_gap = gap_series.nlargest(5)
        
        data_gap = []
        for num, days in top_gap.items():
            data_gap.append({"Cặp số": f"{num:02d}", "Số ngày chưa về": days, "Trạng thái": "Cực kỳ khan"})
        st.table(pd.DataFrame(data_gap))

if __name__ == "__main__":
    main()
