import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    csv_file_path = os.path.join("csv", "xxmb - xuly.csv")
    if not os.path.exists(csv_file_path):
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu '{csv_file_path}'!")
        sys.exit(1)
        
    # Đọc dữ liệu
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"❌ Lỗi khi đọc file CSV: {e}")
        sys.exit(1)
        
    target_col = 'g1-extract'
    if target_col not in df.columns:
        print(f"❌ Lỗi: Cột '{target_col}' không tồn tại trong file CSV!")
        sys.exit(1)
        
    # Làm sạch dữ liệu và parse ngày tháng
    df_clean = df.dropna(subset=[target_col]).copy()
    df_clean[target_col] = df_clean[target_col].astype(int)
    
    try:
        df_clean['date_parsed'] = pd.to_datetime(df_clean['day'], format='%d-%m-%Y')
    except Exception:
        df_clean['date_parsed'] = pd.to_datetime(df_clean['day'])
        
    # Sắp xếp theo ngày tăng dần
    df_clean = df_clean.sort_values('date_parsed').reset_index(drop=True)
    
    # Lấy 60 ngày gần nhất
    df_last_60 = df_clean.tail(360).copy()
    
    if len(df_last_60) < 90:
        print(f"⚠️ Cảnh báo: Chỉ có {len(df_last_60)} ngày dữ liệu khả dụng (ít hơn 60 ngày). Sẽ hiển thị tất cả.")
    
    print(f"📊 Đang vẽ biểu đồ đường cột '{target_col}' cho {len(df_last_60)} ngày gần đây nhất...")
    print(f"  - Từ ngày: {df_last_60['day'].iloc[0]}")
    print(f"  - Đến ngày: {df_last_60['day'].iloc[-1]}")
    
    # Thiết lập giao diện Premium Dark Mode cho biểu đồ
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#11151e')
    
    # Vẽ đường biểu diễn chính
    days = df_last_60['day'].values
    values = df_last_60[target_col].values
    x_indices = np.arange(len(values))
    
    # Vẽ vùng phủ bóng (gradient-like fill) dưới đường
    ax.fill_between(x_indices, values, color='#1f85de', alpha=0.15)
    
    # Vẽ đường line chính với hiệu ứng mượt và màu neon xanh dương sang trọng
    line, = ax.plot(x_indices, values, color='#1f85de', linewidth=2.5, marker='o', 
                    markersize=6, markerfacecolor='#e57373', markeredgecolor='white', 
                    label='Số Đề (G1-Extract)')
    
    # Thống kê trung bình, lớn nhất, nhỏ nhất
    avg_val = np.mean(values)
    max_idx = np.argmax(values)
    min_idx = np.argmin(values)
    
    # Đường trung bình nằm ngang nét đứt
    ax.axhline(avg_val, color='#ffb74d', linestyle='--', linewidth=1.5, alpha=0.7, 
               label=f'Trung bình: {avg_val:.2f}')
    
    # Đánh dấu và chú thích điểm Max và Min
    ax.annotate(f'Cực đại: {values[max_idx]:02d}', 
                xy=(max_idx, values[max_idx]), 
                xytext=(max_idx, values[max_idx] + 5 if values[max_idx] < 92 else values[max_idx] - 7),
                arrowprops=dict(facecolor='#e57373', shrink=0.08, width=1.5, headwidth=6),
                color='#e57373', fontsize=10, fontweight='bold', ha='center')
                
    ax.annotate(f'Cực tiểu: {values[min_idx]:02d}', 
                xy=(min_idx, values[min_idx]), 
                xytext=(min_idx, values[min_idx] - 7 if values[min_idx] > 8 else values[min_idx] + 5),
                arrowprops=dict(facecolor='#81c784', shrink=0.08, width=1.5, headwidth=6),
                color='#81c784', fontsize=10, fontweight='bold', ha='center')
    
    # Thiết lập các nhãn trục và tiêu đề
    ax.set_title(f'BIỂU ĐỒ ĐƯỜNG DIỄN BIẾN {len(values)} KỲ QUAY GẦN NHẤT (CỘT G1-EXTRACT)', 
                 fontsize=15, fontweight='bold', pad=20, color='#1f85de')
    ax.set_xlabel('Ngày quay thưởng', fontsize=12, labelpad=12, color='#b0bec5')
    ax.set_ylabel('Giá trị số đề (00-99)', fontsize=12, labelpad=12, color='#b0bec5')
    
    # Cấu hình nhãn trục X: hiển thị cách quãng để không bị đè chữ
    step = max(1, len(values) // 10)
    ax.set_xticks(x_indices[::step])
    ax.set_xticklabels(days[::step], rotation=30, ha='right', fontsize=9, color='#cfd8dc')
    
    # Nhãn trục Y từ 00 đến 100
    ax.set_ylim(-5, 105)
    ax.set_yticks(range(0, 101, 10))
    ax.tick_params(colors='#cfd8dc', labelsize=10)
    
    # Thêm lưới nét đứt nhẹ nhàng
    ax.grid(True, linestyle='--', alpha=0.2, color='gray')
    
    # Bỏ viền hộp ngoài để biểu đồ thoáng hơn
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.legend(facecolor='#11151e', edgecolor='none', fontsize=10, loc='upper left')
    
    plt.tight_layout()
    
    # Lưu file đồ họa chất lượng cao
    output_dir = "csv"
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, "g1_extract_last_60_days.png")
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"💾 Đã lưu biểu đồ thành công tại: {output_image_path}")
    
    # Hiển thị biểu đồ ra màn hình
    plt.show()

if __name__ == "__main__":
    main()
