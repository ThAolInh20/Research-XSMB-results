# Nghiên cứu và Phân tích Định lượng Kết quả Xổ số Miền Bắc (XSMB)

## Mục tiêu
- Thực hiện phân tích định lượng và nghiên cứu học máy trên chuỗi kết quả xổ số miền Bắc (XSMB), đặc biệt tập trung vào chuỗi số giải đặc biệt.
- Kiểm thử và so sánh hiệu suất của các mô hình học sâu, mô hình xác suất và các thuật toán học máy để kiểm chứng tính ngẫu nhiên của dữ liệu.
- Đưa ra các bằng chứng định lượng khoa học để chứng minh rằng kết quả xổ số là hoàn toàn ngẫu nhiên, không có quy luật tuần hoàn có thể khai thác lâu dài về mặt kinh tế.

## Kiến trúc hệ thống và Các thành phần
Hệ thống được thiết kế mô-đun hóa, kết hợp giữa thu thập dữ liệu tự động và phân tích học máy nâng cao:

1. **Thu thập dữ liệu (Data Crawling):**
   - Sử dụng công cụ tự động hóa quy trình n8n chạy trên môi trường Docker để lấy kết quả xổ số tự động và đồng bộ hóa liên tục.
   - Dữ liệu sau khi lấy về được làm sạch và lưu trữ dưới dạng CSV trong thư mục `csv/` (tệp tin nguồn: `xxmb - xuly.csv`).

2. **Giao diện Dashboard Streamlit (Streamlit UI - main.py):**
   - Cung cấp một dashboard giao diện tối giản, sang trọng, cho phép người dùng tương tác trực quan với các kết quả dự báo thời gian thực.
   - Tích hợp công cụ backtest linh hoạt (backtest ngược về quá khứ bằng cách dùng dữ liệu trước ngày mục tiêu để huấn luyện) và dự báo tự hồi quy đa bước (Multi-step autoregressive forecasting) cho tương lai.
   - Biểu diễn xác suất dự báo dưới dạng biểu đồ trực quan hóa và liệt kê các thống kê về tần suất xuất hiện cũng như độ gan của các cặp số.

3. **Kiểm thử và So sánh Hiệu năng (Backtesting & Correlation CLI - correlation_analysis.py):**
   - Chạy kiểm thử trên toàn bộ tập kiểm thử (chiếm 20% dữ liệu cuối), đánh giá độ chính xác và tính toán các hệ số tương quan như hệ số Pearson và Spearman.
   - Tính toán sai số trung bình tuyệt đối (MAE), sai số bình phương trung bình (RMSE).
   - Thực hiện kiểm định độ calibration (mối quan hệ giữa độ tin cậy và tỷ lệ trúng thực tế) để đánh giá tính ổn định của mô hình.

## Các mô hình huấn luyện dự báo
Dự án thực nghiệm bốn cách tiếp cận khác nhau:

- **PyTorch LSTM Neural Network (lstm.py):** Mạng LSTM chạy trên PyTorch được xây dựng theo dạng đa nhiệm (multi-task training), đồng thời học chuỗi số giải nhất qua việc tách hàng chục và hàng đơn vị để dự báo phân bố xác suất tương quan. Mô hình sử dụng chuỗi time-step độ dài 7 để huấn luyện và trích xuất các đặc trưng nhất quán.
- **Bayes Markov Chain Model (markov_predict.py):** Xây dựng ma trận xác suất chuyển trạng thái động, đánh giá tỷ lệ chuyển từ số này sang số khác qua các vòng quay trước đó.
- **GBDT / LightGBM Classifier (lgbm_predict.py):** Sử dụng các đặc trưng trễ (lag features) gồm chuỗi số thực tế trước đó, các đặc trưng chữ số hàng chục, hàng đơn vị và tổng chữ số để huấn luyện cây quyết định phân loại.
- **Consensus/Ensemble Model (Mô hình Đồng thuận):** Lấy trung bình xác suất dự báo của cả ba phương pháp LSTM, Markov Chain và LightGBM để giảm thiểu sai số cá biệt và nâng cao độ ổn định của kết quả.

## Hướng dẫn huấn luyện và Chạy ứng dụng

### Yêu cầu môi trường
Cài đặt các thư viện phụ thuộc cần thiết cho môi trường Python:
```bash
pip install torch pandas numpy lightgbm scikit-learn matplotlib seaborn streamlit
```

### Thu thập dữ liệu
Khởi động n8n trên docker compose để chạy quy trình crawl:
```bash
docker compose up -d
```
Sử dụng file kịch bản để tự động cấu hình tunnel nếu chạy trên máy cá nhân:
- Trên Windows: Chạy file `start.ps1` bằng PowerShell.
- Trên Linux/MacOS: Chạy file `start.sh` bằng Bash.

### Chạy các mô hình độc lập và CLI Backtest
Chạy các kịch bản python để kiểm tra từng mô hình đơn lẻ:
- Chạy mạng nơ-ron LSTM:
```bash
python lstm.py
```
- Chạy mô hình chuỗi Markov:
```bash
python markov_predict.py
```
- Chạy mô hình LightGBM:
```bash
python lgbm_predict.py
```
- Thực hiện backtest hàng loạt kiểm tra sai số và hệ số tương quan:
```bash
python correlation_analysis.py
```

### Khởi chạy Giao diện Streamlit Web App
Chạy lệnh sau để khởi động dashboard tương tác:
```bash
streamlit run main.py
```

## Kết luận và Đóng góp Khoa học
- Hệ thống được huấn luyện trên tập dữ liệu thực tế trong nhiều năm. Các biểu đồ tương quan và backtest thống kê (lưu tại `csv/ensemble_correlation_report.png`) cho thấy:
  - Độ chính xác Top-1 luôn dao động xung quanh mức 1%.
  - Độ chính xác Top-5 luôn dao động xung quanh mức 5%.
  - Độ chính xác Top-10 luôn dao động xung quanh mức 10%.
- Hệ số tương quan tuyến tính Pearson (R) và Spearman (rho) giữa giá trị thực tế và giá trị dự đoán hội tụ về gần 0. Các sai số MAE và RMSE luôn ở mức lớn và phân bố đồng đều trên dải số từ 00-99.
- Tất cả các chỉ số này đều khớp với kỳ vọng toán học của một biến ngẫu nhiên phân bố đều.
- **Kết luận khoa học:** Kết quả xổ số miền Bắc là hoàn toàn ngẫu nhiên và độc lập qua từng kỳ quay. Mặc dù các mô hình học máy có thể tìm thấy các tín hiệu nhiễu cục bộ hoặc tối ưu hóa trong ngắn hạn, không tồn tại bất kỳ mô hình toán học nào có thể vượt qua được xác suất ngẫu nhiên tự nhiên khi dự báo dài hạn.