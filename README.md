# SROIE OCR - Pure PyTorch Pipeline (EAST & CRNN)

Dự án này là một quy trình nhận dạng chữ trong ảnh (Optical Character Recognition - OCR) hoàn chỉnh, được xây dựng 100% bằng **PyTorch nguyên bản (Pure PyTorch)**. Phiên bản hiện tại đã được nâng cấp lên kiến trúc **EAST** cho phần phát hiện và **CRNN** cho phần nhận dạng, mang lại độ ổn định và hiệu suất cao.

## 🏗 Kiến trúc Model
Pipeline được chia làm 2 giai đoạn chính:

1. **Detection (Phát hiện vùng chữ): EAST (Efficient and Accurate Scene Text detector)**. 
   - Sử dụng backbone **VGG16** (BN) được tối ưu hóa.
   - Đầu ra dự đoán trực tiếp các bản đồ điểm số (Score Map) và bản đồ hình học (Geometry Map - RBOX).
   - Hỗ trợ khôi phục đa giác chữ chính xác ngay cả với chữ bị nghiêng.

2. **Recognition (Nhận dạng ký tự): CRNN (Convolutional Recurrent Neural Network)**.
   - Kết hợp giữa mạng tích chập (CNN) để trích xuất đặc trưng và Bi-LSTM để học trình tự ký tự.
   - Sử dụng hàm Loss **CTC (Connectionist Temporal Classification)** giúp nhận dạng các từ có độ dài biến thiên mà không cần gán nhãn từng ký tự.

## 🚀 Tính năng nổi bật
- **Tối ưu hóa huấn luyện (Performance Boost)**:
    - **AMP (Automatic Mixed Precision)**: Sử dụng FP16 giúp huấn luyện nhanh hơn 2-3 lần và tiết kiệm VRAM.
    - **Multi-worker DataLoader**: Nạp dữ liệu song song, loại bỏ hiện tượng "đói dữ liệu" của GPU.
- **Script tự động hóa (`run.sh`)**: Hỗ trợ chạy trên WSL/Ubuntu để đạt hiệu năng tốt nhất.
- **Hệ thống Metrics chuyên sâu**: 
    - EAST: Precision, Recall, F1-Score, IoU, FPS.
    - CRNN: Word Accuracy, Character Accuracy, NED, Inference Time.
- **Logging tự động**: Lưu log chi tiết ra tệp CSV sau mỗi Epoch để theo dõi quá trình hội tụ.

## 📂 Hướng dẫn sử dụng

### 1. Chuẩn bị Môi trường và Dữ liệu
Nếu bạn dùng Linux hoặc WSL:
```bash
chmod +x run.sh
./run.sh
```
- **Chọn 1**: Để tự động khởi tạo môi trường ảo `.venv` và cài đặt thư viện (`torch`, `torchvision`, `opencv`, `shapely`, `pandas`,...).
- **Chọn 2**: Chuẩn bị dữ liệu (Prepare Dataset).

> **Lưu ý**: Dữ liệu ảnh SROIE cần được đặt trong `data/Stage1train/` và `data/Stage2train/`. Quá trình chuẩn bị dữ liệu sẽ tạo ra các tệp chỉ mục trong `ocr_dataset/`.

### 2. Huấn luyện (Training)
Mô hình sẽ tự động lưu trọng số tốt nhất (`best.pth`) và nhật ký huấn luyện (`log.csv`) vào thư mục `weights/`.

- **Huấn luyện EAST**: `python -m src.train_east` (hoặc chọn option 3 trong `run.sh`).
- **Huấn luyện CRNN**: `python -m src.train_crnn` (hoặc chọn option 4 trong `run.sh`).

### 3. Kiểm thử (Inference)
Sử dụng script `inference.py` để chạy thử trên ảnh thực tế:
```bash
python inference.py --image "đường/dẫn/đến/ảnh.jpg"
```
Mặc định, nếu không truyền `--image`, hệ thống sẽ chọn ngẫu nhiên một ảnh trong tập test để dự đoán. Kết quả trực quan sẽ được lưu tại `inference_result.jpg`.
