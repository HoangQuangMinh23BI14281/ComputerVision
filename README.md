# SROIE OCR - Pure PyTorch Pipeline (DBNet++ & SVTR-Tiny)

Dự án này là một quy trình nhận dạng chữ trong ảnh (Optical Character Recognition - OCR) hoàn chỉnh, được xây dựng 100% bằng **PyTorch nguyên bản (Pure PyTorch)**, không phụ thuộc vào các framework bọc sẵn nặng nề. Dự án được tối ưu hóa đặc biệt cho mục đích chạy trên hệ điều hành Windows, loại bỏ triệt để các lỗi rò rỉ bộ nhớ (Memory Leak) và thắt cổ chai luồng (Thread Contention) khi huấn luyện.

## 🏗 Kiến trúc Model
Pipeline được chia làm 2 giai đoạn chính:

1. **Detection (Phát hiện vùng chữ): DBNet++**. 
   - Backbone được thu gọn và dùng hàm Loss chuyên dụng (`DBLoss` hỗ trợ OHEM) để cắt ra các bounding box đa giác chứa chữ một cách cực kỳ chính xác.
2. **Recognition (Nhận dạng ký tự): SVTR-Tiny**.
   - Phiên bản SVTR đã được tinh gọn cấu trúc Transformer (chỉ còn ~4 Triệu tham số) với các chiều `embed_dims=[64, 128, 256]`. Điều này mang lại tốc độ inference và train cực nhanh, tiêu hao bộ nhớ thấp nhưng vẫn giữ nguyên sức mạnh của cơ chế Local-Global Mixing.

## 🚀 Tính năng nổi bật
- **Script tự động hóa (`run.bat`)**: Chỉ cần click đúp chuột để chạy từ A-Z mọi quy trình (Cài đặt, Chuẩn bị dữ liệu, Huấn luyện, Inference) trên Windows.
- **Tối ưu hóa RAM/VRAM cực mạnh**: Không xảy ra hiện tượng chậm dần qua từng Epoch nhờ khóa luồng OpenCV (`num_threads=0`) và Main Thread DataLoader trên Windows.
- **Nhận dạng thông minh**: Cơ chế Inference ngẫu nhiên (chỉ cần Enter là lấy 1 ảnh test bất kỳ để quét) hoặc nhận dạng chủ động một bức ảnh chỉ định.

## 📂 Hướng dẫn sử dụng

### 1. Chuẩn bị Môi trường và Dữ liệu
Hãy clone dự án về máy và nhấp đúp vào file `run.bat` tại thư mục gốc. Bạn sẽ thấy 6 tùy chọn. 
- **Bấm phím 1**: Để hệ thống tự động tải về Python 3.12 (môi trường ảo), CUDA 12.1, PyTorch, và OpenCV.
- **Bấm phím 2**: Để chuyển đổi data thô thành cấu trúc thư mục phù hợp cho việc train.

> **Lưu ý**: Dữ liệu ảnh cần nằm trong thư mục `data/Stage1train/` (hoặc sửa đổi đường dẫn trong `src/config.py`). Trọng số gốc (nếu có) nằm trong `weights/`. Bọn mình đã dùng `.gitignore` chặn đẩy ảnh và model nặng lên GitHub.

### 2. Quá trình Huấn luyện (Training)
Mô hình hỗ trợ tự động resume (chạy tiếp) nếu quá trình train bị gián đoạn.

- **Bấm phím 3**: Huấn luyện mô hình DBNet (Detection). Hệ thống sẽ trích xuất `thresh_map` và đo lường tỷ lệ F1-Score tự động.
- **Bấm phím 4**: Huấn luyện mô hình SVTR (Recognition). Cấu trúc ảnh nhận dạng là `32x320` và CTC Loss được kiểm soát chặt để xóa lỗi inf.

*Các tệp trọng số `best.pth` và `latest.pth` sẽ được lưu trong `weights/dbnet/` và `weights/svtr/` tương ứng.*

### 3. Kiểm thử (Inference)
- **Bấm phím 5**: Để chạy nhận dạng 1 bức ảnh.
  - Bạn có thể **để trống và nhấn Enter** để hệ thống bốc ngẫu nhiên một bức ảnh trong tập Test.
  - Hoặc dán đường dẫn trực tiếp (vd: `data/Stage1train/X00016469612.jpg`) để nhận dạng ảnh mong muốn.
  
Hệ thống sẽ vẽ Box lên ảnh và hiển thị chữ ra terminal cùng mức độ tự tin (Confidence). Ảnh kết quả vẽ bounding box nằm tại `inference_result.jpg`.
