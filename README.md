# Bài 11 - Lọc ảnh trong miền tần số bằng DFT/FFT

Thư mục này chứa chương trình Python hoàn chỉnh cho bài thực hành xử lý ảnh trong miền tần số. Chương trình đọc ảnh `anh-vinh-ha-long-63.jpg`, tính DFT/FFT, hiển thị phổ tần số, tạo bộ lọc Gaussian Highpass Filter và so sánh FFT tự viết với các hàm có sẵn của NumPy/OpenCV.

## 1. Nội dung trong thư mục

```text
bai_11_vinh_ha_long/
├── bai_11.py
├── anh-vinh-ha-long-63.jpg
├── README.md
└── output/
```

Ý nghĩa:

- `bai_11.py`: file chương trình chính, chạy trực tiếp để tạo toàn bộ kết quả.
- `anh-vinh-ha-long-63.jpg`: ảnh đầu vào mặc định.
- `README.md`: file hướng dẫn sử dụng.
- `output/`: thư mục chứa kết quả sau khi chạy chương trình.

## 2. Chương trình dùng gì để làm?

Chương trình sử dụng:

- `Python`: ngôn ngữ lập trình chính.
- `NumPy`: tính FFT/IFFT và xử lý ma trận ảnh.
- `Matplotlib`: hiển thị và lưu các hình tổng hợp.
- `Pillow`: đọc ảnh grayscale và lưu ảnh kết quả.
- `OpenCV`: tính DFT bằng `cv2.dft` để so sánh với FFT tự viết.

Các nội dung xử lý ảnh được thực hiện:

- Đọc ảnh grayscale.
- Tính DFT trực tiếp cho ảnh nhỏ `32x32`.
- Tự cài đặt FFT 1D bằng thuật toán Cooley-Tukey.
- Mở rộng FFT 1D thành FFT 2D.
- So sánh Manual DFT, Manual FFT, NumPy `fft2` và OpenCV `dft`.
- Hiển thị magnitude spectrum theo thang log.
- Dùng `fftshift` để đưa tần số thấp về tâm ảnh.
- Tạo Gaussian Highpass Filter với `D0 = 5`.
- Lọc ảnh trong miền tần số.
- Dùng `ifftshift` và `ifft2` để đưa ảnh về miền không gian.

## 3. Clone về chạy như thế nào?

Sau khi clone hoặc tải project về máy, đi vào thư mục bài:

```bash
cd XU_LY_ANH/bai_11_vinh_ha_long
```

Nếu chưa có môi trường ảo Python, tạo mới:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo:

Trên Linux/macOS:

```bash
source .venv/bin/activate
```

Trên Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Cài các thư viện cần thiết:

```bash
python -m pip install numpy matplotlib pillow opencv-python
```

Chạy chương trình:

```bash
python bai_11.py
```

Nếu đang dùng virtual environment nằm ở thư mục cha giống máy hiện tại, có thể chạy:

```bash
../.venv/bin/python bai_11.py
```

## 4. Ảnh đầu vào của chương trình

Ảnh đầu vào mặc định:

```text
anh-vinh-ha-long-63.jpg
```

Trong code, đường dẫn ảnh được cấu hình tại biến:

```python
PREFERRED_IMAGE_PATH = Path("anh-vinh-ha-long-63.jpg")
```

Nếu muốn đổi sang ảnh khác, đặt ảnh mới vào cùng thư mục với `bai_11.py` và sửa biến trên thành tên file ảnh mới.

Ví dụ:

```python
PREFERRED_IMAGE_PATH = Path("ten_anh_moi.jpg")
```

## 5. Các output được tạo ra

Sau khi chạy, chương trình tạo hoặc cập nhật thư mục:

```text
output/
```

Bên trong có các file:

```text
output/bai_11_dft_fft_comparison.png
output/bai_11_frequency_domain_summary.png
output/bai_11_01_original.png
output/bai_11_02_fft_spectrum.png
output/bai_11_03_gaussian_highpass_filter.png
output/bai_11_04_filtered_spectrum.png
output/bai_11_05_ifft_result.png
```

Ý nghĩa từng file:

- `bai_11_dft_fft_comparison.png`: hình tổng hợp so sánh Manual DFT, Manual FFT, NumPy FFT2 và OpenCV DFT. Hình này có 12 khung ảnh và bảng sai số/thời gian ở phía dưới.
- `bai_11_frequency_domain_summary.png`: hình tổng hợp quy trình lọc ảnh trong miền tần số bằng Gaussian Highpass Filter.
- `bai_11_01_original.png`: ảnh gốc sau khi đọc và chuyển sang grayscale.
- `bai_11_02_fft_spectrum.png`: phổ FFT của ảnh theo thang log.
- `bai_11_03_gaussian_highpass_filter.png`: mặt nạ lọc Gaussian Highpass với `D0 = 5`.
- `bai_11_04_filtered_spectrum.png`: phổ ảnh sau khi nhân với bộ lọc.
- `bai_11_05_ifft_result.png`: ảnh kết quả sau khi biến đổi ngược IFFT về miền không gian.

## 6. Giải thích nhanh kết quả

Trong hình `bai_11_dft_fft_comparison.png`:

- `Original Grayscale Image`: ảnh đầu vào dạng mức xám.
- `DFT Input 32x32`: ảnh resize nhỏ để tính DFT trực tiếp.
- `FFT Input 32x32`: ảnh resize nhỏ để tính FFT tự viết.
- `Reconstructed from DFT`: ảnh khôi phục lại bằng IDFT.
- `DFT Magnitude`: phổ biên độ DFT trước khi shift.
- `Centered DFT Magnitude`: phổ biên độ sau `fftshift`.
- `Centered DFT Phase`: phổ pha sau `fftshift`.
- `DFT Difference`: sai khác giữa ảnh gốc `32x32` và ảnh khôi phục.
- `Manual FFT Spectrum`: phổ FFT từ thuật toán tự viết.
- `NumPy FFT2 on FFT Input`: phổ FFT tính bằng NumPy.
- `OpenCV DFT Spectrum`: phổ DFT tính bằng OpenCV.
- `FFT Difference`: sai khác giữa FFT tự viết và NumPy FFT.

Bảng phía dưới cho biết:

- thời gian tính bằng code tự viết,
- thời gian tính bằng hàm thư viện,
- sai số trung bình,
- sai số lớn nhất.

Sai số rất nhỏ là bình thường vì máy tính dùng số thực dấu phẩy động. Thứ tự cộng/nhân số phức trong code tự viết khác với NumPy/OpenCV nên kết quả không trùng tuyệt đối từng chữ số, nhưng nếu sai số rất nhỏ thì thuật toán là đúng.

## 7. Lỗi thường gặp

### Lỗi thiếu thư viện

Nếu gặp lỗi dạng:

```text
ModuleNotFoundError: No module named 'numpy'
```

hoặc:

```text
ModuleNotFoundError: No module named 'cv2'
```

hãy cài lại thư viện:

```bash
python -m pip install numpy matplotlib pillow opencv-python
```

### Lỗi không mở được cửa sổ hình

Nếu chương trình chạy xong nhưng không bật cửa sổ hiển thị, hãy mở trực tiếp file trong thư mục `output/`:

```text
output/bai_11_dft_fft_comparison.png
output/bai_11_frequency_domain_summary.png
```

### Lỗi không tìm thấy ảnh

Đảm bảo file ảnh đầu vào nằm cùng thư mục với `bai_11.py`:

```text
anh-vinh-ha-long-63.jpg
```

Nếu đổi tên ảnh, cần sửa lại biến `PREFERRED_IMAGE_PATH` trong `bai_11.py`.

## 8. Lệnh chạy nhanh

Nếu đã cài đủ thư viện:

```bash
cd bai_11_vinh_ha_long
python bai_11.py
```

Nếu dùng virtual environment ở thư mục cha:

```bash
cd bai_11_vinh_ha_long
../.venv/bin/python bai_11.py
```

