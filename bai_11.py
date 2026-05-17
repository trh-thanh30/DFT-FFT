"""
Bai thuc hanh xu ly anh trong mien tan so.

Yeu cau:
1. Doc anh grayscale.
2. Tinh FFT bang numpy va hien thi magnitude spectrum.
3. Tao Gaussian Highpass Filter voi D0 = 5.
4. Loc anh trong mien tan so.
5. IFFT ve mien khong gian.
6. Tu cai dat FFT 1D/2D bang Cooley-Tukey.
7. So sanh FFT tu viet voi numpy.fft.fft2.

Chay:
    python bai_11.py
"""

from pathlib import Path
import shutil
import subprocess
from time import perf_counter

try:
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
except ImportError as exc:
    missing_name = exc.name
    print(f"Thieu thu vien: {missing_name}")
    print("Cai dat bang lenh:")
    print("    python -m pip install numpy matplotlib pillow")
    raise SystemExit(1)


IMAGE_DIR = Path("image")
OUTPUT_DIR = Path("output")
PREFERRED_IMAGE_PATH = Path("anh-vinh-ha-long-63.jpg")

D0 = 5
MAX_FFT_SIZE = 256
COMPARISON_SIZE = 32


def find_input_image(image_dir):
    """Tim anh dau tien trong thu muc image/."""
    if PREFERRED_IMAGE_PATH.exists():
        return PREFERRED_IMAGE_PATH

    image_extensions = ("*.bmp", "*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.webp")
    image_paths = []

    for extension in image_extensions:
        image_paths.extend(image_dir.glob(extension))

    if not image_paths:
        raise FileNotFoundError(f"Khong tim thay anh trong thu muc: {image_dir}")

    return sorted(image_paths)[0]


def previous_power_of_two(value):
    """Lay luy thua cua 2 lon nhat khong vuot qua value."""
    power = 1
    while power * 2 <= value:
        power *= 2
    return power


def load_grayscale_pil(image_path):
    """Doc anh grayscale tu file dau vao."""
    return Image.open(image_path).convert("L")


def read_grayscale_image(image_path, max_size=MAX_FFT_SIZE):
    """
    Doc anh grayscale va resize ve kich thuoc luy thua cua 2.

    FFT Cooley-Tukey de quy trong bai nay yeu cau chieu rong/chieu cao
    la luy thua cua 2. Gioi han max_size giup FFT tu viet chay nhanh.
    """
    image = load_grayscale_pil(image_path)
    width, height = image.size
    target_width = previous_power_of_two(min(width, max_size))
    target_height = previous_power_of_two(min(height, max_size))

    if image.size != (target_width, target_height):
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return np.asarray(image, dtype=np.float64)


def magnitude_spectrum(fft_shifted):
    """Tinh magnitude spectrum theo thang log de de quan sat."""
    return np.log1p(np.abs(fft_shifted))


def create_gaussian_highpass_filter(shape, d0):
    """
    Tao Gaussian Highpass Filter:
        H(u, v) = 1 - exp(-(D(u, v)^2) / (2 * D0^2))

    Tam tan so nam o giua anh vi pho da duoc fftshift.
    """
    rows, cols = shape
    center_y = rows // 2
    center_x = cols // 2

    y = np.arange(rows) - center_y
    x = np.arange(cols) - center_x
    x_grid, y_grid = np.meshgrid(x, y)
    distance_squared = x_grid**2 + y_grid**2

    return 1 - np.exp(-distance_squared / (2 * d0**2))


def normalize_for_save(image):
    """Chuan hoa mang ve [0, 255] de luu anh."""
    image = np.asarray(image, dtype=np.float64)
    min_value = image.min()
    max_value = image.max()

    if max_value == min_value:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image - min_value) * 255 / (max_value - min_value)
    return normalized.astype(np.uint8)


def save_image(path, image):
    Image.fromarray(normalize_for_save(image)).save(path)


def resize_grayscale_image(image_path, size):
    image = load_grayscale_pil(image_path)
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float64)


def open_image_window(image_path):
    """Mo anh ket qua bang trinh xem anh mac dinh cua he dieu hanh."""
    image_path = image_path.resolve()

    if shutil.which("xdg-open"):
        subprocess.Popen(
            ["xdg-open", str(image_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    Image.open(image_path).show()


def fft_1d(signal):
    """
    FFT 1D tu cai dat bang thuat toan Cooley-Tukey de quy.

    Dau vao phai co do dai la luy thua cua 2.
    """
    signal = np.asarray(signal, dtype=np.complex128)
    n = signal.shape[0]

    if n == 1:
        return signal
    if n % 2 != 0:
        raise ValueError("FFT tu viet yeu cau do dai la luy thua cua 2")

    even = fft_1d(signal[0::2])
    odd = fft_1d(signal[1::2])
    factor = np.exp(-2j * np.pi * np.arange(n // 2) / n)

    first_half = even + factor * odd
    second_half = even - factor * odd
    return np.concatenate([first_half, second_half])


def fft_2d(image):
    """Ap dung FFT 1D theo hang, sau do theo cot de tao FFT 2D."""
    image = np.asarray(image, dtype=np.complex128)

    row_fft = np.array([fft_1d(row) for row in image], dtype=np.complex128)
    col_fft = np.array([fft_1d(col) for col in row_fft.T], dtype=np.complex128).T

    return col_fft


def ifft_1d(signal):
    """IFFT 1D dua tren cong thuc lien hop cua FFT."""
    signal = np.asarray(signal, dtype=np.complex128)
    return np.conjugate(fft_1d(np.conjugate(signal))) / signal.shape[0]


def ifft_2d(spectrum):
    """Ap dung IFFT 1D theo hang va cot de khoi phuc anh."""
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    row_ifft = np.array([ifft_1d(row) for row in spectrum], dtype=np.complex128)
    col_ifft = np.array([ifft_1d(col) for col in row_ifft.T], dtype=np.complex128).T
    return col_ifft


def dft_1d(signal, inverse=False):
    """DFT 1D truc tiep theo cong thuc dinh nghia."""
    signal = np.asarray(signal, dtype=np.complex128)
    n = signal.shape[0]
    sign = 1 if inverse else -1
    output = np.zeros(n, dtype=np.complex128)

    for k in range(n):
        total = 0j
        for index in range(n):
            angle = sign * 2j * np.pi * k * index / n
            total += signal[index] * np.exp(angle)
        output[k] = total / n if inverse else total

    return output


def dft_2d(image, inverse=False):
    """DFT/IDFT 2D bang cach ap dung DFT 1D theo hang roi theo cot."""
    image = np.asarray(image, dtype=np.complex128)
    row_dft = np.array([dft_1d(row, inverse=inverse) for row in image], dtype=np.complex128)
    col_dft = np.array([dft_1d(col, inverse=inverse) for col in row_dft.T], dtype=np.complex128).T
    return col_dft


def opencv_dft_2d(image):
    """Tinh DFT bang OpenCV va chuyen ve mang phuc."""
    cv_input = np.asarray(image, dtype=np.float32)
    cv_output = cv2.dft(cv_input, flags=cv2.DFT_COMPLEX_OUTPUT)
    return cv_output[:, :, 0] + 1j * cv_output[:, :, 1]


def create_dft_fft_comparison_figure(image_path):
    """Tao cua so ket qua so sanh DFT/FFT nhu hinh mau."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    original_display = read_grayscale_image(image_path, max_size=MAX_FFT_SIZE)
    dft_input = resize_grayscale_image(image_path, COMPARISON_SIZE)
    fft_input = dft_input.copy()

    dft_start = perf_counter()
    manual_dft = dft_2d(dft_input)
    dft_elapsed = perf_counter() - dft_start

    numpy_dft_start = perf_counter()
    numpy_dft = np.fft.fft2(dft_input)
    numpy_dft_elapsed = perf_counter() - numpy_dft_start

    dft_reconstructed = np.real(dft_2d(manual_dft, inverse=True))
    dft_difference = np.abs(dft_reconstructed - dft_input)

    fft_start = perf_counter()
    manual_fft = fft_2d(fft_input)
    fft_elapsed = perf_counter() - fft_start

    numpy_fft_start = perf_counter()
    numpy_fft = np.fft.fft2(fft_input)
    numpy_fft_elapsed = perf_counter() - numpy_fft_start

    opencv_start = perf_counter()
    opencv_dft = opencv_dft_2d(fft_input)
    opencv_elapsed = perf_counter() - opencv_start

    dft_mean_error = np.mean(np.abs(manual_dft - numpy_dft))
    dft_max_error = np.max(np.abs(manual_dft - numpy_dft))
    fft_mean_error = np.mean(np.abs(manual_fft - numpy_fft))
    fft_max_error = np.max(np.abs(manual_fft - numpy_fft))
    opencv_mean_error = np.mean(np.abs(manual_fft - opencv_dft))
    opencv_max_error = np.max(np.abs(manual_fft - opencv_dft))

    panels = [
        ("Original Grayscale Image", original_display),
        (f"DFT Input {COMPARISON_SIZE}x{COMPARISON_SIZE}", dft_input),
        (f"FFT Input {COMPARISON_SIZE}x{COMPARISON_SIZE}", fft_input),
        ("Reconstructed from DFT", dft_reconstructed),
        ("DFT Magnitude", magnitude_spectrum(manual_dft)),
        ("Centered DFT Magnitude", magnitude_spectrum(np.fft.fftshift(manual_dft))),
        ("Centered DFT Phase", np.angle(np.fft.fftshift(manual_dft))),
        ("DFT Difference", dft_difference),
        ("Manual FFT Spectrum", magnitude_spectrum(np.fft.fftshift(manual_fft))),
        ("NumPy FFT2 on FFT Input", magnitude_spectrum(np.fft.fftshift(numpy_fft))),
        ("OpenCV DFT Spectrum", magnitude_spectrum(np.fft.fftshift(opencv_dft))),
        ("FFT Difference", np.abs(manual_fft - numpy_fft)),
    ]

    table_rows = [
        [
            "Manual DFT",
            f"{COMPARISON_SIZE}x{COMPARISON_SIZE}",
            f"{dft_elapsed:.6f}",
            "NumPy fft2",
            f"{numpy_dft_elapsed:.6f}",
            f"{dft_mean_error:.6e}",
            f"{dft_max_error:.6e}",
        ],
        [
            "Manual FFT",
            f"{COMPARISON_SIZE}x{COMPARISON_SIZE}",
            f"{fft_elapsed:.6f}",
            "NumPy fft2",
            f"{numpy_fft_elapsed:.6f}",
            f"{fft_mean_error:.6e}",
            f"{fft_max_error:.6e}",
        ],
        [
            "Manual FFT",
            f"{COMPARISON_SIZE}x{COMPARISON_SIZE}",
            f"{fft_elapsed:.6f}",
            "OpenCV dft",
            f"{opencv_elapsed:.6f}",
            f"{opencv_mean_error:.6e}",
            f"{opencv_max_error:.6e}",
        ],
    ]

    fig = plt.figure(figsize=(19.2, 10.8))
    fig.suptitle("DFT and FFT Comparison", fontsize=16, y=0.96)
    grid = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.42], hspace=0.18, wspace=0.25)

    for index, (title, image) in enumerate(panels):
        row = index // 4
        col = index % 4
        ax = fig.add_subplot(grid[row, col])
        ax.imshow(image, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    table_ax = fig.add_subplot(grid[3, :])
    table_ax.axis("off")
    table = table_ax.table(
        cellText=table_rows,
        colLabels=[
            "Method",
            "Image size",
            "Manual time (s)",
            "Built-in function",
            "Built-in time (s)",
            "Mean error",
            "Max error",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)

    comparison_path = OUTPUT_DIR / "bai_11_dft_fft_comparison.png"
    fig.savefig(comparison_path, dpi=150)

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)
        open_image_window(comparison_path)

    print("Bang so sanh DFT/FFT:")
    print(f"Manual DFT vs NumPy fft2: mean={dft_mean_error:.6e}, max={dft_max_error:.6e}")
    print(f"Manual FFT vs NumPy fft2: mean={fft_mean_error:.6e}, max={fft_max_error:.6e}")
    print(f"Manual FFT vs OpenCV dft: mean={opencv_mean_error:.6e}, max={opencv_max_error:.6e}")
    print(f"Da luu hinh so sanh: {comparison_path}")

    return comparison_path


def show_and_save_results(
    original,
    fft_spectrum,
    highpass_filter,
    filtered_spectrum,
    result_image,
):
    """Hien thi va luu tat ca ket qua cua bai thuc hanh."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    save_image(OUTPUT_DIR / "bai_11_01_original.png", original)
    save_image(OUTPUT_DIR / "bai_11_02_fft_spectrum.png", fft_spectrum)
    save_image(OUTPUT_DIR / "bai_11_03_gaussian_highpass_filter.png", highpass_filter)
    save_image(OUTPUT_DIR / "bai_11_04_filtered_spectrum.png", filtered_spectrum)
    save_image(OUTPUT_DIR / "bai_11_05_ifft_result.png", result_image)

    titles = [
        "Anh goc",
        "Pho FFT (log scale)",
        f"Gaussian Highpass Filter (D0={D0})",
        "Pho sau khi loc",
        "Anh sau IFFT",
    ]
    images = [
        original,
        fft_spectrum,
        highpass_filter,
        filtered_spectrum,
        result_image,
    ]

    plt.figure(figsize=(15, 7))
    for index, (title, image) in enumerate(zip(titles, images), start=1):
        plt.subplot(2, 3, index)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    summary_path = OUTPUT_DIR / "bai_11_frequency_domain_summary.png"
    plt.savefig(summary_path, dpi=150)

    # Thu hien thi bang Matplotlib. Neu backend khong co GUI, mo file PNG
    # bang trinh xem anh mac dinh de van co cua so ket qua.
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close()
        open_image_window(summary_path)

    return summary_path


def main():
    image_path = find_input_image(IMAGE_DIR)
    original = read_grayscale_image(image_path)

    # Buoc 2: FFT bang numpy.
    numpy_start = perf_counter()
    fft_numpy = np.fft.fft2(original)
    numpy_elapsed = perf_counter() - numpy_start
    fft_shifted = np.fft.fftshift(fft_numpy)
    fft_spectrum = magnitude_spectrum(fft_shifted)

    # Buoc 3: Tao Gaussian Highpass Filter.
    highpass_filter = create_gaussian_highpass_filter(original.shape, D0)

    # Buoc 4: Loc trong mien tan so.
    filtered_fft_shifted = fft_shifted * highpass_filter
    filtered_spectrum = magnitude_spectrum(filtered_fft_shifted)

    # Buoc 5: IFFT ve anh.
    filtered_fft = np.fft.ifftshift(filtered_fft_shifted)
    ifft_image = np.fft.ifft2(filtered_fft)
    result_image = np.real(ifft_image)

    # Buoc 6: Tu viet FFT 2D.
    custom_start = perf_counter()
    fft_custom = fft_2d(original)
    custom_elapsed = perf_counter() - custom_start

    # Buoc 7: So sanh FFT tu viet va numpy FFT.
    mean_error = np.mean(np.abs(fft_custom - fft_numpy))
    max_error = np.max(np.abs(fft_custom - fft_numpy))

    comparison_path = create_dft_fft_comparison_figure(image_path)

    summary_path = show_and_save_results(
        original=original,
        fft_spectrum=fft_spectrum,
        highpass_filter=highpass_filter,
        filtered_spectrum=filtered_spectrum,
        result_image=result_image,
    )

    print(f"Anh dau vao: {image_path}")
    print(f"Kich thuoc xu ly: {original.shape[1]}x{original.shape[0]}")
    print(f"Thoi gian numpy.fft.fft2: {numpy_elapsed:.6f} giay")
    print(f"Thoi gian FFT tu viet: {custom_elapsed:.6f} giay")
    print(f"Sai so trung binh giua FFT tu viet va numpy FFT: {mean_error:.6e}")
    print(f"Sai so lon nhat: {max_error:.6e}")
    print(
        "Giai thich: sai so rat nho xuat hien do tinh toan so thuc dau phay dong "
        "va thu tu cong/nhan phuc cua thuat toan FFT tu viet khac voi numpy."
    )
    print(f"Da luu ket qua vao thu muc: {OUTPUT_DIR}")
    print(f"Da mo cua so so sanh DFT/FFT: {comparison_path}")
    print(f"Da mo cua so hien thi file: {summary_path}")


if __name__ == "__main__":
    main()
