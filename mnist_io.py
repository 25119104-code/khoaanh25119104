# ============================================================
# mnist_io.py — Đọc MNIST test set TRỰC TIẾP từ file raw (idx)
# ============================================================
# Vì sao không dùng torchvision?
#   - Golden model C sau này cũng phải tự đọc file raw này. Đọc bằng
#     tay ở Python trước thì biết chính xác từng byte nằm ở đâu.
#   - inference_python.py cần chạy KHÔNG CÓ PyTorch.
#
# Định dạng idx (big-endian):
#   images: [magic 4B][số ảnh 4B][số hàng 4B][số cột 4B][pixel uint8 ...]
#   labels: [magic 4B][số nhãn 4B][nhãn uint8 ...]
#   -> header ảnh 16 byte, header nhãn 8 byte. Pixel lưu theo hàng.
# ============================================================

import numpy as np

IMG_FILE = "data/MNIST/raw/t10k-images-idx3-ubyte"
LBL_FILE = "data/MNIST/raw/t10k-labels-idx1-ubyte"

# Hằng số chuẩn hoá — PHẢI giống transforms.Normalize((0.1307,), (0.3081,))
# trong model_comparison.py. Sai ở đây thì model nhận ảnh "lạ" và đoán sai.
MEAN = 0.1307
STD = 0.3081


def load_test_set():
    """Trả về (images uint8 [N,28,28], labels uint8 [N])."""
    with open(IMG_FILE, "rb") as f:
        raw = f.read()
    n = int.from_bytes(raw[4:8], "big")
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, 28, 28)

    with open(LBL_FILE, "rb") as f:
        raw = f.read()
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    return images, labels


def normalize(img_u8):
    """uint8 [28,28] -> float32 [28,28], giống hệt ToTensor() + Normalize().

    Tính bằng float32 đúng thứ tự PyTorch làm: (x/255 - mean) / std.
    Lưu ý cho FPGA: đây là phép biến đổi affine y = a*x + b với a, b hằng số
    tính trước (a = 1/(255*std), b = -mean/std) -> không cần phép chia trên chip.
    """
    x = img_u8.astype(np.float32) / np.float32(255.0)
    return (x - np.float32(MEAN)) / np.float32(STD)
