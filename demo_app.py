# ============================================================
# DEMO THỰC TẾ: Vẽ số trên canvas -> model đoán trực tiếp
# Dùng lại đúng kiến trúc + trọng số đã train (digit_cnn.pth)
# ============================================================
#
# CÁCH DÙNG:
#   python3 demo_app.py
#   -> mở link http://127.0.0.1:7860 trên trình duyệt, vẽ 1 chữ số, xem model đoán
#
# LƯU Ý QUAN TRỌNG VỀ TIỀN XỬ LÝ:
# 1) Canvas vẽ ra ảnh NÉT ĐEN trên NỀN TRẮNG, nhưng MNIST train bằng ảnh
#    NÉT TRẮNG trên NỀN ĐEN -> phải đảo màu, nếu không model đoán bậy.
# 2) MNIST không đơn giản resize cả ảnh về 28x28. Digit gốc được:
#    tìm bounding box (vùng có nét vẽ) -> cắt sát -> scale vừa khung ~20x20
#    (giữ tỉ lệ) -> dán vào GIỮA ảnh nền đen 28x28.
#    Nếu chỉ resize nguyên canvas (canvas to, nét vẽ chỉ chiếm 1 góc nhỏ),
#    chữ số sẽ bị co thành 1 vệt tí xíu trong ảnh 28x28 -> model chưa từng
#    thấy dữ liệu tỉ lệ như vậy lúc train -> đoán sai dù vẽ rất rõ ràng.

import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [1] KIẾN TRÚC MODEL — PHẢI GIỐNG HỆT LÚC TRAIN
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = DigitCNN().to(device)
model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
model.eval()  # tắt Dropout, cần dự đoán ổn định

# %% [2] CHUẨN HÓA GIỐNG HỆT LÚC TRAIN (Normalize MNIST)
MEAN, STD = 0.1307, 0.3081
INK_THRESHOLD = 25  # ngưỡng để coi 1 pixel là "có nét vẽ" (sau khi đã đảo màu, nền = 0)


def preprocess(img: Image.Image):
    """
    Nhận 1 ảnh PIL (từ canvas vẽ tay) -> trả về tensor [1,1,28,28]
    đã qua đúng các bước tiền xử lý giống MNIST, hoặc None nếu canvas trống.
    """
    img = img.convert("L")  # ảnh xám (1 kênh màu)
    arr = np.array(img).astype(np.float32)

    # Canvas mặc định: nền TRẮNG (255), nét vẽ ĐEN (0) -> đảo lại giống MNIST
    if arr.mean() > 127:
        arr = 255.0 - arr

    # --- Bước quan trọng: crop theo bounding box của nét vẽ, rồi căn giữa ---
    mask = arr > INK_THRESHOLD
    if not mask.any():
        return None  # canvas trống, chưa vẽ gì

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    digit = arr[y0:y1, x0:x1]

    # Scale digit vừa khung 20x20 (giữ tỉ lệ khung hình, giống chuẩn MNIST)
    h, w = digit.shape
    if h >= w:
        new_h = 20
        new_w = max(1, round(w * (20.0 / h)))
    else:
        new_w = 20
        new_h = max(1, round(h * (20.0 / w)))

    digit_img = Image.fromarray(digit.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)

    # Dán vào giữa ảnh nền đen 28x28 (28 = 20 + lề 4px mỗi bên, đúng chuẩn MNIST)
    canvas28 = Image.new("L", (28, 28), color=0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas28.paste(digit_img, (paste_x, paste_y))

    arr28 = np.array(canvas28).astype(np.float32) / 255.0  # về [0,1] giống ToTensor()
    arr28 = (arr28 - MEAN) / STD  # Normalize giống lúc train

    tensor = torch.tensor(arr28, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,28,28]
    return tensor.to(device)


# %% [3] HÀM DỰ ĐOÁN — GỌI MỖI KHI NGƯỜI DÙNG VẼ XONG
def predict(editor_value):
    if editor_value is None:
        return "Hãy vẽ 1 chữ số vào ô bên trái.", None

    img = editor_value["composite"] if isinstance(editor_value, dict) else editor_value
    if img is None:
        return "Hãy vẽ 1 chữ số vào ô bên trái.", None

    tensor = preprocess(img)
    if tensor is None:
        return "Canvas đang trống, hãy vẽ 1 chữ số.", None

    with torch.no_grad():
        output = model(tensor)                       # logits thô, chưa qua softmax
        probs = torch.softmax(output, dim=1)[0]       # đổi sang xác suất 0-1 cho từng lớp
        pred = int(probs.argmax().item())
        confidence = float(probs[pred].item()) * 100

    label_text = f"Model đoán: {pred}  (độ tin cậy {confidence:.1f}%)"
    prob_dict = {str(i): float(probs[i].item()) for i in range(10)}
    return label_text, prob_dict


# %% [4] GIAO DIỆN GRADIO
with gr.Blocks(title="Nhận diện chữ số viết tay") as demo:
    gr.Markdown("## Demo: Vẽ 1 chữ số (0-9) rồi bấm 'Đoán'")
    gr.Markdown(
        "Model CNN train trên MNIST, accuracy ~99.2% trên tập test. "
        "Vẽ chữ số to, rõ, giữa canvas để có kết quả tốt nhất."
    )

    with gr.Row():
        canvas = gr.Sketchpad(
            label="Vẽ số ở đây",
            type="pil",
            image_mode="L",
            brush=gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed"),
        )
        with gr.Column():
            result_text = gr.Textbox(label="Kết quả")
            prob_plot = gr.Label(label="Xác suất từng lớp (0-9)", num_top_classes=10)

    with gr.Row():
        predict_btn = gr.Button("Đoán", variant="primary")
        clear_btn = gr.Button("Xóa")

    predict_btn.click(fn=predict, inputs=canvas, outputs=[result_text, prob_plot])
    clear_btn.click(fn=lambda: (None, "", None), inputs=None, outputs=[canvas, result_text, prob_plot])

if __name__ == "__main__":
    demo.launch()
