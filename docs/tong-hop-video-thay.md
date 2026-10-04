# Tổng hợp playlist của Thầy — "Build team – Build Project (Full Flow: AI-IC-ES)"

> Nguồn: 9 video kênh MrH3 Center – Kèm học kỹ thuật, playlist `PLKxQsK6-3Src`.
> Tổng hợp qua Gemini Notebook (sổ "Video Thầy - Build Project Full Flow AI-IC-ES", 9 nguồn), ngày 04/10/2026.
> Notebook đọc **bản chép lời tự động** của YouTube. Chỗ nào ghi ⚠️ là cần xem lại video để chắc.

---

## 1. Tóm tắt từng video

| # | Video | Thời lượng | Nội dung chính | Sản phẩm SV phải ra |
|---|---|---|---|---|
| 1 | Step0 – Lộ trình 4 năm (Overall) | 18:32 | Làm 1 project chạy trọn AI → IC → ES để tìm mảng mình thích. Bắt đầu sớm từ năm 1–2 | Đề tài + model nhỏ |
| 2 | Step0 – Định hướng 4 năm (MoreDetail) | 19:47 | 4 giai đoạn: (1) baseline full flow, (2) tối ưu + ĐATN, (3) NCKH, (4) đào sâu 1 mảng theo JD doanh nghiệp | Baseline chạy end-to-end AI → RTL → kit |
| 3 | Step1 – Hiểu model AI (DL) đơn giản | 15:48 | Lập trình truyền thống (người viết luật) vs Deep Learning (máy học từ dữ liệu). Weight/bias, forward → loss → backprop → inference | Hiểu khái niệm |
| 4 | Step1 – Hiểu model AI (more detail) | 14:38 | 5 bước xây model; CNN = trích đặc trưng (Conv, ReLU, MaxPool) + phân loại (Flatten, FC, Softmax) | **Tính tay weight/bias trên giấy**; model accuracy > 90% |
| 5 | **Step2 – Full flow (AI, IC, ES)** | 21:17 | Quy trình chi tiết từ model baseline tới chạy trên board. **Dùng MNIST làm bài mẫu** | Xem mục 2 |
| 6 | **Step3 – Ba giải pháp thiết kế HW** | 10:14 | 3 kiến trúc RTL, từ đơn giản tới linh hoạt | Code RTL |
| 7 | Step4 – Tối ưu HT + Đăng ký NCKH | 6:49 | Đọc paper, tìm bottleneck, tối ưu (quantization, pruning, sparsity, pipelining, data reuse). NCKH cấp 1 ~5 triệu (baseline), cấp 2 ~15 triệu (đã tối ưu) | Báo cáo so sánh latency / power / area |
| 8 | Step5 – Build kit nhúng full flow | 6:04 | SoC FPGA: PS (ARM) + PL (FPGA) nối bằng AXI/DMA. Build Linux: U-Boot → kernel → device tree → driver → rootfs → app | Kit chạy demo |
| 9 | Hướng dẫn tải paper quốc tế | 10:09 | IEEE Xplore, lọc Journal, lấy DOI, request qua phiantu.com | PDF paper |

---

## 2. Step2 chi tiết — quy trình full flow Thầy dạy (bài mẫu MNIST)

| Bước | Việc | Sản phẩm |
|---|---|---|
| 1 | Build model AI baseline, train / validate / test | File model |
| 2 | Phân tích công thức toán từng khối: Conv, BatchNorm, ReLU, MaxPool, FC, Softmax | Bảng phân tích phép toán + độ phức tạp |
| 3 | Làm model **hardware-friendly**: float32 → int8 / fixed-point; thay phép chia bằng nhân + dịch bit; gộp BatchNorm vào Conv; **Softmax → ArgMax** | Model tối ưu cho phần cứng |
| 4 | Trích tham số (weight, bias, multiplier, shift, zero point) ra file text → viết **Golden Model C/C++** thuần | File tham số + chương trình C |
| 5 | Viết **Verilog/RTL**: ALU, datapath, controller (FSM), khối nạp feature/weight, readback | Mã `.v` |
| 6 | Deploy lên **SoC FPGA**: Linux + driver + app C gọi khối gia tốc, nạp bitstream, đo accuracy và latency trên tập test | Kit chạy thật |

Những điều Thầy nói rõ (đã hỏi lại notebook để đối chiếu nguyên văn):

- **Framework:** Thầy demo bằng TensorFlow/TFLite, nhưng nói "PyTorch cũng được, tìm hiểu cái gì được là được". → Bạn dùng PyTorch là đúng.
- **Thứ tự:** trong video, **quantization làm TRƯỚC**, rồi trích tham số int8 để viết golden model C. Video **không** nói tới golden model float.
- **So golden model với gì:** Thầy nói về so **accuracy** xem có sụt không. **Không** nhắc so từng lớp.
- **Board:** chỉ nói chung "kit FPGA / kit SoC", **không** nêu tên board cụ thể.
- **Số ảnh test:** Thầy nói "mười mấy ngàn… 15.000 input". ⚠️ MNIST test chuẩn có 10.000 ảnh, có thể Thầy nói ước chừng.
- **Testbench / mô phỏng:** có nhắc "CPU testbench" và "chạy mô phỏng", không nêu tên phần mềm (Vivado, ModelSim…).

## 3. Step3 — Ba giải pháp thiết kế HW

| Giải pháp | Cách làm | Ưu | Nhược |
|---|---|---|---|
| **1. Direct layer mapping** | Mỗi layer là 1 khối HW riêng, nối tiếp | Dễ hiểu, dễ debug | Tốn diện tích |
| 2. PE dùng chung + FSM | 1 khối tính (MAC, ReLU, MaxPool) dùng lại cho mọi layer, FSM điều khiển, BRAM chứa data/weight | Giảm diện tích nhiều | Đổi model là phải sửa RTL |
| 3. Accelerator theo tập lệnh | Controller giải mã lệnh trong "context BRAM" | Đổi model chỉ cần nạp lệnh + data mới | Phức tạp nhất |

**Thầy khuyên SV mới làm giải pháp 1 trước.**

---

## 4. Mình đang ở đâu so với Step2

| Bước Step2 | Trạng thái project | Ghi chú |
|---|---|---|
| 1. Baseline | ✅ | SmallCNN, 5.018 params, test 98,82% (> 90% Thầy yêu cầu) |
| 2. Phân tích phép toán | ✅ | 6 operator, 414.265 phép/ảnh, `operators/` |
| 3. Hardware-friendly | 🟡 làm một phần | ✅ Softmax → ArgMax. ✅ Đổi thứ tự relu/pool. ✅ Không có BatchNorm nên không phải gộp. ✅ Chọn đệm 0 thay `if`. ⬜ **Quantization int8** — Thầy dặn trực tiếp "chưa cần" |
| 4. Trích tham số + golden C | 🟡 | ✅ Trích tham số **float**, 2 file. ✅ Bản Python thuần khớp PyTorch. ⬜ **Golden model C** — việc kế tiếp |
| 5. RTL | ⬜ | Theo Step3: bắt đầu bằng giải pháp 1 |
| 6. Deploy SoC FPGA | ⬜ | Step5 |
| (Step1) Tính tay weight/bias | ⬜ | Thầy giao trong Step1 more detail và nhắc lại buổi 23/09 |

**Vị trí hiện tại: giữa bước 3 và bước 4 của Step2.**

## 5. Chỗ lệch giữa video và lời Thầy nói trực tiếp — cần hỏi lại

1. **Quantization trước hay sau golden model?** Video: int8 trước, golden C viết thẳng bằng int8.
   Thầy nói trực tiếp (22/09): chưa cần quantization.
   → Đề xuất: làm **golden C float32 trước** (đã có bản Python làm chuẩn), viết hàm tách riêng theo
   từng operator để sau đổi sang int8 chỉ phải sửa phần số học. Gặp Thầy hỏi: "Em làm golden float
   trước rồi mới int8 được không, hay theo video lượng tử hoá luôn?"
2. **Tiêu chí đạt của golden model:** video chỉ nói so accuracy. Nên đề xuất với Thầy: so **trùng dự
   đoán trên 10.000 ảnh** là tiêu chí chính, so từng lớp là công cụ debug.
3. **Chuẩn hoá ảnh trên chip:** video nói "thay phép chia bằng nhân + dịch bit". Khớp với nhận xét
   chuẩn hoá là `a·x + b` — khi lên int8, `a` sẽ thành cặp multiplier + shift.

## 6. Hướng làm tiếp theo video

1. Tính tay 1 điểm conv1 (Step1 more detail).
2. Golden model C float32 → so với `golden/`, chạy 10.000 ảnh.
3. Hỏi Thầy về quantization (mục 5.1). Nếu làm: PyTorch quantization → trích weight, bias, multiplier, shift, zero point → golden C int8.
4. RTL giải pháp 1 (Direct layer mapping): mỗi layer 1 module Verilog, testbench so với golden C.
5. Sau baseline chạy được: Step4 (tối ưu, NCKH cấp 1), Step5 (kit SoC).
