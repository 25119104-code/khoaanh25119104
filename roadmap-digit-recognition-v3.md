# Roadmap v3: Nhận diện chữ số viết tay — từ AI đến Golden Model (C)

> ⚠️ **Vẫn là giả thuyết cá nhân, chưa qua Thầy duyệt.** v3 khác v2 ở một điểm lớn: **mở rộng phạm vi project 1** từ "chỉ AI/DL, không lấn IC/ES" sang "trọn công đoạn AI theo định nghĩa của Thầy" — tức đi tới golden model C. Lý do ở mục 0. Phần cần Thầy xác nhận nằm ở mục 8.
>
> 🔄 **Cập nhật 21/09/2026 sau feedback tuần 2 của Thầy** — 4 nội dung:
> 1. Bắt buộc thêm **validation set**.
> 2. **Giảm params**: model hiện tại 206,922 params, Thầy nêu ví dụ một bạn khác chỉ 5,018 → thêm **Phase 1.7 (thu gọn kiến trúc)**.
> 3. **"Tối ưu trọng số khi đưa vào chip"** → quantization lên trục chính, thành **Phase 2E**.
> 4. **"PyTorch hay TensorFlow đều được"** → chốt giữ PyTorch, rủi ro `.h5` bị loại.

---

## 0. Vì sao có v3 — 6 khoảng lệch giữa v2 và tài liệu Thầy

| # | v2 đang viết | Tài liệu Thầy | Xử lý ở v3 |
|---|---|---|---|
| 1 | "Project này ở lại trong AI/DL, **không lấn IC/ES**" | Công đoạn AI gồm cả: viết operator → thay phép toán khó cho FPGA → trích tham số → **build golden model bằng C**. Câu "Công đoạn học về IC, ES sẽ diễn ra sau khi done bước 1" xác nhận golden model C vẫn thuộc bước 1 | **Mở rộng scope.** v2 hiểu nhầm ranh giới: golden model C *không phải* phần cứng, nó là chương trình thuần giải thuật chạy trên PC. Vạch phân chia AI/IC nằm sau golden model, không phải trước nó |
| 2 | Phase 1.5 dừng ở "hiểu vì sao dùng thành phần này" | Bước "Viết các operator trong model: **phép toán + mã giả**" | Phase 1.5 được kéo thành 2 nấc: hiểu → viết công thức + mã giả. Nấc 2 là đầu vào trực tiếp cho golden model C |
| 3 | Không có bước nào về FPGA-friendly | "Thay các phép toán không triển khai/khó triển khai xuống FPGA → **train lại**" | Thêm Phase 2A. Lưu ý: bước này **sửa ngược lại kiến trúc model** và buộc train lại — càng phát hiện muộn càng tốn |
| 4 | Không phân biệt model train vs model inference | "Model giờ khác gì model lúc train? vd: Dropout (bỏ), batch-norm (nên gộp với conv, FC)" | Thêm Phase 2B — bắt buộc trước khi viết C |
| 5 | Phase 2 backlog: EMNIST, transfer learning, framework thứ 2 | Không nằm trên trục AI → IC → ES | Hạ xuống mục "Ngoài trục", chỉ làm nếu dư thời gian |
| 6 | Quantization/pruning nằm ở Phase 3 backlog | ~~Thầy xếp vào Giai đoạn 2 (HK2 năm 2)~~ → **Feedback tuần 2: "tối ưu trọng số khi đưa vào chip"** | 🔄 **Đã đổi 21/09.** Quantization lên trục chính thành **Phase 2E**, đặt **sau** golden model C float32 |

> ⚠️ **Cảnh báo về mục "Ngoài trục".** Cách phân loại đó dựng trên suy đoán ý Thầy, và đã sai 2 lần liên tiếp: quantization (→ Phase 2E) và thu gọn kiến trúc (→ Phase 1.7) đều từng nằm trong "Ngoài trục" rồi bị Thầy kéo lên trục chính. Đừng tin phần phân loại đó cho tới khi hỏi Thầy xong.

---

## 1. Định vị: mình đang đứng ở đâu trong bức tranh lớn

**Chuỗi end-to-end Thầy vẽ:**

```
AI              Model            IC / FPGA         Embedded System      Application
(Train model) → (Parameters) → (RTL, Synthesis) → (Integration)   →   (Real-world)
```

**Chuỗi chi tiết của Giai đoạn 1 (Hiểu Full Flow):**

```
Train model (AI) → Export parameters → Golden model (C) → Thiết kế RTL (IC/FPGA)
→ Simulation & Verification → Triển khai trên FPGA → Tích hợp hệ nhúng
→ Đo/đánh giá (accuracy, latency, resource, power)
```

**Project 1 phụ trách 3 ô đầu.** Ô thứ 4 (RTL) trở đi là project 2/3 của năm.

| Giai đoạn (theo Thầy) | Thời điểm | Mục tiêu | Mình |
|---|---|---|---|
| 1. Hiểu Full Flow AI–IC–ES | Năm 1 → HK1 năm 2 (3–6 tháng) | 1 hệ thống baseline chạy end-to-end (VD: CNN nhỏ trên FPGA) | **Đang ở đây**, ở phần AI |
| 2. Tối ưu 1 bài toán cụ thể | HK2 năm 2 (3–6 tháng) | Đồ án / LVTN sớm, prototype có số liệu so sánh | Chưa |
| 3. Đăng ký NCKHSV | Năm 2 và năm 3 (2 lần) | Đề tài NCKH SV, báo cáo/poster/paper | Chưa |
| 4. Học theo "Nhà tuyển dụng cần" | HK2 năm 3 → HK1 năm 4 | CV + GitHub + Project sẵn sàng phỏng vấn | Chưa |

---

## 2. Quy trình chuẩn công đoạn AI (theo Thầy) — 8 bước và trạng thái thật

| # | Bước | Nội dung cụ thể | Trạng thái |
|---|---|---|---|
| 1 | **Build model AI (train)** | Gọi thư viện, train ra model chạy được | ✅ Xong — CNN PyTorch, acc 99.13% (epoch 8/8) |
| 2 | **Tìm hiểu kiến trúc** | Vì sao dùng thành phần này mà không dùng cái khác | 🔄 Đang làm (tuần 3) |
| 2b | **Thu gọn kiến trúc** 🔄 | Giảm params từ 206,922 xuống cỡ 5k | 🔄 Tuần 3 — **mới, từ feedback tuần 2** |
| 3 | **Viết các operator** | Phép toán + mã giả cho từng lớp | ⬜ Chưa |
| 4 | **Thay phép toán khó triển khai xuống FPGA** | Thay rồi **train lại** | ⬜ Chưa |
| 5 | **Xác định model/kiến trúc cho inference** | Model inference khác model train ở chỗ nào | ⬜ Chưa |
| 6 | **Trích tham số (weight, bias)** | Xuất từ file model ra định dạng C đọc được | ⬜ Chưa |
| 7 | **Build golden model (C)** | Dùng operator + tham số, chạy inference, thuần giải thuật | ⬜ Chưa — **đích của project 1** |
| 8 | **Tối ưu trọng số cho chip** 🔄 | Quantization/fixed-point, giảm bộ nhớ tham số | ⬜ Chưa — **mới, từ feedback tuần 2** |

**Nguyên tắc thứ tự:** bước 2b phải xong trước bước 3. Viết mã giả và golden model C cho kiến trúc 206k rồi mới đổi sang 5k = viết lại từ đầu.

---

## 3. Params — vấn đề Thầy nêu ở buổi báo cáo tuần 2

### 3.1 Model hiện tại: 206,922 params

| Lớp | Công thức | Params | % tổng |
|---|---|---|---|
| `conv1` Conv2d(1→16, 3×3) | 16×1×3×3 + 16 | 160 | 0.08% |
| `conv2` Conv2d(16→32, 3×3) | 32×16×3×3 + 32 | 4,640 | 2.24% |
| **`fc1` Linear(1568→128)** | 1568×128 + 128 | **200,832** | **97.06%** |
| `fc2` Linear(128→10) | 128×10 + 10 | 1,290 | 0.62% |
| | **Tổng** | **206,922** | 100% |

Float32: **828 KB** tham số. BRAM trên FPGA cỡ nhỏ thường chỉ vài trăm KB → không nhét vừa. Đây là ràng buộc phần cứng thật, không phải Thầy khó tính.

### 3.2 Kiến trúc tham chiếu (bản của anh năm 4): 5,018 params, test acc 98.88%

| Lớp | Shape ra | Params |
|---|---|---|
| conv1 Conv2D(1→8, 3×3) | 28×28×8 | 80 |
| pool1 | 14×14×8 | 0 |
| conv2 Conv2D(8→16, 3×3) | 14×14×16 | 1,168 |
| pool2 | 7×7×16 | 0 |
| **conv3 Conv2D(16→16, 3×3)** | 7×7×16 | 2,320 |
| **pool3** | **3×3×16** | 0 |
| flatten | 144 | 0 |
| Dense(144→10) | 10 | 1,450 |
| | **Tổng** | **5,018** (19.6 KB) |

**Hai chiêu chính — không phải quantization, không phải pruning:**

1. **Thêm conv3 + pool3** → ép spatial từ 7×7 xuống 3×3 **trước** Flatten. Flatten ra 144 thay vì 1,568 → giảm 11 lần ngay tại đầu vào FC.
2. **Bỏ hẳn lớp FC ẩn** → đi thẳng `Dense(144→10)` thay vì `fc1(1568→128) → fc2(128→10)`.

Cộng thêm channel nhỏ hơn (8/16/16 thay vì 16/32). Không có Dropout.

### 3.3 Cái gì giảm params, cái gì không

| Thứ | Đổi cái gì | Giảm số params? |
|---|---|---|
| **Data augmentation** | Số lượng/biến thể ảnh đầu vào | ❌ Không |
| **Hyperparameter** (lr, batch, epochs, optimizer) | Cách tìm giá trị weight | ❌ Không |
| **Kiến trúc** (channel, số neuron, bỏ/thêm lớp) | Số lượng weight | ✅ **Chỉ cái này** |
| **Quantization** | Số byte mỗi weight (4 → 1) | ❌ Giảm **bộ nhớ** 4×, số params giữ nguyên |

→ Phase 2E (quantization) **không** giải quyết được yêu cầu này nếu Thầy so *số params*. Phải sửa kiến trúc, tức Phase 1.7.

→ Augmentation và hyperparameter chỉ có ích ở **bước sau**: bù accuracy sau khi đã cắt params.

### 3.4 Đối chiếu với tiêu chí FPGA-friendly

Phân tích chuẩn bị cho bước 4 — áp dụng cho cả 2 kiến trúc.

| Thành phần | Phép toán thật sự | Mức thân thiện phần cứng | Ghi chú |
|---|---|---|---|
| `Conv2d(3×3, padding=1)` | MAC (nhân–cộng tích luỹ) | ✅ Rất tốt | Đúng thứ mà PE/MAC array sinh ra để làm |
| `ReLU` | `max(x, 0)` | ✅ Rất tốt | 1 comparator + 1 mux |
| `MaxPool2d(2)` | So sánh | ✅ Rất tốt | May mắn đã chọn max thay vì average. Lưu ý 7//2 = 3 (floor), không phải 3.5 |
| `Linear` (fc) | MAC | ✅ Tốt về toán / ⚠️ Tuỳ về bộ nhớ | Với 206k: `fc1` chiếm 97.06% → thảm hoạ. Với 5k: `Dense` chỉ 1,450 (28.9%) → chấp nhận được |
| `Flatten` | Đổi cách đánh chỉ số | ✅ Miễn phí | Trong C chỉ là đọc mảng theo thứ tự khác |
| `Dropout` | — | ✅ | Bỏ hoàn toàn khi inference, chi phí phần cứng = 0. Kiến trúc 5k không có Dropout → Phase 2B nhẹ hơn |
| `Softmax` | `exp` + phép chia | ❌ Đắt nhất | **Có thể bỏ**: `argmax(softmax(z)) = argmax(z)` vì softmax đơn điệu tăng. Golden model chỉ cần argmax trên logits. Chỉ giữ nếu muốn xuất % tin cậy (demo Gradio) |
| `CrossEntropyLoss`, `Adam` | — | — | Chỉ tồn tại lúc train, không bao giờ xuống phần cứng |

**Kết luận:** cả 2 kiến trúc đều đã FPGA-friendly về mặt *phép toán*. Vấn đề duy nhất là **bộ nhớ**, và nó được giải bằng Phase 1.7 (giảm số params) + Phase 2E (giảm byte mỗi param). Bước 4 của Thầy với model này nhiều khả năng chỉ còn 1 việc: bỏ softmax — không cần train lại. Cần Thầy xác nhận (câu hỏi 2, mục 8).

---

## 4. Thuật ngữ bổ sung cho v3

(17 thuật ngữ nền ở v2 giữ nguyên, không lặp lại ở đây.)

| Thuật ngữ | Giải thích ngắn gọn |
|---|---|
| **Operator** | Một phép biến đổi trong model (Conv2d, ReLU, MaxPool, Linear, Softmax). Ở tầng AI nó là 1 dòng gọi thư viện; ở tầng phần cứng nó là 1 khối mạch |
| **Mã giả (pseudocode)** | Mô tả operator bằng vòng lặp + phép toán cơ bản, không phụ thuộc thư viện. Cầu nối giữa công thức toán và code C/RTL |
| **Golden model** | Bản inference viết bằng C thuần (không thư viện DL), làm **chuẩn tham chiếu**. Sau này RTL cho kết quả lệch golden model = bug phần cứng. Không có golden model thì không verify được RTL |
| **FPGA-friendly** | Phép toán rẻ trên phần cứng: cộng, nhân, so sánh, dịch bit. Đắt: chia, `exp`, `log`, căn bậc hai, hàm siêu việt (sigmoid, tanh, softmax) |
| **Validation set** | Tập riêng để theo dõi và ra quyết định (chọn epoch, chọn kiến trúc). Không có nó thì mọi quyết định đều dựa trên test set = data leakage, con số cuối mất ý nghĩa |
| **Fold BatchNorm** | Gộp tham số BatchNorm vào weight/bias của Conv hoặc Linear khi inference. Toán học tương đương, bớt được 1 lớp, kết quả không đổi |
| **Fixed-point / Quantization** | Thay float32 bằng số nguyên (int8/int16) để tiết kiệm tài nguyên phần cứng. 🔄 Việc của **Phase 2E** |
| **PTQ / QAT** | Post-Training Quantization: lượng tử hoá model đã train xong, nhanh, không train lại. Quantization-Aware Training: mô phỏng sai số lượng tử hoá ngay khi train, chính xác hơn nhưng phải train lại. Làm PTQ trước, chỉ dùng QAT nếu PTQ rớt accuracy quá nhiều |
| **Scale / zero-point** | 2 tham số chuyển float ↔ integer: `real ≈ scale × (q − zero_point)`. Khi xuất tham số sang C phải xuất kèm 2 số này cho từng lớp, thiếu là sai toàn bộ |
| **FLOPs** | Số phép tính mỗi lần suy luận → quyết định **tốc độ và năng lượng** |
| **Params** | Số trọng số phải lưu → quyết định **bộ nhớ**. Params lớn ≠ FLOPs lớn: `fc1` chiếm 97% params nhưng phần lớn FLOPs lại nằm ở các lớp Conv (vì conv dùng lại 1 kernel trên cả ảnh) |

---

## 5. Kế hoạch theo phase — v3

**Nguyên tắc (giữ từ v2):** chỉ phase hiện tại + phase kế tiếp có nội dung cụ thể. Phần sau là backlog định hướng, viết lại sau mỗi buổi gặp Thầy (thứ 4 hàng tuần).

### Phase 0 — Baseline ✅ hoàn thành
- Setup Git/GitHub, commit đều hàng tuần.
- CNN MNIST chạy được (`mnist_digit_recognition.py`), acc **99.13%** (epoch 8/8).
- Nắm chắc và tự giải thích lại được các khái niệm nền.

### Phase 1 — Đánh giá đúng cách 🔄 còn 1 việc bắt buộc
- ✅ Confusion matrix 10×10, accuracy theo từng chữ số, top cặp nhầm (`error_analysis.py`).
- ✅ **Kết quả thật:** acc tổng 99.21%; cặp nhầm nhiều nhất 2→7 (8 lần), 4→9 (6 lần), 9→7 (4 lần); lớp yếu nhất là 9 (98.61%) và 2 (98.74%). Xuất `error_analysis.png`, `confusion_matrix.png`.
- ⬜ **BẮT BUỘC — Thầy yêu cầu trực tiếp ở feedback tuần 2:** tách validation set (`random_split` 50k train / 10k val), `evaluate()` dùng val thay vì test khi train; test set chỉ chạm **1 lần** cuối cùng. Tỷ lệ 50k/10k/10k ≈ 71/14/14, khớp ~70/15/15 Thầy đưa.
  - Kèm theo: lưu checkpoint theo **val accuracy cao nhất**, không phải epoch cuối (sửa lỗi epoch 7 = 99.21% nhưng file `.pth` giữ epoch 8 = 99.13%).
  - Đã code sẵn trong `model_comparison.py`.
- ⬜ **Còn lại:** so sánh phiên bản trước/sau (làm được sau khi có validation set). *Data augmentation và tinh chỉnh hyperparameter ở mục "Ngoài trục".*

### Phase 1.5 — Hiểu sâu từng thành phần 🔄 đang làm (yêu cầu Thầy, tuần 3)
Phương pháp "thực chiến": code đã chạy → phân tích vì sao dùng thành phần này → hiểu phép toán bên trong. Chọn 2–3 thành phần/tuần, không ôm hết.

- **Tuần 3:** `Conv2d(kernel=3, padding=1)` (vì sao 3×3, công thức output size), `MaxPool2d` (vì sao max không average), `CrossEntropyLoss` (log-softmax + NLL, vì sao không MSE).
- **Bổ sung 21/09 — câu hỏi sinh ra từ việc so 2 kiến trúc** (có 2 model thật để đối chiếu, không phải học chay):
  - Vì sao 3 lớp conv tốt hơn 2 lớp ở cùng ngân sách params?
  - Vì sao bỏ được lớp FC ẩn mà accuracy chỉ rớt ~0.3%?
  - Vì sao channel 8/16/16 chứ không phải 16/32? Quy tắc nhân đôi channel sau mỗi pool từ đâu ra?
  - Công thức output size qua pool: vì sao 7×7 → 3×3 chứ không phải 3.5 (floor)?
- **Backlog kế tiếp:** `Adam` vs `SGD`; cơ chế toán học của `Dropout` (Bernoulli mask + scale).
- **Nấc 2 (nối sang Phase 2A):** với mỗi thành phần đã hiểu, viết luôn **công thức toán + mã giả** vào `/operators/<tên>.md`. Hiểu xong mà không ghi lại thì tới lúc viết C phải học lại từ đầu.

### Phase 1.7 — Thu gọn kiến trúc ⬜ 🔄 (MỚI 21/09, bước 2b — từ feedback tuần 2)

Thầy nêu trực tiếp ở buổi báo cáo: 206,922 params là quá nhiều, một bạn khác chỉ 5,018. Đây là yêu cầu, không phải gợi ý.

**Điều kiện tiên quyết: validation set phải xong trước.** Đổi kiến trúc rồi so accuracy trên test set = lặp lại đúng lỗi data leakage Thầy vừa bắt. Sẽ chọn kiến trúc dựa trên test set, và con số cuối cùng mất ý nghĩa.

Thứ tự làm:
1. **Validation set** (Phase 1) — xong trước, không thương lượng.
2. **Đo baseline** kiến trúc hiện tại trên val: 206,922 params / val acc bao nhiêu.
3. **Port `SmallCNN`** (5,018 params) sang PyTorch, train cùng seed / cùng epoch / cùng optimizer.
   - Keras `padding='same'` + kernel 3×3 = PyTorch `padding=1`.
   - Keras `MaxPooling2D` `padding='valid'` = PyTorch `MaxPool2d(2,2)`, 7//2 = 3.
4. **Bảng so sánh**: params / KB / val acc / test acc / epoch tốt nhất.
5. **Chốt kiến trúc cuối cùng** — mọi phase sau (2A → 2E) chạy trên kiến trúc này.

**Xong là gì:** 1 bảng đánh đổi để báo cáo Thầy — *giảm bao nhiêu lần params, mất bao nhiêu % accuracy*. Con số đáng nói không phải accuracy cao hơn, mà là tỷ lệ đánh đổi.

**Lưu ý khi báo cáo:** 98.88% của bạn kia **thấp hơn** 99.21% hiện tại của mình. Đừng trình bày như "em thua". Trình bày: *41× ít params, đổi 0.33% accuracy*. Và 98.88% không phải trần của 5k params — augmentation nhẹ + train lâu hơn có thể vượt 99%. Nếu đưa được **5,018 params / 99.0%+** thì mạnh hơn cả hai bên hiện có.

**Không copy suông.** Thầy chấm độ hiểu, mà kiến trúc này không phải của mình. Chạy được là bước 1; giải thích được vì sao từng lựa chọn (các câu hỏi ở Phase 1.5) mới là phần của mình.

**File:** `model_comparison.py` — đã có sẵn validation set, cả 2 kiến trúc, bảng đếm params theo lớp, và bảng so sánh cuối.

### Phase 2A — Viết operator ⬜ (bước 3–4 của Thầy)
- Chạy trên **kiến trúc đã chốt ở Phase 1.7**, không phải kiến trúc 206k.
- Mỗi operator có 1 file trong `/operators/`: công thức toán → mã giả (vòng lặp thuần) → đánh giá mức FPGA-friendly.
- Danh sách: `Conv2d`, `ReLU`, `MaxPool2d`, `Flatten`, `Linear` (+ `argmax`). `Softmax` chỉ nếu giữ lại.
- Sau khi viết xong mã giả: rà lại phép toán nào đắt (mục 3.4), quyết định thay/bỏ. **Nếu phải thay thì train lại** — chi phí thật, phải tính trước.
- **Xong là gì:** đủ mã giả để một người không biết PyTorch vẫn code lại được model bằng C.

### Phase 2B — Xác định model inference ⬜ (bước 5 của Thầy)
- Liệt kê rõ: model inference **khác** model train ở chỗ nào.
  - `Dropout` → bỏ. *(Kiến trúc 5k không có Dropout → nếu chốt kiến trúc đó thì bước này không còn việc.)*
  - `BatchNorm` → fold vào Conv/Linear (cả 2 kiến trúc hiện chưa có BN).
  - `Softmax` → bỏ, thay bằng `argmax` trên logits (trừ khi cần % tin cậy).
  - Loss + optimizer → không tồn tại.
- **Xong là gì:** 1 sơ đồ kiến trúc inference cuối cùng, chốt danh sách operator phải code trong C.

### Phase 2C — Trích tham số ⬜ (bước 6 của Thầy)
- Viết script xuất weight + bias của từng lớp ra file C đọc được.
- ✅ **Đã chốt (feedback tuần 2):** Thầy nói PyTorch hay TensorFlow đều được → **giữ PyTorch**, xuất thẳng từ `.pth` ra `.bin`/`.txt`/header C. Không cần qua `.h5`, không đổi framework.
- Ghi rõ **thứ tự chiều** của tensor (PyTorch Conv weight là `[out_ch, in_ch, kH, kW]`, Keras là `[kH, kW, in_ch, out_ch]` — khác nhau!). Sai thứ tự chiều là lỗi phổ biến nhất khi chuyển sang C, và nó không crash, chỉ cho kết quả sai.
- **Xong là gì:** bộ file tham số + 1 file mô tả shape/thứ tự chiều của từng tensor.

### Phase 2D — Golden model C ⬜ (bước 7 của Thầy — **đích chính của project 1**)
- Code inference bằng C thuần: đọc tham số → đọc 1 ảnh 28×28 → chạy qua từng operator → in ra chữ số dự đoán.
- **float32 trước, fixed-point sau.** Nhảy thẳng vào fixed-point thì không phân biệt nổi "sai công thức" với "sai do lượng tử hoá" — mất hàng tuần debug nhầm chỗ.
- Verify theo chiến lược ở mục 6.
- **Xong là gì:** chương trình C chạy được toàn bộ 10.000 ảnh test, accuracy khớp PyTorch (chênh lệch ≤ vài ảnh do làm tròn float).

### Phase 2E — Tối ưu trọng số cho chip ⬜ 🔄 (bước 8 — từ feedback tuần 2)

Thầy yêu cầu "tối ưu trọng số khi đưa vào chip". Phase 1.7 giảm **số** params; phase này giảm **số byte mỗi** param. Hai việc khác nhau, cộng dồn được: 5,018 params float32 = 19.6 KB → int8 = 4.9 KB.

**Điều kiện tiên quyết: Phase 2D float32 phải chạy đúng trước.** Không có bản float32 làm chuẩn thì khi accuracy rớt, không phân biệt được lỗi công thức với sai số lượng tử hoá.

Thứ tự làm:
1. **Đo baseline:** kích thước tham số (float32), accuracy, phân bố giá trị weight từng lớp.
2. **PTQ int8 trong PyTorch** (`torch.ao.quantization`) — nhanh, không train lại. Đo lại accuracy.
3. **Xuất scale/zero-point** kèm tham số int8 sang C (Phase 2C phải mở rộng format file).
4. **Golden model C bản fixed-point**, verify theo mục 6 nhưng so với **bản float32 của chính mình**, không so thẳng với PyTorch.
5. **QAT chỉ khi cần:** nếu PTQ làm rớt accuracy > ~0.5%, lúc đó mới train lại với QAT.
6. **Pruning: chưa làm.** Chỉ mở ra nếu bộ nhớ vẫn không đủ sau quantization, và phải hỏi Thầy trước. *(Với 5k params thì nhiều khả năng không cần.)*

**Xong là gì:** bảng so sánh float32 vs int8 — kích thước (KB), accuracy, và (nếu đo được) latency.

**Cần hỏi Thầy:** int8 hay int16, và FPGA mục tiêu có ràng buộc bit-width cụ thể không (câu hỏi 4, mục 8).

### Phase 3 — Đóng gói & tổng kết
- **Đã xong trước lịch:** demo Gradio vẽ tay (`demo_app.py`) — canvas → grayscale → tự đảo màu theo nền → resize 28×28 → normalize cùng mean/std lúc train → dự đoán kèm % tin cậy.
  - ⚠️ Nếu chốt kiến trúc mới ở Phase 1.7 thì phải cập nhật `demo_app.py` theo (đổi class model + file `.pth`).
- Gộp report từng tuần thành báo cáo tổng, chuẩn bị trình bày trước khi qua project mảng tiếp theo.

### Ngoài trục — chỉ làm nếu dư thời gian
Những mục này **không nằm trên đường AI → IC → ES**. Xem cảnh báo ở mục 0 trước khi tin cách phân loại này.
- So sánh kiến trúc: CNN vs MLP thuần; optimizer/activation khác. *(Phần so sánh CNN sâu/nông đã lên Phase 1.7.)*
- Mở rộng dữ liệu: EMNIST, hoặc tự viết/chụp chữ số của mình để test.
- Transfer learning, thử framework thứ 2 (PyTorch ↔ Keras) — **không còn cần thiết**, Thầy đã xác nhận framework nào cũng được.
- **Data augmentation, tinh chỉnh hyperparameter** — không giảm params, nhưng là công cụ **bù accuracy sau khi đã cắt params**. Nếu model 5k rớt xuống dưới 99% và muốn kéo lại thì đây là chỗ dùng. Làm sau Phase 1.7, không phải trước.
- ~~Global Average Pooling thay `fc1`~~ — **bỏ**, kiến trúc tham chiếu dùng conv3 + pool3 + Dense thẳng, không dùng GAP.

---

## 6. Chiến lược verification — chỗ dễ mất hàng tuần nhất

Golden model chỉ có giá trị khi chứng minh được nó khớp với model Python. Cách làm sai phổ biến: chạy cả test set rồi so accuracy cuối. Accuracy khớp không có nghĩa là đúng, và accuracy lệch thì không biết lệch ở lớp nào.

**Cách đúng — so từng lớp:**

1. Từ PyTorch: chọn **1 ảnh cố định**, dump tensor đầu ra của từng lớp ra file (`conv1_out.txt`, `pool1_out.txt`, ...).
2. C đọc cùng ảnh đó, in ra tensor cùng vị trí.
3. So sai số tuyệt đối lớn nhất giữa 2 bên.
4. **Ngưỡng:** float32 thì `max_abs_diff < 1e-4` là bình thường. Lệch lớn hơn = sai công thức, không phải sai số làm tròn.
5. Debug theo thứ tự `Conv1 → ReLU → Pool1 → ... → FC cuối`. **Dừng ở lớp đầu tiên lệch** — các lớp sau lệch là hệ quả, sửa chúng vô nghĩa.
6. Chỉ khi cả chuỗi khớp mới chạy toàn bộ 10.000 ảnh test và so accuracy.

**Với bản fixed-point (Phase 2E):** ngưỡng `1e-4` không dùng được, sai số lượng tử hoá lớn hơn thế nhiều. Thay bằng: so **bản int8 với bản float32 của chính mình**, tiêu chí là số ảnh dự đoán lệch trên 10.000 ảnh test, không phải sai số tensor.

3 lỗi hay gặp nhất khi port sang C, theo thứ tự tần suất: sai thứ tự chiều tensor → sai padding → quên normalize (mean/std) ở đầu vào.

---

## 7. Cấu trúc repo — cập nhật

```
/notebooks      # code Python (train, error analysis, demo, so sánh kiến trúc)
/reports        # mỗi buổi/tuần 1 file .md
/models         # checkpoint .pt (digit_cnn.pth, digit_cnn_val.pth, small_cnn.pth)
/operators      # công thức + mã giả từng operator (.md)
/params         # weight/bias đã trích + file mô tả shape (+ scale/zero-point khi quantize)
/golden_model   # source C + Makefile + kết quả đối chiếu (float32 và fixed-point)
```

**File chính:**

| File | Vai trò |
|---|---|
| `mnist_digit_recognition.py` | CNN baseline tuần 1 (206,922 params). Giữ làm mốc so sánh |
| `model_comparison.py` | 🔄 Validation set + DigitCNN vs SmallCNN + đếm params + bảng đánh đổi |
| `error_analysis.py` | Confusion matrix, accuracy theo lớp |
| `demo_app.py` | Demo Gradio vẽ tay |

**Checkpoint — đừng để đè nhau:**

| File | Do file nào tạo | Train trên | Cách lưu |
|---|---|---|---|
| `digit_cnn.pth` | `mnist_digit_recognition.py` | 60k | epoch cuối |
| `digit_cnn_val.pth` | `model_comparison.py` | 50k (tách 10k val) | best val acc |
| `small_cnn.pth` | `model_comparison.py` | 50k | best val acc |

`demo_app.py` và `error_analysis.py` đang đọc `digit_cnn.pth` — nếu đổi sang model mới thì phải sửa cả tên file lẫn class model.

**Cách vận hành (giữ từ v2):**
- Mỗi buổi làm là commit, không đợi hết tuần.
- Mỗi thứ 4: report đủ 3 phần (kết quả – vấn đề – kế hoạch), rồi mới viết lại kế hoạch 1 phase/tuần tới.
- Gặp khái niệm mới → hỏi → hiểu → áp dụng vào code, không học chay lý thuyết.

---

## 8. Câu hỏi cần chốt với Thầy

### ✅ Đã được trả lời ở feedback tuần 2

- **Định dạng file tham số / framework.** Thầy: *"thư viện PyTorch hay TF đều được, không sao cả"* → `.h5` chỉ là ví dụ, không bắt buộc. **Giữ PyTorch**, xuất thẳng `.pth` → `.bin`/`.txt`. Rủi ro #3 (mục 9) bị loại.
- **Ưu tiên: Phase 1.5 hay validation set trước?** Thầy: *"viết code có thêm validation set"* → validation set là yêu cầu đã chốt, làm song song với Phase 1.5 tuần 3.
- **Params 206,922 có phải vấn đề không?** Thầy nêu trực tiếp ở buổi báo cáo, kèm ví dụ model 5,018 params → phải giảm. Sinh Phase 1.7.

### ⬜ Còn phải hỏi (buổi thứ 4 tới)

1. **Golden model C có nằm trong project 1 (mảng AI) không, hay là đầu việc của project IC?** Em đọc tài liệu Thầy thì thấy nó vẫn thuộc công đoạn AI ("công đoạn học về IC, ES sẽ diễn ra sau khi done bước 1"), nên em mở rộng phạm vi roadmap. Nhờ Thầy xác nhận. *(Câu này quyết định toàn bộ v3 đúng hay sai.)*
2. **Bước "thay phép toán khó triển khai xuống FPGA":** kiến trúc của em chỉ có `Softmax` là đắt (exp + chia), mà softmax bỏ được khi inference vì `argmax` cho cùng kết quả. Vậy bước này còn việc gì không, hay Thầy muốn em thêm lớp khác (BatchNorm chẳng hạn) để có cái mà xử lý cho đúng quy trình?
3. **Mức "thuần" của golden model C:** chỉ dùng thư viện chuẩn C (`stdio.h`, `stdlib.h`), hay được dùng `math.h` (`expf`, `sqrtf`)? Câu trả lời đổi cách em viết softmax nếu vẫn phải giữ nó.
4. **"Tối ưu trọng số khi đưa vào chip" — cụ thể tới đâu?**
   - Thầy muốn int8, int16 hay fixed-point Qm.n tự chọn?
   - FPGA mục tiêu là board nào, có ràng buộc bit-width / tài nguyên DSP cụ thể không?
   - Việc này làm **sau** khi golden model C float32 chạy đúng, hay Thầy muốn làm ngay từ lúc trích tham số? *(Em nghiêng về làm sau — cần bản float32 làm chuẩn để verify.)*
5. **Ngưỡng accuracy chấp nhận được là bao nhiêu?** *(mới)* Model 5,018 params của bạn kia đạt 98.88%, model 206,922 params của em đạt 99.21%. Em nên chạy theo con số params (càng nhỏ càng tốt), hay có sàn accuracy phải giữ? Nếu em giảm còn 5k mà kéo accuracy lên 99% thì có đạt không?
6. **Em có nên dùng đúng kiến trúc của bạn kia không**, hay Thầy muốn em tự thiết kế một kiến trúc khác cùng ngân sách params? *(mới)*

---

## 9. Rủi ro — nói thẳng

**Rủi ro số 1: chạy trước Thầy.** v2 đã ghi nhận Phase 1 và demo Gradio làm sớm ngoài lịch. Tuần 3 Thầy nói rõ: **không** tối ưu, **không** thêm tính năng, mà đào sâu cái đã có. Golden model C và quantization đều hấp dẫn hơn hẳn việc ngồi viết công thức `Conv2d`, và cái bẫy là nhảy vào đó rồi bỏ dở phần Thầy đang yêu cầu.

> **Nguyên tắc v3: golden model C và Phase 2E là đích ngắm, không phải việc của tuần này.**
>
> Việc tuần 3 đúng là: **validation set → Phase 1.7 (thu gọn kiến trúc) → Phase 1.5 (hiểu sâu)**. Cả 3 đều là thứ Thầy trực tiếp yêu cầu, không phải mình tự thêm.

**Rủi ro số 2: backlog phình to.** v2 đã có sẵn 7 mục backlog. v3 cộng thêm 6 phase mới (1.7, 2A–2E). Cách xử lý: mục "Ngoài trục" ở cuối mục 5 — đã tách riêng, không trộn vào đường chính.

**Rủi ro số 3: framework mismatch (`.h5` vs `.pt`).** ✅ **ĐÃ LOẠI (21/09).** Thầy xác nhận PyTorch hay TF đều được.

**Rủi ro số 4: verify sai cách.** Viết xong 500 dòng C rồi mới đi so accuracy → sai ở đâu không biết. Mục 6 tồn tại để chặn rủi ro này.

**Rủi ro số 5: quantization debug mù.** Nếu gộp fixed-point vào lúc viết golden model C, khi accuracy rớt sẽ không biết do sai công thức hay do lượng tử hoá. Chặn bằng: float32 chạy đúng và verify xong mới đụng tới int8.

**Rủi ro số 6: chốt kiến trúc muộn.** 🔄 *(mới)* Nếu viết `/operators/*.md` và golden model C cho kiến trúc 206k rồi mới đổi sang 5k, toàn bộ phần đó phải viết lại — mã giả khác, số lớp khác, shape khác. Chặn bằng: **Phase 1.7 phải xong trước Phase 2A**, không có ngoại lệ.

**Rủi ro số 7: copy kiến trúc mà không hiểu.** 🔄 *(mới)* Kiến trúc 5,018 params là của bạn khác. Chạy được nó không chứng minh được gì với Thầy — người đang chấm độ hiểu. Chặn bằng: các câu hỏi bổ sung ở Phase 1.5, phải tự trả lời được trước khi mang đi báo cáo.

---

## 10. Ứng dụng thực tế (giữ từ v2)

OCR số hoá văn bản viết tay · Ngân hàng đọc số tiền trên séc · Bưu chính đọc mã bưu điện (gốc gác MNIST, từ USPS) · Chấm điểm/nhập liệu tự động · Nền tảng mở rộng: biển số xe, khuôn mặt, computer vision công nghiệp · MNIST-CNN cũng là bài tập kinh điển của TinyML — hướng cho project ES sau này.

---

## Phụ lục — Ghi nhận các buổi gặp Thầy

### Tuần 2 → 3 (buổi gặp)
- **Vì sao model chưa có validation set** (chỉ train/test) — đã tự phát hiện ở report tuần 2 (data leakage nhẹ do dùng test set theo dõi trong lúc train); Thầy xác nhận cần sửa sớm.
- **Params vs FLOPs** — vì sau này làm project IC/ES trên nền project này (khả năng chạy trên FPGA), params lớn (`fc1` ~97% tổng params) là vấn đề bộ nhớ phần cứng, cần phân biệt rõ với FLOPs (ảnh hưởng tốc độ/năng lượng).
- **Định hướng tuần 3** — KHÔNG phải tối ưu/thêm tính năng. Theo phương pháp "thực chiến": từ code đã chạy → phân tích vì sao dùng thành phần này → hiểu phép toán bên trong. Report/slide làm cá nhân, không làm chung dù có thể trao đổi.

### Feedback tuần 2 (21/09/2026)
- **"Viết code có thêm validation set"** → Phase 1 còn 1 việc bắt buộc, làm tuần 3.
- **Params 206,922 là quá nhiều** — Thầy nêu ví dụ model của một bạn năm 4 chỉ **5,018 params, test acc 98.88%** (Keras/TF, 3 conv + 3 pool + Dense(144→10) thẳng, không FC ẩn, không Dropout). → sinh **Phase 1.7**.
- **"Tối ưu trọng số khi đưa vào chip"** → sinh **Phase 2E**; quantization rời "Ngoài trục" lên trục chính, đặt sau golden model C float32.
- **"Thư viện PyTorch hay TF đều được, không sao cả"** → chốt giữ PyTorch; rủi ro `.h5` bị loại; bỏ mục "thử framework thứ 2" khỏi backlog.
