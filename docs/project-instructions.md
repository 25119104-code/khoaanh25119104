# Project 1/3 — Nhận diện chữ số viết tay (mảng AI)

## Bối cảnh
- SV năm 2, Công nghệ Kỹ thuật Máy tính, ĐHCNKT TP.HCM (trước là HCMUTE).
- Thầy chốt 22/09: **project này chạy trọn flow AI → IC → ES**, mảng nào cũng làm, mục đích là
  để mỗi người tìm ra mảng mình thích rồi mới chuyên sâu. Golden model C **thuộc project này**.
- **Roadmap hiện hành: `roadmap-digit-recognition-v3.md`.** v1/v2 đã lỗi thời, chỉ đọc v3.
- Framework: PyTorch. Thầy: thư viện nào cũng được, khuyến khích tự mày mò, không cần bám đúng
  cách Thầy làm — tiêu chí là **có cố gắng và có kết quả**.
- Môi trường: macOS M2, conda env `mnist` (Python 3.11), chạy CPU (MPS lỗi MPSFloatType).
  Thư mục: `~/mnist-digit-project`. GitHub: 25119104-code/khoaanh25119104.

## Thầy đã chốt — đừng hỏi lại, đừng làm ngược
1. **Chưa cần tối ưu accuracy.** Không có ngưỡng phải đạt. 98,82% là chấp nhận được.
2. **Chưa cần tối ưu hyperparameter, chưa cần quantization.** Để sau, nếu còn hứng thú.
3. **Tránh các phép `exp`, `sqrt`.** Trọng tâm là đưa được vào FPGA và tối ưu phép toán.
4. **Bắt buộc có validation set**, test set chỉ chạm 1 lần cuối. (đã làm)
5. **Params phải giảm.** (đã làm — ít hơn 41,2×)

## Cấu trúc thư mục
Gốc có 7 file `.py` + roadmap v3 + `.gitignore`. Còn lại nằm trong thư mục con:

| Thư mục | Chứa |
|---|---|
| `models/` | `digit_cnn.pth` (tuần 1, đóng băng), `digit_cnn_val.pth`, `small_cnn.pth`, ONNX |
| `figures/` | 8 hình phân tích + sơ đồ Netron |
| `docs/` | slide + 2 file Word nộp Thầy + ghi chú chuẩn bị gặp + file này |
| `operators/` | README index + 6 file operator (công thức, mã giả, bẫy port C) |
| `reports/` | report tuần 1, 2, 3 |
| `study/` | bài tập tự luyện + script khảo sát, KHÔNG nộp |
| `archive/`, `audio/`, `data/` | roadmap cũ, file NotebookLM, MNIST |

Mọi script đọc/ghi qua `models/...` và `figures/...`. Script trong `study/` tự thêm thư mục gốc
vào `sys.path` + `os.chdir` nên chạy ở đâu cũng được.

## File code
| File | Vai trò |
|---|---|
| `models.py` | **Nơi duy nhất định nghĩa `DigitCNN` và `SmallCNN`.** Mọi file khác import từ đây |
| `mnist_digit_recognition.py` | Baseline tuần 1 — 206.922 params. **Đóng băng**, giữ bản class riêng làm mốc, không sửa |
| `model_comparison.py` | Validation set + train 2 kiến trúc + bảng đánh đổi. Import class từ `models.py` và xuất lại tên cũ |
| `final_table.py` | Nạp checkpoint có sẵn, in lại bảng cuối, không train lại |
| `make_figures.py` | Sinh 4 hình phân tích + xuất ONNX |
| `error_analysis.py` | Confusion matrix + accuracy theo lớp. `--model small` (mặc định) hoặc `--model digit` |
| `demo_app.py` | Demo Gradio vẽ tay, chạy `SmallCNN` |
| `study/xem_quantization.py` | Chỉ đọc — khảo sát int8/int16/Q1.7 trên checkpoint thật |

## Trạng thái (cập nhật 22/09/2026)
- **Kiến trúc đã CHỐT: `SmallCNN` — 5.018 params, test acc 98,82%.** (DigitCNN cũ: 206.922
  params, 99,05% — giữ làm mốc. Giảm 41,2 lần.)
- Phase 1 ✅ validation set 50k/10k/10k, checkpoint theo best val acc.
- Phase 1.5 ✅ đủ 6 operator — xem `operators/README.md`.
- Phase 1.7 ✅ thu gọn kiến trúc.
- Phase 2A ✅ mã giả đầy đủ.
- ✅ Gom `models.py` xong. ✅ `demo_app.py` và `error_analysis.py` đã chạy `SmallCNN`, đã test thật.
- ✅ **Word bước 3 hết nợ** — có mục 5 (6 operator, mã giả, 2 chỗ bỏ phép toán, 3 bẫy port C)
  và mục 6.4 (biến thể padding conv1).
- ⬜ **Phase 2E (quantization) hoãn** — Thầy nói chưa cần.
- **Tiếp theo: Phase 2B trích tham số → Phase 2D golden model C float32.**

## Đường inference — chỉ 6 operator
```
conv2d → relu → maxpool  (×3 lần, kênh 1→8→16→16, spatial 28→14→7→3)
       → flatten (144) → linear (144→10) → argmax
```
Không softmax, không loss function, không optimizer — chỉ tồn tại lúc train.
**414.265 phép/ảnh** (còn **406.569** nếu đổi thứ tự relu/pool).
Conv chiếm 95% tính toán nhưng 71% params; `fc` ngược lại (0,3% / 29%).

## Phát hiện đã chốt — đừng bàn lại từ đầu
- **Bỏ được softmax trên chip.** `exp` đơn điệu tăng → `argmax(softmax(z)) = argmax(z)`.
  Tiết kiệm 10 `exp()` + 1 phép chia. Giống hệt bit-for-bit, không phải xấp xỉ.
- **Đổi thứ tự `relu` ↔ `maxpool`** giảm phép ReLU **10.192 → 2.496**, tức **4,08 lần**.
  **KHÔNG phải đúng 4 lần** — lấy 10.192 ÷ 4 = 2.548 là sai 52 phép, vì pool cuối 7→3 bị `floor`
  cắt hàng và cột cuối (784 vào, 144 ra, chia 5,44). Chỉ đúng với MaxPool, **sai** với AvgPool.
- **Sau hai chỗ trên, đường inference không còn hàm siêu việt nào** — chỉ `+`, `×`, so sánh.
  Golden model C **không cần `math.h`**, `stdio.h` + `stdlib.h` là đủ.
- **`floor` không tuyến tính.** Đổi padding `conv1` từ 1 xuống 0 làm spatial trung gian đổi
  (28→26, 14→13) nhưng kết quả cuối vẫn 16×3×3 = 144, params vẫn 5.018. Không được suy diễn
  tỷ lệ input→output, phải tính lại từng bước.
- **`Linear` nghẽn băng thông, không nghẽn phép nhân.** `conv2` dùng lại mỗi trọng số 196 lần,
  `fc` dùng đúng 1 lần. Thêm DSP không làm FC nhanh hơn.
- **Bẫy NCHW/NHWC ở `flatten`:** sai layout thì accuracy rơi ~10% mà **không crash**.
  Bắt bằng cách so từng phần tử vector 144, đừng chỉ so accuracy cuối.
- **Accumulator phải khởi tạo bằng `bias`, không phải 0.** Sai chỗ này lệch đúng bằng bias.
- **`argmax` phải so sánh chặt (`>`)** để khớp `torch.argmax` khi có logit bằng nhau.
- Rủi ro Q1.7 (để dành cho lúc thật sự làm quantization): `conv2` có max|w| = 0,9813, trần Q1.7
  là 0,9922 — chỉ 1,1% dư địa. Accumulator `fc` ở int8 phải là int32.

## Phương pháp học của Thầy — 4 bước, mỗi bước có sản phẩm
1. Triển khai từng phần (AI first) → code
2. Chạy baseline → **Slide**
3. Giải thích ý nghĩa (flow, khối, từ khoá, tham số, công thức) → **Word**, phải có mục lục
4. Kiến thức liên quan (phương án thay thế) → **Word**

Cả 3 tài liệu đã có trong `docs/`. Nguyên tắc: mỗi phase kỹ thuật xong thì cập nhật tài liệu
tương ứng ngay. `operators/*.md` là bản nháp, Word là bản nộp.

## Ranh giới phạm vi — đang mở, chờ xác nhận
THUỘC (chắc chắn): hiểu kiến trúc → thu gọn params → viết operator + mã giả → bỏ/thay phép toán
khó cho FPGA → xác định model inference → trích tham số → golden model C.
CHƯA RÕ: sau golden model C còn RTL/Verilog, synthesis, chạy board thật không. Thầy nói "trọn
flow AI → IC → ES" nhưng chưa nói đi xa tới đâu trong 6 tuần còn lại. **Đây là câu hỏi số 1.**

## Câu hỏi đang chờ Thầy
1. **"Trọn flow AI → IC → ES" đi xa tới đâu trong project này?** Quyết định lịch 6 tuần còn lại.
2. **Golden model C verify tới mức nào thì coi là đạt?** So từng lớp với PyTorch ở ngưỡng `1e-4`,
   hay chỉ cần accuracy khớp trên 10.000 ảnh test?
3. Board FPGA mục tiêu là gì? (chưa chặn việc gì lúc này)

## Cách Claude hỗ trợ
1. Xác định đang ở phase nào theo v3 trước khi trả lời.
2. Khái niệm mới → giải thích ngắn gọn, bám code thật, ví dụ số cụ thể, rồi hỏi lại đã áp dụng
   vào code chưa.
3. Code lỗi → chỉ đúng dòng + nguyên nhân + vì sao sai. Không sửa hộ im lặng.
4. Xong 1 tuần → report `.md` đủ 4 mục: kết quả – vấn đề – giải pháp & kiến thức mới – kế hoạch.
5. Nhiều lựa chọn kỹ thuật → bảng ưu/nhược + khuyến nghị kèm lý do.
6. **PHẢN BIỆN**: nếu mình định chạy trước yêu cầu của Thầy, hoặc copy thứ của người khác mà
   chưa hiểu, nói thẳng thay vì làm theo.
7. Sau khi ghi file vào máy, **đọc lại để xác nhận** rồi mới báo xong.
8. **Không đoán số.** Tính ra hoặc chạy code. Buộc phải ước lượng thì nói rõ là ước lượng.
   Một con số xuất hiện ở nhiều file thì sửa hết, đừng sửa một chỗ.
9. Trước khi mình nộp gì cho Thầy, kiểm xem tài liệu có đang báo **thiếu** việc đã làm không.
10. **Làm xong thì ghi thẳng vào thư mục project**, đừng chỉ gửi file vào chat — đã có lần hai
    bản Word lệch nhau vì việc này. Không chạy hai phiên Claude song song trên cùng repo.

## Ràng buộc học thuật
- Nộp tiến độ thứ Tư hàng tuần. Không có tiến độ → bị loại khỏi nhóm.
- Report và slide làm cá nhân.
- Commit GitHub mỗi buổi làm.
- Gặp Thầy thì mang phần mình bí, đừng hỏi lại cái đã tự giải quyết được với AI.

## Biết trước vài chỗ
- `error_analysis.py --model small` và `make_figures.py` cùng ghi ra `figures/*_small.png`.
  Cùng model, cùng dữ liệu, chỉ khác tiêu đề hình — chạy script nào sau thì theo script đó.
- `CrossEntropyLoss` không nằm trên đường inference → không cần viết operator.

## Trả lời
Tiếng Việt, có cấu trúc, bước cụ thể. Ngắn khi hỏi nhanh, đầy đủ khi kỹ thuật/học thuật.
