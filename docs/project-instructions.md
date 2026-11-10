# Project 1/3 — Nhận diện chữ số viết tay (mảng AI)

## Bối cảnh
- SV năm 2, Công nghệ Kỹ thuật Máy tính, ĐHCNKT TP.HCM (trước là HCMUTE).
- Thầy chốt 22/09: **project này chạy trọn flow AI → IC → ES**, mảng nào cũng làm, mục đích là
  để mỗi người tìm ra mảng mình thích rồi mới chuyên sâu. Golden model C **thuộc project này**.
- Gặp Thầy 23/09: xác nhận **đi full flow** → phải hiểu bản chất thuật toán, vì lên mạch sẽ gặp lại.
  Thầy: **cứ nhờ AI làm, học dần hiểu dần.** Ghi chú: `docs/ghi-chu-gap-thay-20260923.md`.
- **Roadmap hiện hành: `roadmap-digit-recognition-v3.md`.** v1/v2 đã lỗi thời, chỉ đọc v3.
- Framework: PyTorch. Thầy: thư viện nào cũng được, khuyến khích tự mày mò, không cần bám đúng
  cách Thầy làm — tiêu chí là **có cố gắng và có kết quả**.
- Môi trường: macOS M2, conda env `mnist` (Python 3.11), chạy CPU (MPS lỗi MPSFloatType).
  Thư mục: `~/mnist-digit-project`. GitHub: 25119104-code/khoaanh25119104.
- **Vivado không cài được trên Mac M2** (chỉ có bản x86 Windows/Linux). MacBook Air M2, SSD 256 GB, trống 28 GiB
  (03/11). RTL + mô phỏng làm trên Mac bằng Icarus Verilog/Verilator; synthesis/nạp board dùng máy x86 hoặc Docker.
  Phương án đầy đủ: mục 5 của `reports/report-dot2-tuan01-20261104.md`.
- **Công cụ RTL đã cài (10/11):** Icarus Verilog 13 + Surfer 0.7 (brew). Chạy bài: `./chay.sh <thư mục bài>` trong
  `rtl/bai-tap-nhap-mon/`. Chưa cài Yosys. **Đang là prelab, chưa vào lab: mọi việc làm trên máy cá nhân**, không có
  máy lab để cài Vivado. Mua SSD ngoài / Docker Vivado: **hoãn**, chờ Thầy cho biết khi nào bắt buộc cần Vivado.

## Thầy đã chốt — đừng hỏi lại, đừng làm ngược
1. **Chưa cần tối ưu accuracy.** Không có ngưỡng phải đạt. 98,82% là chấp nhận được.
2. **Chưa cần tối ưu hyperparameter.** Quantization: 22/09 Thầy nói chưa cần, nhưng **07/10 Thầy giao làm PTQ + QAT,
   rồi sang C, rồi RTL** → đã làm (tuần 6). Đường inference hiện tại là **int8**.
3. **Tránh các phép `exp`, `sqrt`.** Trọng tâm là đưa được vào FPGA và tối ưu phép toán.
4. **Bắt buộc có validation set**, test set chỉ chạm 1 lần cuối. (đã làm)
5. **Params phải giảm.** (đã làm — ít hơn 41,2×)

## Cấu trúc thư mục
Gốc có 13 file `.py` + roadmap v3 + `.gitignore`. Còn lại nằm trong thư mục con:

| Thư mục | Chứa |
|---|---|
| `models/` | `digit_cnn.pth` (tuần 1, đóng băng), `digit_cnn_val.pth`, `small_cnn.pth`, `small_cnn_qat_int8.pth` / `_int4.pth` (tuần 6), ONNX |
| `params_int8/` | Tham số PTQ int8 (cấu hình B) + `tham_so.txt` (zp, M0, shift) + `golden_int8/` (ảnh mẫu từng lớp + `fc_acc_test.txt` 10.000 ảnh) — **sinh tự động, không sửa tay, không calibrate lại trên máy khác** |
| `params_int8_qat/` | Cùng định dạng, từ checkpoint QAT int8 (`qat.py --bits 8 --xuat`) |
| `figures/` | 8 hình phân tích + sơ đồ Netron + 3 hình `conv_*.png` (tính 1 điểm conv, vùng đệm) |
| `params/` | `weights.txt` (4.968) + `biases.txt` (50) + README bảng offset — **sinh tự động, không sửa tay** |
| `golden/` | Ảnh mẫu + đầu ra PyTorch từng lớp + logit cả test set — chuẩn để so Python/C |
| `golden_c/` | **Golden model C float32** `golden_model.c` + **int8** `golden_int8.c` + README. File chạy được không commit |
| `docs/` | slide + 2 file Word nộp Thầy + ghi chú gặp Thầy + `tong-hop-video-thay.md` + **`on-tap-flow-7-buoc.md`** (tài liệu ôn theo slide 7 bước, 11 bài tính tay) + **`on-tap-int8-ptq-qat-c.md`** (ôn PTQ/QAT/golden C int8, có tự kiểm + đáp án) + file này. `tien-do-on-tap.md` chỉ có trong Project |
| `operators/` | README index + 6 file operator (công thức, mã giả, bẫy port C) |
| `PDF/` | Bản in của `operators/*.md`, sinh bằng `tools/xuat_pdf.py` — sửa `.md` rồi chạy lại |
| `reports/` | report tuần 1–6 (đợt 1) + đợt 2 tuần 1 (`report-dot2-tuan01-20261104.md`) |
| `rtl/bai-tap-nhap-mon/` | 10 bài nhập môn Verilog Thầy giao 07/11. Mỗi bài 1 thư mục: RTL, testbench tự kiểm PASS/FAIL, `report.md`, `song.png`. `chay.sh` + `tools/ve_song.py` (VCD → PNG). `sim`, `*.vcd` không commit |
| `study/` | bài tập tự luyện + script khảo sát, KHÔNG nộp |
| `tools/` | `xuat_pdf.py` — sinh PDF từ `operators/*.md` |
| `archive/`, `audio/`, `data/` | roadmap cũ, file NotebookLM, MNIST |

Mọi script đọc/ghi qua `models/...` và `figures/...`. Script trong `study/` tự thêm thư mục gốc
vào `sys.path` + `os.chdir` nên chạy ở đâu cũng được. Comment và chuỗi in trong code viết
**tiếng Việt có dấu**.

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
| `mnist_io.py` | Đọc MNIST test **từ file raw idx** (không torchvision) + chuẩn hoá giống PyTorch |
| `export_params.py` | Phase 2B: trích tham số → `params/`, sinh `golden/`. Tự kiểm đọc lại khớp từng bit |
| `inference_python.py` | Inference **vòng lặp thuần, không PyTorch**, mảng 1 chiều như C. 2 bản conv: `if` và đệm 0. Bản nháp của golden model C |
| `make_conv_figures.py` | Vẽ 3 hình tính 1 điểm conv bằng số thật |
| `ptq_int8.py` | PTQ int8 **số nguyên thuần** (không dùng `torch.ao`), so cấu hình A–F trên val, xuất `params_int8/`. `--chi-tham-chieu`: sinh lại `fc_acc_test.txt` từ tham số đã xuất, không calibrate lại |
| `qat.py` | QAT fake-quant + STE, đo trên đúng đường số nguyên của `ptq_int8.py`. `--bits 8 --xuat`: xuất `params_int8_qat/` từ checkpoint, không train |
| `study/xem_quantization.py` | Chỉ đọc — khảo sát int8/int16/Q1.7 trên checkpoint thật |
| `study/thu_batchnorm.py` | Thí nghiệm thêm BN + gộp BN vào Conv. Kết quả: `study/thi-nghiem-batchnorm.md` |
| `study/kiem_tra_tinh_tay.py` | Đáp án 11 bài tính tay của `docs/on-tap-flow-7-buoc.md` |
| `study/tra-cuu-smallcnn.xlsx` | Excel tra cứu 1 điểm mọi lớp: địa chỉ in/w, dòng trong file txt, acc cộng dồn, so PyTorch |

## Trạng thái (cập nhật 10/11/2026)
- **Kiến trúc đã CHỐT: `SmallCNN` — 5.018 params, test acc 98,82%.** (DigitCNN cũ: 206.922
  params, 99,05% — giữ làm mốc. Giảm 41,2 lần.)
- Phase 1 ✅ validation set 50k/10k/10k, checkpoint theo best val acc.
- Phase 1.5 ✅ đủ 6 operator — xem `operators/README.md`.
- Phase 1.7 ✅ thu gọn kiến trúc.
- Phase 2A ✅ mã giả đầy đủ. `operators/conv2d.md` (05/10): mã giả chính là **đệm 0, luôn 9 tap**
  (khớp golden C), bản `if` chỉ để tham khảo; có quy ước zero-padding dưới công thức 1 điểm.
- ✅ Gom `models.py` xong. ✅ `demo_app.py` và `error_analysis.py` đã chạy `SmallCNN`, đã test thật.
- ✅ **Word bước 3 hết nợ** — có mục 5 (6 operator, mã giả, 2 chỗ bỏ phép toán, 3 bẫy port C)
  và mục 6.4 (biến thể padding conv1).
- ✅ **Phase 2E quantization (tuần 6, 14/10)** — `ptq_int8.py` + `qat.py`. PTQ int8 98,84% (float 98,82%),
  khác float 9 ảnh; QAT int8 98,91%. Cấu hình B: ReLU gộp vào kẹp (zp = −128), M0 31 bit, per-tensor.
  Report `reports/report-tuan06-20261014.md`, slide `docs/slide-quantization-tuan06.pptx`.
- ✅ **Golden model C int8 (03/11)** — `golden_c/golden_int8.c`. Khớp Python **từng bit**: ảnh mẫu 11/11 file,
  10 acc lớp fc của 10.000/10.000 ảnh, cả PTQ (`params_int8/`) và QAT (`params_int8_qat/`). Đã chạy lại trên Mac 10/11:
  ra y hệt máy ảo Linux. Report `reports/report-dot2-tuan01-20261104.md`.
- ✅ **Phase 2B trích tham số** — 2 file `weights.txt` + `biases.txt` theo ý Thầy, `%.9g`, khớp từng bit.
- ✅ **Inference Python** — khớp PyTorch từng lớp; 10.000 ảnh: 98,82%, trùng dự đoán 10.000/10.000.
- ✅ **Report tuần 4** (`reports/report-tuan04-20260930.md`) đã commit. PDF operators đã sinh lại.
- ✅ **Phase 2D golden model C float32** (`golden_c/`) — chỉ stdio + stdlib, 395.136 MAC conv/ảnh, khớp PyTorch từng lớp;
  10.000 ảnh: 98,82%, trùng dự đoán 10.000/10.000, logit lệch ≤ 3,05e-05.
- ✅ Thí nghiệm BatchNorm (study): không tăng accuracy, train nhanh ×2, gộp BN → 0/10.000 ảnh khác. Giữ model không BN.
- ✅ Tổng hợp 9 video của Thầy (`docs/tong-hop-video-thay.md`): đang ở bước 6/7 của flow.
- ✅ `study/tra-cuu-smallcnn.xlsx` — tra cứu 1 điểm conv1/conv2/conv3/maxpool/fc + argmax bằng số thật,
  có số dòng trong file txt; khớp PyTorch. Dùng để tự kiểm bài tính tay.
- Đợt 2 bắt đầu 01/11. Báo cáo 04/11: `reports/report-dot2-tuan01-20261104.md`.
- ✅ **GitHub đã đồng bộ (10/11)**: commit `fd4666a` (INT8, golden C int8, report, Bài 1) + `e1848b7`. Repo sạch.
- **Đang ở: phần Vi mạch (IC).** 07/11 Thầy giao 10 bài nhập môn Verilog, deadline ≤ 1 tuần, mỗi bài có report +
  "học được gì", **làm chậm, hiểu rõ rồi mới qua bài mới**. ✅ Bài 1 PASS 8/8 (chạy trên Mac 10/11); chờ user trả lời
  4 câu tự kiểm rồi mới qua Bài 2.
- 09/11 Thầy nhắn (cả nhóm): chuyển INT8 bằng PTQ rồi QAT + PPTX ý nghĩa và kết quả + đồng bộ GitHub → **đã có đủ**
  (slide nộp: `docs/slide-quantization-tuan06.pptx`; slide 9–10 hơi cũ, user chọn không làm PPTX mới, ôn bằng chép tay).
- Gặp Thầy dự kiến tuần sau (16/11 trở đi), lịch chưa chốt.
- **Lịch học tuần 10–15/11:** T3 Bài 1 + ôn INT8 mục 1–4 + đọc `ptq_int8.py`; T4 ôn mục 5–7 + `qat.py` +
  `golden_int8.c`; T5 Bài 2–3; T6 Bài 4–5; T7 Bài 6–7; CN Bài 8–10.
- Chưa làm: RTL cho CNN (sau 10 bài, bắt đầu từ khối requantize); QAT nhiều seed + đối chứng QAT (từ tuần 6);
  cập nhật Word bước 3 với phần int8.

## Đường inference — chỉ 6 operator

    conv2d → relu → maxpool  (×3 lần, kênh 1→8→16→16, spatial 28→14→7→3)
           → flatten (144) → linear (144→10) → argmax

Không softmax, không loss function, không optimizer — chỉ tồn tại lúc train.
**414.265 phép/ảnh** (còn **406.569** nếu đổi thứ tự relu/pool).
Conv chiếm 95% tính toán nhưng 71% params; `fc` ngược lại (0,3% / 29%).

## Phát hiện đã chốt — đừng bàn lại từ đầu
- **FPGA: đệm 0, luôn đủ 9 tap — không dùng `if` bỏ padding** (Thầy góp ý 23/09). 9 bộ nhân song
  song mất 1 chu kỳ dù bỏ bớt tap; `if` tốn thêm LUT so sánh + MUX. Giá: 395.136 thay vì 351.008
  MAC conv (+11,2%). Hai bản ra giống hệt. 414.265 phép/ảnh vốn đã đếm theo kiểu đủ 9 tap.
- **Giá trị đệm = 0 sau chuẩn hoá**, không phải nền ảnh (−0,4242).
- **Tính 1 điểm output conv:** `bias[oc] + Σ_ic Σ_ky Σ_kx in·w`. conv1: 9 MAC, conv2: 72, conv3: 144.
  Mọi kênh vào dồn vào **một accumulator** → một số duy nhất.
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
- Python float64 vs PyTorch float32 lệch ~1e-5 — là độ chính xác, không phải sai thuật toán.
- ~~Rủi ro Q1.7~~ hết hiệu lực: int8 dùng scale per-tensor theo dải thật, không dùng Q1.7.

### Phát hiện int8 (tuần 6 + 03/11)
- **Đầu vào chip = `pixel − 128`** (đảo bit cao nhất), khớp lượng tử hoá đầy đủ ở 7.840.000/7.840.000 pixel.
  Chuẩn hoá biến mất vào scale/zp. conv1 nhận `x = q − zp = pixel − 33` ∈ [−33, 222] (9 bit có dấu).
- **zp = −128 ⇒ `q − zp` = đảo bit cao nhất**, ra [0, 255]. Trừ zp ở conv2, conv3, fc không tốn bộ cộng.
- **Đệm: trừ zp trước, rồi mới đệm 0.** Đệm số nguyên 0 thẳng lên q là sai khi zp ≠ 0.
- **ReLU gộp vào bước kẹp** (scale theo dải sau ReLU → zp = −128): kẹp `max(q, −128)` chính là ReLU.
- **Requantize đặt SAU maxpool được, giống hệt từng bit** (hàm không giảm giao hoán với max): 10.192 → 2.496 lần
  requantize/ảnh, kiểm 0/10.000 ảnh khác ở cả PTQ và QAT. Giá: maxpool so sánh trên acc 20 bit thay vì 8 bit.
- **Accumulator 20 bit có dấu đủ cho MỌI ảnh** (chặn trên tính từ trọng số thật). Đo trên test: 17–19 bit.
- **acc × M0 cần 51 bit** (20 + 31). M0 15 bit cho val bằng 31 bit (thí nghiệm D) → vừa 1 DSP48 (25 × 18).
- **Dịch phải số âm:** Python `>>` và Verilog `>>>` là floor; C `>>` trên số âm là implementation-defined, C `/` làm tròn về 0.
- **Golden C / RTL phải đọc tham số từ `params_int8/`**, không tự tính lại scale/M0: calibrate lại khác máy lệch M0.
  Cần sinh lại tham chiếu thì dùng `python ptq_int8.py --chi-tham-chieu`.

## Phương pháp học của Thầy — 4 bước, mỗi bước có sản phẩm
1. Triển khai từng phần (AI first) → code
2. Chạy baseline → **Slide**
3. Giải thích ý nghĩa (flow, khối, từ khoá, tham số, công thức) → **Word**, phải có mục lục
4. Kiến thức liên quan (phương án thay thế) → **Word**

Cả 3 tài liệu đã có trong `docs/`. Nguyên tắc: mỗi phase kỹ thuật xong thì cập nhật tài liệu
tương ứng ngay. `operators/*.md` là bản nháp, Word là bản nộp.

## Ranh giới phạm vi
THUỘC (chắc chắn): hiểu kiến trúc → thu gọn params → viết operator + mã giả → bỏ/thay phép toán
khó cho FPGA → xác định model inference → trích tham số → golden model C.
23/09: Thầy xác nhận đi full flow. Mức cụ thể (RTL, synthesis, board thật) và lịch cho từng
bước vẫn nên chốt khi gặp lần tới.

## Câu hỏi đang chờ Thầy
1. ~~Trọn flow đi xa tới đâu~~ → 23/09: đi full flow. Còn hỏi: lịch từng bước sau golden model C.
2. ~~Golden model C verify tới mức nào~~ → bản int8 khớp **từng bit** (mức cao nhất), nên không cần ngưỡng.
3. ~~Chuẩn hoá ảnh làm ở đâu~~ → sau lượng tử, chip nhận `uint8` và dùng `pixel − 128` (tuần 6).
4. **Board FPGA mục tiêu** — giờ đã chặn việc: Kria bắt buộc Vivado; Zynq-7000 còn openXC7 dự phòng.
5. ~~Quantization trước hay sau golden C~~ → 07/10 Thầy giao PTQ/QAT trước, rồi C.
6. **Phương án Vivado trên Mac** (đề xuất: mô phỏng trên Mac ngay, synthesis ở máy x86/Docker) — hỏi 04/11.
7. **RTL dùng tham số PTQ hay QAT** (đề xuất PTQ) — hỏi 04/11.
8. Các câu tuần 6 (per-tensor/per-channel, ReLU gộp, đầu vào `uint8`, bề rộng M0, bỏ Model Composer) — chưa ghi nhận câu trả lời.
9. **Phần IC tới bước nào thì bắt buộc phải có Vivado** (tổng hợp, timing, nạp board)? Giai đoạn này là **prelab,
   chưa vào lab** → chỉ có máy cá nhân. Nếu bắt buộc thì phương án là Docker + SSD ngoài (APFS, ≥ 256 GB, nên 500 GB)
   hoặc mượn máy x86 của bạn. Hỏi buổi gặp tới, cùng với câu 4 (board).

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
11. Code và tài liệu viết **tiếng Việt có dấu** — kể cả comment, chuỗi in, README sinh tự động.
12. Không để lại `.git/index.lock` — chỉ chạy `git --no-optional-locks status`, không tự commit.
13. **Bài tập RTL — nhịp mỗi bài:** user lập bảng chân trị / sơ đồ trạng thái trước → Claude viết RTL + testbench tự
    kiểm (bảng mong đợi viết tay theo đề, không chép biểu thức của mạch) → user chạy `chay.sh` trên Mac → user trả lời
    câu tự kiểm → commit. **Không làm nhiều bài một lượt** (yêu cầu của Thầy: hiểu rõ rồi mới qua bài mới).

## Ràng buộc học thuật
- Nộp tiến độ thứ Tư hàng tuần. Không có tiến độ → bị loại khỏi nhóm.
- Report và slide làm cá nhân.
- Commit GitHub mỗi buổi làm.
- Gặp Thầy thì mang phần mình bí, đừng hỏi lại cái đã tự giải quyết được với AI.

## Biết trước vài chỗ
- `error_analysis.py --model small` và `make_figures.py` cùng ghi ra `figures/*_small.png`.
  Cùng model, cùng dữ liệu, chỉ khác tiêu đề hình — chạy script nào sau thì theo script đó.
- `CrossEntropyLoss` không nằm trên đường inference → không cần viết operator.
- Hình cũ trong `figures/` do `make_figures.py` sinh vẫn có tiêu đề không dấu, chạy lại là có dấu.
- Mỗi lần chạy `tools/xuat_pdf.py`, cả 7 PDF đều hiện "modified" vì Chrome ghi ngày giờ mới vào file.

## Trả lời
Tiếng Việt, có cấu trúc, bước cụ thể. Ngắn khi hỏi nhanh, đầy đủ khi kỹ thuật/học thuật.
