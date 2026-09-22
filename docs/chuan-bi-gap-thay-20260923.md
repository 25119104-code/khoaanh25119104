# Chuẩn bị gặp Thầy — 23/09/2026

> Viết lại 22/09 sau khi Thầy trả lời. Bản trước xếp 6 câu hỏi; 5 câu đã đóng, nên phần lớn
> tài liệu này giờ là **báo cáo** chứ không còn là **hỏi**.

---

## 1. Thầy đã chốt gì — và nó đổi việc gì

| Thầy nói | Đổi gì trong project |
|---|---|
| Project chạy trọn flow **AI → IC → ES**, mảng nào cũng làm để tìm mảng mình thích | Golden model C thuộc project 1. Phạm vi v3 đúng hướng — nhưng có thể còn rộng hơn, xem mục 3 |
| **Chưa quan trọng accuracy**, chỉ cần đưa được vào FPGA và tối ưu phép toán | Không còn sàn accuracy. 98,82% đủ dùng. Không train lại để đẩy điểm |
| **Chưa cần** tối ưu hyperparameter hay quantization | Phase 2E rời đường găng, chỉ làm nếu còn thời gian |
| **Tránh `exp`, `sqrt`** | Bỏ softmax là đúng yêu cầu Thầy, không phải sáng kiến riêng. Golden model C không cần `math.h` |
| Thư viện nào cũng được, **tự mày mò**, cốt là có cố gắng và có kết quả | Tự thiết kế kiến trúc là được, miễn giải thích được từng lựa chọn |

---

## 2. Báo cáo — làm được gì từ tuần 3 tới giờ

- **6 operator trên đường inference** đã có tài liệu đầy đủ trong `operators/` (README + 6 file):
  công thức, mã giả, số cụ thể, mức FPGA-friendly, bẫy khi port sang C.
- **Hai chỗ bỏ được phép toán, cả hai giống hệt bit-for-bit:**

  | Bỏ gì | Tiết kiệm | Vì sao đúng |
  |---|---|---|
  | Softmax ở lớp cuối | 10 `exp()` + 1 phép chia | `exp` đơn điệu tăng → `argmax` không đổi thứ tự |
  | Đổi thứ tự `relu` ↔ `maxpool` | 10.192 → 2.496 phép ReLU (4,08×) | `max` và `relu` đều đơn điệu không giảm |

  Sau hai chỗ này, đường inference chỉ còn `+`, `×`, so sánh. **Không còn `exp`, `sqrt`, hay
  phép chia nào** — đúng cái Thầy dặn.
- **Gom `models.py`** — class model từng bị chép ở 4 file, nay còn một bản duy nhất.
- **`demo_app.py` và `error_analysis.py` đã đổi sang `SmallCNN`**, không còn chạy model tuần 1.
- **Word bước 3 đã bổ sung mục 5** — 6 operator, mã giả, hai chỗ bỏ phép toán, ba bẫy port C.

---

## 3. Ba câu còn phải hỏi

1. **"Trọn flow AI → IC → ES" đi xa tới đâu trong project này?** Golden model C là đích của mảng
   AI. Nếu còn phải viết RTL và chạy trên board thật thì 6 tuần còn lại phải chia khác hẳn.
   Đây là câu quyết định lịch, thay chỗ câu "golden C thuộc project nào" đã đóng.
2. **Golden model C verify tới mức nào thì coi là đạt?** So từng lớp với PyTorch ở ngưỡng `1e-4`,
   hay chỉ cần accuracy khớp trên 10.000 ảnh test? Không có tiêu chí thì không biết khi nào dừng.
3. **Board FPGA mục tiêu là gì?** Chưa chặn việc gì lúc này, nhưng quyết định bit-width khi thật
   sự bước sang phần tối ưu trọng số.

---

## 4. Mang đi

- `docs/slide-baseline-cnn.pptx` — 13 trang
- `docs/buoc3-giai-thich-y-nghia.docx` (đã có mục 5 mới), `docs/buoc4-kien-thuc-lien-quan.docx`
- `reports/report-tuan03-20260921.md`
- `operators/` — mở sẵn README, nó có sơ đồ 12 bước
- `roadmap-digit-recognition-v3.md`
- `study/bai-tap-01-conv-params.md` — nếu đã làm xong bài 1 và bài 3

---

## 5. Số liệu phải nhớ, không tra máy

| | DigitCNN (tuần 1) | SmallCNN (đã chốt) |
|---|---|---|
| Params | 206.922 | **5.018** — ít hơn **41,2×** |
| Test acc | 99,05% | **98,82%** |
| Bộ nhớ float32 | 808,3 KB | 19,6 KB |

- Đường inference: **6 operator**, **414.265** phép/ảnh (còn **406.569** nếu đổi thứ tự relu/pool).
- Conv: 95% tính toán nhưng 71% params. `fc`: 0,3% tính toán, 29% params.
- Ba bẫy port sang C, cả ba **không crash** mà chỉ ra kết quả sai: layout NCHW/NHWC ở `flatten`;
  thứ tự chiều trọng số (PyTorch `[out][in]`, Keras `[in][out]`); accumulator khởi tạo bằng 0
  thay vì `bias`.
