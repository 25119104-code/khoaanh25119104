# Bài tập nhập môn RTL Verilog (10 bài)

Bộ bài Thầy giao cho phần IC (07/11/2026), deadline ≤ 1 tuần. Yêu cầu: mỗi bài có report + "học được gì";
**làm chậm, hiểu rõ rồi mới qua bài mới**.

## Công cụ (thay Vivado trên MacBook M2)

Đề chỉ yêu cầu **mô phỏng hành vi**, không tổng hợp, không XDC, không nạp board. Phần này không cần Vivado:

| Việc | Vivado | Ở đây |
|---|---|---|
| Biên dịch + mô phỏng | Run Behavioral Simulation | `iverilog` + `vvp` (Icarus Verilog) |
| Xem waveform | cửa sổ Wave | `surfer song.vcd` (tương tác) hoặc `song.png` (vẽ sẵn cho report) |
| Ghi waveform | tự động | phải có `$dumpfile` / `$dumpvars` trong testbench |

Cài một lần:

```bash
brew install icarus-verilog surfer
```

## Chạy một bài

```bash
cd ~/mnist-digit-project/rtl/bai-tap-nhap-mon
./chay.sh bai01_bao_dong
```

`chay.sh` biên dịch mọi file `.v` trong thư mục bài, chạy mô phỏng (in ra màn hình và lưu `ket_qua.txt`),
rồi vẽ `song.png` bằng `tools/ve_song.py` (cần python3 có matplotlib, ví dụ `conda activate mnist`).

## Cấu trúc mỗi bài

| File | Nội dung |
|---|---|
| `<ten>.v` | RTL (Design Source) |
| `tb_<ten>.v` | Testbench tự kiểm tra, in PASS/FAIL (Simulation Source) |
| `report.md` | Đề, bảng chân trị/sơ đồ trạng thái, giải thích, kết quả, waveform, bài học |
| `song.png`, `ket_qua.txt` | Kết quả chạy (sinh lại bằng `chay.sh`) |

## Tiến độ

| Bài | Nội dung | Trạng thái |
|---|---|---|
| 1 | Mạch báo động (AND/OR) | ✅ PASS 8/8 |
| 2 | Full Adder 1 bit | ⬜ |
| 3 | MUX 4:1 + decoder 2:4 | ⬜ |
| 4 | Bộ tính toán 4 bit | ⬜ |
| 5 | Thanh ghi 8 bit có enable | ⬜ |
| 6 | Reset đồng bộ vs bất đồng bộ | ⬜ |
| 7 | Đếm modulo-10 + chia xung | ⬜ |
| 8 | Thanh ghi dịch SIPO | ⬜ |
| 9 | FSM phát hiện 101 | ⬜ |
| 10 | Testbench tự kiểm tra Full Adder | ⬜ |
