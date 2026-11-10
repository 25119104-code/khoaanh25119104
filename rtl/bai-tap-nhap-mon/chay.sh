#!/bin/bash
# ============================================================
# chay.sh — biên dịch + mô phỏng + vẽ waveform cho 1 bài
# Dùng (đứng trong rtl/bai-tap-nhap-mon):   ./chay.sh bai01_bao_dong
# Thay cho "Run Behavioral Simulation" của Vivado.
# ============================================================
set -e
d="${1%/}"
[ -d "$d" ] || { echo "Không có thư mục $d"; exit 1; }
cd "$d"

# -g2012: chuẩn ngôn ngữ; -Wall: bật mọi cảnh báo (dây không nối, sai độ rộng bit...)
iverilog -g2012 -Wall -o sim *.v
vvp -n sim | tee ket_qua.txt

if python3 ../tools/ve_song.py song.vcd song.png "$d" 2>/dev/null; then :
else echo "(Không vẽ được song.png: cần python3 có matplotlib. Vẫn xem được bằng: surfer $d/song.vcd)"; fi
echo "Xem waveform tương tác: surfer $d/song.vcd"
