// ============================================================
// bao_dong.v — Bài 1: mạch báo động bằng cổng logic
// ============================================================
// Đề: 3 cảm biến A, B, C. alarm bật khi (A và B cùng bật) hoặc C bật.
//
//     alarm = (A AND B) OR C
//
// Mạch TỔ HỢP: không có clock, không nhớ gì. Đầu ra đổi ngay khi đầu vào đổi.
// Phần cứng: đúng 1 cổng AND 2 vào + 1 cổng OR 2 vào (trên FPGA gộp vào 1 LUT).
// ============================================================

`timescale 1ns / 1ps    // đơn vị thời gian / độ phân giải, mọi file phải giống nhau
`default_nettype none   // gõ sai tên tín hiệu thì báo lỗi, không tự sinh dây mới

module bao_dong (
    input  wire A,
    input  wire B,
    input  wire C,
    output wire alarm
);

    // assign = nối dây liên tục: alarm luôn bằng biểu thức bên phải
    assign alarm = (A & B) | C;

endmodule

`default_nettype wire   // trả lại mặc định cho các file khác
