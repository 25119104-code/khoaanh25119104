// ============================================================
// tb_bao_dong.v — Testbench tự kiểm tra cho Bài 1
// ============================================================
// Testbench chỉ để MÔ PHỎNG, không tổng hợp lên FPGA: không có cổng vào/ra,
// được dùng #10 (trì hoãn), $display, $finish.
//
// Đưa lần lượt đủ 8 tổ hợp A, B, C, so alarm với bảng chân trị viết TAY theo đề.
// Không so với biểu thức (A & B) | C: chép lại biểu thức của mạch để kiểm chính nó
// thì sai đâu cũng "PASS".
// ============================================================

`timescale 1ns / 1ps
`default_nettype none

module tb_bao_dong;

    reg  A, B, C;          // reg: testbench tự gán giá trị
    wire alarm;            // wire: do mạch (DUT) điều khiển

    // DUT = Device Under Test, nối theo TÊN cổng cho khỏi nhầm thứ tự
    bao_dong dut (.A(A), .B(B), .C(C), .alarm(alarm));

    // Bảng chân trị theo đề. Bit thứ i = alarm khi {A,B,C} = i:
    //   i  : 7   6   5   4   3   2   1   0
    //  ABC : 111 110 101 100 011 010 001 000
    // alarm: 1   1   1   0   1   0   1   0
    localparam [7:0] MONG_DOI = 8'b1110_1010;

    integer i;
    integer so_sai = 0;

    initial begin
        // Icarus không tự ghi waveform như Vivado: phải bảo nó ghi ra file .vcd
        $dumpfile("song.vcd");
        $dumpvars(0, tb_bao_dong);

        $display(" A B C | alarm | mong đợi | kết quả");
        for (i = 0; i < 8; i = i + 1) begin
            {A, B, C} = i[2:0];   // ghép 3 bit: A là bit cao, C là bit thấp
            #10;                  // chờ 10 ns cho mạch ổn định rồi mới đọc
            $display(" %b %b %b |   %b   |    %b     | %s",
                     A, B, C, alarm, MONG_DOI[i],
                     (alarm === MONG_DOI[i]) ? "PASS" : "FAIL");
            // === so cả x và z: alarm = x (chưa nối dây) cũng bị tính là sai
            if (alarm !== MONG_DOI[i]) so_sai = so_sai + 1;
        end

        if (so_sai == 0) $display("KẾT QUẢ: PASS 8/8 tổ hợp");
        else             $display("KẾT QUẢ: FAIL %0d/8 tổ hợp", so_sai);
        $finish;
    end

endmodule

`default_nettype wire
