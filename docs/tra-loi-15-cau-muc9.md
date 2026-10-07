# Trả lời 15 câu Thầy có thể hỏi (mục 9 của `on-tap-flow-7-buoc.md`)

Dự án: nhận diện chữ số viết tay MNIST bằng SmallCNN, 5.018 tham số, độ chính xác 98,82%.
Đường inference chỉ có 6 operator: conv2d, relu, maxpool, flatten, linear, argmax.
Đọc thành tiếng, mỗi câu khoảng 30 giây.

## Câu 1. Train và inference khác nhau ở đâu? FPGA làm phần nào?

Khi train, trọng số thay đổi sau mỗi batch. Chiều tính là forward, rồi tính loss, rồi backward để lấy gradient, rồi optimizer cập nhật trọng số. Cần có nhãn, loss, learning rate. Khi inference, trọng số cố định, chỉ chạy forward, đầu vào là ảnh và trọng số, đầu ra là một số từ 0 đến 9. FPGA chỉ làm inference, nên trên chip không có softmax, loss, backward hay optimizer. Cả đường inference là 414.265 phép mỗi ảnh.

## Câu 2. Tính 1 điểm output của conv2 thế nào? Bao nhiêu phép nhân?

Một điểm output là một số duy nhất. Công thức: bias của kênh ra cộng tổng, qua 8 kênh vào, 3 hàng, 3 cột, của pixel nhân trọng số. Tức là 8 nhân 9 bằng 72 phép nhân-cộng. Tất cả các kênh vào dồn vào một accumulator, không kênh vào nào cho ra output riêng. Conv1 là 9 phép, conv3 là 144 phép cho mỗi điểm. Với ảnh mẫu, điểm đã tính tay ra −11,0190, khớp PyTorch.

## Câu 3. Vì sao accumulator khởi tạo bằng bias?

Output bằng bias cộng tổng các tích. Khởi tạo bằng bias rồi cộng dồn từng tích thì ra đúng công thức. Khởi tạo bằng 0 mà quên cộng bias thì kết quả lệch đúng bằng bias, chương trình không báo lỗi, chỉ ra số sai. Đây là một trong ba bẫy khi viết C.

## Câu 4. Vì sao conv1 ít params mà vẫn nhiều phép tính?

Conv1 có 72 trọng số và 8 bias, tổng 80 tham số. Nhưng cùng bộ 9 trọng số của mỗi kênh được dùng lại ở mọi vị trí trên ảnh 28 nhân 28. Số phép nhân là 8 kênh nhân 784 điểm nhân 9 tap bằng 56.448. Ít tham số nhưng dùng lại rất nhiều lần, nên tính toán nhiều. Ngược lại lớp fc có 1.440 trọng số và mỗi trọng số chỉ dùng đúng 1 lần.

## Câu 5. Vì sao bỏ được softmax? Có phải xấp xỉ không?

Không phải xấp xỉ. Softmax của z_i bằng e mũ z_i chia cho tổng S. Hàm exp tăng nghiêm ngặt và mẫu số S dương, chung cho mọi lớp, nên thứ tự các phần tử không đổi. Vậy argmax của softmax bằng argmax của logit, chính xác từng bit. Tiết kiệm 10 phép exp và 1 phép chia, đường inference không còn hàm siêu việt nào, nên golden model C không cần math.h.

## Câu 6. Vì sao đổi relu và maxpool được? Sao không đúng 4 lần? Sao AvgPool không làm được?

ReLU là hàm đơn điệu không giảm, nên relu của max bằng max của relu. Đổi thứ tự thì ReLU chạy trên ảnh đã nhỏ: số phép ReLU giảm từ 10.192 xuống 2.496, tức 4,08 lần. Không đúng 4 lần vì pool cuối 7 xuống 3 bị floor, cắt bỏ hàng và cột cuối, 784 phần tử vào nhưng 144 ra. Với AvgPool thì sai, vì relu của (a cộng b) không bằng relu a cộng relu b. Ví dụ [−2, 2, 0, 0]: relu rồi trung bình ra 0,5, trung bình rồi relu ra 0.

## Câu 7. Vì sao đệm 0 tốt hơn `if` trên FPGA, dù nhiều phép nhân hơn?

Đệm 0 luôn 9 tap, tổng 395.136 phép nhân. Dùng `if` bỏ tap ở biên chỉ còn 351.008, ít hơn 11,2%. Nhưng trên FPGA 9 bộ nhân chạy song song, xong 1 điểm trong 1 chu kỳ. Bỏ bớt tap thì các bộ nhân ngồi không, vẫn mất 1 chu kỳ. Còn `if` phải thêm bộ so sánh chỉ số và MUX, tốn LUT và làm luồng điều khiển không đều. Hai bản ra kết quả giống hệt vì cộng thêm 0 nhân w không đổi tổng. Bài học: số phép là thước đo phần mềm, phần cứng đo bằng chu kỳ, diện tích và timing.

## Câu 8. Đệm bằng giá trị gì? Vì sao không phải màu nền?

Đệm bằng 0 sau khi chuẩn hoá. PyTorch đệm 0 sau bước Normalize. Màu nền MNIST sau chuẩn hoá là −0,4242 chứ không phải 0. Nếu đệm bằng −0,4242 thì ảnh trông đúng nhưng số ở viền lệch: ví dụ điểm góc (0,0) của conv1 kênh 0 đổi từ +0,04875 thành −0,02587, sau ReLU từ bật thành tắt.

## Câu 9. BatchNorm là gì? Gộp vào Conv thế nào? Model em có BN không?

BatchNorm chuẩn hoá đầu ra theo trung bình và phương sai, rồi nhân gamma cộng beta. Khi inference, các thống kê cố định nên BN là phép affine, gộp được vào conv: a bằng gamma chia căn của (phương sai cộng epsilon), trọng số mới bằng a nhân w, bias mới bằng beta trừ a nhân trung bình, cộng a nhân bias cũ. Model em không có BN. Em đã thử thêm BN: accuracy không tăng, train nhanh gấp đôi, gộp BN vào Conv thì 0 trên 10.000 ảnh khác dự đoán. Nên giữ model không BN.

## Câu 10. Dropout lúc inference làm gì?

Không làm gì, là phép đồng nhất. Lúc train, Dropout tắt ngẫu nhiên một phần tử và nhân phần còn lại với 1 chia (1 trừ p) để giữ kỳ vọng. Lúc inference đầu vào đi qua nguyên vẹn, nên không có gì để đưa lên chip.

## Câu 11. Fixed-point Q1.7 là gì? Đổi 0,319 sang Q1.7?

Q1.7 là số nguyên 8 bit có dấu, giá trị thật bằng q chia 128. Bước nhỏ nhất là 1/128, bằng 0,0078125, giá trị lớn nhất là 127/128, bằng 0,9921875. Đổi bằng cách nhân 128, làm tròn, rồi kẹp vào khoảng −128 đến 127. Với 0,319: nhân 128 ra 40,8, làm tròn thành q bằng 41, giá trị thật là 0,3203125. Sai số làm tròn tối đa nửa bước, 0,0039. Phải kẹp, vì 1,03 nhân 128 ra 132, không kẹp thì tràn thành −124, số dương thành số âm mà không báo lỗi. Ký hiệu Q là quy ước của ARM, không có chuẩn ISO duy nhất, nên khi nói rõ là 8 bit có dấu, chia 128.

## Câu 12. Chia cho hằng số trên phần cứng làm thế nào?

Chia cho 2 mũ n thì dịch phải n bit: 200 chia 8 bằng 200 dịch phải 3 bằng 25, trên phần cứng chỉ là nối dây. Chia cho hằng số khác thì nhân với nghịch đảo tính sẵn trên PC. Ví dụ chuẩn hoá ảnh chia 255 và 0,3081 được tính trước thành a nhân px cộng b, với a bằng 0,01272823 và b bằng −0,42421292, mỗi pixel chỉ 1 nhân và 1 cộng. Dịch bit thường làm tròn xuống, muốn làm tròn gần nhất thì cộng 2 mũ (n−1) trước khi dịch.

## Câu 13. Vì sao 2 file tham số mà không tách theo lớp? Vì sao in `%.9g`?

Trên FPGA toàn bộ trọng số nằm trong một bộ nhớ ROM hoặc BRAM, mỗi lớp chỉ là một địa chỉ gốc. Một file tuần tự cộng bảng offset mô phỏng đúng cách phần cứng truy cập: weights bắt đầu ở 0, 72, 1.224, 3.528; biases ở 0, 8, 24, 40. Thầy dặn weight và bias 2 file riêng. In `%.9g` vì 9 chữ số là tối thiểu để float32 đi qua dạng chữ rồi quay lại mà không mất bit; in ít hơn thì C lệch PyTorch ngay từ lúc đọc file.

## Câu 14. Golden model C kiểm chứng thế nào? Lệch 1e-05 có phải lỗi không?

Kiểm 2 tầng. Tầng 1: so từng lớp trên ảnh mẫu với đầu ra PyTorch trong thư mục golden, mọi lớp lệch dưới 1e-4. Tầng 2: chạy 10.000 ảnh test, accuracy 98,82%, trùng dự đoán PyTorch 10.000 trên 10.000, logit lệch tối đa khoảng 3e-05. Lệch 1e-05 không phải lỗi: float32 cộng khác thứ tự thì làm tròn khác. Sai thuật toán (quên bias, sai layout) lệch từ cỡ 0,1 trở lên, lớn hơn 3 bậc. Khoảng cách top-1 và top-2 nhỏ nhất trên 10.000 ảnh là 0,019, lớn hơn độ lệch cỡ 600 lần, nên dự đoán chắc chắn trùng. Ngưỡng 1e-4 do em tự chọn, em muốn hỏi Thầy.

## Câu 15. Bước tiếp theo là gì?

Em đề xuất làm RTL trước theo giải pháp 1 của Thầy: mỗi layer một khối, bắt đầu từ ReLU rồi conv 3×3, testbench so với thư mục golden. Quantization INT8 để sau vì Thầy nói chưa cần, nhưng video Step2 làm int8 trước golden C nên em xin Thầy chốt thứ tự. Câu em cần hỏi Thầy: tiêu chí verify golden model, chuẩn hoá ảnh làm ở PC hay trên chip, và board FPGA mục tiêu.
