/* ============================================================
 * golden_int8.c — Golden model C số nguyên (int8) cho SmallCNN
 * ============================================================
 * Dịch từ đường số nguyên của ptq_int8.py (hàm suy_luan_int), cấu hình B:
 * ReLU gộp vào bước kẹp, M0 31 bit. Bản float32 vẫn giữ ở golden_model.c.
 *
 * Mục đích: làm CHUẨN ĐỐI CHIẾU cho RTL. Trên đường inference chỉ còn số nguyên:
 *     int8 × int8 → cộng dồn int32 → nhân M0, cộng, dịch bit → kẹp về int8
 *
 *   - Chỉ dùng <stdio.h>, <stdlib.h>, <stdint.h>. stdint.h chỉ để có tên kiểu
 *     int8_t, int16_t, int32_t, int64_t, cho thấy rõ độ rộng bit, không có hàm nào.
 *     Không <math.h>. Đường inference không có float/double.
 *   - KHÔNG tính lại scale hay M0. Mọi hằng số đọc từ thư mục tham số:
 *     params_int8/ (do ptq_int8.py sinh) hoặc params_int8_qat/ (do qat.py --xuat sinh).
 *     Đổi bộ tham số không phải sửa code.
 *   - Mảng 1 chiều, layout [C][H][W] như PyTorch (NCHW), giống golden_model.c.
 *   - Conv luôn đủ 9 tap trên ảnh đã đệm (Thầy góp ý 23/09).
 *
 * Khác bản float: phải KHỚP TỪNG BIT với Python, không có dung sai nào.
 *   [a] Ảnh mẫu: so 11 file trong <thư mục tham số>/golden_int8/.
 *   [b] Nhiều ảnh: so cả 10 acc lớp fc của từng ảnh với golden_int8/fc_acc_test.txt.
 *   [c] Thí nghiệm cho RTL: maxpool trên acc int32 trước, requantize sau.
 *
 * BIÊN DỊCH (đứng ở thư mục gốc project):
 *   cc -O2 -std=c99 -Wall -Wextra -o golden_c/golden_int8 golden_c/golden_int8.c
 * CHẠY:
 *   ./golden_c/golden_int8                         # ảnh mẫu + 10.000 ảnh, tham số params_int8/
 *   ./golden_c/golden_int8 100                     # chỉ 100 ảnh đầu
 *   ./golden_c/golden_int8 0                       # chỉ ảnh mẫu
 *   ./golden_c/golden_int8 10000 params_int8_qat   # bộ tham số QAT
 * ============================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* ---------- [1] Kích thước và tham số ---------- */
#define K          3        /* kernel 3x3 */
#define PAD        1        /* đệm 1 → H_ra = H_vào */
#define FC_IN      144      /* 16 x 3 x 3 */
#define FC_OUT     10
#define N_TEST     10000
#define BUF_MAX    6272     /* mảng lớn nhất: conv1 ra 8 x 28 x 28 */
#define PADDED_MAX 2048     /* ảnh đã đệm lớn nhất: 8 x 16 x 16 (vào conv2) */
#define POOL_MAX   1568     /* pool1 ra 8 x 14 x 14 */

/* Một lớp conv: kích thước + tham số số nguyên + hằng số requantize */
typedef struct {
    const char *ten;
    int IC, OC, H, W;          /* vào IC x H x W, ra OC x H x W (pad 1, stride 1) */
    int8_t  w[16 * 16 * 9];    /* [OC][IC][3][3], int8 đối xứng [-127, 127], zp_w = 0 */
    int32_t b[16];             /* bias int32, scale = s_vào x s_w */
    int32_t zp_vao, zp_ra;     /* zero-point đầu vào, đầu ra */
    int32_t M0;                /* M = s_vào·s_w / s_ra ≈ M0 · 2^-shift, 0 < M0 < 2^31 */
    int     shift;
} LopConv;

static LopConv conv[3] = {
    {.ten = "conv1", .IC = 1,  .OC = 8,  .H = 28, .W = 28},
    {.ten = "conv2", .IC = 8,  .OC = 16, .H = 14, .W = 14},
    {.ten = "conv3", .IC = 16, .OC = 16, .H = 7,  .W = 7},
};
static int8_t  fc_w[FC_OUT * FC_IN];   /* [10][144] */
static int32_t fc_b[FC_OUT];
static int32_t fc_zp_vao;
static int32_t zp_in;                  /* zero-point của ảnh đầu vào */
static int     relu_rieng = 0;         /* 0: ReLU gộp vào bước kẹp (cấu hình B). 1: ReLU riêng như slide */
static int     bits_M0 = 0;

/* Bộ đếm, in ra ở ảnh mẫu */
static long mac_conv = 0;              /* phải là 395.136 / ảnh, bằng bản float */
static long so_requantize = 0;         /* số lần requantize / ảnh */
static long acc_max[4];                /* |acc| lớn nhất: conv1, conv2, conv3, fc */

/* ---------- [2] Đọc file ---------- */
static long tam[BUF_MAX];              /* bộ đệm đọc file, dùng chung */
static char duong_dan[512];

/* "<thư mục>/<tên>". Trả về bộ đệm tĩnh: dùng ngay, trước lần gọi sau. */
static const char *ghep(const char *thu_muc, const char *ten)
{
    snprintf(duong_dan, sizeof duong_dan, "%s/%s", thu_muc, ten);
    return duong_dan;
}

/* Đọc ĐÚNG n số nguyên vào tam[]. Kiểm từng số nằm trong [lo, hi] và file không thừa số. */
static void doc_so_nguyen(const char *path, int n, long lo, long hi)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Không mở được %s\n", path); exit(1); }
    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%ld", &tam[i]) != 1) {
            fprintf(stderr, "%s: chỉ đọc được %d/%d số\n", path, i, n);
            exit(1);
        }
        if (tam[i] < lo || tam[i] > hi) {
            fprintf(stderr, "%s: số thứ %d = %ld nằm ngoài [%ld, %ld]\n", path, i, tam[i], lo, hi);
            exit(1);
        }
    }
    long thua;
    if (fscanf(f, "%ld", &thua) == 1) {
        fprintf(stderr, "%s: có nhiều hơn %d số, sai kích thước\n", path, n);
        exit(1);
    }
    fclose(f);
}

static void doc_trong_so(const char *path, int8_t *dst, int n)
{
    doc_so_nguyen(path, n, -127, 127);      /* đối xứng: -128 không bao giờ xuất hiện */
    for (int i = 0; i < n; i++) dst[i] = (int8_t)tam[i];
}

static void doc_bias(const char *path, int32_t *dst, int n)
{
    doc_so_nguyen(path, n, INT32_MIN, INT32_MAX);
    for (int i = 0; i < n; i++) dst[i] = (int32_t)tam[i];
}

/* tham_so.txt: chỉ lấy zp, M0, shift (số nguyên). Các số thực s, M trong file là để người
 * đọc. Golden C bỏ qua chúng bằng %*f, không đưa vào phép tính nào. */
static void doc_tham_so(const char *thu_muc)
{
    const char *path = ghep(thu_muc, "tham_so.txt");
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Không mở được %s\n", path); exit(1); }
    char dong[512], relu[16];
    int du = 0;                             /* cần đủ 6 dòng: đầu file, input, conv1..3, fc */
    int i, zv, zr, sh;
    long long m0;
    while (fgets(dong, sizeof dong, f)) {
        if (sscanf(dong, "# relu=%15s bits_M0=%d", relu, &bits_M0) == 2) {
            if (relu[0] == 'g') relu_rieng = 0;
            else if (relu[0] == 'r') relu_rieng = 1;
            else { fprintf(stderr, "tham_so.txt: relu=%s không hiểu\n", relu); exit(1); }
            if (bits_M0 < 2 || bits_M0 > 31) { fprintf(stderr, "tham_so.txt: bits_M0=%d phải trong [2, 31]\n", bits_M0); exit(1); }
            du++;
        } else if (sscanf(dong, "input s=%*f zp=%d", &zv) == 1) {
            zp_in = zv;
            du++;
        } else if (sscanf(dong, "conv%d s_w=%*f zp_vao=%d s_ra=%*f zp_ra=%d M=%*f M0=%lld shift=%d",
                          &i, &zv, &zr, &m0, &sh) == 5) {
            if (i < 1 || i > 3 || m0 <= 0 || m0 >= (1LL << 31) || sh < 1 || sh > 62) {
                fprintf(stderr, "tham_so.txt: dòng conv%d có M0 hoặc shift ngoài dải\n", i);
                exit(1);
            }
            LopConv *L = &conv[i - 1];
            L->zp_vao = zv;
            L->zp_ra  = zr;
            L->M0     = (int32_t)m0;
            L->shift  = sh;
            du++;
        } else if (sscanf(dong, "fc s_w=%*f zp_vao=%d", &zv) == 1) {
            fc_zp_vao = zv;
            du++;
        }
    }
    fclose(f);
    if (du != 6) { fprintf(stderr, "tham_so.txt: chỉ đọc được %d/6 dòng cần thiết\n", du); exit(1); }
}

/* ---------- [3] Các operator số nguyên ---------- */

/* floor(x / 2^n): dịch phải SỐ HỌC, viết tường minh.
 * Chuẩn C để `>>` trên số âm cho trình biên dịch tự quyết (implementation-defined).
 * gcc/clang dịch số học, nhưng golden model không được dựa vào chuyện đó.
 * Python `>>` và Verilog `>>>` (trên tín hiệu signed) đều là dịch số học, khớp hàm này. */
static int64_t dich_phai_so_hoc(int64_t x, int n)
{
    if (x >= 0) return x >> n;
    return -((-x - 1) >> n) - 1;
}

/* Requantize int32 → int8:  q = kẹp(((acc·M0 + 2^(shift−1)) >> shift) + zp_ra, −128, 127)
 * Cộng 2^(shift−1) rồi mới dịch = làm tròn nửa lên, giống floor(x + 0,5) ở Python.
 * acc·M0 phải tính bằng 64 bit: acc cỡ 19 bit nhân M0 31 bit ra cỡ 50 bit, tràn int32. */
static int8_t requantize(int32_t acc, const LopConv *L)
{
    int64_t t = (int64_t)acc * L->M0 + ((int64_t)1 << (L->shift - 1));
    int64_t q = dich_phai_so_hoc(t, L->shift) + L->zp_ra;
    if (q < INT8_MIN) q = INT8_MIN;     /* BƯỚC KẸP (slide thiếu). Khi zp_ra = −128, đây chính là ReLU */
    if (q > INT8_MAX) q = INT8_MAX;
    if (relu_rieng && q < L->zp_ra) q = L->zp_ra;   /* ReLU riêng như slide: q < zp thì q = zp */
    return (int8_t)q;
}

/* Conv 3x3 số nguyên, stride 1, pad 1, luôn đủ 9 tap. Ra acc int32, CHƯA requantize.
 *
 * BẪY ĐỆM: số thực 0 trong miền int8 là zp_vao, KHÔNG phải số nguyên 0.
 * Nên trừ zp TRƯỚC (x = q − zp_vao) rồi mới đệm 0 lên x, tương đương đệm giá trị zp.
 *   conv1: zp_vao = −95  → x = q + 95 = pixel − 33, nằm trong [−33, 222]: cần 9 bit có dấu.
 *   conv2, conv3, fc: zp_vao = −128 → x = q + 128 ∈ [0, 255]: chỉ là đảo bit cao nhất của q,
 *   phần cứng không tốn bộ cộng nào.
 * Accumulator khởi tạo bằng BIAS, không phải 0.
 * acc int32 không thể tràn: kể cả khi mọi trọng số là ±127, |acc| ≤ 144·255·127 + |bias|,
 * cỡ 4,7 triệu (24 bit). Với trọng số thật, chặn trên chỉ còn 20 bit (hàm chan_tren, mục [b]). */
static void conv3x3_int(const int8_t *q, const LopConv *L, int32_t *acc_ra)
{
    static int16_t xp[PADDED_MAX];
    const int IC = L->IC, H = L->H, W = L->W, HP = H + 2 * PAD, WP = W + 2 * PAD;

    for (int i = 0; i < IC * HP * WP; i++) xp[i] = 0;                 /* đệm 0 SAU khi trừ zp */
    for (int ic = 0; ic < IC; ic++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                xp[(ic * HP + y + PAD) * WP + x + PAD] = (int16_t)(q[(ic * H + y) * W + x] - L->zp_vao);

    for (int oc = 0; oc < L->OC; oc++)
        for (int oy = 0; oy < H; oy++)
            for (int ox = 0; ox < W; ox++) {
                int32_t acc = L->b[oc];                  /* khởi tạo bằng BIAS */
                for (int ic = 0; ic < IC; ic++)
                    for (int ky = 0; ky < K; ky++)
                        for (int kx = 0; kx < K; kx++) {
                            acc += (int32_t)xp[(ic * HP + oy + ky) * WP + ox + kx]
                                 * L->w[((oc * IC + ic) * K + ky) * K + kx];
                            mac_conv++;
                        }
                acc_ra[(oc * H + oy) * W + ox] = acc;
            }
}

/* MaxPool 2x2 stride 2 trên int8. max không đổi scale/zp nên chạy thẳng trên số nguyên.
 * H_ra = H / 2 (chia nguyên = floor): 7 → 3, hàng/cột cuối bị bỏ. */
static void maxpool2_int8(const int8_t *in, int C, int H, int W, int8_t *out)
{
    const int HO = H / 2, WO = W / 2;
    for (int c = 0; c < C; c++)
        for (int oy = 0; oy < HO; oy++)
            for (int ox = 0; ox < WO; ox++) {
                int8_t m = in[(c * H + 2 * oy) * W + 2 * ox];
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        int8_t v = in[(c * H + 2 * oy + dy) * W + 2 * ox + dx];
                        if (v > m) m = v;
                    }
                out[(c * HO + oy) * WO + ox] = m;
            }
}

/* Như trên nhưng trên acc int32 — chỉ dùng cho thí nghiệm [c]. */
static void maxpool2_int32(const int32_t *in, int C, int H, int W, int32_t *out)
{
    const int HO = H / 2, WO = W / 2;
    for (int c = 0; c < C; c++)
        for (int oy = 0; oy < HO; oy++)
            for (int ox = 0; ox < WO; ox++) {
                int32_t m = in[(c * H + 2 * oy) * W + 2 * ox];
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        int32_t v = in[(c * H + 2 * oy + dy) * W + 2 * ox + dx];
                        if (v > m) m = v;
                    }
                out[(c * HO + oy) * WO + ox] = m;
            }
}

/* Lớp fc: acc = bias + Σ (q − zp_vao)·w. KHÔNG requantize: argmax thẳng trên acc int32,
 * vì nhân cùng một scale dương không đổi thứ tự. Flatten không tốn phép nào. */
static void linear_int(const int8_t *q, int32_t *acc_ra)
{
    for (int o = 0; o < FC_OUT; o++) {
        int32_t acc = fc_b[o];
        for (int i = 0; i < FC_IN; i++)
            acc += (int32_t)(q[i] - fc_zp_vao) * fc_w[o * FC_IN + i];
        acc_ra[o] = acc;
    }
}

/* So sánh CHẶT (>): bằng nhau thì giữ chỉ số đầu, giống np.argmax và torch.argmax. */
static int argmax_int(const int32_t *z, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (z[i] > z[best]) best = i;
    return best;
}

/* ---------- [4] Ghép lại ---------- */
/* Đầu ra từng lớp, giữ lại để so với golden_int8/ */
static int8_t  L_in[784];
static int32_t L_acc1[6272], L_acc2[3136], L_acc3[784], L_fc[FC_OUT];
static int8_t  L_q1[6272], L_q2[3136], L_q3[784];         /* sau requantize (= sau ReLU) */
static int8_t  L_p1[1568], L_p2[784], L_p3[144];
static int32_t *const L_acc[3] = {L_acc1, L_acc2, L_acc3};
static int8_t  *const L_q[3]   = {L_q1, L_q2, L_q3};
static int8_t  *const L_p[3]   = {L_p1, L_p2, L_p3};

static void cap_nhat_max(int lop, const int32_t *a, int n)
{
    for (int i = 0; i < n; i++) {
        long v = a[i] < 0 ? -(long)a[i] : (long)a[i];
        if (v > acc_max[lop]) acc_max[lop] = v;
    }
}

/* Đầu vào: chip nhận pixel uint8 và dùng q = p − 128 (đảo bit cao nhất).
 * ptq_int8.py đã kiểm: lượng tử hoá đầy đủ round(x/s_in) + zp_in cho ra đúng p − 128
 * ở cả 7.840.000 pixel test, nên trên chip không cần chuẩn hoá bằng số thực. */
static void dau_vao(const unsigned char *px, int8_t *q)
{
    for (int i = 0; i < 784; i++) q[i] = (int8_t)((int)px[i] - 128);
}

static int infer_int(const unsigned char *px)
{
    dau_vao(px, L_in);
    const int8_t *q = L_in;
    for (int l = 0; l < 3; l++) {
        const LopConv *L = &conv[l];
        const int n = L->OC * L->H * L->W;
        conv3x3_int(q, L, L_acc[l]);
        cap_nhat_max(l, L_acc[l], n);
        for (int i = 0; i < n; i++) L_q[l][i] = requantize(L_acc[l][i], L);
        so_requantize += n;
        maxpool2_int8(L_q[l], L->OC, L->H, L->W, L_p[l]);
        q = L_p[l];
    }
    linear_int(q, L_fc);                     /* L_p3 đã nằm theo [16][3][3] = vector 144 */
    cap_nhat_max(3, L_fc, FC_OUT);
    return argmax_int(L_fc, FC_OUT);
}

/* [c] Đổi thứ tự: maxpool trên acc int32 TRƯỚC, requantize SAU.
 * requantize là hàm không giảm (M0 > 0; cộng hằng, floor, kẹp đều không giảm), nên
 * max(rq(a), rq(b)) = rq(max(a, b)). Phải ra y hệt, mà chỉ requantize 1/4 số điểm. */
static int8_t  E_p1[1568], E_p2[784], E_p3[144];
static int8_t  *const E_p[3] = {E_p1, E_p2, E_p3};
static int32_t E_acc[BUF_MAX], E_pacc[POOL_MAX], E_fc[FC_OUT];
static long so_requantize_c = 0;

static void infer_pool_truoc(const unsigned char *px)
{
    static int8_t in[784];
    dau_vao(px, in);
    const int8_t *q = in;
    for (int l = 0; l < 3; l++) {
        const LopConv *L = &conv[l];
        const int n = L->OC * (L->H / 2) * (L->W / 2);
        conv3x3_int(q, L, E_acc);
        maxpool2_int32(E_acc, L->OC, L->H, L->W, E_pacc);
        for (int i = 0; i < n; i++) E_p[l][i] = requantize(E_pacc[i], L);
        so_requantize_c += n;
        q = E_p[l];
    }
    linear_int(q, E_fc);
}

/* ---------- [5] Tiện ích kiểm tra ---------- */

/* Số phần tử KHÁC giữa mảng của C và file golden (n số nguyên). */
static int so_khac_int8(const int8_t *a, const char *path, int n)
{
    doc_so_nguyen(path, n, INT32_MIN, INT32_MAX);
    int khac = 0;
    for (int i = 0; i < n; i++) khac += (a[i] != tam[i]);
    return khac;
}

static int so_khac_int32(const int32_t *a, const char *path, int n)
{
    doc_so_nguyen(path, n, INT32_MIN, INT32_MAX);
    int khac = 0;
    for (int i = 0; i < n; i++) khac += (a[i] != tam[i]);
    return khac;
}

/* Chặn trên |acc| cho MỌI ảnh có thể, không chỉ ảnh test.
 * Mỗi đầu vào x = q − zp nằm trong [−128 − zp, 127 − zp]. Với từng trọng số, chọn đầu mút cho
 * tích lớn nhất (hoặc nhỏ nhất) rồi cộng dồn. Ô đệm có x = 0, nằm trong dải đó, nên không vượt.
 * Đây là độ rộng thanh ghi acc AN TOÀN cho RTL. Số "đo" chỉ đúng với các ảnh đã chạy. */
static long chan_tren(const int8_t *w, const int32_t *b, int OC, int n_tap, int32_t zp_vao)
{
    const long xmin = INT8_MIN - (long)zp_vao, xmax = INT8_MAX - (long)zp_vao;
    long m = 0;
    for (int oc = 0; oc < OC; oc++) {
        long hi = b[oc], lo = b[oc];
        for (int t = 0; t < n_tap; t++) {
            long a = w[oc * n_tap + t] * xmin, c = w[oc * n_tap + t] * xmax;
            hi += a > c ? a : c;
            lo += a < c ? a : c;
        }
        if (labs(hi) > m) m = labs(hi);
        if (labs(lo) > m) m = labs(lo);
    }
    return m;
}

/* Số bit CÓ DẤU để chứa mọi giá trị trong [−v, v]: bit_length(v) + 1, cùng cách tính với Python. */
static int so_bit(long v)
{
    int n = 1;
    while (v > 0) { n++; v >>= 1; }
    return n;
}

/* ---------- [6] main ---------- */
int main(int argc, char **argv)
{
    int n_anh = (argc > 1) ? atoi(argv[1]) : N_TEST;
    const char *thu_muc = (argc > 2) ? argv[2] : "params_int8";
    if (n_anh < 0 || n_anh > N_TEST) n_anh = N_TEST;
    char ten[64];

    /* Nạp tham số */
    doc_tham_so(thu_muc);
    for (int l = 0; l < 3; l++) {
        LopConv *L = &conv[l];
        snprintf(ten, sizeof ten, "conv%d_w.txt", l + 1);
        doc_trong_so(ghep(thu_muc, ten), L->w, L->OC * L->IC * K * K);
        snprintf(ten, sizeof ten, "conv%d_b.txt", l + 1);
        doc_bias(ghep(thu_muc, ten), L->b, L->OC);
    }
    doc_trong_so(ghep(thu_muc, "fc_w.txt"), fc_w, FC_OUT * FC_IN);
    doc_bias(ghep(thu_muc, "fc_b.txt"), fc_b, FC_OUT);

    /* Chuỗi zero-point phải nối liền: zp_ra lớp trước = zp_vao lớp sau */
    if (conv[0].zp_vao != zp_in || conv[1].zp_vao != conv[0].zp_ra ||
        conv[2].zp_vao != conv[1].zp_ra || fc_zp_vao != conv[2].zp_ra) {
        fprintf(stderr, "tham_so.txt: zp_ra lớp trước không bằng zp_vao lớp sau\n");
        return 1;
    }
    /* q = p − 128 chỉ đúng khi dải đầu vào calibrate là trọn [0, 255], tức zp_in = −95 */
    if (zp_in != -95) {
        fprintf(stderr, "zp đầu vào = %d, không phải -95: q = p - 128 không còn đúng, xem lại dau_vao()\n", (int)zp_in);
        return 1;
    }

    printf("Tham số: %s/  4968 trọng số int8 + 50 bias int32, M0 %d bit, ReLU %s\n",
           thu_muc, bits_M0, relu_rieng ? "riêng (như slide)" : "gộp vào bước kẹp");
    printf("  %-6s %7s %6s %11s %6s\n", "lớp", "zp_vào", "zp_ra", "M0", "shift");
    printf("  %-6s %7d\n", "input", (int)zp_in);
    for (int l = 0; l < 3; l++)
        printf("  %-6s %7d %6d %11d %6d\n", conv[l].ten, (int)conv[l].zp_vao, (int)conv[l].zp_ra,
               (int)conv[l].M0, conv[l].shift);
    printf("  %-6s %7d   (không requantize, argmax trên acc)\n", "fc", (int)fc_zp_vao);

    /* [a] Ảnh mẫu = ảnh test index 0, đọc thẳng file raw giống như chip sẽ nhận */
    FILE *fi = fopen("data/MNIST/raw/t10k-images-idx3-ubyte", "rb");
    FILE *fb = fopen("data/MNIST/raw/t10k-labels-idx1-ubyte", "rb");
    if (!fi || !fb) { fprintf(stderr, "Thiếu file MNIST raw trong data/MNIST/raw/\n"); return 1; }
    unsigned char px[784], lb;
    fseek(fi, 16, SEEK_SET);                       /* header ảnh 16 byte */
    fseek(fb, 8, SEEK_SET);                        /* header nhãn 8 byte */
    if (fread(px, 1, 784, fi) != 784 || fread(&lb, 1, 1, fb) != 1) { fprintf(stderr, "Lỗi đọc ảnh mẫu\n"); return 1; }

    mac_conv = 0;
    so_requantize = 0;
    int pred = infer_int(px);
    const long mac_mau = mac_conv, rq_mau = so_requantize;
    so_requantize_c = 0;
    infer_pool_truoc(px);
    const long rq_c_mau = so_requantize_c;

    doc_so_nguyen(ghep(thu_muc, "golden_int8/label.txt"), 1, 0, 9);
    const long nhan_golden = tam[0];

    printf("\n[a] Ảnh mẫu (test index 0, nhãn %d): so TỪNG SỐ với %s/golden_int8/\n", lb, thu_muc);
    printf("  %-18s %7s %8s\n", "file", "số pt", "số khác");
    struct { const char *file; const int8_t *q8; const int32_t *q32; int n; } bang[] = {
        {"input.txt",        L_in, NULL,   784},
        {"01_conv1_acc.txt", NULL, L_acc1, 6272},
        {"02_relu1.txt",     L_q1, NULL,   6272},
        {"03_pool1.txt",     L_p1, NULL,   1568},
        {"04_conv2_acc.txt", NULL, L_acc2, 3136},
        {"05_relu2.txt",     L_q2, NULL,   3136},
        {"06_pool2.txt",     L_p2, NULL,   784},
        {"07_conv3_acc.txt", NULL, L_acc3, 784},
        {"08_relu3.txt",     L_q3, NULL,   784},
        {"09_pool3.txt",     L_p3, NULL,   144},
        {"10_fc_acc.txt",    NULL, L_fc,   10},
    };
    int tong_khac = 0;
    for (int i = 0; i < 11; i++) {
        snprintf(ten, sizeof ten, "golden_int8/%s", bang[i].file);
        int k = bang[i].q8 ? so_khac_int8(bang[i].q8, ghep(thu_muc, ten), bang[i].n)
                           : so_khac_int32(bang[i].q32, ghep(thu_muc, ten), bang[i].n);
        tong_khac += k;
        printf("  %-18s %7d %8d\n", bang[i].file, bang[i].n, k);
    }
    printf("  Đoán %d, nhãn %d (golden ghi %ld). MAC conv/ảnh: %ld (bản float: 395136). Requantize/ảnh: %ld\n",
           pred, lb, nhan_golden, mac_mau, rq_mau);
    const int mau_dat = (tong_khac == 0 && nhan_golden == lb);
    printf("  KẾT LUẬN ảnh mẫu: %s\n", mau_dat ? "KHỚP TỪNG BIT ở cả 11 file" : "CÓ CHỖ LỆCH");

    if (n_anh == 0) { fclose(fi); fclose(fb); return mau_dat ? 0 : 1; }

    /* [b] Nhiều ảnh: đọc lại từ đầu file raw */
    fseek(fi, 16, SEEK_SET);
    fseek(fb, 8, SEEK_SET);
    FILE *ft = fopen(ghep(thu_muc, "golden_int8/fc_acc_test.txt"), "r");   /* tham chiếu Python */
    FILE *fp = fopen("golden/logits_pytorch.txt", "r");                     /* logit float32 PyTorch */
    const int co_ref = (ft != NULL);

    int dung = 0, khop_acc = 0, trung_py = 0, khac_c = 0;
    int khac_float = 0, dung_thanh_sai = 0, sai_thanh_dung = 0;
    for (int a = 0; a < n_anh; a++) {
        if (fread(px, 1, 784, fi) != 784 || fread(&lb, 1, 1, fb) != 1) {
            fprintf(stderr, "Hết dữ liệu ở ảnh %d\n", a);
            return 1;
        }
        const int p = infer_int(px);
        dung += (p == lb);

        if (ft) {                                   /* so cả 10 acc với Python, từng bit */
            int32_t ref[FC_OUT];
            int giong = 1;
            for (int k = 0; k < FC_OUT; k++) {
                long v;
                if (fscanf(ft, "%ld", &v) != 1) { fprintf(stderr, "fc_acc_test.txt thiếu số ở ảnh %d\n", a); return 1; }
                ref[k] = (int32_t)v;
                if (ref[k] != L_fc[k]) giong = 0;
            }
            khop_acc += giong;
            trung_py += (p == argmax_int(ref, FC_OUT));
        }
        if (fp) {
            /* float CHỈ xuất hiện ở đây, để so với PyTorch float32. Không nằm trên đường inference. */
            float z[FC_OUT];
            int pf = 0;
            for (int k = 0; k < FC_OUT; k++) {
                if (fscanf(fp, "%f", &z[k]) != 1) { fprintf(stderr, "Lỗi đọc golden/logits_pytorch.txt\n"); return 1; }
                if (z[k] > z[pf]) pf = k;
            }
            khac_float     += (p != pf);
            dung_thanh_sai += (pf == lb && p != lb);
            sai_thanh_dung += (pf != lb && p == lb);
        }

        /* [c] cùng ảnh, thứ tự maxpool → requantize */
        infer_pool_truoc(px);
        int giong_c = 1;
        for (int l = 0; l < 3; l++) {
            const int n = conv[l].OC * (conv[l].H / 2) * (conv[l].W / 2);
            for (int i = 0; i < n; i++) if (E_p[l][i] != L_p[l][i]) giong_c = 0;
        }
        for (int k = 0; k < FC_OUT; k++) if (E_fc[k] != L_fc[k]) giong_c = 0;
        khac_c += !giong_c;
    }
    fclose(fi);
    fclose(fb);

    /* Phần trăm × 100, làm tròn, tính bằng số nguyên */
    const long pt = (dung * 10000L + n_anh / 2) / n_anh;
    printf("\n[b] %d ảnh đầu test set\n", n_anh);
    printf("  Accuracy golden C int8          : %d/%d = %ld.%02ld%%\n", dung, n_anh, pt / 100, pt % 100);
    if (ft) {
        fclose(ft);
        printf("  Khớp cả 10 acc lớp fc với Python : %d/%d ảnh (từng bit)\n", khop_acc, n_anh);
        printf("  Trùng dự đoán Python int8       : %d/%d\n", trung_py, n_anh);
    } else {
        printf("  (Chưa có %s/golden_int8/fc_acc_test.txt nên chưa so từng ảnh với Python.\n"
               "   Sinh bằng: python ptq_int8.py --chi-tham-chieu)\n", thu_muc);
    }
    if (fp) {
        fclose(fp);
        printf("  So với float32 PyTorch          : khác %d ảnh (float đúng → int8 sai: %d, float sai → int8 đúng: %d)\n",
               khac_float, dung_thanh_sai, sai_thanh_dung);
    }

    printf("\n  Accumulator |acc|: đo trên các ảnh đã chạy, và chặn trên cho MỌI ảnh\n");
    printf("  %-6s %9s %5s %12s %5s\n", "lớp", "đo", "bit", "chặn trên", "bit");
    for (int l = 0; l < 4; l++) {
        const long ct = l < 3 ? chan_tren(conv[l].w, conv[l].b, conv[l].OC, conv[l].IC * K * K, conv[l].zp_vao)
                              : chan_tren(fc_w, fc_b, FC_OUT, FC_IN, fc_zp_vao);
        printf("  %-6s %9ld %5d %12ld %5d\n", l < 3 ? conv[l].ten : "fc", acc_max[l], so_bit(acc_max[l]), ct, so_bit(ct));
    }

    printf("\n[c] Đổi thứ tự: maxpool trên acc int32 trước, requantize sau\n");
    printf("  Requantize/ảnh: %ld → %ld (ít hơn %ld.%02ld lần)\n",
           rq_mau, rq_c_mau, rq_mau / rq_c_mau, (rq_mau * 100 / rq_c_mau) % 100);
    printf("  Số ảnh có pool1/pool2/pool3/fc khác thứ tự gốc: %d/%d\n", khac_c, n_anh);

    const int dat = mau_dat && khac_c == 0 && (!co_ref || khop_acc == n_anh);
    return dat ? 0 : 1;
}
