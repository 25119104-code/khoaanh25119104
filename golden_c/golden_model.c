/* ============================================================
 * golden_model.c — Golden model C (float32) cho SmallCNN
 * ============================================================
 * Phase 2D. Dịch dòng-đổi-dòng từ inference_python.py (bản conv ĐỆM 0).
 *
 * Mục đích: làm CHUẨN ĐỐI CHIẾU cho phần cứng (RTL) sau này. Mọi phép toán
 * viết tay bằng vòng lặp, không thư viện, để thấy rõ phần cứng phải làm gì.
 *
 *   - Chỉ dùng <stdio.h> và <stdlib.h>. KHÔNG cần <math.h>: đường inference
 *     chỉ còn +, ×, so sánh (đã bỏ softmax/exp).
 *   - Mảng 1 chiều, tự tính chỉ số, layout [C][H][W] giống PyTorch (NCHW).
 *   - Conv luôn tính đủ 9 tap trên ảnh đã đệm 0 (Thầy góp ý 23/09).
 *
 * BIÊN DỊCH (đứng ở thư mục gốc project):
 *   cc -O2 -std=c99 -Wall -o golden_c/golden_model golden_c/golden_model.c
 * CHẠY:
 *   ./golden_c/golden_model          # ảnh mẫu (so từng lớp) + toàn bộ 10.000 ảnh test
 *   ./golden_c/golden_model 100      # chỉ 100 ảnh đầu
 *
 * Cần có sẵn (do export_params.py sinh): params/weights.txt, params/biases.txt,
 * các file trong golden/. Ảnh test đọc thẳng từ data/MNIST/raw/ (2 file t10k).
 * ============================================================ */

#include <stdio.h>
#include <stdlib.h>

/* ---------- [1] Kích thước và bảng offset — PHẢI khớp params/README.md ---------- */
#define K    3          /* kernel 3x3 */
#define PAD  1          /* đệm 1 → H_out = H_in */

#define N_WEIGHTS 4968
#define N_BIASES  50

/* offset trong weights.txt / biases.txt */
#define C1_W 0
#define C1_B 0
#define C2_W 72
#define C2_B 8
#define C3_W 1224
#define C3_B 24
#define FC_W 3528
#define FC_B 40

#define FC_IN  144      /* 16 x 3 x 3 */
#define FC_OUT 10

/* Bộ đệm lớn nhất cần: conv1 ra 8 x 28 x 28 = 6272 phần tử */
#define BUF_MAX 6272
/* Ảnh đã đệm lớn nhất: 1 x 30 x 30 = 900, 8 x 16 x 16 = 2048, 16 x 9 x 9 = 1296 */
#define PADDED_MAX 2048

static float W[N_WEIGHTS];
static float B[N_BIASES];

/* Đếm số phép nhân-cộng thực sự chạy — phải ra 395.136 cho conv */
static long mac_conv = 0;

/* ---------- [2] Đọc file ---------- */
static void doc_float(const char *path, float *dst, int n)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Không mở được %s\n", path); exit(1); }
    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%f", &dst[i]) != 1) {
            fprintf(stderr, "%s: chỉ đọc được %d/%d số\n", path, i, n);
            exit(1);
        }
    }
    fclose(f);
}

/* ---------- [3] Sáu operator ---------- */

/* Conv2d 3x3, stride 1, pad 1. Bước 1: tạo ảnh đã đệm 0. Bước 2: luôn 9 tap. */
static void conv2d(const float *in, int IC, int H, int Wd,
                   int OC, int w_off, int b_off, float *out)
{
    static float padded[PADDED_MAX];
    const int HP = H + 2 * PAD, WP = Wd + 2 * PAD;

    for (int i = 0; i < IC * HP * WP; i++) padded[i] = 0.0f;       /* đệm 0 */
    for (int ic = 0; ic < IC; ic++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < Wd; x++)
                padded[(ic * HP + y + PAD) * WP + x + PAD] = in[(ic * H + y) * Wd + x];

    for (int oc = 0; oc < OC; oc++)
        for (int oy = 0; oy < H; oy++)
            for (int ox = 0; ox < Wd; ox++) {
                float acc = B[b_off + oc];                  /* khởi tạo bằng BIAS, không phải 0 */
                for (int ic = 0; ic < IC; ic++)
                    for (int ky = 0; ky < K; ky++)
                        for (int kx = 0; kx < K; kx++) {
                            acc += padded[(ic * HP + oy + ky) * WP + ox + kx]
                                 * W[w_off + ((oc * IC + ic) * K + ky) * K + kx];
                            mac_conv++;
                        }
                out[(oc * H + oy) * Wd + ox] = acc;
            }
}

static void relu(float *x, int n)
{
    for (int i = 0; i < n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;               /* chỉ là 1 phép so sánh */
}

/* MaxPool 2x2 stride 2. H_out = H / 2 (chia nguyên = floor): 7 → 3, hàng/cột cuối bị bỏ. */
static void maxpool2d(const float *in, int C, int H, int Wd, float *out)
{
    const int HO = H / 2, WO = Wd / 2;
    for (int c = 0; c < C; c++)
        for (int oy = 0; oy < HO; oy++)
            for (int ox = 0; ox < WO; ox++) {
                float m = in[(c * H + 2 * oy) * Wd + 2 * ox];
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        float v = in[(c * H + 2 * oy + dy) * Wd + 2 * ox + dx];
                        if (v > m) m = v;
                    }
                out[(c * HO + oy) * WO + ox] = m;
            }
}

/* Flatten: mảng đã nằm theo [C][H][W] nên KHÔNG làm gì — chỉ đọc tiếp như vector 144. */

static void linear(const float *x, float *z)
{
    for (int o = 0; o < FC_OUT; o++) {
        float acc = B[FC_B + o];
        for (int i = 0; i < FC_IN; i++)
            acc += x[i] * W[FC_W + o * FC_IN + i];
        z[o] = acc;
    }
}

/* So sánh CHẶT (>) để khớp torch.argmax khi có logit bằng nhau. */
static int argmax(const float *z, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (z[i] > z[best]) best = i;
    return best;
}

/* ---------- [4] Ghép lại ---------- */
/* Lưu đầu ra từng lớp để so với golden/ (chỉ dùng cho ảnh mẫu) */
static float L_conv1[6272], L_pool1[1568], L_conv2[3136], L_pool2[784],
             L_conv3[784], L_pool3[144];

static int infer(const float *img, float *z)
{
    conv2d(img, 1, 28, 28, 8, C1_W, C1_B, L_conv1);   relu(L_conv1, 6272);
    maxpool2d(L_conv1, 8, 28, 28, L_pool1);           /* 8 x 14 x 14 */
    conv2d(L_pool1, 8, 14, 14, 16, C2_W, C2_B, L_conv2); relu(L_conv2, 3136);
    maxpool2d(L_conv2, 16, 14, 14, L_pool2);          /* 16 x 7 x 7 */
    conv2d(L_pool2, 16, 7, 7, 16, C3_W, C3_B, L_conv3);  relu(L_conv3, 784);
    maxpool2d(L_conv3, 16, 7, 7, L_pool3);            /* 16 x 3 x 3 = 144 */
    linear(L_pool3, z);                               /* flatten không tốn phép nào */
    return argmax(z, FC_OUT);
}

/* ---------- [5] Tiện ích kiểm tra ---------- */
static float absf(float v) { return v < 0.0f ? -v : v; }

static float lech_max(const float *a, const char *golden_path, int n)
{
    static float g[BUF_MAX];
    doc_float(golden_path, g, n);
    float m = 0.0f;
    for (int i = 0; i < n; i++) if (absf(a[i] - g[i]) > m) m = absf(a[i] - g[i]);
    return m;
}

/* Tiền xử lý — LÀM TRÊN PC, không phải trên chip: (x/255 − 0,1307) / 0,3081, float32
 * đúng thứ tự PyTorch. Có phép chia nên câu hỏi "chuẩn hoá làm ở đâu" vẫn đang chờ Thầy. */
static void chuan_hoa(const unsigned char *px, float *img)
{
    for (int i = 0; i < 784; i++)
        img[i] = ((float)px[i] / 255.0f - 0.1307f) / 0.3081f;
}

/* ---------- [6] main ---------- */
int main(int argc, char **argv)
{
    int n_anh = (argc > 1) ? atoi(argv[1]) : 10000;
    if (n_anh < 0 || n_anh > 10000) n_anh = 10000;

    doc_float("params/weights.txt", W, N_WEIGHTS);
    doc_float("params/biases.txt", B, N_BIASES);
    printf("Đọc %d weight + %d bias = %d params\n", N_WEIGHTS, N_BIASES, N_WEIGHTS + N_BIASES);

    /* (a) Ảnh mẫu: so TỪNG LỚP với PyTorch (sau ReLU, vì relu ghi đè tại chỗ) */
    float img[784], z[10];
    doc_float("golden/input.txt", img, 784);
    mac_conv = 0;
    int pred = infer(img, z);

    FILE *fl = fopen("golden/label.txt", "r");
    int nhan = -1;
    if (!fl || fscanf(fl, "%d", &nhan) != 1) { fprintf(stderr, "Lỗi đọc golden/label.txt\n"); return 1; }
    fclose(fl);

    printf("\n[a] Ảnh mẫu (nhãn %d) — lệch lớn nhất so với PyTorch\n", nhan);
    printf("  %-12s %8s %14s\n", "lớp", "số pt", "lệch max");
    struct { const char *ten, *file; const float *a; int n; } bang[] = {
        {"relu1", "golden/02_relu1.txt", L_conv1, 6272},
        {"pool1", "golden/03_pool1.txt", L_pool1, 1568},
        {"relu2", "golden/05_relu2.txt", L_conv2, 3136},
        {"pool2", "golden/06_pool2.txt", L_pool2, 784},
        {"relu3", "golden/08_relu3.txt", L_conv3, 784},
        {"pool3", "golden/09_pool3.txt", L_pool3, 144},
        {"flatten", "golden/10_flatten.txt", L_pool3, 144},
        {"logits", "golden/11_fc_logits.txt", z, 10},
    };
    float lech_lon_nhat = 0.0f;
    for (int i = 0; i < 8; i++) {
        float d = lech_max(bang[i].a, bang[i].file, bang[i].n);
        if (d > lech_lon_nhat) lech_lon_nhat = d;
        printf("  %-10s %7d %12.2e\n", bang[i].ten, bang[i].n, (double)d);
    }
    printf("  Đoán %d, nhãn đúng %d. Số MAC conv / ảnh: %ld (phải là 395136)\n", pred, nhan, mac_conv);
    printf("  KẾT LUẬN ảnh mẫu: %s\n", lech_lon_nhat < 1e-4f ? "KHỚP PyTorch ở mọi lớp (< 1e-4)" : "CÓ LỚP LỆCH");

    if (n_anh == 0) return 0;

    /* (b) Nhiều ảnh: đọc thẳng file raw idx, so dự đoán với logit PyTorch */
    FILE *fi = fopen("data/MNIST/raw/t10k-images-idx3-ubyte", "rb");
    FILE *fb = fopen("data/MNIST/raw/t10k-labels-idx1-ubyte", "rb");
    FILE *fp = fopen("golden/logits_pytorch.txt", "r");
    if (!fi || !fb || !fp) { fprintf(stderr, "Thiếu file MNIST raw hoặc golden/logits_pytorch.txt\n"); return 1; }
    fseek(fi, 16, SEEK_SET);                       /* header ảnh 16 byte */
    fseek(fb, 8, SEEK_SET);                        /* header nhãn 8 byte */

    unsigned char px[784], lb;
    float zt[10];
    int dung = 0, trung = 0;
    float lech_logit = 0.0f;
    for (int i = 0; i < n_anh; i++) {
        if (fread(px, 1, 784, fi) != 784 || fread(&lb, 1, 1, fb) != 1) { fprintf(stderr, "Hết dữ liệu\n"); return 1; }
        for (int k = 0; k < 10; k++)
            if (fscanf(fp, "%f", &zt[k]) != 1) { fprintf(stderr, "Lỗi đọc logits_pytorch\n"); return 1; }
        chuan_hoa(px, img);
        int p = infer(img, z);
        dung += (p == lb);
        trung += (p == argmax(zt, 10));
        for (int k = 0; k < 10; k++) if (absf(z[k] - zt[k]) > lech_logit) lech_logit = absf(z[k] - zt[k]);
    }
    fclose(fi); fclose(fb); fclose(fp);

    printf("\n[b] %d ảnh đầu test set\n", n_anh);
    printf("  Accuracy golden model C : %.2f%%\n", 100.0 * dung / n_anh);
    printf("  Trùng dự đoán PyTorch   : %d/%d\n", trung, n_anh);
    printf("  Logit lệch lớn nhất     : %.2e\n", (double)lech_logit);
    return 0;
}
