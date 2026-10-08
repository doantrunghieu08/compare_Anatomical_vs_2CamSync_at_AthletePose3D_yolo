# Hướng dẫn sửa sai số cho pipeline AthletePose3D (Uncalibrated 2-Camera)

Tài liệu này dựa trên kết quả ablation của bạn (`ablation_methods_summary.csv`, `ablation_per_joint_summary.csv`, `ablation_per_frame_joint_errors.csv`) và kiến trúc uncalibrated mới nhất.

---

## 0. Tóm tắt chẩn đoán

| Quan sát từ dữ liệu | Ý nghĩa |
|---|---|
| MPJPE ≈ 650 mm nhưng PA-MPJPE ≈ 67 mm | Chênh lệch gần 10 lần; MPJPE đang đo độ lệch hệ tọa độ, không đo chất lượng pose |
| Tương quan MPJPE và PA-MPJPE trên 138 frame = −0.02 | MPJPE không phản ánh hình dạng pose |
| MPJPE theo khớp tăng theo khoảng cách tới pelvis (hông ~180, gối 630–750, cổ chân 1080–1200 mm) | Dấu hiệu của một phép quay lớn giữa hai hệ tọa độ, không phải nhiễu hay sai tỷ lệ |
| 6 khớp suy ra (Head, Spine, Neck, 2 Hip, Thorax) có PA trung bình ~104 mm; 10 khớp còn lại ~46 mm | Khoảng 58% sai số thật đến từ khác biệt định nghĩa khớp COCO và H36M |
| DST / Physics / SeqRefine cải thiện ≤ 1.14% PA (dưới 1 mm), win rate 50–60% | Nằm trong nhiễu; chưa đáng tune thêm |
| Dữ liệu chỉ có subject S1, 5 cặp camera, 138 frame | Quá nhỏ để kết luận chắc chắn |

**Giả thuyết chính (cần kiểm chứng ở Bước 1):** pipeline xuất 3D trong hệ camera 1 (vì `P1 = K1 [I | 0]`, trục Y hướng xuống), còn GT MoCap nằm trong hệ thế giới (Z hướng lên). MPJPE hiện chỉ căn gốc pelvis nên không loại được phép quay này.

---

## Thứ tự thực hiện

1. Kiểm chứng giả thuyết lỗi hệ quy chiếu
2. Sửa cách đánh giá (MPJPE căn quay theo chuỗi)
3. Đưa kết quả về hệ Z-up không cần GT
4. Tách sai số theo nhóm khớp
5. Bộ hồi quy khớp COCO sang H36M
6. Quét tiêu cự theo từng cặp camera
7. Kiểm định thống kê và mở rộng dữ liệu
8. Kiểm tra keyframe có liên tiếp không
9. Cập nhật tài liệu kiến trúc

---

## Bước 1. Kiểm chứng: đo góc quay giữa dự đoán và GT

Với mỗi cặp camera, gộp toàn bộ frame của chuỗi, căn pelvis về gốc, rồi tìm phép quay Kabsch (không co giãn) tốt nhất giữa dự đoán và GT.

```python
import numpy as np

def kabsch(P, G, with_scale=False):
    """P, G: (N, 3), đã căn pelvis về gốc. Trả về R sao cho G ≈ s * (P @ R.T)."""
    H = P.T @ G
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    s = 1.0
    if with_scale:
        s = (S * np.diag(D)).sum() / (P ** 2).sum()
    return R, s

def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

def sequence_diagnosis(pred, gt):
    """pred, gt: (T, 17, 3), pelvis = khớp 0."""
    P = (pred - pred[:, :1])[:, 1:].reshape(-1, 3)
    G = (gt - gt[:, :1])[:, 1:].reshape(-1, 3)
    R, s = kabsch(P, G, with_scale=True)
    aligned = s * (P @ R.T)
    err = np.linalg.norm(aligned - G, axis=1).mean()
    return {"angle_deg": rotation_angle_deg(R), "scale": s, "mpjpe_seq_rot_mm": err}
```

Cách đọc kết quả:

- `angle_deg` lớn (hàng chục đến hơn 100 độ) và khác nhau giữa các cặp camera: đúng là lỗi hệ quy chiếu.
- `scale` khác 1 nhiều: có thêm lỗi tỷ lệ riêng (xem Bước 6).
- `mpjpe_seq_rot_mm` gần mức PA-MPJPE (khoảng 70–100 mm): xác nhận phần lớn 650 mm chỉ là phép quay.

---

## Bước 2. Sửa cách đánh giá

Thay MPJPE "chỉ căn pelvis" bằng ba chỉ số báo cáo cùng nhau:

| Chỉ số | Cách tính | Dùng để |
|---|---|---|
| MPJPE-SeqRot | Một phép quay Kabsch (không co giãn) cho cả chuỗi, rồi tính MPJPE | Sai số thật khi chưa biết hướng |
| PA-MPJPE | Giữ nguyên | Chất lượng hình dạng pose |
| Scale ratio | Hệ số co giãn Procrustes giữa dự đoán và GT | Phát hiện lỗi tỷ lệ riêng |

Lưu ý: phép quay là **một cho cả chuỗi**, không phải mỗi frame. Căn quay từng frame sẽ che mất lỗi thật và đồng nghĩa với PA-MPJPE.

---

## Bước 3. Đưa kết quả về hệ Z-up không cần GT

Ước lượng hướng "lên" từ chính pose rồi xoay về trục Z. Cách này hợp lệ cho bài toán không dùng MoCap lúc chạy.

```python
def estimate_up_axis(poses):
    """poses: (T, 17, 3) trong hệ camera. Dùng hướng pelvis -> neck trung bình."""
    v = poses[:, 9] - poses[:, 0]            # H36M: 0 = pelvis, 9 = neck
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    up = np.median(v, axis=0)
    return up / np.linalg.norm(up)

def rotation_a_to_b(a, b):
    """Ma trận quay đưa vector đơn vị a về b (Rodrigues)."""
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(a, a)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)

def to_z_up(poses):
    R = rotation_a_to_b(estimate_up_axis(poses), np.array([0.0, 0.0, 1.0]))
    return poses @ R.T
```

Sau bước này chỉ còn lệch hướng quay quanh trục dọc (yaw), không thể biết nếu không có tham chiếu. Khi đánh giá, chỉ căn riêng yaw cho cả chuỗi:

```python
def best_yaw(P, G):
    """P, G: (N, 3) đã ở hệ Z-up. Tìm góc quay quanh Z (2D Kabsch trên XY)."""
    H = P[:, :2].T @ G[:, :2]
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R2 = Vt.T @ np.diag([1.0, d]) @ U.T
    R = np.eye(3)
    R[:2, :2] = R2
    return R
```

Chú ý với động tác thể thao (nhảy Axel): hướng thân người nghiêng nhiều trong lúc bay, nên dùng trung vị trên cả chuỗi (đã làm ở trên) hoặc dùng các frame đứng thẳng. Nếu muốn chắc hơn, có thể ước lượng mặt đất từ cổ chân.

---

## Bước 4. Tách sai số theo nhóm khớp

Tính PA-MPJPE riêng cho khớp rõ ràng và khớp bị suy ra, trực tiếp từ CSV:

```python
import pandas as pd

d = pd.read_csv("ablation_per_frame_joint_errors.csv")

DERIVED = [1, 4, 7, 8, 9, 10]        # Hip R/L, Spine, Thorax, Neck, Head
CLEAR   = [2, 3, 5, 6, 11, 12, 13, 14, 15, 16]

def pa_cols(idx):
    return [c for c in d.columns if c.startswith("PA_Joint_")
            and int(c.split("_")[2]) in idx]

for name, idx in [("derived", DERIVED), ("clear", CLEAR)]:
    d[f"PA_{name}"] = d[pa_cols(idx)].mean(axis=1)

print(d.groupby("Method")[["PA_derived", "PA_clear"]].mean().round(1))
```

Kết quả hiện tại với DLT Baseline: khớp suy ra khoảng 104 mm, khớp rõ ràng khoảng 46 mm. Báo cáo cả hai con số này cạnh PA-MPJPE tổng để biết thuật toán thật sự đang tốt đến đâu.

---

## Bước 5. Bộ hồi quy khớp COCO sang H36M

COCO không có đỉnh đầu, cột sống, cổ; vị trí hông COCO cũng khác tâm khớp hông H36M. Đây là sai số hệ thống nên ràng buộc xương hay vật lý không sửa được. Cách xử lý: học một ánh xạ tuyến tính cố định từ 3D khớp COCO sang 3D khớp H36M.

```python
from sklearn.linear_model import Ridge

def fit_joint_regressor(X, Y, lam=1.0):
    """
    X: (N, K, 3)  khớp COCO 3D, đã căn pelvis, đã xoay về cùng hệ với Y
    Y: (N, J, 3)  khớp H36M cần dự đoán (chủ yếu 1, 4, 7, 8, 9, 10)
    Trả về ma trận trọng số W (J, K): Y_j ≈ sum_k W[j, k] * X_k
    """
    N, K, _ = X.shape
    J = Y.shape[1]
    # hồi quy riêng cho mỗi trục x, y, z chung một W
    Xf = X.transpose(0, 2, 1).reshape(N * 3, K)
    W = np.zeros((J, K))
    for j in range(J):
        yj = Y[:, j].reshape(N * 3)
        W[j] = Ridge(alpha=lam, fit_intercept=False).fit(Xf, yj).coef_
    return W

def apply_regressor(W, X):
    return np.einsum("jk,nkc->njc", W, X)
```

Lưu ý quan trọng:

- Huấn luyện trên các subject khác với subject kiểm tra (chia theo subject, không chia theo frame). Hiện bạn chỉ có S1, nên cần thêm dữ liệu trước khi dùng kết quả này làm số liệu cuối.
- Bộ hồi quy dùng GT chỉ ở giai đoạn huấn luyện. Lúc chạy pipeline vẫn không cần GT, nên vẫn hợp lệ với mục tiêu "độc lập với MoCap" nếu bạn nêu rõ điều này.
- Nếu chưa muốn dùng học, tối thiểu hãy kiểm tra lại cách suy ra pelvis, vì mọi MPJPE đều căn theo pelvis.

---

## Bước 6. Quét tiêu cự theo từng cặp camera

Hiện tại `f = 1.2 * max(W, H)` được đặt cố định. Tiêu cự sai làm méo độ sâu và các bước sau không bù lại được.

Thuật toán gợi ý (tên hàm lấy từ tài liệu kiến trúc, bạn thay bằng chữ ký thật trong `uncalibrated.py`):

```python
def bone_instability(poses, bones):
    """Độ biến thiên chiều dài xương theo thời gian (nhỏ hơn là tốt hơn)."""
    L = np.stack([np.linalg.norm(poses[:, a] - poses[:, b], axis=1)
                  for a, b in bones], axis=1)          # (T, B)
    return float(np.mean(L.std(axis=0) / L.mean(axis=0)))

best = None
for k in np.linspace(0.6, 2.5, 20):                    # f = k * max(W, H)
    f = k * max(W, H)
    K = intrinsics_from_focal(f, W, H)                 # giống approximate_intrinsics
    F = estimate_fundamental_matrix(pts1, pts2, conf)  # tính một lần cho cả cặp
    R, t = recover_relative_cameras(F, K, K, pts1, pts2)
    P1, P2 = solve_metric_scale(K, R, t, pts1, pts2)
    poses = triangulate_all_frames(P1, P2, pts1, pts2)
    score = bone_instability(poses, H36M_BONES)
    if best is None or score < best[0]:
        best = (score, f)
```

Việc nên làm cùng lúc:

- Ước lượng `F` **một lần cho cả cặp** từ mọi frame đã đồng bộ (RANSAC có trọng số độ tin cậy YOLO), không ước lượng theo nhóm 30 frame nhỏ.
- Chỉ dùng các khớp COCO thật khi ước lượng `F` (loại Pelvis, Spine, Thorax, Neck, Head vì đó là khớp suy ra).
- Vẽ PA-MPJPE theo `f` để xem độ nhạy. Nếu đường cong phẳng thì tiêu cự không phải điểm nghẽn.
- Có thể dùng độ biến thiên chiều dài xương làm chỉ số chất lượng hiệu chuẩn và loại cặp camera có giá trị quá cao.

---

## Bước 7. Kiểm định thống kê và mở rộng dữ liệu

Các cải thiện của DST, Physics, SeqRefine dưới 1 mm. Hãy kiểm tra xem chúng có khác nhau về mặt thống kê không:

```python
from scipy.stats import wilcoxon

key = ["Motion", "Subject", "Cam_A", "Cam_B", "Frame"]
base = d[d.Method == "1_DLT_Baseline"].set_index(key)

for m in d.Method.unique():
    if m == "1_DLT_Baseline":
        continue
    cur = d[d.Method == m].set_index(key)
    diff = (base["PA_MPJPE_mm"] - cur["PA_MPJPE_mm"]).dropna()
    stat, p = wilcoxon(diff)
    print(f"{m:28s} mean gain = {diff.mean():+.2f} mm   p = {p:.3g}")
```

Nếu p lớn thì bước đó chưa chứng minh được giá trị. Các việc nên làm thêm:

- Chạy trên nhiều subject và nhiều động tác hơn (hiện chỉ S1, 5 cặp camera, 138 frame).
- Báo cáo khoảng tin cậy bootstrap (chọn lại frame theo cặp camera) thay vì chỉ trung bình.
- Chỉ tune `σ`, `τ`, ngưỡng xung đột DST, các trọng số Physics **sau khi** hoàn tất Bước 1–6, và tune bằng chia theo subject.

---

## Bước 8. Kiểm tra keyframe có liên tiếp không

Độ trơn gia tốc chỉ bật khi `frame_nums` liên tiếp (`np.diff == 1`). SeqRefine đang làm xấu PA ở khoảng 38% số frame. Kiểm tra:

```python
frames = d[(d.Method == "7_DST_Physics_SeqRefine")
           & (d.Motion == "Axel_1") & (d.Cam_A == 2) & (d.Cam_B == 6)]["Frame"].values
print(np.diff(np.sort(frames)))
```

Nếu khoảng cách giữa các frame lớn hơn 1, thành phần độ trơn đang bị tắt, và SeqRefine thực chất chỉ còn các ràng buộc xương/đối xứng/tia. Khi đó hãy xử lý các đoạn liên tục dài (hoặc toàn bộ chuỗi) thay vì 30 keyframe thưa.

---

## Bước 9. Cập nhật tài liệu kiến trúc

Các chỗ cần sửa trong `Kien_Truc_Tong_The_Mo_Hinh.md`:

- **Mục 5.6 (MPJPE):** thay định nghĩa "chỉ căn pelvis" bằng MPJPE-SeqRot; thêm Scale ratio; ghi rõ lý do (không có hiệu chuẩn nên hệ quy chiếu tùy ý).
- **Mục 2 (Uncalibrated Recovery):** thêm bước ước lượng hướng Z-up và bước chọn tiêu cự; ghi rõ `F` được ước lượng một lần cho cả cặp.
- **Mục 4.1 hoặc 5.x:** ghi rõ khớp nào của H36M là khớp suy ra, và bộ hồi quy COCO sang H36M (nếu dùng).
- **Mục 6 (bảng tham số):** thêm `focal_scale_range`, `eval_alignment`, `joint_regressor`.

---

## Kết quả kỳ vọng sau khi sửa

| Chỉ số | Hiện tại | Kỳ vọng |
|---|---|---|
| MPJPE-SeqRot | (chưa đo; MPJPE hiện 650 mm là sai hệ quy chiếu) | Gần mức PA-MPJPE, cỡ 70–100 mm |
| PA-MPJPE khớp rõ ràng | ~46 mm | Giữ hoặc giảm nhờ tiêu cự tốt hơn |
| PA-MPJPE khớp suy ra | ~104 mm | Giảm rõ rệt nhờ bộ hồi quy khớp |
| PA-MPJPE tổng | 67.7 mm | Giảm theo hai dòng trên |
| Đóng góp DST/Physics/SeqRefine | ≤ 1.14% | Đo lại bằng kiểm định cặp, không kết luận trước |

Đây là ước lượng định tính, giá trị thật chỉ có sau khi chạy Bước 1. Nếu góc quay ở Bước 1 nhỏ và ổn định thì giả thuyết sai và cần tìm nguyên nhân khác (ví dụ lỗi quy ước đơn vị hoặc hoán trục), nên đừng bỏ qua bước này.
