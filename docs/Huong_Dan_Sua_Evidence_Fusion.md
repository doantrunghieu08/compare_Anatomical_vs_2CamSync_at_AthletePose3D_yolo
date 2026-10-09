       

# Hướng dẫn sửa `evidence_fusion.py` (thay và bổ sung bằng chứng cho DST)

Tài liệu này hướng dẫn sửa theo từng bước, có kiểm tra sau mỗi bước, để bạn biết thay đổi nào thật sự giúp ích. Chữ ký trả về của `fuse_evidences` giữ nguyên `(weights, total_conflict, is_outlier)` nên `physics_refine.py` không phải sửa.

## 0. Chuẩn bị

1. Tạo nhánh hoặc sao lưu file:
   ```bash
   cp algorithms/evidence_fusion.py algorithms/evidence_fusion_backup.py
   ```
2. Chạy pipeline bản hiện tại trên một tập nhỏ cố định (vài subject/motion), tắt sequence refinement, lưu `outputs/results.csv` thành `results_A_baseline.csv`. Đây là mốc so sánh.
3. Ghi lại: MPJPE, PA-MPJPE, số frame hợp lệ, tỷ lệ khớp bị đánh outlier.

## 1. Các vấn đề trong code hiện tại (tại sao cần sửa)

| # | Vấn đề                                                                                                                                              | Hậu quả                                                                                         |
| - | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| 1 | `_joint_bone_error` trả `0.0` khi khớp không có xương nối hoặc thiếu key trong `bone_lengths` (ví dụ key lưu theo chiều `(b, a)`) | `err = 0` nên `m_v = 0.8`: thiếu thông tin lại thành tin tưởng cao                     |
| 2 | `compute_epipolar_bba` đo khoảng cách tia bằng mm, phụ thuộc độ sâu và scale metric (scale lại lấy từ bone prior)                       | `sigma_mm = 60` lệch khi người đứng xa hoặc scale sai                                     |
| 3 | Bone error của một xương được chia đều cho cả hai đầu                                                                                      | Một khớp sai làm các khớp kề nó đúng cũng bị giảm bằng chứng                        |
| 4 | `fuse_evidences` bỏ phần `u` (không biết) sau khi kết hợp, chỉ dùng `vf`                                                                 | Mất thông tin về độ bất định                                                              |
| 5 | Dempster đã chuẩn hóa để loại xung đột, rồi`weights = vf * (1 - total_conflict)` phạt thêm lần nữa                                     | Phạt xung đột hai lần (có thể cố ý, nhưng cần ablation)                                 |
| 6 | Bone đo trên`initial_3d` dựng từ chính các tia dùng cho epipolar                                                                              | Hai nguồn không độc lập; việc lấy trung bình thay vì Dempster là hợp lý, giữ nguyên |

## 2. Các bước sửa

### Bước 1 — Giữ bản cũ để làm baseline, đổi tên hàm bone cũ

Trong `evidence_fusion.py`, đổi tên `compute_bone_bba` hiện tại thành `compute_bone_bba_legacy`. Bản cũ cần để chạy lại biến thể A khi so sánh.

### Bước 2 — Thêm các hàm tiện ích (`error_to_bba`, `discount`, `combine`)

Dán vào sau `triangulate_ray_midpoint`:

```python
def error_to_bba(err, scale, cap=0.85, floor_theta=0.05):
    """Chuyển sai số >= 0 thành (valid, not-valid, unknown). Không cần vòng for."""
    err = np.asarray(err, dtype=float)
    v = cap * np.exp(-(err ** 2) / (2.0 * scale ** 2))
    nv = cap * (1.0 - np.exp(-(err ** 2) / (8.0 * scale ** 2)))
    u = np.maximum(floor_theta, 1.0 - v - nv)
    s = v + nv + u
    return v / s, nv / s, u / s


def discount(bba, alpha):
    """Shafer discounting: nguồn kém tin cậy thì đẩy khối lượng sang Theta."""
    v, nv, u = bba
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha * v, alpha * nv, 1.0 - alpha * (v + nv)


def combine(b1, b2, rule="yager"):
    """Trả (v, nv, u, conflict). rule: 'yager' hoặc 'dempster'."""
    v1, nv1, u1 = b1
    v2, nv2, u2 = b2
    v0 = v1 * v2 + v1 * u2 + u1 * v2
    nv0 = nv1 * nv2 + nv1 * u2 + u1 * nv2
    u0 = u1 * u2
    k = v1 * nv2 + nv1 * v2
    if rule == "yager":  # xung đột dồn vào "không biết"
        return v0, nv0, u0 + k, k
    d = np.maximum(1.0 - k, 1e-4)
    return v0 / d, nv0 / d, u0 / d, k
```

**Kiểm tra nhanh:** ba thành phần `(v, nv, u)` trả về từ `error_to_bba` và `discount` phải cộng bằng 1 trên mọi khớp.

### Bước 3 — Thay bone BBA (sửa vấn đề 1 và 3)

Thêm hàm mới (giữ bản legacy đã đổi tên):

```python
def compute_bone_bba(points_3d, bone_lengths, joint_reliability=None, tolerance=0.22):
    n = len(points_3d)
    rel = np.ones(n) if joint_reliability is None else np.asarray(joint_reliability, dtype=float)
    num, den = np.zeros(n), np.zeros(n)
    for a, b in H36M_BONES:
        L = bone_lengths.get((a, b)) or bone_lengths.get((b, a))
        if not L or L <= 0:
            continue
        r = abs(np.linalg.norm(points_3d[a] - points_3d[b]) - L) / L
        # lỗi của xương được chia theo độ tin cậy của khớp ở đầu kia
        num[a] += rel[b] * r; den[a] += rel[b]
        num[b] += rel[a] * r; den[b] += rel[a]
    has = den > 1e-6
    err = np.where(has, num / np.maximum(den, 1e-6), 0.0)
    v, nv, u = error_to_bba(err, tolerance, cap=0.80)
    # khớp không có prior -> hoàn toàn "không biết"
    return np.where(has, v, 0.0), np.where(has, nv, 0.0), np.where(has, u, 1.0)
```

**Kiểm tra:** in ra số khớp có `has == False`. Nếu thấy nhiều khớp hơn mong đợi, kiểm tra định dạng key của `bone_lengths` (chiều `(a, b)` hay `(b, a)`).

**Chạy biến thể B:** chỉ thay bone ở bước này, giữ epipolar ray_mm và Dempster. Lưu kết quả `results_B.csv`.

### Bước 4 — Thêm epipolar bằng pixel (Sampson) và discount theo góc tam giác

```python
def fundamental_from_projections(p1, p2):
    c1 = np.linalg.svd(p1)[2][-1]          # tâm camera 1 (thuần nhất)
    e2 = p2 @ c1
    ex = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
    return ex @ p2 @ np.linalg.pinv(p1)


def compute_sampson_bba(p1, p2, points1, points2, sigma_px=None):
    F = fundamental_from_projections(p1, p2)
    h1 = np.c_[points1, np.ones(len(points1))]
    h2 = np.c_[points2, np.ones(len(points2))]
    fx1, ftx2 = h1 @ F.T, h2 @ F
    num = np.sum(h2 * fx1, axis=1) ** 2
    den = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    dist = np.sqrt(num / np.maximum(den, 1e-12))
    if sigma_px is None:  # theo chiều cao người trong ảnh, không phụ thuộc độ phân giải
        h = 0.5 * (np.ptp(points1[:, 1]) + np.ptp(points2[:, 1]))
        sigma_px = max(2.0, 0.015 * h)
    return error_to_bba(dist, sigma_px, cap=0.85)


def triangulation_alpha(p1, p2, points1, points2, full_deg=15.0, floor=0.2):
    """Hai tia gần song song -> chiều sâu kém ổn định -> giảm alpha."""
    n = len(points1)
    alpha = np.ones(n)
    for i in range(n):
        _, d1 = _camera_center_and_ray(p1, points1[i])
        _, d2 = _camera_center_and_ray(p2, points2[i])
        ang = np.degrees(np.arccos(np.clip(abs(np.dot(d1, d2)), 0.0, 1.0)))
        alpha[i] = np.clip(ang / full_deg, floor, 1.0)
    return alpha
```

**Kiểm tra quan trọng (sanity check Sampson):** với một frame, tính `dist` trên các điểm đã triangulate rồi chiếu lại. Sai số Sampson phải nhỏ (vài pixel trở xuống) khi hai camera đã được ước lượng tốt. Nếu `dist` lớn đều ở mọi khớp, nghi ngờ P1/P2 hoặc quy ước hệ tọa độ pixel chứ không phải keypoint.

Lưu ý: `_camera_center_and_ray` đang giả định `P[:, :3]` khả nghịch và P đã bao gồm intrinsics (tọa độ pixel). `compute_sampson_bba` dùng cùng giả định đó.

**Chạy biến thể C:** bone mới + Sampson + discount góc. Lưu `results_C.csv`.

### Bước 5 — Thêm nguồn nhất quán theo thời gian (tùy chọn, biến thể E)

```python
def compute_temporal_bba(pts, prev, nxt, sigma_px):
    err = np.linalg.norm(pts - 0.5 * (prev + nxt), axis=1)
    return error_to_bba(err, sigma_px, cap=0.75)
```

Chỉ truyền `temporal` khi ba frame (t−1, t, t+1) thật sự liên tiếp. Với keyframe thưa, truyền `None`; nguồn này tự thành neutral. Chuyển động nhanh làm second-difference lớn một cách tự nhiên, nên bắt đầu với sigma rộng (`0.02 * chiều cao người trong ảnh`) rồi chỉnh.

### Bước 6 — Viết lại `fuse_evidences`

Thay toàn bộ hàm bằng bản sau. Bản này đã sửa một lỗi so với lần trước: nếu tắt nguồn `epipolar`, độ tin cậy khớp truyền cho bone không còn là mảng 0 (làm mọi khớp thành "không biết"), mà là `None` (xem dòng `joint_rel`).

```python
def fuse_evidences(
    conf_a, conf_b, p1, p2, points1, points2, bone_lengths, *,
    initial_3d=None, conflict_threshold=0.65,
    sources=("detector", "epipolar", "bone"),   # thêm "temporal" để bật
    epi_mode="sampson",                          # "sampson" | "ray_mm"
    rule="yager",                                # "yager" | "dempster"
    unknown_trust=0.3,
    temporal=None,                               # (prev1, next1, prev2, next2) hoặc None
):
    n = len(points1)
    neutral = (np.zeros(n), np.zeros(n), np.ones(n))
    use = set(sources)
    if initial_3d is None:
        initial_3d = triangulate_ray_midpoint(p1, p2, points1, points2)

    # --- Nguồn detector (+ temporal nếu có) ---
    det = compute_yolo_bba(conf_a, conf_b) if "detector" in use else neutral
    if "temporal" in use and temporal is not None:
        pv1, nx1, pv2, nx2 = temporal
        h = 0.5 * (np.ptp(points1[:, 1]) + np.ptp(points2[:, 1]))
        s = max(2.0, 0.02 * h)
        t1 = compute_temporal_bba(points1, pv1, nx1, s)
        t2 = compute_temporal_bba(points2, pv2, nx2, s)
        det = combine(det, t1, "dempster")[:3]
        det = combine(det, t2, "dempster")[:3]

    # --- Nhóm hình học: epipolar + bone ---
    geo, epi, bone = [], neutral, neutral
    if "epipolar" in use:
        epi = (compute_sampson_bba(p1, p2, points1, points2) if epi_mode == "sampson"
               else compute_epipolar_bba(p1, p2, points1, points2))
        epi = discount(epi, triangulation_alpha(p1, p2, points1, points2))
        geo.append(epi)
    if "bone" in use:
        # độ tin cậy khớp dùng để chia lỗi xương; sàn 0.1 để tránh chia cho ~0
        joint_rel = (0.1 + 0.9 * epi[0]) if "epipolar" in use else None
        bone = compute_bone_bba(initial_3d, bone_lengths, joint_reliability=joint_rel)
        geo.append(bone)
    geometry = (tuple(sum(c) / len(geo) for c in zip(*geo)) if geo else neutral)

    geometry_conflict = (epi[0] * bone[1] + epi[1] * bone[0]) if len(geo) == 2 else np.zeros(n)
    vf, nvf, uf, k = combine(det, geometry, rule)
    total_conflict = np.maximum(k, geometry_conflict)

    if rule == "yager":
        weights = np.clip(vf + unknown_trust * uf, 0.02, 1.0)
    else:
        weights = np.clip(vf * (1.0 - total_conflict), 0.02, 1.0)
    is_outlier = (total_conflict > conflict_threshold) | (nvf > vf)
    return weights, total_conflict, is_outlier
```

Ghi chú:

- Với Yager, trọng số cuối là `v + λ·u` (λ = `unknown_trust`). λ cao thì khớp mơ hồ vẫn được tin; λ thấp thì gần hành vi cũ.
- Với Dempster, công thức giữ nguyên bản gốc (vẫn phạt xung đột lần hai, vấn đề 5) để biến thể A so sánh được. Muốn ablation vấn đề 5, thêm một tùy chọn `double_penalty=False` rồi dùng `weights = vf` khi tắt.

## 3. Nối vào config và pipeline

Tôi chưa thấy `pipeline.py` và `settings.py`, nên đây là hướng chung. Bạn cần điều chỉnh theo code thật.

1. **`configs/default.yml`**, trong nhóm `dst_physics`, thêm:
   ```yaml
   dst_physics:
     # ... các khóa hiện có ...
     fusion_sources: [detector, epipolar, bone]   # thêm temporal để bật
     fusion_epi_mode: sampson                     # sampson | ray_mm
     fusion_rule: yager                           # yager | dempster
     fusion_unknown_trust: 0.3
   ```
2. **`settings.py`**: đọc các khóa trên vào dataclass cấu hình, kèm kiểm tra giá trị (`fusion_rule` và `fusion_epi_mode` thuộc tập cho phép; các nguồn thuộc tập `{detector, epipolar, bone, temporal}`). Đặt mặc định là bản cũ nếu muốn không đổi hành vi.
3. **Nơi gọi `fuse_evidences`** (trong `pipeline.py` hoặc `physics_refine.py`): truyền thêm
   ```python
   fuse_evidences(..., sources=cfg.fusion_sources, epi_mode=cfg.fusion_epi_mode,
                  rule=cfg.fusion_rule, unknown_trust=cfg.fusion_unknown_trust,
                  temporal=temporal_args)
   ```

   `temporal_args` là `None` mặc định. Chỉ dựng nó khi đang xử lý frame liên tiếp và nguồn `temporal` bật.
4. **Biến thể A cần chạy lại bản cũ.** Cách đơn giản nhất: giữ `evidence_fusion_backup.py` và chạy A từ file backup, hoặc thêm `bone_mode: legacy` gọi `compute_bone_bba_legacy`.

## 4. Kiểm tra trước khi chạy dữ liệu thật

Tạo script nhỏ `tests/check_fusion.py` (hoặc chạy tạm trong notebook):

```python
import numpy as np
from algorithms import evidence_fusion as ef

# 1) BBA luôn cộng bằng 1
v, nv, u = ef.error_to_bba(np.array([0.0, 1.0, 10.0]), 2.0)
assert np.allclose(v + nv + u, 1.0)

# 2) discount cũng cộng bằng 1
v2, nv2, u2 = ef.discount((v, nv, u), 0.5)
assert np.allclose(v2 + nv2 + u2, 1.0)

# 3) khớp thiếu prior -> "không biết" hoàn toàn
pts = np.random.randn(17, 3) * 100
v, nv, u = ef.compute_bone_bba(pts, {})   # bone_lengths rỗng
assert np.allclose(u, 1.0) and np.allclose(v, 0.0)

# 4) trọng số nằm trong [0.02, 1]
# (gọi fuse_evidences trên một frame thật rồi kiểm tra weights.min()/max())
print("OK")
```

Nếu mục 4 cho thấy `weights` gần như luôn bằng 0.02 hoặc luôn bằng 1, thang sigma đang sai (xem mục 6).

## 5. Thực nghiệm so sánh công bằng

Chạy trên cùng sequence, cặp camera, tập frame GT hợp lệ, tắt sequence refinement khi so sánh nguồn bằng chứng.

| Biến thể    | Cấu hình                                                                                   |
| ------------- | -------------------------------------------------------------------------------------------- |
| A (baseline)  | `epi_mode="ray_mm"`, `rule="dempster"`, bone legacy                                      |
| B             | A + bone mới (sửa vấn đề 1, 3)                                                          |
| C             | B +`epi_mode="sampson"` + discount góc tam giác                                          |
| D             | C +`rule="yager"`                                                                          |
| E             | D +`sources` thêm `"temporal"`                                                          |
| Leave-one-out | từ biến thể tốt nhất, bỏ lần lượt`detector`, `epipolar`, `bone`, `temporal` |

Với mỗi biến thể ghi: MPJPE, PA-MPJPE, số frame hợp lệ, tỷ lệ khớp bị đánh outlier, và so với DLT trên cùng frame. Cách đọc:

- MPJPE giảm nhưng PA-MPJPE không đổi: cải thiện nằm ở vị trí/scale, không phải hình dáng.
- Tỷ lệ outlier tăng mạnh mà MPJPE không cải thiện: ngưỡng `conflict_threshold` hoặc sigma quá gắt.
- Một biến thể tốt hơn ở subject này nhưng xấu hơn ở subject khác: kiểm tra offset đồng bộ và chất lượng cặp camera trước khi kết luận về nguồn bằng chứng.
- Cần chạy ít nhất vài subject/motion; một sequence không đủ để kết luận.

## 6. Tham số cần chỉnh và triệu chứng

| Tham số                      | Giá trị khởi đầu                              | Triệu chứng cần chỉnh                                                                                                     |
| ----------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `sigma_px` (Sampson)        | `max(2, 0.015 × chiều cao người trong ảnh)` | Hầu hết khớp có`v ≈ 0`: tăng. Mọi khớp đều `v` cao kể cả khớp sai: giảm                                     |
| `full_deg` (góc tam giác) | 15°                                               | Nếu cặp camera chọn thường có góc nhỏ, discount làm yếu toàn bộ epipolar: giảm`full_deg` hoặc nâng `floor` |
| `tolerance` (bone)          | 0.22                                               | Hệ số 22% dung sai độ dài xương; giảm nếu bone prior đáng tin                                                      |
| `unknown_trust` (Yager)     | 0.3                                                | Quét 0.0, 0.2, 0.3, 0.5                                                                                                      |
| `conflict_threshold`        | 0.65                                               | Quét 0.5–0.8, theo dõi tỷ lệ outlier                                                                                     |
| sigma temporal                | `0.02 × chiều cao người`                     | Chuyển động nhanh làm nhiều khớp bị phạt oan: tăng                                                                   |

## 7. Hạn chế cần nhớ

- Độ dài xương không phát hiện được khớp lệch vuông góc với xương; phần này do epipolar và temporal đảm nhận.
- Bone đo trên `initial_3d` nên vẫn phụ thuộc DLT khởi tạo. Nếu DLT tốt, bone evidence ít thông tin mới.
- Sampson dựa trên F suy ra từ P1/P2 ước lượng từ pose; nếu P1/P2 sai thì mọi nguồn hình học đều sai theo.
- Kết quả so với DLT không đảm bảo luôn thắng. Khi 2D và đồng bộ tốt, DLT vẫn là baseline mạnh.

## 8. Nếu muốn bỏ hẳn DST (hướng nghiên cứu)

Thay phần kết hợp bằng trọng số nghịch đảo phương sai trong ray loss của `physics_refine.py`: mỗi khớp `w = 1/σ²`, với σ² tổng hợp từ sai số Sampson, độ bất định detector và bone. Cách này dễ gỡ lỗi hơn nhưng mất cơ chế đánh dấu xung đột. Chỉ thử sau khi đã có kết quả của các biến thể ở mục 5, và so sánh lại bằng cùng quy trình.
