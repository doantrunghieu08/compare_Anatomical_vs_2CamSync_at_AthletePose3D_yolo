# Hướng dẫn sửa đổi repo `compare_Anatomical_vs_2CamSync_at_AthletePose3D_yolo`

Tài liệu này biến phần nhận xét thành các bước sửa cụ thể. Mỗi mục gồm: **vấn đề**, **cách sửa** (kèm code), **cách kiểm chứng**.
Thứ tự mục = thứ tự ưu tiên. Làm hết mục 1–5 trước khi viết thêm bất kỳ kết luận nào trong README.

> Ghi chú: các đoạn code bên dưới là gợi ý dựa trên phiên bản repo tôi đọc ngày 10/10/2026. Hãy đọc lại file thật trước khi dán.

---

## Mục lục

1. [Sửa `sync_local_radius: 0` (đang bị ép thành 1)](#1)
2. [Làm lại ablation cho sạch (tách chiều cao khỏi phương pháp)](#2)
3. [Báo cáo metric trung thực (raw / rigid / cấp chuỗi / PA)](#3)
4. [Làm mượt thời gian theo từng cặp frame liên tiếp](#4)
5. [Mốc tham chiếu camera và thống nhất `f_scale`](#5)
6. [Cho `anatomical` có `lr` cấu hình được và quét thử](#6)
7. [Thống kê: CI theo motion, mở rộng S2/S3, tách dev/test](#7)
8. [Dọn lỗi nhỏ trong code](#8)
9. [Test thật sự (pytest)](#9)
10. [Sửa README và docs](#10)
11. [Checklist nghiệm thu](#11)

---

<a id="1"></a>
## 1. Sửa `sync_local_radius: 0`

**Vấn đề.** Trong `pipeline.py`, hàm `_fps_scaled_sync_params`:

```python
"local_radius": max(1, int(round(config.sync_local_radius * scale))),
```

Giá trị 0 trong YAML luôn thành 1, nên "radius 0" mà README ca ngợi thực ra là radius 1. Giá trị này còn được dùng làm `validation_radius` khi chọn cặp camera.

**Cách sửa.**

```python
# pipeline.py -> _fps_scaled_sync_params
"local_radius": max(0, int(round(config.sync_local_radius * scale))),
# giữ nguyên max(1, ...) cho dynamic_radius vì DP cần ít nhất 1 offset lân cận
```

Kiểm tra các hàm dùng `radius` chấp nhận 0:
`select_synced_pose` (`range(base-0, base+1)` → 1 offset, ổn), `_best_validation_match` (ổn), `_validation_frames` (margin = `|offset| + 0 + 2`, ổn).

**Kiểm chứng.** Chạy lại 3 giá trị `sync_local_radius ∈ {0, 1, 2}` trên S1 với config 03.
Giả thuyết cũ ("radius 2 làm tăng +1.20 mm") chỉ được giữ lại nếu số mới còn ủng hộ. Cập nhật mục "Bài học 1" trong README theo kết quả thật.

---

<a id="2"></a>
## 2. Làm lại ablation cho sạch

**Vấn đề.**
- Config 01 (DLT) dùng `subject_height_mm: auto` (chiều cao hiệu chuẩn), còn bước 02 dùng 1730 mm chung (theo docs). Chiều cao ảnh hưởng tới **thang mét của P2** (`solve_metric_scale`) và điểm chọn frame, không chỉ bone prior. Vì vậy bước 02 đổi hai thứ cùng lúc.
- Docs viết ablation "đảm bảo sai số giảm đơn điệu": nên bỏ, vì ablation phải báo cáo kết quả, không được thiết kế để ra kết quả định sẵn. Thực tế PA-MPJPE đã không đơn điệu (50.52 → 50.87).
- Bước 5 ghi `sync_local_radius: 0` như thay đổi mới trong khi 01/03/04 đã có sẵn.

**Cách sửa: thiết kế 2×2 rồi thêm các thành phần.**

| ID | Phương pháp | Chiều cao | Sequence refine | Ghi chú |
|----|-------------|-----------|-----------------|---------|
| A1 | `dlt` | generic 1730 | tắt | baseline, chiều cao kém |
| A2 | `dlt` | calibrated | tắt | baseline, chiều cao tốt |
| B1 | `anatomical` | generic 1730 | tắt | |
| B2 | `anatomical` | calibrated | tắt | |
| C1 | B2 | calibrated | bật, chỉ gia tốc | |
| C2 | C1 | calibrated | bật, gia tốc + vận tốc | |

Cách đọc:
- **Hiệu ứng phương pháp** = A2 → B2 (cùng chiều cao).
- **Hiệu ứng chiều cao** = A1 → A2 và B1 → B2.
- **Hiệu ứng refine** = B2 → C1, C1 → C2.

Mỗi config chỉ khác nhau đúng một khóa so với config đứng trước nó trong phép so sánh. Nên sinh config bằng script thay vì sao chép tay 5 file YAML dài (xem bên dưới).

```python
# scripts/make_ablation_configs.py (gợi ý)
import copy, yaml
from pathlib import Path

base = yaml.safe_load(Path("configs/default.yml").read_text(encoding="utf-8"))

def variant(name, method, height, refine, smooth=0.15, vel=0.0):
    cfg = copy.deepcopy(base)
    cfg["output"]["results_csv"] = f"outputs/ablation/{name}.csv"
    cfg["output"]["overwrite"] = True
    cfg["pipeline"]["subject_height_mm"] = height          # 1730.0 hoặc "auto"
    cfg["pipeline"]["sequence_refinement"].update(
        enabled=refine, smoothness_weight=smooth, velocity_weight=vel)
    cfg["method"] = {"name": method, "options": {} if method == "dlt"
                     else {"iterations": 80, "bone_weight": 1.0}}
    Path(f"configs/ablation/{name}.yml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

variant("A1_dlt_generic",      "dlt",        1730.0, False)
variant("A2_dlt_calibrated",   "dlt",        "auto", False)
variant("B1_anat_generic",     "anatomical", 1730.0, False)
variant("B2_anat_calibrated",  "anatomical", "auto", False)
variant("C1_refine_accel",     "anatomical", "auto", True, vel=0.0)
variant("C2_refine_accel_vel", "anatomical", "auto", True, vel=0.03)
```

Sau đó cập nhật `DEFAULT_CONFIGS` trong `scripts/run_ablation.py`.

**Lưu ý quan trọng.** Với `dlt`, cột "Selected_Method" và "Baseline_DLT" là cùng một thứ (delta = 0%). Khi báo cáo, hãy báo **số tuyệt đối từng config**, đừng báo `Delta_vs_DLT` của từng config riêng.

---

<a id="3"></a>
## 3. Báo cáo metric trung thực

**Vấn đề.** `mpjpe()` mặc định gọi `rigid_mpjpe`, tức căn xoay + tịnh tiến về GT cho **từng frame**. Số này không phải MPJPE thuần, và nó triệt tiêu sai số xoay từng frame. Bảng README gọi nó là "MPJPE".

**Cách sửa.**

1. Đổi tên cho rõ nghĩa và giữ tương thích:

```python
# geometry.py
def mpjpe(prediction, ground_truth, align_rotation=True):
    """DEPRECATED alias. Dùng rigid_mpjpe() hoặc raw_mpjpe() để rõ nghĩa."""
    return rigid_mpjpe(prediction, ground_truth) if align_rotation else raw_mpjpe(prediction, ground_truth)
```

2. Trong `pipeline.py`, ghi thêm `raw_mpjpe` vào mỗi kết quả frame (hàm `_evaluate_methods` và `_dlt_synced_baseline`):

```python
from .algorithms.geometry import raw_mpjpe, rigid_mpjpe
...
results[name] = {
    "recon_3d": reconstruction,
    "mpjpe": rigid_mpjpe(reconstruction, ground_truth),   # đổi tên cột sang Rigid_MPJPE khi xuất
    "raw_mpjpe": raw_mpjpe(reconstruction, ground_truth),
    "pa_mpjpe": pa_mpjpe(reconstruction, ground_truth),
    ...
}
```

3. **Metric cấp chuỗi** (căn một lần cho cả chuỗi, trung thực hơn khi đánh giá độ ổn định theo thời gian). `sequence_diagnosis` đã có, nhưng mới chỉ `print`. Hãy ghi ra CSV, cho cả phương pháp chính và DLT:

```python
# io/reporting.py
from ..algorithms.geometry import sequence_diagnosis

def sequence_metrics_table(results):
    groups = defaultdict(list)
    for r in results:
        dlt = (r.get("all_methods", {}).get("DLT (baseline)") or {}).get("recon_3d")
        if dlt is None or r.get("gt_3d") is None:
            continue
        key = (r["subject"], r["motion"], r["cam_a"], r["cam_b"])
        groups[key].append((r["recon_3d"], dlt, r["gt_3d"]))
    rows = []
    for (subject, motion, cam_a, cam_b), items in groups.items():
        pred, dlt, gt = (np.stack(x) for x in zip(*items))
        m, d = sequence_diagnosis(pred, gt), sequence_diagnosis(dlt, gt)
        rows.append({
            "Subject": subject, "Motion": motion, "Cam_A": cam_a, "Cam_B": cam_b,
            "Frames": len(items),
            "Method_SeqRot_MPJPE": m["mpjpe_seq_rot_mm"], "DLT_SeqRot_MPJPE": d["mpjpe_seq_rot_mm"],
            "Method_SeqSim_MPJPE": m["mpjpe_seq_sim_mm"], "DLT_SeqSim_MPJPE": d["mpjpe_seq_sim_mm"],
            "Method_Scale": m["scale_ratio"], "Method_RotAngle_deg": m["angle_deg"],
        })
    return pd.DataFrame(rows)
```

Gọi trong `CsvResultReporter.finish()` và lưu thành `<stem>_sequence_metrics.csv`.

4. **README phải có đủ 4 cột**: `Raw`, `Rigid (per-frame)`, `Seq-Rot`, `PA`. Chỉ khi cả 4 cùng cải thiện mới nên nói "tốt hơn".

**Kiểm chứng.** Với DLT trên S1, `Raw ≥ Rigid ≥ PA` phải đúng cho mọi frame (assert trong test).

---

<a id="4"></a>
## 4. Làm mượt thời gian theo từng cặp frame liên tiếp

**Vấn đề.** Trong `refinement.py`, các thành phần gia tốc/vận tốc chỉ bật khi **toàn bộ** chuỗi liên tục (`np.all(np.diff(frame_nums) == 1)`). `_sample_key_frames` chỉ chọn cửa sổ liên tục theo *danh sách key frame*, và `_frame_inputs` còn loại các frame có `dynamic_score > 155`. Kết quả: chỉ cần thủng 1 frame là toàn bộ làm mượt tắt, mà không có cảnh báo. Tương tự, `prev_recon_3d` trong `_process_motion` được truyền qua cả những chỗ nhảy frame.

**Cách sửa A: log tình trạng liên tục** (làm trước để biết mức độ vấn đề).

```python
# refinement.py -> refine_results, ngay sau items.sort(...)
frames = np.array([r["frame"] for _, r in items])
n_gaps = int(np.sum(np.diff(frames) != 1)) if len(frames) > 1 else 0
print(f"[refine] {subject}/{motion} frames={len(frames)} gaps={n_gaps}")
```

**Cách sửa B: mặt nạ theo cặp liên tiếp** (thay cho điều kiện tất cả-hoặc-không).

```python
# refinement.py -> _optimize_sequence_torch
frame_nums = np.array([res["frame"] for _, res in items])
gap_ok = torch.as_tensor(np.diff(frame_nums) == 1, dtype=torch.float32, device=device)  # (T-1,)

def closure():
    ...
    centered = poses - poses[:, :1]
    if len(frame_nums) >= 3 and (accel_w := weights.get("acceleration", 0.0)) > 0:
        acc_ok = gap_ok[:-1] * gap_ok[1:]                       # cần cả hai khoảng liên tiếp
        accel = (centered[:-2] - 2.0 * centered[1:-1] + centered[2:]) * acc_ok[:, None, None]
        residuals.append((accel_w * accel).reshape(-1))
    if len(frame_nums) >= 2 and weights.get("velocity", 0.0) > 0:
        vel = (centered[1:] - centered[:-1]) * gap_ok[:, None, None]
        residuals.append((weights["velocity"] * vel).reshape(-1))
    ...
```

**Reset prior thời gian khi nhảy frame:**

```python
# pipeline.py -> _process_motion
prev_recon_3d, prev_frame = None, None
for frame in context["frames"]:
    inputs = _frame_inputs(...)
    if inputs is None:
        continue
    if prev_frame is not None and frame != prev_frame + 1:
        prev_recon_3d = None
    ...
    prev_frame = frame
```

**Kiểm chứng.** Sau khi sửa, chạy lại bước C1/C2. Nếu lợi ích của vận tốc (trước đây −1.10 mm) biến mất hoặc đổi dấu, đó là thông tin quan trọng: lợi ích cũ có thể đến từ thành phần khác.

---

<a id="5"></a>
## 5. Mốc tham chiếu camera và thống nhất `f_scale`

**Vấn đề.**
- P1, P2 được phục hồi từ ma trận F của chính các pose 2D, với intrinsics giả định (`1920×1088`, điểm chính ở giữa, `f = f_scale × 1920`) và thang mét lấy từ bone prior theo chiều cao, dùng chỉ 5 frame mẫu. Sai số của bước này có thể lớn hơn nhiều so với vài mm mà bước tối ưu hóa cải thiện.
- `uncalibrated_sync_score` gọi `uncalibrated_triangulation` mà không truyền `f_scale` (mặc định 1.2), trong khi phục hồi P1/P2 dùng `auto`.
- Độ phân giải 1920×1088 cố định, không đọc từ video/metadata.

**Cách sửa.**

1. **Truyền `f_scale` nhất quán.**

```python
# uncalibrated.py
def uncalibrated_sync_score(pts1, pts2, conf1, conf2, bone_lengths,
                            min_confidence=0.25, min_valid=8, f_scale=1.2):
    ...
    points_3d, p1, p2 = uncalibrated_triangulation(
        pts1, pts2, conf1, conf2, bone_lengths, f_scale=f_scale)
```

Trong `synchronization.estimate_pair_time_offset`, hàm `score(delta)` nên truyền cùng `f_scale`.
Chế độ `auto` tốn kém (11 ứng viên × mỗi lần gọi), nên **giải một lần cho mỗi cặp camera** (ở offset toàn cục thô), rồi cố định số đó cho các lần chấm điểm sau.

2. **Đọc độ phân giải từ metadata** nếu có (`*.json` thường chứa kích thước), và ghi cảnh báo khi phải dùng mặc định.

3. **Thêm mốc "camera chuẩn".** Nếu bộ dữ liệu cung cấp tham số camera (README cũ nhắc `cam_param.json`), thêm tùy chọn:

```yaml
pipeline:
  calibration: estimated   # estimated | reference
```

Khi `reference`, bỏ qua `uncalibrated_triangulation` và dùng P1/P2 chuẩn. Chạy A2/B2 với cả hai chế độ:

| | estimated | reference |
|---|---|---|
| DLT | ? | ? |
| Anatomical | ? | ? |

Hai kết quả này cho biết **bao nhiêu mm sai số đến từ hiệu chuẩn** và bao nhiêu từ triangulation. Nếu khoảng cách giữa hai cột lớn hơn nhiều so với khoảng cách giữa hai hàng, thì tối ưu Anatomical chỉ là hiệu ứng bậc hai.

4. **Log chẩn đoán hiệu chuẩn** cho mỗi cặp: `f_scale` chọn được, trung vị Sampson, số joint hợp lệ, hệ số thang mét. Ghi vào CSV `all_camera_pairs`.

---

<a id="6"></a>
## 6. Cho `anatomical` có `lr` cấu hình được và quét thử

**Vấn đề.** `triangulate_anatomical` dùng `Adam(lr=0.1)` (đơn vị mm) và 80 vòng. Mỗi bước Adam dịch cỡ `lr`, nên mỗi tọa độ chỉ dịch khoảng vài mm đến vài chục mm so với DLT, rất khó cho ràng buộc xương phát huy. Trong khi `physics_refine` mặc định `lr=1.0`. Hai phương pháp không so được công bằng.

**Cách sửa.**

```python
# geometry.py
def triangulate_anatomical(p1, p2, points1, points2, confidence1, confidence2, *,
                           bone_lengths=None, bone_weight=1.0, iterations=80, lr=0.1, **_kwargs):
    ...
    optimizer = torch.optim.Adam([points_3d], lr=lr)
```

```python
# pipeline.py -> _selected_triangulation
"anatomical": {"iterations", "bone_weight", "lr"},
```

(`lr` đã nằm trong danh sách kiểm tra của `_validate_method_options`.)

Quét thử trên **tập dev** (xem mục 7): `lr ∈ {0.1, 0.5, 1, 2, 5}` × `iterations ∈ {80, 200}`. Ghi lại cả thời gian chạy. Khi chọn xong, đóng băng giá trị trước khi chạy tập test.

---

<a id="7"></a>
## 7. Thống kê, mở rộng subject, tách dev/test

**Vấn đề.** Chỉ S1, không có khoảng tin cậy; nhiều ngưỡng (155.0, 7.5, các trọng số refinement) dường như được chỉnh trên đúng tập báo cáo.

**Cách sửa.**

1. **Tách dev/test theo motion**, cố định bằng hash để tái lập:

```python
import hashlib
def split_of(subject, motion):
    h = int(hashlib.md5(f"{subject}/{motion}".encode()).hexdigest(), 16)
    return "dev" if h % 2 == 0 else "test"
```

Mọi quyết định chỉnh tham số (radius, `lr`, trọng số refine, các ngưỡng) chỉ dựa trên **dev**. Số báo cáo trong README chỉ lấy từ **test**. Ghi rõ danh sách tham số đã đóng băng và ngày đóng băng.

2. **Chạy S1, S2, S3** (`python scripts/run_ablation.py` không kèm `-S1`), báo cáo từng subject.

3. **CI theo motion** (frame trong cùng motion tương quan mạnh, nên không bootstrap theo frame):

```python
# scripts/run_ablation.py (thêm vào)
def bootstrap_ci(deltas, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(deltas, dtype=float)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), np.percentile(means, [2.5, 97.5]).round(2).tolist()

# df: bảng từng frame của một config
per_motion = (df.assign(d=df["Baseline_DLT_MPJPE"] - df["Selected_Method_MPJPE"])
                .groupby(["Subject", "Motion"])["d"].mean())
mean_gain, ci = bootstrap_ci(per_motion.values)
```

Báo cáo: `mean_gain [CI 95%]` và số motion. Một cải thiện mà CI chứa 0 thì không được gọi là cải thiện.

4. Báo cáo số frame và số motion ngay trong bảng README. Cũng báo cáo **trên tập frame không lọc** (không áp `max_dynamic_sync_score`), vì điểm lọc này chứa số hạng lỗi độ dài xương, dễ làm lệch tập đánh giá theo hướng có lợi cho phương pháp dùng bone prior.

5. Ghi rõ giả định: bảng tỉ lệ xương hiện là `MALE_H36M_BONE_RATIOS`, trong khi S1 là nữ (theo docs). Cân nhắc thêm bảng tỉ lệ theo giới và chạy thử.

---

<a id="8"></a>
## 8. Dọn lỗi nhỏ trong code

| # | Vấn đề | Sửa |
|---|--------|-----|
| a | `select_best_pair` luôn ghi `outputs/all_camera_pairs.csv` (theo CWD, chế độ append, không qua config) | Truyền `record_all_csv=None` mặc định trong `_process_motion`, hoặc thêm `output.pairs_csv` vào config và `OutputSettings`; xóa file cũ khi `--overwrite` |
| b | `try/except TypeError` trong `_evaluate_methods` có thể che lỗi thật và chạy lại lần hai | Dùng `inspect.signature` (xem bên dưới) |
| c | Tham số DST còn sót: `belief_*`, `weights_dst`, `weights_cam1/2`, `is_outlier` của `physics_refine` không bao giờ được truyền | Xóa hoặc ghi chú rõ "chưa nối"; đổi tên cột `Belief_*` thành `Conf_A/Conf_B/Stereo_Conf` |
| d | Nhãn `"Anatomical (SOTA)"` | Đổi thành `"Anatomical"`; chỉ ghi SOTA nếu có so sánh với phương pháp công bố |
| e | `refine_results` im lặng trả về chuỗi gốc khi `max_drift` bị vượt, nhưng vẫn gắn nhãn "+ SequenceRefine" | Log cảnh báo và đánh dấu `refine_applied=False` trong kết quả |
| f | `triangulate_ransac` không phải RANSAC | Đổi tên (`reweighted_dlt`) hoặc cài RANSAC thật |
| g | Ba đoạn SVD lặp lại | Gom vào một hàm `_weighted_dlt(p1, p2, pt1, pt2, w1, w2)` |
| h | Cột `User_Info` ghi `getpass.getuser()` | Bỏ, hoặc băm, trước khi chia sẻ CSV |
| i | Comment kiểu `# ponytail:` rải khắp code | Đổi thành giải thích thật hoặc xóa |
| j | `_optimal_offset_path` tính lại `offset_change` trong mỗi vòng lặp | Đưa ra ngoài vòng lặp |

Gợi ý cho mục (b):

```python
import inspect
from functools import partial

def _accepts(fn, name):
    target = fn.func if isinstance(fn, partial) else fn
    params = inspect.signature(target).parameters.values()
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in params)

# _evaluate_methods
kwargs = {"prev_pose_3d": prev_pose_3d} if _accepts(triangulate, "prev_pose_3d") else {}
reconstruction = triangulate(p1, p2, left["kps_h36m"], right["kps_h36m"],
                             left["conf_h36m"], right["conf_h36m"], **kwargs)
```

---

<a id="9"></a>
## 9. Test thật sự (pytest)

**Vấn đề.** 13 "unit test" nằm trong `scratch/`, chạy bằng `print`, và phần lớn (3, 4, 7, 8, 11) chỉ kiểm tra shape/hữu hạn. README gọi là "bảo đảm tính đúng đắn".

**Cách sửa.**

1. Chuyển sang `tests/`, thêm vào `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

2. Giữ các test tốt (5: loss vector hóa; 9: dấu kinematic prior). Thêm test **kiểm tra hành vi**:

```python
# tests/test_sync_params.py
def test_local_radius_zero_is_honored(config_factory):
    cfg = config_factory(sync_local_radius=0)
    params = _fps_scaled_sync_params({"subject": "S1", "cameras": []}, cfg)
    assert params["local_radius"] == 0

# tests/test_refinement.py
def test_temporal_terms_are_masked_across_gaps():
    # chuỗi có 1 chỗ nhảy frame: số residual thời gian phải > 0 trên phần liên tục
    ...

# tests/test_metrics.py
def test_metric_ordering():
    pred, gt = make_noisy_pose_pair(seed=0)
    assert raw_mpjpe(pred, gt) >= rigid_mpjpe(pred, gt) >= pa_mpjpe(pred, gt)

# tests/test_triangulation.py
def test_dlt_recovers_exact_point_without_noise():
    ...  # dữ liệu tổng hợp, sai số < 1e-6
def test_anatomical_does_not_increase_bone_error_on_noisy_input():
    ...  # bone error sau < bone error trước
```

3. Chạy trong CI (GitHub Actions) để có huy hiệu trạng thái.

---

<a id="10"></a>
## 10. Sửa README và docs

Danh sách chỉnh sửa cụ thể:

- [ ] Sơ đồ ghi `PA-MPJPE ~50.87` nhưng bảng ghi `50.40`: thống nhất số.
- [ ] Sơ đồ ghi `radius=0` cho DP, nhưng `dynamic_local_radius` thực tế là 5: sửa lại cho khớp.
- [ ] Gọi "DTW" nhưng thuật toán là Viterbi/DP có phạt thay đổi offset: dùng tên đúng.
- [ ] "YOLOv8" trong sơ đồ so với tên repo/dữ liệu `yolo26x`: thống nhất.
- [ ] Bỏ câu "đảm bảo sai số giảm đơn điệu" trong `Huong_Dan_Ablation.md`.
- [ ] Gọi "MPJPE" thì phải nói rõ là rigid từng frame; thêm cột Raw và Seq-Rot (mục 3).
- [ ] Ghi rõ **số frame, số motion, subject nào** cho mỗi bảng.
- [ ] Mục "Bài học" phải chạy lại sau khi sửa mục 1, 4 và đánh dấu "đã xác nhận / bị bác bỏ".
- [ ] Ghi rõ giả định: dùng chiều cao thật của vận động viên (thông tin ngoài ảnh), và DLT baseline cũng hưởng thang mét từ chiều cao này.
- [ ] Thay link `file:///D:/...` bằng đường dẫn tương đối trong docs.
- [ ] Thêm phần "Hạn chế" (calibration ước lượng, quy mô nhỏ, bảng xương nam).
- [ ] Thêm LICENSE, mô tả repo, trích dẫn bộ dữ liệu AthletePose3D.
- [ ] Đổi "13 unit tests bảo đảm tính đúng đắn" thành mô tả đúng sau khi hoàn thành mục 9.

---

<a id="11"></a>
## 11. Checklist nghiệm thu

Thứ tự thực hiện đề xuất và điều kiện "xong":

| Bước | Việc | Xong khi |
|------|------|----------|
| 1 | Mục 1 (radius) | Test `local_radius == 0` qua; có bảng radius 0/1/2 |
| 2 | Mục 3 (metric) | CSV có Raw, Rigid, Seq-Rot, PA; test thứ tự metric qua |
| 3 | Mục 4 (làm mượt) | Log `gaps` cho mọi motion; mặt nạ theo cặp frame |
| 4 | Mục 2 (ablation 2×2) | 6 config sinh bằng script; không còn câu "đơn điệu" |
| 5 | Mục 6 (`lr`) | Có bảng quét `lr` trên dev, giá trị được đóng băng |
| 6 | Mục 5 (calibration) | Có bảng estimated vs reference |
| 7 | Mục 7 (thống kê) | S1–S3, CI theo motion, tập test tách riêng |
| 8 | Mục 8, 9, 10 | Dọn lỗi, test pytest chạy trong CI, README khớp với số liệu |

**Nguyên tắc chung khi báo cáo lại:**

1. Mỗi kết luận phải nêu: số motion, số frame, subject, metric nào, và CI.
2. Mọi thay đổi tham số sau khi nhìn kết quả test phải được ghi lại như một lần chỉnh trên test (không coi là kết quả sạch).
3. Nếu sau khi sửa mà cải thiện nhỏ hơn CI, hãy báo "chưa kết luận được", đó là kết quả hợp lệ.
