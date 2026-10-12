# AthletePose3D: Local Two-Camera 3D Pose Benchmark

Hệ thống phục dựng tư thế người 3D từ hệ thống 2 camera không đồng bộ phần cứng trên tập dữ liệu thể thao tốc độ cao **AthletePose3D**. Pipeline thực thi 100% local, tối ưu hóa toán học thuần túy (không phụ thuộc mạng nơ-ron 3D cồng kềnh hay cloud), được thiết kế theo triết lý tối giản (**Ponytail**: tối thiểu mã nguồn, tối đa hiệu quả tính toán).

---

## 1. Kiến Trúc Pipeline (End-to-End Pipeline)

```text
2D Keypoints (Cam A & Cam B)
            │
            ▼
[Dynamic Viterbi DP Synchronization]  --> Tìm quỹ đạo offset tối ưu (dynamic_radius=5, transition_weight=7.5, local_radius=0)
            │
            ▼
[Relative Camera Pose Recovery]       --> Ma trận cơ bản F, ma trận thiết yếu E, giải tỉ lệ metric từ chiều cao VĐV
            │
            ▼
[Anatomical Triangulation]            --> Khởi tạo DLT + Adam tối ưu Huber ray loss (delta=5mm) & H36M bone priors (delta=10mm)
            │
            ▼
[Kinematic Sequence Refinement]       --> L-BFGS chuỗi liên tục (Gia tốc + Vận tốc cản dao động có Gap Masking)
            │
            ▼
   3D Pose Reconstruction & Đánh giá 4 cấp Metric (Raw, Rigid, Seq-Rot, PA-MPJPE)
```

### Các thành phần cốt lõi:
1. **Dynamic Viterbi DP Synchronization** ([synchronization.py](src/athlete_pose3d/algorithms/synchronization.py)): Ước lượng độ trôi pha thời gian giữa 2 camera bằng quy hoạch động Viterbi trên ma trận chi phí biểu kiến. Đặt `dynamic_local_radius: 5`, `dynamic_transition_weight: 7.5` và khóa `sync_local_radius: 0` để triệt tiêu rung giật pha giữa các khung hình liên tiếp.
2. **Relative Camera Pose Recovery** ([uncalibrated.py](src/athlete_pose3d/algorithms/uncalibrated.py)): Ước lượng Fundamental matrix $F$ qua 8-điểm chuẩn hóa + RANSAC, phân rã Essential matrix $E$ thành $(R, t)$, và khôi phục tỷ lệ metric thực từ chiều cao vận động viên (`solve_metric_scale`).
3. **Anatomical Triangulation** ([geometry.py](src/athlete_pose3d/algorithms/geometry.py)): Khởi tạo từ DLT, tối ưu hóa phi tuyến bằng Adam với hàm mất mát Huber trên khoảng cách tia camera tiền tính toán (`_camera_rays`) và độ dài xương chuẩn H36M theo chiều cao thực của từng VĐV (`S1: 1591mm, S2: 1553mm, S3: 1733mm`).
4. **Kinematic Sequence Refinement** ([refinement.py](src/athlete_pose3d/algorithms/refinement.py)): Tối ưu hóa chuỗi cửa sổ khung hình liên tục (`contiguous window`) bằng L-BFGS với mặt nạ khoảng trống (`gap masking`):
   - Ràng buộc gia tốc: $\mathcal{L}_{accel} = \sum \mathcal{H}_\delta(X_{t-1} - 2X_t + X_{t+1}) \cdot \mathbb{I}(\Delta t_1 = 1 \land \Delta t_2 = 1)$ (`smoothness_weight: 0.15`).
   - Ràng buộc vận tốc: $\mathcal{L}_{vel} = \sum \mathcal{H}_\delta(X_t - X_{t-1}) \cdot \mathbb{I}(\Delta t = 1)$ (`velocity_weight: 0.03`).
   - Ràng buộc đối xứng xương: $\mathcal{L}_{sym} = \sum |\|X_L\| - \|X_R\||$ (`symmetry_weight: 0.12`).

---

## 2. Kết Quả Thực Nghiệm Chuẩn (Benchmark Results)

Kết quả thực nghiệm trên tập thử nghiệm vận động viên **S1** (15 clips chuyển động phức tạp: Axel, Flip, Combo jump; 30 khung hình/cặp; tổng cộng 450 khung hình):

### 2.1. So sánh Tổng thể 4 cấp Metric (mm)

| Thước đo (Metric) | Mô tả hình học | Baseline DLT | Full Pipeline | Cải thiện ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **Raw MPJPE** | Trừ gốc háng Pelvis, giữ nguyên hướng và thang đo camera | 660.99 mm | 661.88 mm | +0.89 mm (+0.1%) |
| **Rigid MPJPE** | Căn xoay $SO(3)$ và tịnh tiến tối ưu từng khung hình | **58.42 mm** | **53.15 mm** | **-5.27 mm (-9.01%)** |
| **Seq-Rot MPJPE** | Căn xoay $SO(3)$ một lần duy nhất cho toàn bộ chuỗi clip | 68.68 mm | 68.52 mm | -0.16 mm (-0.23%) |
| **PA-MPJPE** | Căn chỉnh Procrustes toàn phần (Xoay, tịnh tiến và co giãn $s$) | **48.71 mm** | **47.63 mm** | **-1.08 mm (-2.23%)** |

> [!NOTE]
> Định lý hình học luôn được bảo đảm trên mọi khung hình:  
> $$\text{Raw MPJPE} \ge \text{Rigid MPJPE} \ge \text{PA-MPJPE}$$

### 2.2. Phân rã Khớp Rõ (Clear) vs Khớp Suy diễn (Derived)

Do dữ liệu 2D đầu vào theo chuẩn COCO-17, các khớp thân giữa của chuẩn H36M (Pelvis, Spine, Thorax, Neck, Head) phải suy diễn hình học từ các khớp lân cận:
- **Khớp rõ quan sát trực tiếp (Clear joints - gối, cổ chân, khuỷu, cổ tay)**: PA-MPJPE đạt **28.95 mm**.
- **Khớp thân giữa suy diễn (Derived joints - pelvis, spine, thorax, head)**: PA-MPJPE đạt **78.75 mm**.

---

## 3. Thiết Kế Thực Nghiệm Ablation (2×2 Matrix)

Bộ thực nghiệm ablation phân tách rõ rệt hiệu ứng phương pháp khỏi hiệu ứng nhân trắc học chiều cao để loại bỏ hoàn toàn các biến gây nhiễu:

| ID | Cấu hình | Phương pháp | Chiều cao | Sequence Refine | Mục đích cô lập biến |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **A1** | [A1_dlt_generic.yml](configs/ablation/A1_dlt_generic.yml) | `dlt` | 1730 mm (generic) | Tắt | Baseline DLT thuần, scale danh định |
| **A2** | [A2_dlt_calibrated.yml](configs/ablation/A2_dlt_calibrated.yml) | `dlt` | Hiệu chuẩn (`auto`) | Tắt | Baseline DLT, scale nhân trắc học |
| **B1** | [B1_anat_generic.yml](configs/ablation/B1_anat_generic.yml) | `anatomical` | 1730 mm (generic) | Tắt | Hiệu ứng Huber ray khi scale danh định |
| **B2** | [B2_anat_calibrated.yml](configs/ablation/B2_anat_calibrated.yml) | `anatomical` | Hiệu chuẩn (`auto`) | Tắt | Hiệu ứng Huber ray + prior xương thật |
| **C1** | [C1_refine_accel.yml](configs/ablation/C1_refine_accel.yml) | `anatomical` | Hiệu chuẩn (`auto`) | Bật (Gia tốc) | Hiệu ứng làm mượt gia tốc thời gian |
| **C2** | [C2_refine_accel_vel.yml](configs/ablation/C2_refine_accel_vel.yml) | `anatomical` | Hiệu chuẩn (`auto`) | Bật (Gia tốc + Vận tốc) | Full Pipeline hoàn chỉnh |

- **Hiệu ứng phương pháp thuần túy**: So sánh `A2` $\to$ `B2` (giữ nguyên chiều cao hiệu chuẩn).
- **Hiệu ứng hiệu chuẩn chiều cao**: So sánh `A1` $\to$ `A2` và `B1` $\to$ `B2`.
- **Hiệu ứng tối ưu chuỗi liên tục**: So sánh `B2` $\to$ `C1` và `C1` $\to$ `C2`.

---

## 4. Cài Đặt và Sử Dụng (Installation & Usage)

### Cài đặt môi trường
Yêu cầu Python $\ge$ 3.10:
```powershell
python -m pip install -e .
python -m pip install -e ".[dev]"
```

### Chạy Benchmark mặc định (Full Pipeline)
```powershell
# Chạy trên vận động viên S1 (mặc định)
python -m athlete_pose3d.main -S1 --overwrite

# Chạy trên toàn bộ test set (S1, S2, S3)
python -m athlete_pose3d.main --overwrite

# Tùy chỉnh số lượng cặp clip và phương pháp
python -m athlete_pose3d.main -S1 --max-pairs 5 --method anatomical --overwrite
```

### Chạy bộ Ablation Suite hoàn chỉnh
```powershell
# Sinh lại 6 cấu hình ma trận chuẩn hóa nếu cần
python scripts/make_ablation_configs.py

# Chạy tự động toàn bộ 6 bước ablation kèm 95% Bootstrap CI
python scripts/run_ablation.py -S1
```
Kết quả tổng hợp được lưu tại:
- `outputs/ablation_summary.csv`
- `outputs/ablation/<config>_sequence_metrics.csv`

### Chạy bộ kiểm thử tự động
```powershell
pytest
```

---

## 5. Cấu Trúc Thư Mục (Repository Structure)

```text
.
├── configs/
│   ├── default.yml                     # Cấu hình tối ưu mặc định của pipeline
│   └── ablation/                       # Bộ 6 cấu hình YAML ma trận ablation 2x2
│       ├── A1_dlt_generic.yml
│       ├── A2_dlt_calibrated.yml
│       ├── B1_anat_generic.yml
│       ├── B2_anat_calibrated.yml
│       ├── C1_refine_accel.yml
│       └── C2_refine_accel_vel.yml
├── docs/                               # Tài liệu thiết kế chi tiết và báo cáo
│   ├── Kien_Truc_Va_Giai_Phap.md       # Cơ sở toán học và luồng xử lý
│   ├── Huong_Dan_Ablation.md           # Hướng dẫn chi tiết phân tích ablation
│   └── HUONG_DAN_SUA_DOI.md            # Sổ tay rà soát và kiểm chứng kỹ thuật
├── scripts/
│   ├── make_ablation_configs.py        # Script sinh 6 file cấu hình YAML ablation
│   └── run_ablation.py                 # Script điều phối chạy ablation & bootstrap CI
├── src/athlete_pose3d/
│   ├── main.py                         # CLI entry point và phân tích tham số
│   ├── pipeline.py                     # Điều phối frame, window, chunk và dynamic sync
│   ├── settings.py                     # Schema cấu hình YAML và ràng buộc hợp lệ
│   ├── algorithms/
│   │   ├── geometry.py                 # DLT, Adam Anatomical, Huber loss, 4 metrics
│   │   ├── refinement.py               # L-BFGS Kinematic Sequence Refinement & Gap Mask
│   │   ├── synchronization.py          # Dynamic Viterbi DP multi-camera synchronization
│   │   ├── uncalibrated.py             # Phân rã F, E và khôi phục metric scale
│   │   └── physics_refine.py           # Thử nghiệm mở rộng ràng buộc vật lý
│   └── io/
│       ├── data.py                     # Đọc pose 2D, ground truth, LRU cache an toàn
│       └── reporting.py                # Xuất báo cáo CSV đa chỉ số & thống kê hệ thống
└── tests/                              # Bộ kiểm thử đơn vị pytest
    ├── test_geometry.py                # Kiểm thử DLT chính xác, lr cấu hình được
    ├── test_metrics.py                 # Kiểm thử tính đúng đắn Raw >= Rigid >= PA MPJPE
    ├── test_refinement.py              # Kiểm thử tính liên tục và gap masking
    └── test_sync_params.py             # Kiểm thử bảo toàn sync_local_radius = 0
```

---

## 6. Điểm Nhấn Kỹ Thuật (Key Technical Highlights)

1. **Bảo toàn `sync_local_radius: 0`**: Đã sửa hàm `_fps_scaled_sync_params` (`max(0, ...)` thay vì ép `max(1, ...)`). Quỹ đạo offset được khóa theo kết quả Viterbi DP toàn cục mượt mà, loại bỏ hiện tượng nhảy offset giữa các khung hình liên tiếp.
2. **Mặt nạ khoảng trống (`Gap Masking`)**: Khi có khung hình bị che khuất hoặc loại bỏ, các số hạng vận tốc ($\Delta x$) và gia tốc ($\Delta^2 x$) chỉ kích hoạt trên các cặp/bộ ba khung hình thực sự liên tiếp ($\Delta t = 1$), bảo toàn tính đúng đắn động học.
3. **Precomputed Camera Rays & LRU Cache**: Tiền tính toán tia camera `_camera_rays` ngoài vòng lặp Adam và quản lý bộ nhớ đệm `PoseRepository` bằng `OrderedDict` LRU, tăng tốc độ xử lý hơn 40%.
4. **An toàn dữ liệu tuyệt đối**: Loại bỏ hoàn toàn `allow_pickle=True` trên toàn bộ module I/O trong `src/` nhằm ngăn chặn lỗ hổng thực thi mã độc qua file pickle.

---

## 7. Giả Định & Hạn Chế (Assumptions & Limitations)

1. **Ngoại tại Camera ước lượng từ Pose 2D**: Ma trận ngoại tại $(R, t)$ và tiêu cự được ước lượng từ tương ứng khớp 2D (chưa dùng thông số hình học chuẩn trong `cam_param.json`). Sai lệch góc camera ngoại tại là nguyên nhân khiến Raw MPJPE cao hơn đáng kể so với Rigid MPJPE.
2. **Tỉ lệ Xương Danh định**: Hệ thống sử dụng mô hình tỉ lệ `MALE_H36M_BONE_RATIOS`, trong khi đối tượng S1 là nữ vận động viên.
3. **Các khớp thân giữa suy diễn**: Do chuẩn COCO-17 không có Pelvis, Thorax, Neck, Head trực tiếp, sai số suy diễn hình học đóng góp phần lớn vào sai số tổng thể.

---

## 8. Bản Quyền & Giấy Phép (License & Citation)

- **Mã nguồn**: Phát hành theo giấy phép tự do [MIT License](LICENSE).
- **Bộ dữ liệu**: AthletePose3D dataset được sử dụng cho mục đích nghiên cứu học thuật.
