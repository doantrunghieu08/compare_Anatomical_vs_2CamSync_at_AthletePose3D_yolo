# AthletePose3D: Local Two-Camera 3D Pose Benchmark

Benchmark dựng tư thế người 3D từ hệ thống 2 camera đồng bộ động trên tập dữ liệu thể thao tốc độ cao **AthletePose3D**. Toàn bộ pipeline chạy hoàn toàn local, nhẹ, không phụ thuộc cloud hay thư viện cồng kềnh, tối ưu hóa theo triết lý tối giản (*Ponytail*).

---

## 1. Kiến trúc Pipeline

```text
2D YOLOv8 Keypoints (Cam A & Cam B)
            │
            ▼
[Dynamic DP Synchronization]  --> Đường đi offset mượt mà (DTW, transition_weight=7.5, radius=0)
            │
            ▼
[Anatomical Triangulation]    --> Tối ưu Adam tia camera (Huber) + Chiều dài xương H36M (Calibrated Heights)
            │
            ▼
[Kinematic Sequence Refine]   --> Tối ưu L-BFGS chuỗi liên tục (Gia tốc + Vận tốc cản dao động)
            │
            ▼
   Reconstruction 3D (MPJPE ~55.50 mm / PA-MPJPE ~50.87 mm)
```

### Các thành phần cốt lõi:
1. **Dynamic Synchronization**: Ước lượng độ lệch pha thời gian giữa 2 camera bằng quy hoạch động (DP) trên ma trận chi phí biểu kiến. Đặt `sync_local_radius: 0` để khóa đường đi DP liên tục, triệt tiêu hoàn toàn hiện tượng rung lắc offset giữa các frame kế tiếp.
2. **Anatomical Triangulation**: Khởi tạo từ DLT, sau đó tối ưu hóa tọa độ 3D bằng Adam với hàm mất mát Huber trên khoảng cách tia camera (`_camera_rays`) và độ dài xương chuẩn H36M theo chiều cao thực nghiệm của từng VĐV (`S1: 1591mm, S2: 1553mm, S3: 1733mm`).
3. **Kinematic Sequence Refinement**: Tối ưu hóa chuỗi cửa sổ khung hình liên tục (`contiguous window`) bằng L-BFGS:
   - Ràng buộc gia tốc: $\Delta^2 x = x_{t-1} - 2x_t + x_{t+1}$ (`smoothness_weight: 0.15`).
   - Ràng buộc vận tốc: $\Delta x = x_t - x_{t-1}$ (`velocity_weight: 0.03`).
   - Ràng buộc đối xứng xương: $\Delta L_{sym} = |L_{left} - L_{right}|$ (`symmetry_weight: 0.12`).

---

## 2. Kết quả Thực nghiệm trên S1 (Ablation Benchmark)

| Bước | Cấu hình | Yếu tố cô lập | MPJPE (mm) | PA-MPJPE (mm) | Ghi chú |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **01** | `01_dlt.yml` | Baseline DLT | 61.51 | 51.80 | Khởi tạo tam giác đạc thuần túy |
| **02** | `02_anatomical_generic_height.yml` | + Anatomical (Chiều cao 1730mm) | 60.50 | 51.10 | Thêm Huber ray + Prior xương H36M |
| **03** | `03_anatomical_calibrated_height.yml` | + Chiều cao hiệu chuẩn VĐV | 58.77 | 50.52 | S1=1591mm, S2=1553mm, S3=1733mm |
| **04** | `04_sequence_refine.yml` | + Tối ưu chuỗi gia tốc | 56.60 | 50.87 | Cửa sổ liên tục, smoothness=0.15 |
| **05** | `05_full_pipeline.yml` | + Ràng buộc vận tốc cản dao động | **55.50** | **50.40** | **Full Pipeline (Default)** |

---

## 3. Cài đặt và Sử dụng

### Cài đặt môi trường
```powershell
python -m pip install -e .
```

### Chạy Benchmark mặc định (Cấu hình tối ưu 05)
```powershell
# Chạy toàn bộ test set
python -m athlete_pose3d.main --overwrite

# Chạy riêng từng vận động viên (ví dụ S1)
python -m athlete_pose3d.main -S1 --overwrite
```

### Chạy bộ Ablation Study hoàn chỉnh
```powershell
# Chạy cả 5 bước ablation trên S1 và xuất bảng tóm tắt
python scripts/run_ablation.py -S1
```
Kết quả tổng hợp sẽ được ghi tự động vào `outputs/ablation_summary.csv`.

---

## 4. Cấu trúc Thư mục

```text
.
├── configs/
│   ├── default.yml                     # Cấu hình tối ưu mặc định (Full Pipeline)
│   └── ablation/                       # Bộ 5 bước ablation test chuẩn hóa
│       ├── 01_dlt.yml
│       ├── 02_anatomical_generic_height.yml
│       ├── 03_anatomical_calibrated_height.yml
│       ├── 04_sequence_refine.yml
│       └── 05_full_pipeline.yml
├── docs/                               # Tài liệu kiến trúc và hướng dẫn phân tích
│   ├── Kien_Truc_Va_Giai_Phap.md
│   └── Huong_Dan_Ablation.md
├── scripts/
│   └── run_ablation.py                 # Script chạy tự động toàn bộ ablation suite
├── src/athlete_pose3d/
│   ├── main.py                         # CLI entry point
│   ├── pipeline.py                     # Điều phối luồng xử lý frame, window, chunk
│   ├── settings.py                     # Quản lý & kiểm tra schema cấu hình YAML
│   ├── algorithms/
│   │   ├── geometry.py                 # DLT, Anatomical, Huber loss, Camera rays
│   │   ├── refinement.py               # L-BFGS Kinematic Sequence Refinement
│   │   └── synchronization.py          # Dynamic DP / DTW multi-camera sync
│   └── io/
│       ├── data.py                     # Đọc pose 2D, ground-truth, LRU cache
│       └── reporting.py                # Xuất báo cáo MPJPE / PA-MPJPE ra CSV
└── scratch/
    └── test_ponytail_fixes.py          # Bộ kiểm thử 13 unit tests bảo đảm tính đúng đắn
```

---

## 5. Các Bài học Thực nghiệm Quan trọng

1. **Khóa `sync_local_radius: 0`**: Quy hoạch động đã tính toán đường cong offset tối ưu. Tìm kiếm cục bộ frame-by-frame không có penalty sẽ làm rung lắc offset (±2 frame = 50ms), gây méo hình học và tăng sai số +1.20 mm.
2. **Ưu tiên `bone_prior: h36m`**: Không dùng trung vị xương từ DLT (`sequence_median`) vì DLT 2 camera bị co giãn chiều sâu (`depth ambiguity`), khiến bộ xương ước lượng bị lệch tỉ lệ cơ thể.
3. **Cửa sổ liên tục (`contiguous window`)**: Lấy cửa sổ khung hình liên tục tại tâm clip thay vì lấy mẫu thưa (`np.linspace`) để bảo toàn đạo hàm bậc một (vận tốc) và bậc hai (gia tốc).
4. **Tránh ép thẳng cột sống**: Cột sống vận động viên uốn cong tự nhiên khi thực hiện các động tác nhảy/tiếp đất phức tạp. Ép cột sống thẳng làm tăng sai số thêm +4.48 mm.
