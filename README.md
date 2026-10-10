# AthletePose3D: Local Two-Camera 3D Pose Benchmark

Hệ thống phục dựng tư thế người 3D từ hệ thống 2 camera đồng bộ động trên tập dữ liệu thể thao tốc độ cao **AthletePose3D**. Toàn bộ pipeline chạy hoàn toàn local, nhẹ, không phụ thuộc cloud hay mô hình học sâu 3D cồng kềnh, được thiết kế và kiểm thử chặt chẽ theo triết lý tối giản (*Ponytail*).

---

## 1. Kiến trúc Pipeline

```text
2D Keypoints (Cam A & Cam B)
            │
            ▼
[Dynamic Viterbi DP Synchronization]  --> Tìm đường đi offset tối ưu toàn cục (transition_weight=7.5, radius=0)
            │
            ▼
[Anatomical Triangulation]            --> Adam tia camera (Huber) + Chiều dài xương H36M (Calibrated Heights)
            │
            ▼
[Kinematic Sequence Refine]           --> L-BFGS chuỗi liên tục (Gia tốc + Vận tốc cản dao động có mặt nạ gap)
            │
            ▼
   Reconstruction 3D (Đánh giá đa chiều: Raw, Rigid-aligned, Seq-Rot, PA-MPJPE)
```

### Các thành phần cốt lõi:
1. **Dynamic DP Synchronization**: Ước lượng độ lệch pha thời gian giữa 2 camera bằng quy hoạch động (Viterbi DP) trên ma trận chi phí biểu kiến. Đặt `sync_local_radius: 0` để khóa đường đi DP liên tục, triệt tiêu hiện tượng rung lắc offset giữa các frame kế tiếp.
2. **Anatomical Triangulation**: Khởi tạo từ DLT, sau đó tối ưu hóa tọa độ 3D bằng Adam với hàm mất mát Huber trên khoảng cách tia camera (`_camera_rays`) và độ dài xương chuẩn H36M theo chiều cao của từng VĐV (`S1: 1591mm, S2: 1553mm, S3: 1733mm`). Tham số `lr` cấu hình được (mặc định 0.1).
3. **Kinematic Sequence Refinement**: Tối ưu hóa chuỗi cửa sổ khung hình liên tục (`contiguous window`) bằng L-BFGS với mặt nạ khoảng trống (`gap masking`):
   - Ràng buộc gia tốc: $\Delta^2 x = x_{t-1} - 2x_t + x_{t+1}$ (`smoothness_weight: 0.15`).
   - Ràng buộc vận tốc: $\Delta x = x_t - x_{t-1}$ (`velocity_weight: 0.03`).
   - Ràng buộc đối xứng xương: $\Delta L_{sym} = |L_{left} - L_{right}|$ (`symmetry_weight: 0.12`).

---

## 2. Thiết Kế Thực Nghiệm Ablation (2×2 Matrix)

Để loại bỏ hoàn toàn các yếu tố gây nhiễu (confounding factors), bộ thực nghiệm ablation được thiết kế theo ma trận 2×2 phân tách rõ rệt hiệu ứng phương pháp khỏi hiệu ứng nhân trắc học chiều cao:

| ID | Cấu hình | Phương pháp | Chiều cao | Sequence Refine | Mục đích cô lập biến |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **A1** | [A1_dlt_generic.yml](configs/ablation/A1_dlt_generic.yml) | `dlt` | 1730 mm (generic) | Tắt | Baseline DLT thuần, scale danh định |
| **A2** | [A2_dlt_calibrated.yml](configs/ablation/A2_dlt_calibrated.yml) | `dlt` | Hiệu chuẩn (`auto`) | Tắt | Baseline DLT, scale nhân trắc học |
| **B1** | [B1_anat_generic.yml](configs/ablation/B1_anat_generic.yml) | `anatomical` | 1730 mm (generic) | Tắt | Hiệu ứng Huber ray khi chưa có scale thật |
| **B2** | [B2_anat_calibrated.yml](configs/ablation/B2_anat_calibrated.yml) | `anatomical` | Hiệu chuẩn (`auto`) | Tắt | Hiệu ứng Huber ray + prior xương thật |
| **C1** | [C1_refine_accel.yml](configs/ablation/C1_refine_accel.yml) | `anatomical` | Hiệu chuẩn (`auto`) | Bật (Gia tốc) | Hiệu ứng làm mượt gia tốc thời gian |
| **C2** | [C2_refine_accel_vel.yml](configs/ablation/C2_refine_accel_vel.yml) | `anatomical` | Hiệu chuẩn (`auto`) | Bật (Gia tốc + Vận tốc) | Full Pipeline tối ưu hoàn chỉnh |

- **Hiệu ứng phương pháp thuần túy**: So sánh `A2` $\to$ `B2` (giữ nguyên chiều cao hiệu chuẩn).
- **Hiệu ứng hiệu chuẩn chiều cao**: So sánh `A1` $\to$ `A2` và `B1` $\to$ `B2`.
- **Hiệu ứng tối ưu chuỗi liên tục**: So sánh `B2` $\to$ `C1` và `C1` $\to$ `C2`.

---

## 3. Cài đặt và Sử dụng

### Cài đặt môi trường
```powershell
python -m pip install -e .
python -m pip install -e ".[dev]"
```

### Chạy Benchmark mặc định (Cấu hình tối ưu Full Pipeline)
```powershell
# Chạy toàn bộ test set
python -m athlete_pose3d.main --overwrite

# Chạy riêng từng vận động viên (ví dụ S1)
python -m athlete_pose3d.main -S1 --overwrite
```

### Chạy bộ Ablation Suite hoàn chỉnh
```powershell
# Chạy toàn bộ 6 bước ablation và xuất bảng tóm tắt kèm khoảng tin cậy 95% Bootstrap CI
python scripts/run_ablation.py -S1
```
Kết quả tổng hợp và chẩn đoán cấp chuỗi sẽ được ghi tự động vào:
- `outputs/ablation_summary.csv`
- `outputs/ablation/<config>_sequence_metrics.csv`

### Chạy kiểm thử tự động
```powershell
pytest
```

---

## 4. Cấu trúc Thư mục

```text
.
├── configs/
│   ├── default.yml                     # Cấu hình tối ưu mặc định
│   └── ablation/                       # Bộ 6 cấu hình ma trận 2x2 chuẩn hóa
│       ├── A1_dlt_generic.yml
│       ├── A2_dlt_calibrated.yml
│       ├── B1_anat_generic.yml
│       ├── B2_anat_calibrated.yml
│       ├── C1_refine_accel.yml
│       └── C2_refine_accel_vel.yml
├── docs/                               # Tài liệu thiết kế kỹ thuật và báo cáo
│   ├── Kien_Truc_Va_Giai_Phap.md
│   ├── Huong_Dan_Ablation.md
│   └── HUONG_DAN_SUA_DOI.md
├── scripts/
│   ├── make_ablation_configs.py        # Script sinh tự động 6 file YAML ablation
│   └── run_ablation.py                 # Script chạy tự động toàn bộ ablation suite
├── src/athlete_pose3d/
│   ├── main.py                         # CLI entry point
│   ├── pipeline.py                     # Điều phối luồng xử lý frame, window, chunk
│   ├── settings.py                     # Schema cấu hình YAML và validation
│   ├── algorithms/
│   │   ├── geometry.py                 # DLT, Anatomical, Huber loss, Camera rays
│   │   ├── refinement.py               # L-BFGS Kinematic Sequence Refinement
│   │   ├── synchronization.py          # Dynamic DP (Viterbi) multi-camera sync
│   │   └── uncalibrated.py             # Khôi phục Fundamental/Essential & Intrinsics
│   └── io/
│       ├── data.py                     # Đọc pose 2D, ground-truth, LRU cache
│       └── reporting.py                # Xuất báo cáo Raw/Rigid/Seq-Rot/PA MPJPE ra CSV
└── tests/                              # Bộ kiểm thử chuẩn pytest
    ├── test_geometry.py
    ├── test_metrics.py
    ├── test_refinement.py
    └── test_sync_params.py
```

---

## 5. Các Lưu Ý Kỹ Thuật Quan Trọng

1. **Khóa `sync_local_radius: 0`**: Đã sửa kiểm tra `max(0, ...)` trong code để đảm bảo giá trị 0 không bị ép thành 1. Quy hoạch động Viterbi đã tìm quỹ đạo offset mượt mà; không tìm kiếm cục bộ frame-by-frame không có ràng buộc chuyển trạng thái để tránh rung giật pha.
2. **Báo cáo trung thực 4 bậc Metric**: Hệ thống ghi nhận đầy đủ 4 thước đo:
   - `Raw MPJPE`: Căn gốc khớp háng (Pelvis), không xoay.
   - `Rigid MPJPE`: Căn xoay $SO(3)$ và tịnh tiến tối ưu từng khung hình (giữ nguyên thang đo metric).
   - `Seq-Rot MPJPE`: Căn xoay và tịnh tiến một lần duy nhất cho toàn bộ chuỗi clip.
   - `PA-MPJPE`: Căn chỉnh Procrustes toàn phần (xoay, tịnh tiến và co giãn tỉ lệ).
3. **Mặt nạ khoảng trống (`Gap Masking`)**: Khi một số khung hình bị loại bỏ do độ tin cậy thấp, các số hạng động học gia tốc và vận tốc được áp dụng cục bộ trên các cặp/bộ ba khung hình thực sự liên tiếp ($\Delta t = 1$), thay vì tắt bỏ toàn bộ chuỗi.

---

## 6. Hạn Chế Nghiên Cứu (Limitations)

1. **Hiệu chuẩn xấp xỉ từ Pose 2D**: Ma trận ngoại tại $(R, t)$ và tiêu cự được ước lượng từ tương ứng khớp 2D (chưa sử dụng thông số camera tham chiếu chuẩn trong `cam_param.json`). Sai số từ bước hiệu chuẩn không căn chuẩn đóng góp phần đáng kể vào sai số tuyệt đối.
2. **Quy mô tập đánh giá**: Mặc dù pipeline hỗ trợ đầy đủ S1, S2, S3, các thử nghiệm phát triển nhanh chủ yếu tập trung trên S1 với các chuyển động xoay nhảy đặc trưng của trượt băng nghệ thuật.
3. **Mô hình nhân trắc học**: Tỉ lệ các đoạn chi hiện đang dựa trên mô hình danh định `MALE_H36M_BONE_RATIOS`, trong khi S1 là nữ vận động viên. Khác biệt hình thái học giới tính có thể tạo ra sai số nền nhỏ trong các ràng buộc chiều dài xương.

---

## 7. Trích Dẫn & Giấy Phép (Citation & License)

- **AthletePose3D Dataset**: Bộ dữ liệu thể thao đa góc nhìn AthletePose3D được sử dụng cho mục đích nghiên cứu học thuật.
- **Mã nguồn**: Phát hành theo giấy phép [MIT License](LICENSE).

