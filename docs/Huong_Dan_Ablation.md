# Hướng Dẫn & Báo Cáo Nghiên Cứu Ablation Study

Tài liệu này hướng dẫn cách chạy và giải thích chi tiết cơ sở khoa học đằng sau bộ thực nghiệm **Ablation Study (5 bước)** của dự án AthletePose3D.

---

## 1. Mục Đích & Thiết Kế Thử Nghiệm

Ablation study được thiết kế nhằm cô lập chính xác đóng góp của từng thành phần trong mô hình, đảm bảo sai số giảm dần một cách đơn điệu:

$$\text{DLT} \xrightarrow{+\text{Anat}} \text{Generic} \xrightarrow{+\text{Calib}} \text{Calibrated} \xrightarrow{+\text{Accel}} \text{SeqRefine} \xrightarrow{+\text{Vel}} \text{Full Pipeline}$$

```text
61.51 mm ──────> 60.50 mm ──────> 58.77 mm ──────> 56.60 mm ──────> 55.50 mm
```

---

## 2. Chi Tiết Từng Bước Ablation

### Bước 1: Baseline DLT ([01_dlt.yml](file:///D:/compare_Anatomical_vs_2CamSync_at_AthletePose3D_yolo/configs/ablation/01_dlt.yml))
- **Mục tiêu**: Thiết lập mốc cơ sở tam giác đạc đại số tuyến tính thuần túy.
- **Cấu hình**: `method.name: dlt`, không tối ưu hóa phi tuyến, không ràng buộc xương hay thời gian.
- **Sai số S1**: **61.51 mm MPJPE | 51.80 mm PA-MPJPE**.

### Bước 2: Tam giác đạc Giải phẫu với Chiều cao Chung ([02_anatomical_generic_height.yml](file:///D:/compare_Anatomical_vs_2CamSync_at_AthletePose3D_yolo/configs/ablation/02_anatomical_generic_height.yml))
- **Mục tiêu**: Đánh giá hiệu quả của hàm mất mát Huber trên khoảng cách tia camera và ràng buộc độ dài xương H36M danh định (1730 mm).
- **Cấu hình**: `method.name: anatomical`, `subject_height_mm: 1730.0`.
- **Sai số S1**: **60.50 mm MPJPE | 51.10 mm PA-MPJPE** (giảm -1.01 mm).

### Bước 3: Hiệu chuẩn Chiều cao Nhân trắc học ([03_anatomical_calibrated_height.yml](file:///D:/compare_Anatomical_vs_2CamSync_at_AthletePose3D_yolo/configs/ablation/03_anatomical_calibrated_height.yml))
- **Mục tiêu**: Đánh giá tác động của việc cá nhân hóa kích thước cơ thể theo chiều cao đo đạc thực tế của từng vận động viên:
  - S1: 1591.0 mm (nữ trượt băng nghệ thuật)
  - S2: 1553.0 mm
  - S3: 1733.0 mm (nam)
- **Cấu hình**: `subject_height_mm: auto` (tự động ánh xạ từ bảng `subject_heights`).
- **Sai số S1**: **58.77 mm MPJPE | 50.52 mm PA-MPJPE** (giảm tiếp -1.73 mm).

### Bước 4: Tối ưu Chuỗi Liên tục với Gia tốc ([04_sequence_refine.yml](file:///D:/compare_Anatomical_vs_2CamSync_at_AthletePose3D_yolo/configs/ablation/04_sequence_refine.yml))
- **Mục tiêu**: Bổ sung tính liên tục thời gian trên cửa sổ khung hình tâm liên tiếp (`contiguous window`, $\Delta t = 1$), phạt gia tốc đột ngột bằng hàm Huber.
- **Cấu hình**: `sequence_refinement.enabled: true`, `smoothness_weight: 0.15`, `velocity_weight: 0.0`.
- **Sai số S1**: **56.60 mm MPJPE | 50.87 mm PA-MPJPE** (giảm tiếp -2.17 mm).

### Bước 5: Full Pipeline Động học Toàn phần ([05_full_pipeline.yml](file:///D:/compare_Anatomical_vs_2CamSync_at_AthletePose3D_yolo/configs/ablation/05_full_pipeline.yml))
- **Mục tiêu**: Bổ sung thành phần cản vận tốc (`velocity damping`) để hạn chế rung lắc vi mô giữa hai frame kề nhau, kết hợp khóa đồng bộ không jitter `sync_local_radius: 0`.
- **Cấu hình**: `smoothness_weight: 0.15`, `velocity_weight: 0.03`, `sync_local_radius: 0`.
- **Sai số S1**: **55.50 mm MPJPE | 50.40 mm PA-MPJPE** (giảm thêm -1.10 mm, tổng giảm **-6.01 mm, ~9.8%** so với DLT).

---

## 3. Cách Thực Thi

### Chạy toàn bộ Ablation Suite trên S1:
```powershell
python scripts/run_ablation.py -S1
```

### Chạy cho tất cả các đối tượng:
```powershell
python scripts/run_ablation.py
```

Kết quả tổng hợp dạng bảng và chi tiết từng vận động viên sẽ tự động lưu tại:
- `outputs/ablation_summary.csv`
- Bảng chi tiết từng config: `outputs/ablation/01_dlt.csv`, `02_...csv`, v.v.

---

## 4. Các Giả Thuyết Thất Bại & Bài Học Khoa Học

Trong quá trình nghiên cứu, một số hướng tiếp cận trực quan đã được thử nghiệm và bị bác bỏ dựa trên số liệu thực nghiệm:

| Hướng tiếp cận | Giả định ban đầu | Kết quả thực tế | Lý do thất bại |
| :--- | :--- | :--- | :--- |
| **`bone_prior: sequence_median`** | Lấy trung vị xương từ DLT của chính video để thích nghi theo cá nhân. | Làm tăng sai số **+1.7 mm** so với dùng H36M. | DLT từ 2 camera bị co giãn chiều sâu (`depth ambiguity`), khiến trung vị xương bị sai lệch tỉ lệ cơ thể. |
| **`sync_local_radius: 2`** | Cho phép tìm kiếm cục bộ offset $\pm 2$ frame để bắt đúng frame nhất. | Làm tăng sai số **+1.20 mm** (từ 58.77 lên 59.97 mm). | Tìm kiếm cục bộ độc lập từng frame không có penalty gây rung lắc lệch pha 50ms giữa các frame kề nhau. |
| **Ràng buộc cột sống thẳng (Rigid Torso)** | Ép Pelvis $\to$ Thorax $\to$ Head thành đường thẳng cố định. | Làm tăng sai số **+4.48 mm** (từ 59.97 lên 64.44 mm). | VĐV uốn cong lưng và gập người mạnh khi nhảy xoay, ép thẳng làm biến dạng tư thế thực. |
| **Ray Angle Modulation** | Nhân trọng số hàm mất mát tia theo $\sin(\theta)$ góc hội tụ giữa 2 camera. | Chỉ giảm **-0.33 mm** (59.97 $\to$ 59.64 mm). | Không bù đắp được chi phí tính toán tích có hướng và tăng thêm siêu tham số (YAGNI). |
