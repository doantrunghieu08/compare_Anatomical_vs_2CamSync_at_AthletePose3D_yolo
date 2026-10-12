# Báo Cáo Xác Thực & Phân Tích Kỹ Thuật Dữ Liệu AthletePose3D Benchmark

**Ngày lập**: 12/10/2026  
**Phạm vi**: 3 đối tượng vận động viên (**S1, S2, S3**), 42 chuỗi chuyển động, 1.260 khung hình đánh giá.  
**Cơ sở đối chiếu**: Dữ liệu thực nghiệm kết hợp rà soát cấu trúc mã nguồn đồ thị (*code_review_graph*).

---

## 1. Tóm Tắt & Kết Quả Thực Nghiệm Định Lượng

### 1.1. Hiệu suất tổng thể (mm, đánh giá sau căn chỉnh Rigid per-frame)

| Đối tượng | Số Frame | Baseline DLT (mm) | Selected Method (mm) | Cải thiện ($\Delta$) | DLT PA (mm) | Method PA (mm) | Tỉ lệ Thắng (MPJPE) | Tỉ lệ Thắng (PA) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** (Trượt băng) | 450 | 58.42 | **53.15** | **-5.27 mm (-9.01%)** | 48.71 | **47.63** | **97.3%** (438/450) | 76.2% |
| **S2** (Chạy cự ly) | 420 | 59.50 | **54.54** | **-4.96 mm (-8.34%)** | 57.47 | **51.89** | **97.6%** (410/420) | 96.0% |
| **S3** (Điền kinh ném) | 390 | 61.29 | **56.08** | **-5.21 mm (-8.50%)** | 50.50 | **48.74** | **94.6%** (369/390) | 75.4% |
| **Toàn bộ** | **1.260** | **59.67** | **54.52** | **-5.15 mm (-8.63%)** | **52.47** | **49.40** | **96.6%** (1.217/1.260) | **82.5%** |

- Phương pháp đề xuất (`Anatomical + SequenceRefine`) cho sai số thấp hơn DLT đồng nhất từ **8.3% đến 9.0%** trên toàn bộ 3 vận động viên và toàn bộ 42 chuỗi động tác (không có chuỗi nào thua DLT khi xét tổng thể).
- Phân bố trung vị sai số của Method: **51.4 mm** (S1), **51.6 mm** (S2), **55.3 mm** (S3).

---

## 2. Giải Mã Cơ Sở Toán Học & Kiểm Chứng Mã Nguồn

### 2.1. Bản chất sai số thô lớn (~660 mm) và sự đồng nhất giữa DLT & Method
- **Cơ chế**: Pipeline là tam giác đạc không hiệu chuẩn ngoại tại ([uncalibrated_triangulation](../src/athlete_pose3d/algorithms/uncalibrated.py)). Hệ tọa độ 3D dựng lại được neo cố định vào quang tâm và hướng của **Camera 1** ($P_1 = K_1 [I \mid 0]$).
- **Nguyên nhân toán học**: Hàm `raw_mpjpe` ([geometry.py](../src/athlete_pose3d/algorithms/geometry.py#L519)) chỉ trừ gốc khớp háng Pelvis ($x - x_{pelvis}$), **hoàn toàn không xoay hệ quy chiếu Camera 1 về hệ quy chiếu thế giới (World MoCap GT)**.
- **Dữ liệu thực chứng**: Góc lệch xoay $SO(3)$ giữa Camera 1 và MoCap GT dao động từ **100° đến 150°** (cột `Method_RotAngle_deg` trong `results_sequence_metrics.csv`). Khi hai khung xương chuẩn bị xoay lệch nhau 100°–150°, khoảng cách Euclid trung bình giữa các khớp tất yếu rơi vào khoảng 600–750 mm. Do cả DLT và Method đều dùng chung hệ quy chiếu Camera 1, sai số thô của hai bên là tương đương.

### 2.2. Sai số hệ thống nhóm khớp thân giữa (Spine, Thorax, Head) và Pelvis
- **Bất đối xứng ngữ nghĩa khớp (Semantic Mismatch)**:
  - Dữ liệu 2D đầu vào theo chuẩn COCO-17 không có các khớp trục thân giữa. Hàm `coco_to_h36m_2d` ([data.py](../src/athlete_pose3d/io/data.py#L124)) phải nội suy/ngoại suy hình học:
    * $\text{Pelvis} = 0.5 \times (\text{R\_Hip} + \text{L\_Hip})$
    * $\text{Thorax} = 0.5 \times (\text{L\_Shoulder} + \text{R\_Shoulder})$
    * $\text{Spine} = 0.5 \times (\text{Pelvis} + \text{Thorax})$ (điểm giữa thân trên mặt phẳng 2D)
    * $\text{Head} = \text{Thorax} + 1.45 \times (\text{Nose} - \text{Thorax})$
  - Trong khi đó, Ground-Truth MoCap của AthletePose3D ([settings.py](../src/athlete_pose3d/settings.py)) gắn marker quang học trực tiếp trên bề mặt cơ thể: `T10`/`T8` (sau lưng), `Manubrium` (xương ức trước ngực), `Vertex` (đỉnh đầu).
  - Khi vận động viên gập người hoặc uốn lưng, điểm giữa thân hình học (nội suy) lệch xa marker đốt sống lưng `T10/T8` $\to$ **Gây ra sai số cố định 94–117 mm ở Spine trên cả 3 đối tượng**. Đây là giới hạn dữ liệu, không phải nhiễu giải thuật.
- **Hiện tượng Pelvis tăng sai số (36 $\to$ 44 mm)**: DLT tam giác đạc khớp Pelvis tự do theo 2D hips; còn `triangulate_anatomical` ràng buộc chiều dài xương H36M danh định, gián tiếp dịch chuyển vị trí Pelvis 3D để dung hòa toàn bộ khung chi dưới.

### 2.3. Giải mã độ lệch 1 mm giữa file per-joint (52.7 mm) và file kết quả (53.15 mm)
- **Công thức tính toán**:
  - `results.csv` tính MPJPE trên **16 khớp đánh giá** chuẩn H36M ([geometry.py](../src/athlete_pose3d/algorithms/geometry.py#L27): `H36M_EVAL_JOINTS = np.arange(1, 17)`, **loại trừ Pelvis joint 0**).
  - `results_per_joint_summary.csv` liệt kê đầy đủ **17 khớp** (gồm cả Pelvis joint 0). Khớp Pelvis có sai số sau căn chỉnh thấp hơn trung bình (44.78 mm ở Method, 36.50 mm ở DLT).
  - Tính trung bình 17 dòng:
    $$\text{Method: } \frac{16 \times 53.15 + 44.78}{17} = 52.66\text{ mm} \approx \mathbf{52.7\text{ mm}}$$
    $$\text{DLT: } \frac{16 \times 58.42 + 36.50}{17} = 57.13\text{ mm} \approx \mathbf{57.1\text{ mm}}$$
  - Khớp tuyệt đối với số liệu thống kê.

### 2.4. Đơn vị của `Global_Sync_Delta` & `Dynamic_Sync_Delta`
- **Đơn vị chuẩn**: **Số khung hình (frames)** (tại [pipeline.py](../src/athlete_pose3d/pipeline.py#L250) và [synchronization.py](../src/athlete_pose3d/algorithms/synchronization.py#L428): `frame_b = frame_a + offset`).
- Ở 60 FPS, độ lệch $N$ frames tương đương $N \times 16.67\text{ ms}$; ở 120 FPS tương đương $N \times 8.33\text{ ms}$.

### 2.5. Phân tích Sequence Metrics: Tỉ lệ co xương (Scale Ratio ~0.945)
- Trong `sequence_diagnosis` ([geometry.py](../src/athlete_pose3d/algorithms/geometry.py#L586)), thước đo `SeqSim` áp dụng căn chỉnh Procrustes toàn chuỗi với hệ số co giãn tự do $s$.
- DLT giữ nguyên tam giác đạc phóng đại tự do nên khi căn Procrustes toàn chuỗi kèm scale $s$, DLT dễ dàng co giãn khớp hình dạng GT.
- Method bị neo cứng bởi độ dài xương danh định nam `MALE_H36M_BONE_RATIOS`. Do đối tượng S1 là nữ vận động viên, mô hình xương nam làm khung chi bị co ngắn khoảng 5–6% so với thực tế, dẫn đến scale ratio trung bình đạt **0.940** (S1) và **0.949** (S3).

### 2.6. Tệp `all_camera_pairs.csv` bị lặp khối và trùng lặp giữa các thư mục
- Hàm `_record_evaluated_pairs_csv` ([synchronization.py](../src/athlete_pose3d/algorithms/synchronization.py#L330)) mở tệp ở chế độ append (`open("a")`).
- Trong pipeline chạy chính ([pipeline.py](../src/athlete_pose3d/pipeline.py#L385)), cờ `record_all_csv` không được truyền vào. Tệp hiện tại là bản ghi tích lũy qua nhiều lần chạy thử nghiệm cục bộ trên S1 trước đây và được sao chép nguyên trạng sang S2, S3.

### 2.7. Tương quan -1.00 giữa `Mean_Confidence` và Số khớp bị che
- Trong `_extract_confidence_and_occlusion` ([reporting.py](../src/athlete_pose3d/io/reporting.py#L104-L106)), mặt nạ che khuất được xác định theo ngưỡng: `occluded_mask = (ca <= 0.3) | (cb <= 0.3)`.
- Giá trị confidence của mô hình phát hiện 2D phân cực mạnh (xấp xỉ 1.0 khi quan sát rõ và rớt xuống $\le 0.3$ khi bị che), khiến `mean_conf` biến thiên nghịch đảo tuyến tính hoàn hảo với số khớp rơi vào vùng che.

---

## 3. Phân Tích Trường Hợp Bất Thường (Outlier Case Study)

- **Trường hợp chuỗi `Running_50` (S2, Cam 1 & Cam 3)**:
  - Sai số cao đột biến: 80.7 mm (Method) và 83.2 mm (DLT).
  - **Nguyên nhân hình học**: Camera 1 và Camera 3 có góc trục quang học hội tụ hẹp (baseline ngắn), gây suy giảm nghiêm trọng độ sâu tam giác đạc (`depth ambiguity`). Khi ma trận Essential $E$ phân rã suy biến, sai số hình học phóng đại dọc theo tia chiếu camera.

---

## 4. Kết Luận & Đề Xuất Kỹ Thuật

1. **Hiệu năng đã chứng minh**: Pipeline `Anatomical + SequenceRefine` vượt trội DLT ổn định **8.6%** trên toàn bộ 1.260 khung hình; các khớp chi rõ (Clear joints) đạt mức chính xác cao (**28.95 mm PA-MPJPE**).
2. **Báo cáo chuẩn mực**: Báo cáo song song `Rigid MPJPE` (đánh giá pose 3D thực tế) và `PA-MPJPE` (đánh giá hình dạng tư thế), tránh dùng Raw MPJPE khi chưa hiệu chuẩn ma trận quay Camera $\to$ World.
3. **Cải tiến trong tương lai**:
   - Tinh chỉnh vector nội suy khớp Spine/Thorax thích ứng theo góc nghiêng thân người.
   - Thêm mô hình nhân trắc học nữ (`FEMALE_H36M_BONE_RATIOS`) cho đối tượng S1 để khắc phục hiện tượng co xương 5%.
   - Truyền tham số đường dẫn riêng cho `all_camera_pairs.csv` theo từng Subject để xếp hạng camera độc lập.
