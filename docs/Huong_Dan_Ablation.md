# Hướng Dẫn & Báo Cáo Nghiên Cứu Ablation Study

Tài liệu này hướng dẫn cách chạy và giải thích chi tiết cơ sở khoa học đằng sau bộ thực nghiệm **Ablation Study (Ma trận 2×2 và các thành phần tối ưu)** của dự án AthletePose3D.

---

## 1. Mục Đích & Thiết Kế Thử Nghiệm

Ablation study được thiết kế nhằm phân tách rõ ràng và cô lập tác động của từng thành phần kỹ thuật, đặc biệt là tách biệt giữa **hiệu ứng phương pháp tam giác đạc** và **hiệu ứng thang đo nhân trắc học (chiều cao)**.

Hệ thống đánh giá qua 4 thước đo chuẩn mực:
1. **Raw MPJPE**: Chỉ trừ gốc Pelvis (joint 0), giữ nguyên hệ tọa độ và thang đo thực tế.
2. **Rigid MPJPE** (per-frame): Căn chỉnh tối ưu phép xoay $SO(3)$ và tịnh tiến từng khung hình (không co giãn thang đo).
3. **Seq-Rot MPJPE**: Căn chỉnh một phép xoay $SO(3)$ duy nhất cho toàn bộ chuỗi clip (đánh giá độ ổn định hướng thời gian).
4. **PA-MPJPE**: Căn chỉnh Procrustes toàn phần (xoay, tịnh tiến và co giãn tỉ lệ $s$).

> [!NOTE]
> Mối quan hệ toán học luôn được bảo đảm trên từng khung hình:  
> $$\text{Raw MPJPE} \ge \text{Rigid MPJPE} \ge \text{PA-MPJPE}$$

---

## 2. Ma Trận Ablation 2×2 và Các Biến Thể

| ID | Cấu hình | Phương pháp | Chiều cao VĐV | Sequence Refine | Mục đích cô lập biến |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **A1** | [A1_dlt_generic.yml](../configs/ablation/A1_dlt_generic.yml) | `dlt` | 1730 mm (generic) | Tắt | Baseline DLT thuần với scale danh định |
| **A2** | [A2_dlt_calibrated.yml](../configs/ablation/A2_dlt_calibrated.yml) | `dlt` | Hiệu chuẩn (`auto`) | Tắt | Baseline DLT với scale cá nhân hóa |
| **B1** | [B1_anat_generic.yml](../configs/ablation/B1_anat_generic.yml) | `anatomical` | 1730 mm (generic) | Tắt | Anatomical Huber ray khi scale danh định |
| **B2** | [B2_anat_calibrated.yml](../configs/ablation/B2_anat_calibrated.yml) | `anatomical` | Hiệu chuẩn (`auto`) | Tắt | Anatomical với prior xương cá nhân hóa |
| **C1** | [C1_refine_accel.yml](../configs/ablation/C1_refine_accel.yml) | `anatomical` | Hiệu chuẩn (`auto`) | Bật (Chỉ gia tốc) | Hiệu ứng làm mượt gia tốc thời gian |
| **C2** | [C2_refine_accel_vel.yml](../configs/ablation/C2_refine_accel_vel.yml) | `anatomical` | Hiệu chuẩn (`auto`) | Bật (Gia tốc + Vận tốc) | Full pipeline hoàn chỉnh |

### Cách phân tích hiệu ứng:
- **Hiệu ứng Phương pháp (Method Effect)**: So sánh `A2` $\to$ `B2` (cùng chiều cao hiệu chuẩn).
- **Hiệu ứng Thang đo Chiều cao (Height Calibration Effect)**: So sánh `A1` $\to$ `A2` và `B1` $\to$ `B2`.
- **Hiệu ứng Tối ưu Chuỗi (Sequence Refinement Effect)**: So sánh `B2` $\to$ `C1` và `C1` $\to$ `C2`.

> [!IMPORTANT]
> Với các cấu hình DLT (`A1` và `A2`), cột kết quả phương pháp chính (`Selected_Method`) và DLT baseline là đồng nhất ($\Delta = 0$). Việc so sánh cần đối chiếu trực tiếp giá trị tuyệt đối giữa các cấu hình thay vì chỉ dựa vào độ lệch nội bộ từng file.

---

## 3. Chi Tiết Từng Cấu Hình

### A1: Baseline DLT Generic ([A1_dlt_generic.yml](../configs/ablation/A1_dlt_generic.yml))
- Tam giác đạc đại số tuyến tính có trọng số tin cậy.
- Sử dụng chiều cao mặc định chung 1730.0 mm để khôi phục độ sâu camera $P_2$.
- Không áp dụng tối ưu hóa phi tuyến hay ràng buộc thời gian.

### A2: Baseline DLT Calibrated ([A2_dlt_calibrated.yml](../configs/ablation/A2_dlt_calibrated.yml))
- Vẫn dùng giải thuật DLT thuần túy.
- Thang đo camera $P_2$ được chuẩn hóa theo chiều cao thực tế của từng vận động viên (`auto`: S1=1591mm, S2=1553mm, S3=1733mm).

### B1: Anatomical Generic ([B1_anat_generic.yml](../configs/ablation/B1_anat_generic.yml))
- Khởi tạo từ DLT, tối ưu hóa bằng Adam với hàm Huber ray loss.
- Độ dài xương tham chiếu lấy theo tỉ lệ cơ thể nam H36M scaled theo chiều cao chung 1730 mm.

### B2: Anatomical Calibrated ([B2_anat_calibrated.yml](../configs/ablation/B2_anat_calibrated.yml))
- Tối ưu hóa bằng Adam với hàm Huber ray loss kết hợp độ dài xương tham chiếu chuẩn theo chiều cao thực của từng vận động viên.
- `lr: 0.1` (có thể cấu hình trong YAML), `bone_weight: 1.0`, `iterations: 80`.

### C1: Sequence Refinement Accel ([C1_refine_accel.yml](../configs/ablation/C1_refine_accel.yml))
- Kế thừa B2, kích hoạt tối ưu hóa chuỗi cửa sổ khung hình liên tục bằng L-BFGS.
- Phạt gia tốc đột ngột với mặt nạ khoảng trống (`gap masking`):
  $$\mathcal{L}_{accel} = \sum_{t} \mathcal{H}_\delta(X_{t-1} - 2X_t + X_{t+1}) \cdot \mathbb{I}(\Delta t_1 = 1 \land \Delta t_2 = 1)$$
- `smoothness_weight: 0.15`, `velocity_weight: 0.0`.

### C2: Sequence Refinement Accel + Velocity ([C2_refine_accel_vel.yml](../configs/ablation/C2_refine_accel_vel.yml))
- Kế thừa C1, bổ sung thành phần cản vận tốc liên khung hình:
  $$\mathcal{L}_{vel} = \sum_{t} \mathcal{H}_\delta(X_t - X_{t-1}) \cdot \mathbb{I}(\Delta t = 1)$$
- `smoothness_weight: 0.15`, `velocity_weight: 0.03`.

---

## 4. Cách Thực Thi Thử Nghiệm

### Chạy toàn bộ ma trận Ablation trên vận động viên S1:
```powershell
python scripts/run_ablation.py -S1
```

### Chạy trên toàn bộ các vận động viên (S1, S2, S3):
```powershell
python scripts/run_ablation.py
```

### Sinh lại các file cấu hình YAML:
```powershell
python scripts/make_ablation_configs.py
```

Kết quả tổng hợp tự động xuất ra:
- `outputs/ablation_summary.csv`: Bảng tổng hợp các chỉ số Raw, Rigid, Seq-Rot, PA và 95% Bootstrap CI theo từng chuyển động.
- `outputs/ablation/<Config>_sequence_metrics.csv`: Báo cáo chẩn đoán cấp chuỗi cho từng cặp camera.

---

## 5. Phân Tích Thất Bại & Bài Học Khoa Học

| Hướng tiếp cận | Giả định ban đầu | Kết quả thực tế | Lý do thất bại |
| :--- | :--- | :--- | :--- |
| **`bone_prior: sequence_median`** | Lấy trung vị xương từ DLT của chính video để thích nghi theo từng cá nhân. | Tăng sai số **+1.7 mm** so với dùng prior H36M. | DLT từ 2 camera bị co giãn chiều sâu (`depth ambiguity`), khiến trung vị xương bị sai lệch tỉ lệ cơ thể nghiêm trọng. |
| **`sync_local_radius: 2`** | Cho phép tìm kiếm cục bộ offset $\pm 2$ frame để bắt frame tốt nhất. | Tăng sai số **+1.20 mm** (từ 58.77 lên 59.97 mm). | Tìm kiếm cục bộ độc lập từng frame không có phạt chuyển trạng thái gây rung lắc pha 50ms giữa các frame kề nhau. Đặt `sync_local_radius: 0` để khóa đường đi Viterbi DP mượt mà. |
| **Ràng buộc cột sống cứng (Rigid Torso)** | Ép Pelvis $\to$ Thorax $\to$ Head thành đoạn thẳng cố định. | Tăng sai số **+4.48 mm** (từ 59.97 lên 64.44 mm). | Vận động viên uốn lưng và gập người mạnh trong các động tác thể thao phức tạp, ràng buộc cứng gây méo mó tư thế thực. |
| **Ray Angle Modulation** | Nhân trọng số hàm mất mát tia theo $\sin(\theta)$ góc hội tụ 2 camera. | Chỉ giảm **-0.33 mm** (59.97 $\to$ 59.64 mm). | Không bù đắp được chi phí tính toán tích có hướng và tăng thêm siêu tham số không cần thiết (YAGNI). |
