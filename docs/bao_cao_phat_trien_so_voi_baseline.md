# Báo cáo phát triển so với baseline gốc

**Ngày:** 07/10/2026  
**Phạm vi:** Pipeline dựng pose 3D hai camera, tập S1 AthletePose3D  
**Baseline so sánh:** DLT trên cùng cặp camera và frame đã đồng bộ  
**Ràng buộc:** Không thay đổi tham số camera.

## 1. Tóm tắt

Các thay đổi đã phát triển thuật toán theo ba hướng: ước lượng prior chiều dài xương riêng từ sequence, cân bằng lại loss của DST+Physics để giữ thông tin từ tia camera và neo DLT, và bổ sung tinh chỉnh pose theo thời gian với reprojection, giải phẫu và độ mượt chuyển động.

Thí nghiệm ablation đã ghi nhận cấu hình **Subject Bone Prior + Tuned** tốt hơn DLT trung bình **0,84 mm Raw MPJPE** và **2,22 mm Rigid MPJPE** trên 57 motion validation (1.578 frame). Kết quả Raw có CI bootstrap 95% `[+0,07; +1,57]` mm. Đây là mức cải thiện có ý nghĩa thống kê theo báo cáo ablation, nhưng **chưa đạt mục tiêu cải thiện 10 mm**.

Tinh chỉnh liên khung và bộ tối ưu PyTorch L-BFGS mới được đưa vào runtime sau thí nghiệm trên. Chưa có kết quả validation đo riêng cho các thay đổi mới này, vì vậy không cộng chúng vào số liệu trên và chưa thể kết luận chúng giúp đạt 10 mm.

## 2. Những phần đã phát triển

### 2.1 Prior xương thích nghi theo sequence

Baseline DLT không áp đặt prior giải phẫu. Phiên bản DST+Physics cũ dùng tỷ lệ xương H36M chung; ablation cho thấy prior chung có thể kéo lệch các khớp thân trên của vận động viên.

Pipeline hiện tại hỗ trợ `bone_prior: sequence_median`: chiều dài xương được ước lượng từ các pose DLT chất lượng đủ tốt trong sequence đang xử lý, không lấy Ground Truth làm prior. Cấu hình này phù hợp hơn với tỷ lệ cơ thể của từng vận động viên.

Trong bước sequence refinement, mục tiêu chiều dài xương lấy từ trung vị của pose trong sequence. Bước này không còn trộn mặc định prior H36M chung vào mục tiêu xương.

### 2.2 Cân bằng lại loss DST+Physics

Cấu hình runtime hiện tại dùng các giá trị:

| Tham số | Giá trị | Vai trò |
|---|---:|---|
| `data_weight` | 1.0 | Giữ ảnh hưởng của khoảng cách tới tia camera |
| `anchor_weight` | 0.3 | Hạn chế pose trôi xa nghiệm DLT |
| `bone_weight` | 0.5 | Áp dụng ràng buộc chiều dài xương ở mức vừa phải |
| `sym_weight` | 0.2 | Khuyến khích chiều dài xương trái–phải tương đồng |
| `bone_reliability_modulation` | `false` | Không tự động tăng trọng số xương khi độ tin cậy thấp |

Thay đổi này xử lý điểm yếu của cấu hình DST+Physics cũ: khi keypoint không đáng tin, cơ chế điều biến trước đây có thể giảm ảnh hưởng dữ liệu camera và tăng quá mức ảnh hưởng prior xương.

### 2.3 Tinh chỉnh liên khung

`sequence_refinement` hiện được bật trong `configs/default.yml`. Bước này tối ưu một đoạn pose theo nhiều frame, kết hợp:

- Giữ pelvis và hình dạng tương đối gần nghiệm ban đầu.
- Reprojection 3D về các tia từ hai camera, có trọng số theo confidence.
- Chiều dài xương và tính đối xứng trái–phải.
- Độ mượt gia tốc theo thời gian.
- Robust loss để giảm ảnh hưởng của residual ngoại lai.

Để tránh chi phí sai phân số lớn của `scipy.least_squares`, phần tối ưu sequence đã chuyển sang PyTorch L-BFGS với gradient tự động và tính khoảng cách tới tia theo batch. Mục tiêu là giữ refinement nhưng giảm thời gian chạy; cần đo benchmark runtime thực tế để xác nhận mức tăng tốc.

## 3. Kết quả ablation đã có

Các số dưới đây được ghi trong `bao_cao_nghien_cuu_ablation_dst_physics.md`. Quy ước `Δ = sai số DLT − sai số phương pháp`, vì vậy số dương nghĩa là cải thiện.

| Phương pháp | Raw MPJPE (mm) | Δ Raw vs DLT (mm) | 95% CI Δ Raw | Rigid MPJPE (mm) | Δ Rigid vs DLT (mm) | Tỷ lệ thắng Raw |
|---|---:|---:|---:|---:|---:|---:|
| DLT baseline | 679.32 | 0.00 | — | 99.58 | 0.00 | 0.0% |
| DST+Physics cũ | 681.14 | -1.82 | [-2.39; -1.29] | 100.18 | -0.60 | 32.2% |
| Subject Bone Prior + Tuned | 678.49 | **+0.84** | **[+0.07; +1.57]** | 97.36 | **+2.22** | **53.2%** |

> Các con số trên là kết quả của protocol ablation: prior được fit trên 6 motion Dev rồi áp dụng cho 57 motion Validation. Chúng không đại diện cho benchmark của toàn bộ cấu hình runtime hiện nay, vốn fit prior theo sequence và đã bật sequence refinement.

Raw MPJPE lớn một phần do khác biệt hệ tọa độ quay giữa camera và MoCap. Trong báo cáo ablation, Rigid MPJPE của DLT là 99,58 mm; do đó cần báo cáo riêng Raw, Rigid và PA-MPJPE để phân biệt sai khác hệ trục với sai số hình dạng pose.

## 4. Tình trạng mục tiêu giảm ít nhất 10 mm

**Chưa đạt và chưa được xác nhận.** Mức tăng được xác nhận trên validation là +0,84 mm Raw và +2,22 mm Rigid, thấp hơn mục tiêu 10 mm. Thay đổi sequence refinement và bộ tối ưu L-BFGS chưa có kết quả validation mới.

Để xác nhận mục tiêu, cần chạy cùng protocol DLT vs phương pháp mới trên cùng 57 motion validation, không dùng GT để fit prior hoặc chọn trọng số. Báo cáo cần gồm Raw/Rigid/PA, bootstrap CI theo motion, win rate, breakdown theo khớp và thời gian chạy.

## 5. Giới hạn và trạng thái kiểm chứng

- Không thay đổi intrinsic/extrinsic hay tham số camera.
- Kết quả ablation là số liệu lịch sử được chép từ báo cáo nghiên cứu hiện có.
- Tại thời điểm viết báo cáo này, benchmark của cấu hình mới chưa chạy xong; môi trường thực thi của agent thiếu `numpy`, còn cache ablation và các CSV mở trong IDE không nằm trong workspace đọc được.
- Mã `refinement.py` đã được kiểm tra cú pháp Python sau khi chuyển sang L-BFGS. Đây không phải xác nhận chất lượng số học, thời gian chạy hay độ chính xác trên S1.
- Báo cáo ablation trước đó ghi nhận 31/31 unit tests đạt trong một lần chạy lịch sử. Trạng thái đó không đồng nghĩa code hiện tại đã được kiểm thử lại.

## 6. Tệp liên quan

- Cấu hình runtime: [`configs/default.yml`](../configs/default.yml)
- Tinh chỉnh liên khung: [`src/athlete_pose3d/algorithms/refinement.py`](../src/athlete_pose3d/algorithms/refinement.py)
- Pipeline: [`src/athlete_pose3d/pipeline.py`](../src/athlete_pose3d/pipeline.py)
- Kết quả ablation gốc: [`docs/bao_cao_nghien_cuu_ablation_dst_physics.md`](bao_cao_nghien_cuu_ablation_dst_physics.md)
