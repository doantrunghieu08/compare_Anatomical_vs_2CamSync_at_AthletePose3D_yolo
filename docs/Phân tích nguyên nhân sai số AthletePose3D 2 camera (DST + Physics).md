# Phân tích nguyên nhân sai số: AthletePose3D 2 camera (DST + Physics)

Oct 8, 2026 · @Hieu

## Phạm vi và mức độ kiểm chứng

Tài liệu này giải thích hai hiện tượng: MPJPE thô khoảng 680 mm trong khi PA-MPJPE chỉ khoảng 70 mm, và DST+Physics gần như không hơn DLT (cải thiện dưới 1 mm theo báo cáo trong repo).

Nguyên nhân được xác định bằng cách đọc code repo `compare_Anatomical_vs_2CamSync_at_AthletePose3D_yolo` (commit 2a396e0). Chưa chạy được pipeline vì repo không có dữ liệu hay file output, và cài `torch` bị proxy chặn. Các con số lấy từ báo cáo của chính repo, chưa được tái tạo.

Nếu "lỗi" cần tìm là một traceback hoặc kết quả sai ở một subject cụ thể, cần thêm thông tin để xác định đúng chỗ.

## 1. MPJPE thô \~680 mm là lỗi hệ quy chiếu, chưa phải lỗi pose

Pose 3D đang nằm trong hệ tọa độ camera 1, còn GT mocap nằm trong hệ có trục Z hướng lên, và MPJPE hiện không loại được phép quay giữa hai hệ.

- `P1 = K[I|0]` (`uncalibrated.py:73`), nên mọi điểm 3D tính theo hệ camera 1.
- `mpjpe()` (`geometry.py:443`) chỉ trừ pelvis, không xoay. Khi hai hệ lệch khoảng 90°, sai số tăng theo khoảng cách tới pelvis: hông thấp, cổ chân cao, đúng với mẫu trong báo cáo.
- Các hàm sửa đã có (`to_z_up`, `best_yaw`, `rigid_mpjpe`, `auto_z_up`) nhưng pipeline không gọi hàm nào. `pipeline.py` chỉ import `mpjpe` và `pa_mpjpe`, nên cột MPJPE trong CSV vẫn là số thô.

**Lỗi thật trong chẩn đoán.** `reporting.py:147-152` gộp mọi motion và mọi cặp camera vào một phép Kabsch duy nhất. Mỗi cặp có hệ camera 1 khác nhau, nên một phép quay chung không có nghĩa, và "Angle" cùng "MPJPE-SeqRot" in ra sẽ sai. Cần nhóm theo `(subject, motion, cam_a, cam_b)`.

Con số `scale_ratio` của chẩn đoán đã sửa cũng cho biết đơn vị GT có khớp không. PA-MPJPE có co giãn nên che mất lệch đơn vị m/mm.

## 2. Phần sai số thật (\~70–100 mm) đến từ khớp suy ra và camera giả định

Sau khi bỏ phần quay, sai số còn lại có ba nguồn, và cả DLT lẫn DST đều chịu chung vì dùng cùng P1/P2.

- **Khớp suy ra từ 2D bằng hằng số cố định.** `data.py:136-145` ngoại suy head và neck từ vector thorax→mũi với hệ số 1.45 và 0.40, độc lập trên từng camera, nên hai view không còn trỏ cùng một điểm 3D và epipolar evidence của head/neck luôn mâu thuẫn. Pelvis, spine, thorax là trung điểm 2D, lệch so với marker GT (Manubrium, C7, T10, Vertex). Báo cáo của repo ghi PA khớp suy ra \~104 mm so với \~46 mm ở khớp rõ ràng.
- **Camera giả định.** Intrinsics cố định 1920×1088, f = 1.2·max(W,H) (`uncalibrated.py:132-137`). Ma trận F được ước lượng từ cả 17 khớp, gồm 5 khớp tổng hợp. Cơ chế quét tiêu cự `f_scale="auto"` đã viết nhưng không nơi nào truyền vào (`pipeline.py:171`, `synchronization.py:198` và `:252`).
- **Scale tuyệt đối** do chiều cao cấu hình nhân tỷ lệ H36M chung quyết định (`solve_metric_scale`, `uncalibrated.py:168`).

## 3. DST+Physics gần như không hơn DLT vì bone prior bị vòng tròn

Mức cải thiện bị chặn trần bởi cách tạo prior và điểm khởi tạo, chứ không phải bởi việc tune thêm tham số.

- **Prior lấy từ chính đầu ra.** `_fit_context_bone_prior` (`pipeline.py:191-213`) lấy median độ dài xương từ DLT của chính P1/P2 đó, mà scale của P1/P2 lại đặt theo prior chung. Với refinement, `refine_results` được gọi không có `bone_lengths` (`pipeline.py:459-464`), nên target xương là median của chính chuỗi đầu vào. Ràng buộc xương chỉ khử được nhiễu độ dài giữa các frame, không sửa được méo hệ thống như tiêu cự hay chiều sâu.
- **Điểm khởi tạo đã gần tối ưu của data term.** Pose bắt đầu từ DLT, khoảng cách tia đã nhỏ, và anchor kéo ngược về DLT. Phần tự do còn lại chủ yếu là hướng chiều sâu, nơi prior lấy từ DLT không giúp được.
- **Prior sai được khuếch đại khi bằng chứng yếu.** Trọng số tia và anchor nhân với `reliability` (`physics_refine.py:202-207`) nhưng trọng số xương cố định bằng 1. Tắt `bone_reliability_modulation` chỉ bỏ một dạng của hiệu ứng này.
- **Sequence refinement mất độ trơn.** `frames_per_pair=30` lấy mẫu thưa từ keyframe (`pipeline.py:226`), nên `is_contiguous` sai (`refinement.py:89`) và thành phần độ trơn bị tắt.
- **Nhỏ:** `fuse_evidences` không nhận `initial_3d`, nên bone evidence tính trên điểm giữa hai tia trong khi tối ưu bắt đầu từ DLT (`physics_refine.py:294-299`).

## 4. Sửa chẩn đoán và camera trước, tune DST/Physics sau

Chừng nào hệ quy chiếu và camera chưa đúng, mọi chênh lệch dưới 1 mm giữa DST và DLT đều nằm trong nhiễu.

1. Sửa chẩn đoán theo từng cặp camera `(subject, motion, cam_a, cam_b)` và ghi `MPJPE-SeqRot` cùng `scale_ratio` vào CSV. Kiểm tra `scale_ratio` gần 1 để chắc đơn vị GT khớp.
2. Nối `f_scale="auto"` vào config và chỉ dùng khớp COCO thật (bỏ 5 khớp tổng hợp) khi ước lượng F.
3. Chạy lại baseline DLT và DST với metric mới, trên cùng frame hợp lệ, tắt sequence refinement khi so sánh.
4. Chỉ sau đó mới tune `sigma`, ngưỡng xung đột và các trọng số physics, và tune theo subject thay vì theo frame.

## Tham chiếu file và dòng code

| Vị trí | Nội dung liên quan |
| --- | --- |
| `uncalibrated.py:73` | \`P1 = K\[I |
| `uncalibrated.py:132-137` | Intrinsics cố định, `f_scale` mặc định 1.2, `auto_z_up=False` |
| `uncalibrated.py:168` | Scale metric từ prior xương chung |
| `geometry.py:443` | `mpjpe()` chỉ trừ pelvis, không xoay |
| `pipeline.py:191-213` | `_fit_context_bone_prior`, prior lấy từ DLT |
| `pipeline.py:226` | Lấy mẫu keyframe thưa |
| `pipeline.py:459-464` | `refine_results` không nhận `bone_lengths` |
| `refinement.py:89` | Điều kiện `is_contiguous` |
| `reporting.py:147-152` | Chẩn đoán Kabsch gộp mọi cặp camera |
| `data.py:136-145` | Ngoại suy head/neck bằng hệ số 1.45 và 0.40 |
| `physics_refine.py:202-207` | Trọng số tia và anchor nhân với `reliability` |
| `physics_refine.py:294-299` | `fuse_evidences` không nhận `initial_3d` |
