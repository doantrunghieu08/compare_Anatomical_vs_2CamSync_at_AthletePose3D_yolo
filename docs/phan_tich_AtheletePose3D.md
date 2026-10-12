# Phân tích dữ liệu AtheletePose3D_Report

Ngày phân tích: 12/10/2026. Phạm vi: 3 folder S1, S2, S3, mỗi folder 4 file CSV (tổng 12 file).

Tài liệu này giải thích tôi đã phân tích thế nào, đọc từng con số ra sao, kết quả là gì, và những chỗ đã phải sửa lại so với nhận định ban đầu.

## 1. Tóm tắt

- Phương pháp được đánh giá ("Method", cấu hình `Anatomical + SequenceRefine`) cho sai số **thấp hơn DLT khoảng 8–9%** theo MPJPE sau khi căn chỉnh rigid từng frame. Điều này đúng ở cả 3 subject và cả 42 sequence (không sequence nào thua DLT).
- Lợi thế **chỉ tồn tại sau khi căn chỉnh**. Sai số thô (chưa căn) của Method và DLT gần như bằng nhau, và khi chỉ dùng một phép căn cho cả sequence thì lợi thế biến mất.
- Với S1 và S3, phần lớn lợi thế đến từ căn chỉnh rigid. Theo PA-MPJPE (chỉ xét hình dạng tư thế), mức cải thiện chỉ 2,2% (S1) và 3,4% (S3), so với khoảng 10% ở S2.
- Lỗi lớn nhất nằm ở nhóm khớp thân trên "Derived" (Spine, Thorax, Head), Method hầu như không cải thiện. Mức cải thiện tập trung ở hông và cổ chân.
- Sequence tệ nhất là `Running_50` (S2, camera 1–3), cả hai phương pháp đều sai nặng.
- `all_camera_pairs.csv` ở S2 và S3 chỉ là bản sao của file S1, nên không có xếp hạng camera cho các môn Running, Discus, Shot put, Javelin.

## 2. Dữ liệu và cách tôi đọc

### 2.1. Cấu trúc

| File | Nội dung | Số dòng dữ liệu |
|---|---|---|
| `results.csv` | Kết quả từng frame: MPJPE, PA, độ tin cậy, che khuất, cấu hình chạy | S1: 450, S2: 420, S3: 390 |
| `results_sequence_metrics.csv` | Một dòng cho mỗi sequence: cặp camera, sai số cấp sequence, scale, góc xoay | S1: 15, S2: 14, S3: 13 |
| `results_per_joint_summary.csv` | 17 khớp: tỉ lệ che khuất, sai số DLT và Method theo khớp | 17 mỗi folder |
| `all_camera_pairs.csv` | Xếp hạng 132 cặp camera cho mỗi motion | 13.992 mỗi folder |

Mỗi sequence có đúng 30 frame. Các motion: S1 gồm Axel (10), Comb (3), Flip (2); S2 là Running (14); S3 gồm Discus, Glide shot put, Javelin và các bản `_error`.

### 2.2. Thuật ngữ

File không có từ điển dữ liệu, nên các định nghĩa dưới đây là suy luận từ tên cột và thông lệ chung, chưa được xác nhận từ phía bạn.

- **DLT**: phương pháp tam giác hóa Direct Linear Transform, dùng làm baseline.
- **Method**: phương pháp được đánh giá (cột `Selected_Method_*`).
- **MPJPE**: sai số vị trí trung bình trên các khớp (mm).
- **PA-MPJPE**: MPJPE sau khi căn Procrustes (tịnh tiến, xoay, scale), tức chỉ đo hình dạng tư thế.
- **Raw MPJPE**: sai số trước khi căn chỉnh. **Rigid MPJPE**: sau căn rigid. Trong dữ liệu, cột `*_MPJPE` trùng với cột `*_Rigid_MPJPE`, nên MPJPE báo cáo ở đây là sai số sau căn rigid.
- **Khớp "Clear" và "Derived"**: nhóm khớp trong file per-joint. Derived gồm hông, Spine, Thorax, Neck, Head; Clear gồm vai, khuỷu, cổ tay, đầu gối, cổ chân.

### 2.3. Quy trình

1. Link Google Drive không đọc được bằng cách tải trang web thông thường, nên tôi dùng kết nối Google Drive.
2. Qua Drive, tôi đọc đầy đủ các file per-joint và sequence-metrics. `results.csv` và `all_camera_pairs.csv` bị cắt ở ~100.000 ký tự đầu, nên đợt này mới chỉ phản ánh một phần.
3. Bạn gửi file zip. Tôi giải nén vào thư mục riêng và phân tích toàn bộ 1.260 frame bằng pandas.
4. Tôi đối chiếu chéo các file với nhau (mục 7) và sửa lại những nhận định ban đầu không còn đứng vững (mục 6).

## 3. Kết quả chính

### 3.1. Sai số theo frame (mm, sau căn rigid)

| | DLT | Method | DLT (PA) | Method (PA) | Method thắng, MPJPE | Method thắng, PA |
|---|---|---|---|---|---|---|
| S1 | 58,42 | 53,15 | 48,71 | 47,63 | 97,3% | 76,2% |
| S2 | 59,50 | 54,54 | 57,47 | 51,89 | 97,6% | 96,0% |
| S3 | 61,29 | 56,08 | 50,50 | 48,74 | 94,6% | 75,4% |

- Số frame Method thua DLT theo MPJPE: 12/450, 10/420, 21/390 (tổng 43/1.260, 3,4%).
- Số frame Method thua DLT theo PA: 107/450, 17/420, 96/390.
- Các giá trị trung bình trên khớp chính xác với dòng `Summary_Mean` có sẵn ở cuối mỗi `results.csv`.

Phân bố sai số Method (p5 / trung vị / p95 / lớn nhất):

| | Method | DLT |
|---|---|---|
| S1 | 45,5 / 51,4 / 64,6 / 71,6 | 50,0 / 56,3 / 71,2 / 81,5 |
| S2 | 40,2 / 51,6 / 75,8 / 87,1 | 48,2 / 56,3 / 79,5 / 88,7 |
| S3 | 46,5 / 55,3 / 65,5 / 69,0 | 48,5 / 62,0 / 73,4 / 74,4 |

### 3.2. Sai số thô và vai trò của căn chỉnh

| | DLT thô | Method thô | DLT sau căn | Method sau căn |
|---|---|---|---|---|
| S1 | 661 | 662 | 58,4 | 53,2 |
| S2 | 622 | 622 | 59,5 | 54,5 |
| S3 | 774 | 773 | 61,3 | 56,1 |

Sai số thô lớn gấp khoảng 10 lần sau căn chỉnh và gần như giống nhau giữa hai phương pháp. Nghĩa là mọi con số báo cáo ở đây phụ thuộc vào bước căn chỉnh, và vị trí hay hướng toàn cục của Method không tốt hơn DLT. Nguyên nhân của sai số thô lớn (khác hệ tọa độ, thiếu hiệu chuẩn, v.v.) chưa được kiểm tra.

### 3.3. Theo khớp (file per-joint)

Trung bình trên 17 khớp (mm):

| | Che khuất TB | DLT | Method | Method (PA) |
|---|---|---|---|---|
| S1 | 5,2% | 57,1 | 52,7 | 47,1 |
| S2 | 8,0% | 57,8 | 53,7 | 51,2 |
| S3 | 19,5% | 59,6 | 55,1 | 47,8 |

Chia theo nhóm khớp (không tính Pelvis), DLT / Method / Method PA:

| | Clear | Derived |
|---|---|---|
| S1 | 42,5 / 39,0 / 29,0 | 84,9 / 76,7 / 78,8 |
| S2 | 46,3 / 44,9 / 41,8 | 81,5 / 70,7 / 68,8 |
| S3 | 42,5 / 38,6 / 25,9 | 92,6 / 85,3 / 86,8 |

- **Hông giảm mạnh nhất:** R_Hip, L_Hip giảm từ khoảng 75–90 mm xuống khoảng 45–58 mm (ví dụ S2 R_Hip: 88,8 → 45,3 mm). Ở S1, hai cổ chân cũng giảm khoảng 20 mm.
- **Spine sai 94–117 mm ở cả 3 subject** và Method không cải thiện (S1 91,8 → 94,1; S2 118,2 → 117,1; S3 109,8 → 111,7). Thorax dao động 74–105 mm và Head 56–97 mm tùy subject; ở S2, Head còn tệ đi (34,1 → 55,8 mm).
- **Pelvis tệ hơn ở cả 3 subject** (36,5 → 44,8; 30,2 → 39,8; 33,3 → 38,7).
- Sai số gần như cố định giữa các subject ở Spine gợi ý lỗi hệ thống (định nghĩa keypoint hoặc offset) thay vì nhiễu, nhưng đây mới là giả thuyết.

### 3.4. Theo sequence (file sequence-metrics)

File này dùng một phép căn cho cả sequence. Trung bình các sequence:

| | Số sequence | SeqRot: Method / DLT | SeqSim: Method / DLT | Scale Method TB | Method thắng SeqSim |
|---|---|---|---|---|---|
| S1 | 15 | 67,8 / 68,7 | 55,8 / 51,6 | 0,940 | 0/15 |
| S2 | 14 | 64,8 / 64,8 | 63,3 / 63,1 | 1,009 | 7/14 |
| S3 | 13 | 66,9 / 69,3 | 55,8 / 53,6 | 0,949 | 3/13 |

Ở cấp sequence, hai phương pháp ngang nhau (SeqRot) hoặc DLT tốt hơn (SeqSim). Scale của Method nhỏ hơn 1 ở S1 và S3 (khoảng 5–6%), tức skeleton bị co lại; nguyên nhân có thể là ràng buộc xương nhưng chưa được kiểm tra.

### 3.5. Theo motion

Sai số trung bình theo nhóm motion (DLT → Method, mm; mỗi sequence có trọng số bằng nhau):

| Folder | Nhóm motion | Số seq | DLT | Method |
|---|---|---|---|---|
| S1 | Axel | 10 | 57,75 | 52,13 |
| S1 | Comb | 3 | 62,71 | 59,16 |
| S1 | Flip | 2 | 55,33 | 49,27 |
| S2 | Running | 14 | 59,50 | 54,54 |
| S3 | Discus | 2 | 56,49 | 54,48 |
| S3 | Discus_error | 2 | 54,71 | 50,93 |
| S3 | Glide_shot_put | 4 | 60,06 | 54,82 |
| S3 | Glide_shot_put_error | 2 | 61,38 | 55,64 |
| S3 | Javelin | 3 | 70,45 | 62,57 |

- **Tệ nhất:** `Running_50` (80,7 mm Method; 83,2 mm DLT; camera 1–3), `Running_46` (68,2), `Javelin_18` (64,8), `Javelin_16` (64,6), `Comb_3` (64,4).
- **Tốt nhất:** `Running_44` (41,7), `Running_49` (43,4), `Running_43` (44,9).
- Tám frame tệ nhất toàn bộ dữ liệu đều thuộc `Running_50` (Method 84–87 mm, DLT 85–89 mm) và gần như không có khớp bị che. Cả hai phương pháp đều sai, nên cặp camera 1–3 (hiệu chuẩn hoặc hình học) là nghi vấn, chưa được kiểm chứng.
- Sai số tăng nhẹ về cuối sequence ở S2 và S3 (Method, 10 frame đầu / giữa / cuối: S2 53,8 / 54,4 / 55,4; S3 55,5 / 55,5 / 57,3).

### 3.6. Che khuất

| | Frame có ≥1 khớp bị che | Số khớp bị che TB mỗi frame | Method, frame không che | Method, frame có che |
|---|---|---|---|---|
| S1 | 24,0% | 0,89 | 53,15 | 53,18 |
| S2 | 40,7% | 1,37 | 50,73 | 60,08 |
| S3 | 90,5% | 3,31 | 52,66 | 56,44 |

- Che khuất ảnh hưởng nhiều nhất ở S2 (+9,4 mm), ít ở S3 (+3,8 mm), gần như không ở S1.
- `Mean_Confidence` có tương quan -1,00 với số khớp bị che, tức nó chỉ là hàm của số khớp bị che chứ không phải tín hiệu độc lập.
- Ở S3, sai số của các khớp bị che (53,6 mm) không cao hơn các khớp thấy rõ (56,1 mm). Có thể do khớp bị che chủ yếu là chân, vốn có sai số thấp hơn Spine/Thorax/Head. Đây là giả thuyết.

### 3.7. Đồng bộ camera (`Global_Sync_Delta`, `Dynamic_Sync_Delta`)

- Độ lệch đồng bộ (giá trị tuyệt đối): S1 trung bình 8,9 (lớn nhất 33); S2 trung bình 16,3 (lớn nhất 89); S3 gần như 0. Đơn vị của cột này chưa được ghi rõ trong file.
- Trên 42 sequence **không có quan hệ ổn định** giữa độ lệch đồng bộ và sai số: Pearson -0,04, Spearman -0,38.
- Theo từng folder: S1 Pearson 0,62 nhưng Spearman 0,08 (do ba sequence lệch lớn Axel_7, Comb_3, Axel_2); S2 Pearson -0,13; S3 -0,06.
- `Running_40` và `Running_45` (S2) lệch -89 và -86 nhưng sai số bình thường (50,8 và 56,2 mm).

## 4. File camera pairs

- MD5 giống hệt nhau ở cả 3 folder, và chỉ chứa Subject S1 (15 motion Axel, Comb, Flip).
- Mỗi folder có 13.992 dòng, chia thành 106 khối × 132 cặp (12 camera × 11). Chỉ có 1.980 khóa (motion, rank) duy nhất; mỗi motion lặp 6–7 lần, riêng `Axel_1` lặp 17 lần.
- Các lần lặp có điểm số khác nhau và không có cột run id, nên đây là nhật ký nối thêm của nhiều lần chạy.
- 34 cặp có điểm `inf` (0,6% số dòng duy nhất).
- Đối chiếu với cặp camera thực dùng trong `results.csv` (S1, 15 motion), lấy khối xuất hiện cuối cùng trong file làm phiên bản mới nhất (suy luận từ thứ tự):
  - 9/15 motion dùng cặp xếp hạng 1.
  - 6 motion dùng cặp xếp hạng 2–10: `Axel_1` (hạng 8), `Axel_9` (5), `Axel_7` (4), `Flip_1` (3), `Comb_2` (2), `Flip_10` (10).

## 5. Thông tin chạy

- Môi trường: Windows 10, Python 3.11.9, NVIDIA RTX 3050 Ti Laptop, 16 lõi CPU, code version `local`.
- Cấu hình: `default.yml; method=Anatomical; iterations=80; bone_weight=1.0`.
- Cột `Best_Method` là `Anatomical + SequenceRefine` ở tất cả 1.260 frame, tức là giá trị cố định, không phải lựa chọn theo từng frame.
- Thời gian chạy: 12:17:11 đến 12:50:17 ngày 12/10/2026.

## 6. Những chỗ tôi đã sửa lại

1. **Mâu thuẫn giữa bảng per-joint và bảng sequence.** Ban đầu tôi coi đây là điểm bất thường. Thực ra hai bảng dùng cách căn chỉnh khác nhau: `results.csv` căn từng frame, sequence-metrics căn một lần cho cả sequence. Xem mục 3.2 và 3.4.
2. **Giả thuyết "lệch đồng bộ gây sai số ở `Axel_2`" đã rút lại.** Dữ liệu đầy đủ không ủng hộ (mục 3.7).
3. **Nhận định "`all_camera_pairs.csv` đã cũ" chỉ đúng một phần.** File là nhật ký nhiều lần chạy; phiên bản cuối khớp cặp camera thực dùng ở 9/15 motion (mục 4).
4. **"Spine, Thorax, Head sai 85–117 mm ở mọi subject" là khái quát quá mức.** Chỉ Spine đúng ở cả 3 subject (94–117 mm); Thorax là 74–105 mm và Head là 56–97 mm (mục 3.3).
5. **Dòng không phải frame trong `results.csv`.** Mỗi file có một dòng `Summary_Mean` (Motion = `ALL`) và một dòng `End` ở cuối. Phải loại hai dòng này trước khi tính, nếu không các cột số bị hiểu thành chuỗi và trung bình bị lệch.

## 7. Kiểm tra chéo đã làm

- Trung bình tính lại từ 1.260 frame khớp với dòng `Summary_Mean` của cả 3 folder.
- Tỉ lệ che khuất theo frame khớp file per-joint: 0,887/17 = 5,2% (S1), 1,367/17 = 8,0% (S2), 3,305/17 = 19,4% (S3, file ghi 19,5%).
- Cặp camera `Cam_A`, `Cam_B` khớp giữa `results.csv` và `results_sequence_metrics.csv` ở cả 42 sequence.
- Không có frame trùng (Folder, Motion, Frame).

## 8. Giới hạn và câu hỏi còn mở

- Đây là sai số do chính pipeline báo cáo. Tôi không có ground truth gốc để kiểm tra độc lập, và không xem mã nguồn của phương pháp căn chỉnh.
- Mọi giải thích về nguyên nhân (Spine, Running_50, scale nhỏ hơn 1, khớp bị che ở S3) đều là giả thuyết.
- Trung bình trong file per-joint (S1: DLT 57,1, Method 52,7) khác trung bình theo frame (58,4 và 53,2) khoảng 1 mm. Tôi chưa giải thích được; có thể do cách lấy trung bình khác nhau.
- File không có từ điển dữ liệu; định nghĩa cột ở mục 2.2 là suy luận từ tên cột.

## 9. Gợi ý bước tiếp theo

- Kiểm tra hiệu chuẩn cặp camera 1–3 trong `Running_50`.
- Xem lại định nghĩa keypoint Spine, Thorax, Head và Pelvis.
- Thêm cột run id vào `all_camera_pairs.csv`, và xuất xếp hạng camera riêng cho S2 và S3.
- Báo cáo song song cả sai số căn từng frame và căn cả sequence, để tránh hiểu nhầm lợi thế của Method.
- Thêm từ điển dữ liệu cho các cột (đặc biệt đơn vị của sync delta).

## 10. Cách tái lập

Đọc `results.csv` mỗi folder, loại dòng tổng hợp, rồi ép kiểu số:

```python
import pandas as pd

folder = "S1"
df = pd.read_csv(f"AtheletePose3D_Report/{folder}/results.csv")
df = df[(df.Subject == folder) & ~df.Motion.isin(["ALL", "End"])].copy()

skip = {"Time", "Motion", "Subject", "Best_Method", "Python_Version", "OS_Type",
        "OS_Version", "Compute_Device", "Code version", "Notes",
        "Conf_A", "Conf_B", "Stereo_Conf"}
for c in df.columns:
    if c not in skip:
        df[c] = pd.to_numeric(df[c], errors="coerce")

print(df[["Baseline_DLT_MPJPE", "Selected_Method_MPJPE",
          "Baseline_DLT_PA", "Selected_Method_PA"]].mean().round(2))
```

Kết quả với `folder = "S1"` phải là 58,42 / 53,15 / 48,71 / 47,63.
