# Kiến trúc tổng thể hệ thống tái dựng pose 3D

## 1. Mục tiêu và phạm vi

Tài liệu này mô tả kiến trúc đang được triển khai trong repository: cấu hình, đọc pose 2D nhiều camera, đồng bộ thời gian, tái dựng pose 3D, tinh chỉnh theo chuỗi, đánh giá với ground truth (GT) và ghi kết quả.

**Lưu ý về phạm vi:** pipeline hiện tại không chạy YOLO để phát hiện pose từ video. File MP4 dùng để nhận diện camera/sequence; tọa độ pose 2D đã được suy luận trước và đọc từ các file NPY. Với mỗi subject/motion, hệ thống chọn một cặp camera tốt nhất. Nó không hợp nhất đồng thời tất cả camera.

GT chỉ dùng để tính sai số sau tái dựng. GT không tham gia chọn camera, ước lượng độ lệch đồng bộ hay tạo bone prior.

## 2. Luồng xử lý chính

~~~mermaid
flowchart TD
    A[configs/default.yml] --> B[main.py: đọc config và CLI]
    B --> C[scan_local_dataset]
    C --> D[Nhóm video theo subject và motion]
    C --> E[PoseRepository: NPY 2D và cache frame]
    D --> F[Chia sequence thành chunk]
    F --> G[pipeline.py: xử lý chunk]
    E --> G
    G --> H[Chọn cặp camera tốt nhất]
    H --> I[Đồng bộ toàn cục và ước lượng P1 P2]
    I --> J[Ước lượng bone prior không dùng GT]
    J --> K[Đồng bộ động theo frame]
    K --> L[Lọc frame và keypoint 2D]
    L --> M[DLT baseline]
    L --> N[Phương pháp được cấu hình]
    N --> O{DST physics?}
    O -->|Có| P[Fusion bằng chứng từ hai camera]
    P --> Q[Tối ưu ray bone symmetry anchor]
    O -->|Không| R[Triangulation của phương pháp khác]
    Q --> S[Sequence refinement nếu bật]
    R --> S
    M --> T[Đánh giá với GT]
    S --> T
    T --> U[CSV: metric và summary]
~~~

## 3. Sơ đồ các tầng module

~~~mermaid
flowchart LR
    subgraph CLI[CLI và cấu hình]
        MAIN[main.py]
        CFG[settings.py]
    end
    subgraph DATA[Dữ liệu]
        SCAN[io/data.py: scan_local_dataset]
        REPO[io/data.py: PoseRepository]
        GT[io/data.py: extract_gt_3d]
    end
    subgraph ORCH[Điều phối]
        PIPE[pipeline.py]
        REPORT[io/reporting.py]
    end
    subgraph ALG[Thuật toán]
        SYNC[algorithms/synchronization.py]
        UNC[algorithms/uncalibrated.py]
        GEO[algorithms/geometry.py]
        FUSE[algorithms/evidence_fusion.py]
        PHYS[algorithms/physics_refine.py]
        SEQ[algorithms/refinement.py]
    end
    MAIN --> CFG
    MAIN --> SCAN
    MAIN --> PIPE
    MAIN --> REPORT
    SCAN --> REPO
    SCAN --> GT
    PIPE --> REPO
    PIPE --> SYNC
    PIPE --> UNC
    PIPE --> GEO
    PIPE --> FUSE
    PIPE --> PHYS
    PIPE --> SEQ
    PIPE --> GT
    PIPE --> REPORT
    SYNC --> UNC
    SYNC --> GEO
~~~

## 4. Trách nhiệm các thành phần

| Thành phần | Trách nhiệm |
|---|---|
| main.py | Điểm vào CLI: đọc config, áp dụng override, quét dữ liệu, lọc subject/motion, chia chunk, gọi pipeline và kết thúc reporter. |
| settings.py | Đọc YAML thành các nhóm cấu hình, gán mặc định và kiểm tra giá trị. |
| io/data.py: scan_local_dataset | Tìm video camera, nhóm theo subject/motion và đọc manifest tiến độ nếu được cấu hình. |
| io/data.py: PoseRepository | Nạp pose 2D NPY theo nhu cầu; hỗ trợ memory map và cache frame LRU có giới hạn. |
| io/data.py: extract_gt_3d | Đọc marker 3D GT và ánh xạ marker thành skeleton 17 khớp. |
| pipeline.py | Điều phối chọn pair, đồng bộ, camera matrix, bone prior, hai nhánh tái dựng, refinement và đánh giá. |
| algorithms/synchronization.py | Tìm lệch frame; chấm chất lượng hình học, xương, đối xứng và confidence. |
| algorithms/uncalibrated.py | Ước lượng fundamental/essential matrix, pose tương đối và scale khi không có calibration cố định. |
| algorithms/geometry.py | Phép chiếu, dựng tia và triangulation hình học như DLT. |
| algorithms/evidence_fusion.py | Tạo và kết hợp bằng chứng cho mỗi khớp từ quan sát của hai camera trong DST. |
| algorithms/physics_refine.py | Tối ưu pose theo ray reprojection, bone, symmetry, anchor và các ràng buộc tùy chọn. |
| algorithms/refinement.py | Tinh chỉnh nhiều frame cùng sequence theo prior và độ trơn thời gian. |
| io/reporting.py | Ghi metric frame, summary, thông tin hệ thống và chẩn đoán sequence vào CSV. |

## 5. Dữ liệu đầu vào và skeleton

### 5.1 Các loại file

Đường dẫn được khai báo trong configs/default.yml:

- Dataset và pose 2D bắt đầu từ thư mục inputs.
- Thư mục con test set là data.
- Manifest tiến độ là manifest_progress.csv.
- Video có đuôi .mp4; metadata camera có đuôi .json.
- Ground truth có đuôi .npy.
- Pose 2D có hậu tố _coco.npy.
- Danh sách keyframe có hậu tố _key_frames.npy.

MP4 giúp phát hiện và ghép camera theo subject/motion. Pose 2D và keyframe được đọc từ NPY tương ứng. Manifest có thể lọc sequence chưa đạt ngưỡng confidence cấu hình.

### 5.2 COCO sang H36M 17 khớp

Pose 2D COCO được đổi sang thứ tự H36M 17 khớp. Một số điểm không có trực tiếp được suy ra từ trung bình hoặc hình học đơn giản: pelvis từ hai hông, thorax từ hai vai, spine từ pelvis và thorax.

| Chỉ số | Khớp |
|---:|---|
| 0 | Pelvis |
| 1–3 | Hông phải, gối phải, cổ chân phải |
| 4–6 | Hông trái, gối trái, cổ chân trái |
| 7–10 | Spine, thorax, neck, head/vertex |
| 11–13 | Vai trái, khuỷu trái, cổ tay trái |
| 14–16 | Vai phải, khuỷu phải, cổ tay phải |

Pose và GT phải cùng thứ tự khớp và cùng đơn vị. Sai mapping hoặc đơn vị có thể làm MPJPE không còn ý nghĩa dù pipeline vẫn chạy.

## 6. Quy trình chi tiết

### Bước 1 — Đọc cấu hình

main.py đọc configs/default.yml, áp dụng override từ CLI rồi tạo cấu hình runtime. Cấu hình gồm input/output, pipeline, đồng bộ, tái dựng, DST physics và sequence refinement.

Một số giá trị hiện tại:

| Nhóm | Giá trị |
|---|---|
| Pipeline | max_pairs 15, chunk_size 5, frames_per_pair 30 |
| Đồng bộ thô | sync_max_offset 45, sync_samples 5, coarse_step 7, local_radius 2 |
| Đồng bộ động | dynamic_local_radius 5, max_candidate_pairs 132, max_dynamic_sync_score 155 |
| Kiểm tra đồng bộ | min confidence 0.25, tối thiểu 8 joint hợp lệ, min pair valid ratio 0.5 |
| Keypoint | min joint confidence 0.3, tối thiểu 8 joint hợp lệ |
| Bone prior | sequence_median, tối thiểu 5 mẫu, confidence 0.4 |
| Phương pháp | dst_physics, 80 iterations |
| Sequence refinement | bật, tối đa 30 lần đánh giá, giới hạn drift 360 mm |

Đây là giá trị trong YAML tại thời điểm viết; override CLI có thể làm runtime khác đi.

### Bước 2 — Quét dataset và chia chunk

Dataset scanner gom các video theo subject và motion rồi ghép đường dẫn camera. Sequence thiếu dữ liệu hoặc không đạt điều kiện manifest sẽ bị bỏ qua. Các sequence hợp lệ được chia theo chunk_size.

PoseRepository nạp NPY theo nhu cầu, có thể dùng memory mapping và giữ frame gần đây trong LRU cache để hạn chế đọc lặp.

### Bước 3 — Chọn cặp camera

Với từng motion có nhiều camera, pipeline chấm các cặp ứng viên dựa trên khả năng tạo pose 3D hợp lý:

- Confidence và số lượng khớp hợp lệ.
- Sai khác hình học hai view.
- Độ dài và tính nhất quán xương.
- Đối xứng trái/phải và chất lượng đồng bộ.

Cặp điểm cao nhất được dùng cho sequence. Bước này không dùng GT. Khi có hơn hai camera, các camera khác không góp trực tiếp vào cùng một phép dựng.

### Bước 4 — Đồng bộ và ước lượng camera

Hệ thống tìm lệch frame trong phạm vi cấu hình. Đồng bộ ban đầu chấm một số frame mẫu, sau đó coarse-to-local thu hẹp quanh offset tốt.

uncalibrated.py có thể ước lượng hình học hai camera:

1. Fundamental matrix bằng normalized 8-point và xử lý outlier.
2. Essential matrix từ fundamental matrix và ước lượng nội tại gần đúng.
3. Pose tương đối bằng SVD và kiểm tra cheirality.
4. Scale metric từ bone prior để chuyển hình học tương đối sang đơn vị thực.

Nếu ước lượng thành công, P1/P2 được tái sử dụng trong đồng bộ và triangulation. Pipeline còn có thể ước lượng camera matrix trên subset frame của motion context và thay ma trận hiện tại nếu nghiệm mới hợp lệ.

### Bước 5 — Tạo bone prior

Mặc định là sequence_median: hệ thống tạo DLT pose trên các frame cùng motion, lọc theo confidence rồi lấy thống kê độ dài xương. GT không được dùng.

Nếu thiếu mẫu đủ tốt, hệ thống dự phòng sang độ dài xương H36M được scale theo chiều cao subject. default.yml hiện có chiều cao cho S1, S2 và S3; map này có thể được ưu tiên hơn chiều cao tự động nếu có.

### Bước 6 — Đồng bộ động và lọc frame

Sau offset toàn cục, thuật toán quy hoạch động tìm offset thay đổi theo frame trong vùng ứng viên cấu hình. Mục đích là giảm ảnh hưởng của drift hoặc nhịp video không đều; phạm vi tìm kiếm vẫn bị giới hạn bởi các ngưỡng đã đặt.

Pipeline ưu tiên keyframe nếu file có sẵn, giới hạn số frame theo frames_per_pair rồi lọc theo confidence và số joint hợp lệ. Frame không đạt điều kiện đồng bộ hoặc pose sẽ bị loại. GT chỉ được lấy cho frame dùng tính metric.

### Bước 7 — Chạy baseline và phương pháp chính

Cả hai nhánh dùng cùng cặp camera, frame, keypoint và projection:

- DLT baseline: triangulation tuyến tính chuẩn, dùng làm mốc. Phép DLT cơ bản không cân trọng số theo confidence detector.
- Phương pháp cấu hình: mặc định dst_physics; pipeline điều phối phương pháp khác nếu config chỉ định.

Như vậy có thể so sánh hai phương pháp trên cùng đầu vào hơn. Frame thiếu dữ liệu hoặc triangulation không hợp lệ có thể không có kết quả ở một hay cả hai nhánh.

### Bước 8 — DST fusion và tối ưu physics

DST tạo bằng chứng cho từng khớp từ hai camera:

1. Confidence detector biểu diễn độ tin cậy của mỗi quan sát 2D.
2. Sai khác hình học giữa hai tia biểu diễn độ phù hợp.
3. Bone consistency quanh nghiệm khởi tạo bổ sung thông tin cấu trúc.
4. Dempster-style combination kết hợp bằng chứng, tạo trọng số và đánh dấu outlier.

Đây là fusion của hai quan sát trong cặp camera, không phải fusion nhiều camera. dst_physics khởi tạo pose 3D từ DLT rồi tối ưu các thành phần:

- Ray/data loss: giảm khoảng cách giữa pose 3D và tia quan sát, dùng Huber và confidence weight.
- Bone loss: giữ độ dài xương gần prior.
- Symmetry loss: hạn chế chênh lệch bất hợp lý giữa cặp xương trái/phải.
- Anchor loss: hạn chế nghiệm đi quá xa DLT ban đầu.
- Ground constraint: tùy chọn; ground_z hiện null.
- NIMBLE prior: tùy chọn; hiện tắt.

Các trọng số và iteration nằm trong dst_physics config. Tăng số vòng lặp không tự động giảm lỗi: prior sai, keypoint nhiễu, lệch đồng bộ hoặc weight mất cân bằng đều có thể làm kết quả xấu hơn.

### Bước 9 — Tinh chỉnh toàn chuỗi

Khi bật, các frame được gom theo subject, motion và cặp camera để tối ưu chung. Objective dùng root/relative data, bone median, symmetry, reprojection và smoothness.

Smoothness gia tốc chỉ áp dụng khi frame được chọn liên tiếp. Với keyframe thưa, các frame nhảy cách quãng không bị coi như liên tiếp để áp ràng buộc này.

Kết quả được chấp nhận nếu hữu hạn và median drift so với chuỗi đầu vào không vượt max_drift_mm. Nếu không đạt, pipeline quay lại chuỗi trước refinement. Refinement cập nhật phương pháp được chọn; DLT baseline vẫn được giữ làm đối chứng.

### Bước 10 — Metric và báo cáo

Metric được tính sau khi lấy GT tương ứng:

- MPJPE: dịch pelvis về gốc rồi tính khoảng cách trung bình trên joint 1–16, đơn vị mm. Không căn chỉnh rotation hoặc scale.
- PA-MPJPE: căn chỉnh Procrustes gồm scale, rotation và translation trước khi đo.
- Các nhóm Clear/Derived được ghi riêng khi phù hợp.
- rigid_mpjpe tồn tại trong code nhưng không thuộc cột CSV tiêu chuẩn hiện tại.

Reporter ghi metric từng frame, summary trung bình Summary_Mean, dấu kết thúc, thông tin hệ thống và chẩn đoán sequence vào outputs/results.csv, theo chế độ overwrite/append. Khi so sánh, cần xem cả số frame hợp lệ và metric; trung bình trên các tập frame khác nhau có thể gây kết luận sai.

## 7. Sơ đồ nhánh tái dựng

~~~mermaid
flowchart TD
    Y[default.yml và CLI overrides] --> D[Video, pose NPY, GT, manifest]
    D --> C[Chọn pair và camera projection]
    C --> K[Keyframe và dynamic sync]
    K --> M1[DLT baseline]
    K --> M2[Phương pháp cấu hình]
    M2 -->|dst_physics| F[Evidence fusion]
    F --> O[Physics optimization]
    M2 -->|Phương pháp khác| T[Triangulation tương ứng]
    O --> R[Sequence refinement nếu bật]
    T --> R
    M1 --> E[MPJPE và PA-MPJPE]
    R --> E
    GT[Ground truth] --> E
    E --> CSV[outputs/results.csv]
~~~

## 8. Cách đọc DST so với DLT

Để kết luận công bằng:

1. So sánh cùng sequence, pair, frame và thứ tự joint.
2. Xác nhận cùng tập frame GT hợp lệ, đồng thời ghi số frame mỗi phương pháp sử dụng.
3. Xem cả MPJPE và PA-MPJPE. PA tốt nhưng MPJPE xấu có thể cho thấy hình dáng tương đối ổn nhưng scale hoặc vị trí còn lệch.
4. Bật/tắt sequence refinement có kiểm soát để tách cải thiện từ DST physics khỏi cải thiện từ tối ưu chuỗi.
5. Đối chiếu bone prior, data_weight, bone_weight, anchor_weight, symmetry_weight, confidence threshold và iteration.
6. Kiểm tra offset đồng bộ, joint hợp lệ, reprojection error và tỷ lệ frame bị loại.

DST physics thêm prior và bước tối ưu nên không được đảm bảo luôn thắng DLT. Anchor hoặc bone prior không phù hợp có thể kéo nghiệm khỏi kết quả DLT tốt. Khi dữ liệu 2D và đồng bộ tốt, DLT vẫn có thể là baseline mạnh.

## 9. Giới hạn khi diễn giải kết quả

- Pose 2D được đọc từ NPY; sai số detector upstream ảnh hưởng trực tiếp đến kết quả.
- Mỗi motion dùng một cặp camera; chưa hợp nhất đồng thời từ ba camera trở lên.
- Hình học camera được ước lượng từ pose, nên cần keypoint đủ tốt và chuyển động đủ đa dạng.
- Dynamic sync chỉ tìm trong phạm vi offset và số ứng viên cấu hình.
- Bone prior median có thể sai nếu sequence có nhiều pose lỗi hoặc quá ít frame.
- Ground constraint và NIMBLE đang tắt theo config hiện tại.
- Confidence threshold thay đổi tập frame dùng tính metric; nên so sánh trên cùng frame hợp lệ.

## 10. Sơ đồ phụ thuộc rút gọn

~~~mermaid
flowchart LR
    CFG[Config] --> INPUT[Dataset và pose NPY]
    INPUT --> SYNC[Pair selection và synchronization]
    SYNC --> CAM[Uncalibrated camera matrices]
    CAM --> TRI[DLT và triangulation chính]
    TRI --> DST[DST fusion và physics]
    DST --> SEQ[Sequence refinement]
    TRI --> METRIC[Đánh giá bằng GT]
    SEQ --> METRIC
    METRIC --> REPORT[CSV report]
    GT[Ground truth] --> METRIC
    PRIOR[Bone prior từ sequence hoặc height] --> DST
    PRIOR --> SYNC
~~~
