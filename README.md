# AthletePose3D: local two-camera benchmark

Pipeline so sánh các phương pháp dựng pose 3D từ hai camera. Dự án chỉ chạy local và ghi kết quả ra CSV.

## Cấu trúc

```text
.
├── configs/
│   └── default.yml                 # Toàn bộ cấu hình runtime
├── src/athlete_pose3d/
│   ├── main.py                     # Điểm chạy duy nhất
│   ├── settings.py                 # Đọc và kiểm tra YAML
│   ├── pipeline.py                 # Điều phối benchmark
│   ├── algorithms/                 # Hình học, đồng bộ, refinement
│   └── io/                         # Đọc dữ liệu local, ghi CSV
└── tests/
```

Mọi giá trị có thể thay đổi khi chạy nằm trong `configs/default.yml`; mã Python chỉ giữ schema và validation.

File này gồm bốn nhóm:

- `input`: thư mục dữ liệu, tên file, phần mở rộng, hậu tố pose 2D và ngưỡng lọc manifest.
- `output`: đường dẫn CSV và quyền ghi đè.
- `pipeline`: giới hạn xử lý, đồng bộ, confidence, chiều cao cơ thể và sequence refinement.
- `method`: thuật toán triangulation cùng các tham số riêng.

## Chuẩn bị dữ liệu

Các đường dẫn tương đối trong config được tính từ project root:

```text
data/
├── unzipped/
│   └── data/data/test_set/
│       └── S3/
│           ├── Javelin_20_cam_1.mp4
│           ├── Javelin_20_cam_1.json
│           └── Javelin_20_cam_1.npy
└── pose2d_yolo26x_h36m/
    ├── cam_param.json
    ├── manifest_progress.csv       # Không bắt buộc
    └── S3/
        ├── Javelin_20_cam_1_h36m_yolo26x.npy
        └── Javelin_20_cam_1_key_frames.npy
```

## Cài đặt và chạy

```powershell
python -m pip install -e .
python -m athlete_pose3d.main
```

Hoặc dùng command được cài cùng package:

```powershell
athlete-pose3d
```

Chương trình tự dùng `configs/default.yml`. Có thể override đường dẫn mà không sửa file:

```powershell
python -m athlete_pose3d.main `
  --dataset-root D:/datasets/AthletePose3D/unzipped `
  --pose2d-root D:/datasets/pose2d_yolo26x_h36m `
  --output outputs/results.csv
```

Nếu output đã tồn tại, thêm `--overwrite` để ghi lại. Muốn thử một cấu hình ngoài repo, truyền `--config path/to/config.yml`.

## Chọn phương pháp

Sửa `method.name` trong `configs/default.yml` thành một trong:

- `dlt`
- `confidence_algebraic`
- `ransac_dlt`
- `iterative_refine`
- `anatomical`
- `dst_anatomical` (Dempster-Shafer Theory Evidence Fusion)
- `physics_refine` (Nimble Physics & Biomechanical Priors)
- `dst_physics` (Unified SOTA: Dempster-Shafer Theory + Nimble Physics)

Hoặc truyền trực tiếp qua CLI với flag `--method <tên_phương_pháp>`.

Các tham số riêng của phương pháp đặt tại `method.options`.

## Cấu hình DST + Physics

Cấu hình runtime duy nhất nằm trong `configs/default.yml`. Các trọng số refinement, ngưỡng ray loss và cách tạo bone prior được đọc từ file này rồi truyền qua pipeline tới thuật toán. `bone_prior: sequence_median` ước lượng prior từ các frame DLT của sequence hiện tại; đây là chế độ thích nghi theo sequence, không phải prior subject-level đóng băng từ tập Dev.
