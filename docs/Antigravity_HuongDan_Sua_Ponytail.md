# Hướng dẫn Antigravity sửa dự án theo báo cáo Ponytail

## 1. Mục tiêu

Hãy sửa dự án:

`compare_Anatomical_vs_2CamSync_at_AthletePose3D_yolo`

dựa trên báo cáo review Ponytail ngày 07/10/2026.

Phương pháp cần tuân thủ:

- Ưu tiên sửa **root cause** thay vì workaround.
- Giữ kiến trúc hiện tại nếu chưa có lý do rõ ràng để thay đổi.
- Không over-engineering.
- Không thêm abstraction/class/config không cần thiết.
- Ưu tiên Standard Library / API native khi phù hợp.
- Không thay đổi thuật toán nghiên cứu ngoài phạm vi các lỗi được nêu nếu không cần thiết.
- Sau mỗi nhóm thay đổi phải kiểm tra tính đúng đắn và regression.
- Không chỉ sửa cho hết warning; phải bảo toàn behavior đúng của pipeline.

Báo cáo gốc xác định 6 vấn đề chính:

1. RCE do `allow_pickle=True`.
2. Phụ thuộc Ground Truth trong đồng bộ camera.
3. Smoothness loss sai khi keyframe không liên tục.
4. Trạng thái `best_pose` bị lệch.
5. Vòng lặp bone loss gây overhead CUDA.
6. Custom cache gây technical debt.

---

# 2. Thứ tự ưu tiên

Thực hiện theo đúng thứ tự:

1. **HIGH — Security:** loại bỏ `allow_pickle=True`.
2. **HIGH — Correctness / Architecture:** loại bỏ phụ thuộc Ground Truth khỏi camera synchronization.
3. **MEDIUM — Mathematical correctness:** sửa smoothness loss cho keyframe không liên tục.
4. **MEDIUM — Optimization correctness:** sửa trạng thái `best_pose`.
5. **MEDIUM — Performance:** vector hóa bone loss.
6. **LOW — Maintainability:** thay custom cache bằng cơ chế chuẩn.

**Không thực hiện tối ưu hóa trước khi xử lý hai vấn đề HIGH.**

---

# 3. Vấn đề 1 — Security: `allow_pickle=True`

## Vị trí

- `src/athlete_pose3d/algorithms/synchronization.py`
- `src/athlete_pose3d/pipeline.py`

Theo báo cáo, các đoạn hiện tại sử dụng:

```python
np.load(..., allow_pickle=True)
```

Báo cáo xác định đây là rủi ro arbitrary code execution khi dữ liệu `.npy` không đáng tin cậy.

## Yêu cầu sửa

Thay các trường hợp liên quan đến pose / ground truth dạng numeric array thành:

```python
allow_pickle=False
```

Ví dụ:

```python
counts = tuple(
    len(
        np.load(
            camera["ground_truth_path"],
            mmap_mode="r",
            allow_pickle=False,
        )
    )
    for camera in (camera_a, camera_b)
)
```

và:

```python
raw_gt = np.load(gt_path, allow_pickle=False)
```

## Kiểm tra bắt buộc

Sau khi sửa:

1. Search toàn project để tìm:

   ```text
   allow_pickle=True
   ```

2. Nếu còn trường hợp nào, kiểm tra xem có thực sự cần pickle hay không.

3. Không được giữ `allow_pickle=True` chỉ để tránh lỗi loading.

4. Kiểm tra shape và dtype của dữ liệu sau khi load không thay đổi ngoài ý muốn.

---

# 4. Vấn đề 2 — Camera synchronization phụ thuộc Ground Truth

## Vị trí

`src/athlete_pose3d/algorithms/synchronization.py`

Hàm liên quan:

```python
_pair_frame_counts(...)
```

Báo cáo chỉ ra rằng synchronization đang kiểm tra:

```python
camera["ground_truth_path"]
```

và lấy số lượng frame từ Ground Truth.

Điều này làm pipeline thất bại khi chạy trên video thực tế không có MoCap Ground Truth.

## Root cause cần sửa

Camera synchronization cần xác định số frame dựa trên dữ liệu video/pose đầu vào, không được bắt buộc Ground Truth.

## Hướng sửa

Nếu kiến trúc hiện tại có `PoseRepository`, ưu tiên sử dụng repository đó.

Ví dụ theo báo cáo:

```python
def _pair_frame_counts(camera_a, camera_b, repository: PoseRepository):
    try:
        p_a = repository.pose_path(camera_a["video_path"])
        p_b = repository.pose_path(camera_b["video_path"])

        if p_a.exists() and p_b.exists():
            n_a = len(np.load(p_a, mmap_mode="r", allow_pickle=False))
            n_b = len(np.load(p_b, mmap_mode="r", allow_pickle=False))

            return (n_a, n_b) if min(n_a, n_b) >= 2 else None

    except (OSError, ValueError):
        pass

    return None
```

## Quan trọng

**Không copy nguyên đoạn trên một cách máy móc.**

Trước tiên hãy đọc:

- signature hiện tại của `_pair_frame_counts`;
- cách `PoseRepository` đang được khởi tạo;
- cách `video_path` và pose path đang được quản lý;
- tất cả nơi gọi `_pair_frame_counts`.

Sau đó sửa theo kiến trúc thực tế của project.

## Kiểm tra bắt buộc

Phải kiểm tra ít nhất hai trường hợp.

### Trường hợp A — Có Ground Truth

Pipeline vẫn phải hoạt động như trước.

### Trường hợp B — Không có Ground Truth

Synchronization vẫn phải có khả năng lấy frame count từ pose/video metadata và không fail chỉ vì thiếu:

```text
ground_truth_path
```

Không được biến lỗi thiếu pose thành lỗi giả tạo do Ground Truth.

---

# 5. Vấn đề 3 — Smoothness loss sai với keyframe thưa

## Vị trí

- `src/athlete_pose3d/algorithms/refinement.py`
- `src/athlete_pose3d/pipeline.py`

Đoạn hiện tại dùng sai phân bậc hai:

```python
centered = poses - poses[:, :1]
smooth = centered[:-2] - 2.0 * centered[1:-1] + centered[2:]
```

Báo cáo chỉ ra rằng `_sample_key_frames(..., limit=30)` có thể tạo các frame cách xa nhau đáng kể.

Sai phân bậc hai ở đây đang ngầm giả định các sample liên tiếp và có khoảng thời gian đều nhau.

## Yêu cầu sửa tối thiểu

Chỉ áp dụng smoothness loss hiện tại khi frame thực sự liên tiếp:

```python
frame_nums = np.array([res["frame"] for _, res in items])

is_contiguous = (
    len(frame_nums) >= 3
    and np.all(np.diff(frame_nums) == 1)
)

if is_contiguous:
    centered = poses - poses[:, :1]

    smooth = (
        centered[:-2]
        - 2.0 * centered[1:-1]
        + centered[2:]
    )

    residuals.append(
        (weights["smoothness"] * smooth).reshape(-1)
    )
```

## Nhưng trước khi sửa

Kiểm tra:

1. `items` thực sự chứa frame index nào.
2. `res["frame"]` có đúng là frame number gốc hay không.
3. smoothness residual đang được dùng ở đâu.
4. Có test nào hiện tại phụ thuộc vào smoothness loss không.

## Không được làm

Không tự ý thay công thức bằng một temporal model phức tạp hơn nếu báo cáo không yêu cầu.

Nếu muốn hỗ trợ keyframe không liên tục bằng `dt` normalization, đó là một thay đổi thuật toán riêng và phải được đánh giá riêng.

---

# 6. Vấn đề 4 — `best_pose` lưu sai trạng thái

## Vị trí

`src/athlete_pose3d/algorithms/physics_refine.py`

Đoạn hiện tại có dạng:

```python
best_pose, best_loss = pose.detach().clone(), float("inf")

for _ in range(iterations):
    candidate = pose.detach().clone()

    loss = _optimize_biomechanics_step(...)

    if np.isfinite(loss) and loss < best_loss:
        best_pose, best_loss = candidate, loss
```

## Root cause

`candidate` được clone trước `_optimize_biomechanics_step()`.

Nếu optimizer cập nhật `pose` bên trong hàm đó, `candidate` là trạng thái trước update nhưng `loss` tương ứng với bước tối ưu.

Điều này làm `best_pose` có thể không phải pose tương ứng với `best_loss`.

## Yêu cầu sửa

Không clone candidate trước optimization.

Ưu tiên:

```python
loss = _optimize_biomechanics_step(...)

if np.isfinite(loss) and loss < best_loss:
    best_pose = pose.detach().clone()
    best_loss = loss
```

## Kiểm tra

Xác minh:

- `_optimize_biomechanics_step()` thực sự update `pose`.
- `loss` được tính trước hay sau `optimizer.step()`.
- `best_loss` và `best_pose` phải biểu diễn cùng một trạng thái nghiệm.

Nếu semantics của `_optimize_biomechanics_step()` khác với báo cáo, phải điều chỉnh implementation để đảm bảo invariant:

> `best_pose` là pose tương ứng với `best_loss`.

---

# 7. Vấn đề 5 — Vector hóa bone loss

## Vị trí

- `src/athlete_pose3d/algorithms/physics_refine.py`
- `src/athlete_pose3d/algorithms/geometry.py`

Hiện tại code lặp Python qua từng bone:

```python
losses = []

for (a, b), target_len in bone_lengths.items():
    if target_len > 0:
        actual_len = torch.linalg.vector_norm(
            points_3d[..., a, :] - points_3d[..., b, :],
            dim=-1,
        )

        target_t = points_3d.new_tensor(target_len)

        loss_item = functional.huber_loss(
            actual_len,
            target_t,
            delta=delta,
        )

        losses.append(loss_item)

return torch.sum(torch.stack(losses))
```

Báo cáo xác định đây là performance debt do nhiều kernel nhỏ và Python loop.

## Yêu cầu

Tìm cách vector hóa dựa trên pattern vectorization đã tồn tại trong project.

Theo báo cáo, project đã có pattern dùng tensor index:

```python
b_idx_a
b_idx_b
```

Ưu tiên tái sử dụng pattern đó thay vì tạo abstraction mới.

Ví dụ mục tiêu:

```python
actual_lens = torch.linalg.vector_norm(
    points_3d[..., b_idx_a, :]
    - points_3d[..., b_idx_b, :],
    dim=-1,
)

bone_loss = torch.sum(
    functional.huber_loss(
        actual_lens,
        target_lens,
        delta=delta,
    )
    * bone_weights_tensor
)
```

## Rất quan trọng

Không được chỉ thay code để "trông vectorized".

Phải đảm bảo:

- thứ tự bone không thay đổi;
- target length đúng với từng bone;
- weight đúng bone;
- bỏ qua bone có target length <= 0 như behavior cũ;
- shape broadcasting đúng;
- dtype/device đúng;
- gradient vẫn truyền được.

## Kiểm tra performance

Nếu project có benchmark hiện tại, chạy benchmark trước và sau.

Không được tuyên bố "10–20x" nếu chưa benchmark thực tế trên project hiện tại.

---

# 8. Vấn đề 6 — Custom LRU cache

## Vị trí

`src/athlete_pose3d/io/data.py`

Code hiện tại:

```python
if len(self._frames) > self.cache_limit:
    for stale in list(self._frames)[:50]:
        self._frames.pop(stale)
```

Báo cáo xác định đây không phải LRU chuẩn.

## Yêu cầu

Đầu tiên xác định behavior thực tế của cache:

- key nào được insert;
- key nào được access;
- cache có cần true LRU không;
- `cache_limit` có ý nghĩa chính xác gì;
- có code nào phụ thuộc vào thứ tự dictionary không.

Nếu thực sự cần LRU, dùng:

```python
from collections import OrderedDict
```

Ví dụ:

```python
self._frames = OrderedDict()
```

và eviction:

```python
while len(self._frames) > self.cache_limit:
    self._frames.popitem(last=False)
```

Nếu có access existing key, phải cập nhật thứ tự bằng:

```python
self._frames.move_to_end(key)
```

## Lưu ý

Không sử dụng `functools.lru_cache` nếu object/state hiện tại của `PoseRepository` không phù hợp với decorator.

Mục tiêu là đơn giản hóa code nhưng vẫn giữ behavior đúng.

---

# 9. Quy trình thực hiện

Antigravity hãy thực hiện theo các bước sau.

## Step 1 — Đọc code trước khi sửa

Đọc toàn bộ context của các file:

```text
src/athlete_pose3d/algorithms/synchronization.py
src/athlete_pose3d/pipeline.py
src/athlete_pose3d/algorithms/refinement.py
src/athlete_pose3d/algorithms/physics_refine.py
src/athlete_pose3d/algorithms/geometry.py
src/athlete_pose3d/io/data.py
experiments/run_dst_ablation_experiment.py
```

Đặc biệt tìm tất cả nơi gọi các hàm bị ảnh hưởng.

## Step 2 — Sửa Security

Sửa toàn bộ `allow_pickle=True` liên quan đến numeric `.npy`.

## Step 3 — Sửa Ground Truth dependency

Tách frame counting khỏi Ground Truth.

## Step 4 — Sửa mathematical correctness

Sửa smoothness condition.

## Step 5 — Sửa optimizer state

Đảm bảo `best_pose` tương ứng với `best_loss`.

## Step 6 — Vectorize performance bottleneck

Tái sử dụng pattern tensor indexing đã có.

## Step 7 — Simplify cache

Chỉ sửa nếu behavior thực tế xác nhận đây là cache cần LRU/FIFO.

---

# 10. Testing bắt buộc

Sau khi sửa, không chỉ chạy formatter.

Hãy tìm test hiện có:

```text
tests/
pytest
```

và chạy test suite phù hợp.

## Security test

Xác nhận không còn:

```text
allow_pickle=True
```

trong các đường code liên quan.

## Synchronization test

Test:

1. Có GT.
2. Không có GT.
3. Có pose files.
4. Thiếu pose files.
5. Pose files không hợp lệ.

## Smoothness test

Test:

1. Frame liên tiếp:

   ```text
   10, 11, 12
   ```

   → smoothness được áp dụng.

2. Frame không liên tiếp:

   ```text
   10, 20, 30
   ```

   → không áp dụng công thức smoothness hiện tại.

## `best_pose` test

Tạo trường hợp optimizer cập nhật pose rõ ràng và xác nhận:

```text
best_pose == pose tại thời điểm best_loss
```

## Vectorization test

So sánh implementation cũ và mới trên cùng input:

```text
old_loss ≈ new_loss
```

trong tolerance số học phù hợp.

Kiểm tra cả gradient nếu bone loss được dùng trong optimization.

## Cache test

Kiểm tra:

- cache không vượt quá `cache_limit`;
- eviction đúng;
- cache hit hoạt động;
- thứ tự LRU đúng nếu dùng `OrderedDict`.

---

# 11. Regression protection

Sau khi sửa, phải kiểm tra rằng các chức năng nghiên cứu chính vẫn hoạt động:

- camera synchronization;
- 2-camera pose processing;
- refinement;
- biomechanics optimization;
- bone-length constraint;
- sequence refinement;
- pipeline chạy với dataset hiện tại.

Không được tự ý thay đổi:

- format dữ liệu;
- model architecture;
- loss weights;
- dataset split;
- evaluation metric;
- experimental protocol;

trừ khi thay đổi đó trực tiếp cần thiết để sửa một lỗi đã nêu.

---

# 12. Nguyên tắc khi gặp bất đồng với báo cáo

Báo cáo là cơ sở để sửa nhưng không được sửa mù quáng.

Nếu code thực tế khác với mô tả trong báo cáo:

1. Đọc implementation hiện tại.
2. Xác định behavior thực tế.
3. Xác định root cause.
4. Chỉ sửa phần cần thiết.
5. Ghi rõ trong final report nếu đề xuất trong báo cáo không thể áp dụng nguyên trạng.

Không được tạo code giả chỉ để khớp với báo cáo.

---

# 13. Báo cáo cuối cùng Antigravity phải trả về

Sau khi hoàn thành, trả về bảng:

| STT | Vấn đề | Đã sửa? | File thay đổi | Test | Kết quả |
|---|---|---|---|---|---|
| 1 | `allow_pickle=True` | | | | |
| 2 | GT dependency | | | | |
| 3 | Smoothness | | | | |
| 4 | `best_pose` | | | | |
| 5 | Bone loss | | | | |
| 6 | Cache | | | | |

Sau bảng phải ghi:

### Files changed

Liệt kê chính xác các file đã sửa.

### Tests executed

Liệt kê command và kết quả.

### Remaining risks

Chỉ ghi những rủi ro thực sự còn tồn tại.

### Research behavior

Xác nhận các thay đổi có làm thay đổi behavior/experimental protocol hay không.

---

# 14. Điều kiện hoàn thành

Chỉ coi task hoàn thành khi:

- [ ] Security issue đã được xử lý.
- [ ] Pipeline không còn phụ thuộc bắt buộc vào Ground Truth để synchronization.
- [ ] Smoothness loss không áp dụng sai cho sparse keyframes.
- [ ] `best_pose` đồng bộ với `best_loss`.
- [ ] Bone loss được vector hóa nếu behavior có thể giữ nguyên.
- [ ] Cache được đơn giản hóa mà không làm thay đổi behavior.
- [ ] Test/regression đã chạy.
- [ ] Không có thay đổi ngoài phạm vi cần thiết.
- [ ] Không có `allow_pickle=True` còn sót lại trong các đường code liên quan.
- [ ] Không tạo abstraction không cần thiết.

**Ưu tiên correctness và security cao hơn performance. Không hy sinh tính đúng đắn của thuật toán để đạt benchmark nhanh hơn.**
