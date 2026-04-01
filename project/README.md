# Imbalanced IDS Benchmark Project

Project này hiện đã được tinh gọn theo một luồng chính:

- benchmark theo hướng paper-style
- dùng dữ liệu dạng bảng cho IDS/anomaly detection
- tập trung vào xử lý mất cân bằng

## Chạy mặc định

```bash
python main.py
```

Lệnh trên sẽ dùng:

- `paper_style_config.yaml`

## Chạy chỉ định config

```bash
python main.py --config paper_style_config.yaml
```

## Các file quan trọng nhất

- `main.py`
- `paper_style_config.yaml`
- `src/`
- `realtime_inference.py`

## Output

Kết quả sẽ được lưu dưới thư mục output khai báo trong config, mặc định là:

- `paper_style_outputs/`
