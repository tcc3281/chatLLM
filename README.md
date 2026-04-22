# chatLLM

Giao diện chat bằng Gradio để test model OpenAI-compatible, hỗ trợ:

- Chat nhiều lượt
- Upload hoặc paste nhiều ảnh
- Lấy danh sách model từ `/v1/models` theo `base_url`
- Chỉnh temperature, max tokens, system prompt
- Ẩn mục `Use via API` trên giao diện
- Lưu lịch sử chat theo từng lượt (kèm cấu hình model tại thời điểm gửi)

## Chạy bằng uv

```bash
uv sync
uv run main.py
```

Mở trình duyệt tại địa chỉ hiển thị trong terminal, mặc định là `http://localhost:7860`.

## Chạy bằng Docker

Dockerfile đã hỗ trợ cả 2 trường hợp:
- Có `uv.lock` -> dùng lock file (`uv sync --frozen --no-dev`)
- Chưa có `uv.lock` -> vẫn build được (`uv sync --no-dev`)

Build image:

```bash
docker build -t chatllm:latest .
```

Run container:

```bash
docker run --rm -p 7860:7860 --env-file .env -v chatllm_data:/app/data chatllm:latest
```

Hoặc dùng compose:

```bash
docker compose up -d --build
```

Lịch sử chat sẽ được lưu tại thư mục `/app/data/chat_history` trong container (đã map volume `chatllm_data` trong compose).
Mỗi dòng trong file `.jsonl` chứa:

- `timestamp`
- `session_id`
- `config` (model, base_url, temperature, max_tokens, system_prompt)
- `messages` gồm user/assistant, trong đó ảnh user được lưu dưới dạng data URL (base64)

## Biến môi trường

- `base_url`: base URL mặc định trong `.env`
- `api_key`: API key mặc định trong `.env`
- `OPENAI_API_KEY`: API key
- `OPENAI_BASE_URL`: base URL cho OpenAI-compatible server
- `HOST`: host lắng nghe, mặc định `0.0.0.0`
- `PORT`: cổng, mặc định `7860`
- `SHARE`: đặt `1` để bật Gradio share link
- `CHAT_HISTORY_DIR`: đường dẫn lưu lịch sử chat, mặc định `./data/chat_history`

