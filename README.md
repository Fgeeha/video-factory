# video-factory

Собирает вертикальный (9:16) клип для VK Клипов из фото + текста:
фото → Ken Burns (наплыв/зум через ffmpeg) → озвучка (piper, офлайн, CPU) →
автосубтитры (faster-whisper, CPU) → микс с фоновой музыкой.

Базовый пайплайн работает без GPU (CPU-only). Опционально сцены можно
генерировать нейросетями на локальной GPU (ComfyUI) вместо реальных фото —
см. раздел ниже.

## Установка

```bash
sudo apt-get install -y ffmpeg   # нужен ffmpeg с libass (стандартная сборка apt подходит)

cd video-factory
uv sync

# голос Piper (RU): скачать .onnx + .onnx.json с
# https://github.com/rhasspy/piper/blob/master/VOICES.md
# и положить в voices/, например voices/ru_RU-irina-medium.onnx(.json)
```

## Использование

1. Положить фото в `shots/`, фоновую музыку (опционально) в `assets/`.
2. Скопировать `example.config.yaml`, отредактировать текст озвучки, список фото, путь к музыке.
3. Запустить:

```bash
uv run make_clip.py example.config.yaml
```

Результат — `out/<имя>.mp4`.

## Проверка логики без ffmpeg/piper/ComfyUI

```bash
uv run test_make_clip.py
```

## Генерация кадров нейросетью на локальной GPU (опционально)

Вместо `file:` в сцене можно сгенерировать кадр или короткое видео через
локально запущенный ComfyUI (проверено на RTX 5070 Ti, 16GB VRAM):

```yaml
images:
  - generate_image:                              # кадр с нуля (Z-Image Turbo)
      prompt: "certified truck service center signage, industrial garage entrance, realistic"
  - generate_video:                              # видео с нуля (Wan2.2 T2V)
      prompt: "mechanic changing a brake pad on a heavy truck wheel, close up, realistic"
      length: 33   # кадров при 16fps; 33 ~= 2с, 81 ~= 5с (нужно 4n+1)
  - generate_video:                              # оживить реальное фото (Wan2.2 I2V)
      image: shots/03-brakes.jpg
      prompt: "the mechanic looks up and smiles at the camera, subtle motion"
      length: 33
  - file: shots/04-lift.jpg                      # обычное фото, без AI
```

Требуется:

1. Запущенный ComfyUI: `cd ComfyUI && venv/bin/python main.py` (по умолчанию
   `http://127.0.0.1:8188`, переопределяется `--comfy-url` или `COMFYUI_URL`).
2. Модели в `ComfyUI/models/` (репаковки Comfy-Org на HuggingFace):
   - `diffusion_models/z_image_turbo_bf16.safetensors` + `text_encoders/qwen_3_4b.safetensors` + `vae/ae.safetensors`
   - T2V: `diffusion_models/wan2.2_t2v_{high,low}_noise_14B_fp8_scaled.safetensors` + `loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_{high,low}_noise.safetensors`
   - I2V (для `generate_video` с `image:`): `diffusion_models/wan2.2_i2v_{high,low}_noise_14B_fp8_scaled.safetensors` + `loras/wan2.2_i2v_lightx2v_4steps_lora_v1_{high,low}_noise.safetensors`
   - Общие для Wan2.2: `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` + `vae/wan_2.1_vae.safetensors`

Ориентировочное время на RTX 5070 Ti: картинка (Z-Image Turbo, 9 шагов) —
~15-20с; видео 33 кадра, T2V или I2V (Wan2.2, 4 шага, MoE high/low-noise) —
~35-90с. Сгенерированное видео нормализуется под общий холст (1080x1920/30fps)
тем же `ffmpeg`, что и остальные сегменты, — конкатенация не отличает
AI-сегменты от фото.

I2V (`image:`) сохраняет композицию исходного фото и покадрово анимирует то,
что описано в `prompt` — это единственная I2V-модель, поэтому 100%
идентичность первого кадра источнику не гарантирована (лёгкий upscale/crop
под `width`/`height` неизбежен), но общая сцена и фон не меняются.

## Что сознательно не сделано

- Нет автопубликации в VK — собранный ролик заливается вручную
  (это позволяет прикрепить трек из официальной библиотеки VK для охвата).
- Ken Burns только зум-ин с фиксированным шагом; добавить панораму/
  зум-аут, если понадобится разнообразие.
