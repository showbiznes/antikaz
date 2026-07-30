# =============================================================================
# Dockerfile — для bothost.ru
# torch устанавливается ЗДЕСЬ, до запуска бота
# =============================================================================

FROM python:3.11-slim

LABEL description="Discord Anti-Spam Bot"

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgomp1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ШАГ 1: Базовые пакеты (быстро)
RUN pip install --no-cache-dir \
    python-dotenv==1.0.1 \
    Pillow==10.3.0 \
    aiohttp==3.9.5 \
    "discord.py==2.3.2" \
    pytesseract==0.3.10 \
    ftfy==6.1.3 \
    "regex==2024.4.28"

# ШАГ 2: PyTorch CPU (медленно, ~700 МБ — отдельный шаг для кэша)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cpu

# ШАГ 3: CLIP (после torch)
RUN pip install --no-cache-dir openai-clip==1.0.1 || \
    echo "CLIP не установлен — используется OCR"

# ШАГ 4: Копируем проект
COPY . .

# Создаём папки
RUN mkdir -p logs/review model dataset/spam dataset/normal

# Переменные окружения
ENV DISCORD_TOKEN=""
ENV GUILD_ID="0"
ENV LOG_CHANNEL_ID="0"
ENV MUTE_DURATION="3600"
ENV CONFIDENCE_THRESHOLD="0.70"

CMD ["python", "-u", "bot.py"]
