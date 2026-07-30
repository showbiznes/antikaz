# =============================================================================
# Dockerfile — для деплоя на bothost.ru
# =============================================================================

FROM python:3.11-slim

LABEL description="Discord Anti-Spam Bot with CLIP + EfficientNet"

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgomp1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Сначала python-dotenv и базовые пакеты
RUN pip install --no-cache-dir python-dotenv==1.0.1

# 2. PyTorch CPU (отдельно, со своим index-url)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Остальные зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем проект
COPY . .

# Создаём нужные папки
RUN mkdir -p logs/review model dataset/spam dataset/normal

# Переменные окружения (переопределяются в панели bothost.ru)
ENV DISCORD_TOKEN=""
ENV GUILD_ID="0"
ENV LOG_CHANNEL_ID="0"
ENV MUTE_DURATION="3600"
ENV CONFIDENCE_THRESHOLD="0.70"

CMD ["python", "-u", "bot.py"]
