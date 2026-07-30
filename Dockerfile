# =============================================================================
# Dockerfile — для деплоя на bothost.ru
# =============================================================================
# Bothost использует Docker-контейнеры.
# Этот Dockerfile устанавливает все зависимости включая tesseract-ocr.
# =============================================================================

FROM python:3.11-slim

# Метаданные
LABEL description="Discord Anti-Spam Bot with CLIP + EfficientNet"

# Системные зависимости:
# - tesseract-ocr + tesseract-ocr-rus: OCR для русских/английских текстов
# - libgomp1: нужен для PyTorch
# - wget: скачивание зависимостей при сборке
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgomp1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория внутри контейнера
WORKDIR /app

# Сначала копируем requirements для кэширования слоёв Docker
COPY requirements.txt .

# Устанавливаем PyTorch CPU-версию (явно, до остальных пакетов)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаём необходимые директории
RUN mkdir -p logs/review model dataset/spam dataset/normal

# Переменные окружения (переопределяются в панели bothost.ru)
ENV DISCORD_TOKEN=""
ENV GUILD_ID="0"
ENV LOG_CHANNEL_ID="0"
ENV MUTE_DURATION="3600"
ENV CONFIDENCE_THRESHOLD="0.70"

# Запуск бота
CMD ["python", "-u", "bot.py"]
