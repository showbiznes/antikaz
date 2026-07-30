# =============================================================================
# detector.py — Модуль определения спам-изображений
# =============================================================================
# Многоуровневая система детекции:
#   1. CLIP (zero-shot) — работает СРАЗУ без обучения,
#      распознаёт казино/гемблинг по визуальным паттернам
#   2. EfficientNet-B0 (fine-tuned) — дообучается на вашем датасете
#   3. OCR (pytesseract) — вспомогательный метод по тексту
#
# Примеры распознаваемых изображений:
#   - Скриншоты казино-сайтов (Rasowin, Mellgams и подобные)
#   - Фейковые новости с рекламой казино (стиль РИА Новости)
#   - Скриншоты "Withdrawal Success" / вывода крипты
#   - Балансы казино и страницы бонусов
# =============================================================================

import io
import logging
import re
from pathlib import Path
from typing import Optional

from PIL import Image

import config

logger = logging.getLogger("antispam.detector")

# Пробуем загрузить torch — если не установлен, бот работает через OCR
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning(
        "torch/torchvision не установлены! "
        "Детекция через нейросеть недоступна. "
        "Работает только OCR + CLIP (если установлен)."
    )
    TORCH_AVAILABLE = False



# ---------------------------------------------------------------------------
# CLIP — промпты для zero-shot классификации
# ---------------------------------------------------------------------------
# Описания СПАМА — что ищет CLIP на изображении
CLIP_SPAM_PROMPTS = [
    # Казино-сайты
    "screenshot of online casino website",
    "casino slots gambling website interface",
    "online gambling website with deposit and withdraw",
    "casino bonus page with promo codes",
    "online casino VIP club bonuses page",
    "casino withdrawal success message",
    "casino withdrawal screen with crypto",
    "casino balance screen with rubles",
    "casino rakeback bonuses interface",

    # Скриншоты выплат (как Rasowin)
    "withdrawal success screen with green checkmark",
    "crypto withdrawal success notification",
    "USDT withdrawal successful casino screen",
    "casino winnings withdrawal screenshot",

    # Фейковые новости
    "fake news article about casino promotion",
    "news article promoting online casino bonus",
    "celebrity endorsing casino with free money offer",
    "fake RIA Novosti news about casino",

    # Мобильные и балансы
    "casino mobile app balance screen",
    "gambling app balance with Russian rubles",
    "casino app pополнить deposit button",
    "Mell coins casino balance screenshot",

    # Общие паттерны
    "gambling advertisement with free money offer",
    "casino registration bonus advertisement",
    "online casino advertisement in Russian",
    "sports betting advertisement",
    "bookmaker advertisement gambling",
]

# Описания НОРМАЛЬНОГО контента
CLIP_NORMAL_PROMPTS = [
    "normal conversation screenshot",
    "gaming screenshot video game",
    "meme funny image",
    "anime art drawing",
    "photo of people friends",
    "landscape nature photo",
    "food photo restaurant",
    "sports event photo",
    "music concert photo",
    "social media post friends",
]

# ---------------------------------------------------------------------------
# Ключевые слова для OCR (все языки)
# ---------------------------------------------------------------------------
SPAM_KEYWORDS = [
    # Казино/гемблинг RU
    "казино", "слоты", "джекпот", "ставки", "букмекер",
    "гемблинг", "покер", "рулетка", "выигрыш", "выиграй",
    "заработай", "халява", "бонус", "фриспин", "фри спин",
    "пополнить", "вывод", "вывести", "баланс казино",
    "mell coins", "мелл", "раздаёт", "раздает",
    # Казино/гемблинг EN
    "casino", "slots", "jackpot", "betting", "bookmaker",
    "gambling", "poker", "roulette", "withdrawal success",
    "withdraw", "deposit", "bonuses", "rakeback", "vip-club",
    "promo code", "free spin", "freespin", "cashback",
    # Конкретные бренды
    "1xbet", "1хbet", "мелбет", "melbet", "winline", "betwinner",
    "fonbet", "фонбет", "marathonbet", "rasowin", "mellgams",
    "mellgames", "bwin", "betway",
    # Крипто-вывод (признак мошенничества)
    "withdrawal of $", "was successful", "usdt", "trc20", "bep20",
    "erc20", "tether", "bnb smart chain",
    # Телеграм-каналы
    "t.me/", "telegram", "телеграм",
    # Фейк-новости
    "регистрации деньги", "на баланс", "mellgams.com",
    "раздаёт 10000", "раздает рублей",
]

# URL-паттерны
URL_PATTERN = re.compile(
    r"(https?://|t\.me/|www\.|\.com|\.ru|\.net|\.io)\S*",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Цветовые признаки казино (анализ без torch)
# ---------------------------------------------------------------------------
# Тёмно-синий интерфейс (Rasowin, Mellgams и подобные)
# HSV: оттенок Hue=200-260, высокая насыщенность, низкая яркость
CASSINO_DARK_BLUE = {
    "hue_min": 190,    # синий/фиолетовый цвет
    "hue_max": 275,
    "sat_min": 40,     # должна быть насыщенной (>серого)
    "val_max": 120,    # тёмная (не светлая)
    "min_ratio": 0.35, # порог: 35% пикселей должны быть тёмно-синими
}


def build_efficientnet(num_classes: int = 2):
    """
    Создаёт EfficientNet-B0 с кастомным классификатором.
    Возвращает None если torch не установлен.
    """
    if not TORCH_AVAILABLE:
        return None
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


class ImageDetector:
    """
    Многоуровневый детектор спам-изображений.

    Уровни детекции:
      1. CLIP zero-shot — работает сразу, без обучения
      2. EfficientNet fine-tuned — после train.py
      3. OCR — по ключевым словам в тексте изображения
    """

    def __init__(self) -> None:
        # Устройство (CPU или GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if TORCH_AVAILABLE else None

        # EfficientNet (дообученная модель)
        self.model = None

        # CLIP модель (zero-shot)
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_text_features = None

        # Трансформации для EfficientNet
        if TORCH_AVAILABLE:
            self.transform = transforms.Compose([
                transforms.Resize(config.TRAIN_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = None

        self._load_model()
        self._load_clip()
        logger.info(
            "ImageDetector запущен (torch=%s, CLIP=%s, EfficientNet=%s)",
            "✓" if TORCH_AVAILABLE else "✗ (не установлен)",
            "✓" if self.clip_model else "✗",
            "✓" if self.model else "✗ (запустите train.py)",
        )

    # -----------------------------------------------------------------------
    # Загрузка моделей
    # -----------------------------------------------------------------------

    def _load_model(self) -> None:
        """Загружает EfficientNet из файла."""
        model_path = Path(config.MODEL_PATH)
        if not model_path.exists():
            logger.warning(
                "EfficientNet не найден: %s. "
                "Запустите train.py для обучения. "
                "Пока используется только CLIP + OCR.",
                model_path,
            )
            self.model = None
            return

        try:
            self.model = build_efficientnet(num_classes=len(config.MODEL_LABELS))
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info("EfficientNet загружен из %s", model_path)
        except Exception as e:
            logger.error("Ошибка загрузки EfficientNet: %s", e)
            self.model = None

    def _load_clip(self) -> None:
        """Загружает CLIP ViT-B/32 и кодирует текстовые промпты."""
        try:
            import clip  # openai-clip
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-B/32", device=self.device
            )
            self.clip_model.eval()
            self._encode_clip_prompts()
            logger.info("CLIP ViT-B/32 загружен")
        except ImportError:
            logger.warning(
                "CLIP не установлен. Установите: pip install openai-clip. "
                "Zero-shot детекция недоступна."
            )
            self.clip_model = None
        except Exception as e:
            logger.error("Ошибка загрузки CLIP: %s", e)
            self.clip_model = None

    def _encode_clip_prompts(self) -> None:
        """Предварительно кодирует все текстовые промпты CLIP."""
        try:
            import clip
            all_prompts = CLIP_SPAM_PROMPTS + CLIP_NORMAL_PROMPTS
            tokens = clip.tokenize(all_prompts).to(self.device)
            with torch.no_grad():
                self.clip_text_features = self.clip_model.encode_text(tokens)
                self.clip_text_features /= self.clip_text_features.norm(dim=-1, keepdim=True)
            logger.debug("CLIP промпты закодированы (%d spam + %d normal)",
                         len(CLIP_SPAM_PROMPTS), len(CLIP_NORMAL_PROMPTS))
        except Exception as e:
            logger.error("Ошибка кодирования CLIP промптов: %s", e)
            self.clip_text_features = None

    def reload(self) -> bool:
        """
        Перезагружает модели с диска (команда !reloadmodel).

        Returns:
            True если хотя бы одна модель загружена.
        """
        logger.info("Перезагрузка моделей...")
        self._load_model()
        self._load_clip()
        return self.model is not None or self.clip_model is not None

    # -----------------------------------------------------------------------
    # Основной метод предсказания
    # -----------------------------------------------------------------------

    def predict(self, image_data: bytes) -> tuple[bool, float, str]:
        """
        Классифицирует изображение с использованием всех доступных методов.

        Args:
            image_data: Бинарные данные изображения.

        Returns:
            (is_spam, confidence, method):
              - is_spam: True если обнаружен спам
              - confidence: уверенность [0.0, 1.0]
              - method: описание сработавших методов
        """
        results = []

        # --- Метод 1: Анализ цвета (без torch, работает всегда) ---
        color_spam, color_conf = self._predict_color(image_data)
        if color_spam:
            results.append(("color", True, color_conf))
        elif color_conf > 0.3:
            # Подозрительный — записываем но не блокируем
            results.append(("color_weak", False, color_conf))

        # --- Метод 2: CLIP zero-shot ---
        clip_spam, clip_conf = self._predict_clip(image_data)
        if self.clip_model is not None:
            results.append(("clip", clip_spam, clip_conf))

        # --- Метод 3: EfficientNet ---
        if self.model is not None:
            eff_spam, eff_conf = self._predict_efficientnet(image_data)
            results.append(("efficientnet", eff_spam, eff_conf))

        # --- Метод 4: OCR ---
        ocr_spam, ocr_keyword = self._predict_ocr(image_data)
        if ocr_spam:
            results.append(("ocr", True, 0.88))

        # --- Нет доступных методов ---
        if not results:
            logger.error("Ни один метод детекции не доступен!")
            return False, 0.0, "none"

        # --- Агрегация результатов ---
        return self._aggregate(results)

    def _aggregate(
        self, results: list[tuple[str, bool, float]]
    ) -> tuple[bool, float, str]:
        """
        Объединяет результаты нескольких методов в итоговое решение.

        Логика:
        - Если несколько методов согласны → повышенная уверенность
        - Если хотя бы один высокоуверенный метод сработал → спам
        - Иначе берём максимальную уверенность
        """
        methods_fired = [name for name, is_spam, conf in results if is_spam]
        max_conf = max((conf for _, is_spam, conf in results if is_spam), default=0.0)
        all_names = "+".join(methods_fired) if methods_fired else "none"

        # Если сработало 2+ метода — бустим уверенность
        if len(methods_fired) >= 2:
            boosted = min(max_conf * 1.15, 1.0)
            logger.info("Мульти-детект: %s → confidence=%.2f", all_names, boosted)
            return True, boosted, all_names

        # Один метод сработал
        if methods_fired:
            is_spam = max_conf >= config.CONFIDENCE_THRESHOLD
            logger.info("Детект: %s → confidence=%.2f, is_spam=%s",
                        all_names, max_conf, is_spam)
            return is_spam, max_conf, all_names

        # Ничего не сработало — берём максимальную из всех
        best = max(results, key=lambda x: x[2])
        return False, best[2], best[0]

    # -----------------------------------------------------------------------
    # Метод 1: CLIP zero-shot
    # -----------------------------------------------------------------------

    def _predict_clip(self, image_data: bytes) -> tuple[bool, float]:
        """
        Zero-shot классификация через CLIP.

        Работает сразу без обучения — CLIP понимает визуальные концепции
        по текстовым описаниям (промптам).

        Признаки казино/спама которые CLIP ищет:
        - Тёмно-синий UI казино (Rasowin, Mellgams)
        - "Withdrawal Success" экраны
        - Фейковые новости с рекламой
        - Балансы с криптой и рублями
        """
        if self.clip_model is None or self.clip_text_features is None:
            return False, 0.0

        try:
            import clip
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Сходство со всеми промптами
                similarities = (100.0 * image_features @ self.clip_text_features.T).softmax(dim=-1)
                similarities = similarities.squeeze()

            # Суммируем сходство с SPAM-промптами
            n_spam = len(CLIP_SPAM_PROMPTS)
            spam_score = similarities[:n_spam].sum().item()
            normal_score = similarities[n_spam:].sum().item()
            total = spam_score + normal_score + 1e-8
            spam_prob = spam_score / total

            # Топ сработавшие промпты (для дебага)
            top_spam_idx = similarities[:n_spam].argmax().item()
            logger.debug(
                "CLIP: spam_prob=%.3f | топ-промпт: %r",
                spam_prob, CLIP_SPAM_PROMPTS[top_spam_idx],
            )

            return spam_prob >= config.CONFIDENCE_THRESHOLD, spam_prob

        except Exception as e:
            logger.error("Ошибка CLIP предсказания: %s", e)
            return False, 0.0

    # -----------------------------------------------------------------------
    # Метод 2: EfficientNet (дообученная)
    # -----------------------------------------------------------------------

    def _predict_efficientnet(self, image_data: bytes) -> tuple[bool, float]:
        """Классификация с помощью дообученной EfficientNet-B0."""
        if self.model is None:
            return False, 0.0

        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze()

            spam_idx = config.MODEL_LABELS.index("spam")
            spam_prob = probs[spam_idx].item()

            logger.debug("EfficientNet: spam_prob=%.3f", spam_prob)
            return spam_prob >= config.CONFIDENCE_THRESHOLD, spam_prob

        except Exception as e:
            logger.error("Ошибка EfficientNet предсказания: %s", e)
            return False, 0.0

    # -----------------------------------------------------------------------
    # Метод 1: Анализ цвета (без torch)
    # -----------------------------------------------------------------------

    def _predict_color(self, image_data: bytes) -> tuple[bool, float]:
        """
        Определяет спам по цветовому составу изображения.
        Работает БЕЗ torch/CLIP, всегда доступен.

        Детектирует:
        - Тёмно-синий интерфейс казино (Rasowin, Mellgams) — >35% тёмно-синих пикселей
        - Сине-зелёные уведомления о выплате (зелёный чекмарк + синий фон)

        Returns:
            (is_spam, confidence): уверенность [0.0, 1.0]
        """
        try:
            import colorsys
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            # Уменьшаем для скорости (100x100 достаточно)
            thumb = image.resize((100, 100))
            pixels = list(thumb.getdata())
            total = len(pixels)

            dark_blue_count = 0
            green_count = 0

            for r, g, b in pixels:
                h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
                hue_deg = h * 360
                sat_pct = s * 100
                val = v * 255

                # Тёмно-синий/фиолетовый (основной признак казино-сайтов)
                if (
                    CASSINO_DARK_BLUE["hue_min"] <= hue_deg <= CASSINO_DARK_BLUE["hue_max"]
                    and sat_pct >= CASSINO_DARK_BLUE["sat_min"]
                    and val <= CASSINO_DARK_BLUE["val_max"]
                ):
                    dark_blue_count += 1

                # Ярко-зелёный (кнопки казино: «Пополнить», «Claim», «Activate»)
                if (
                    80 <= hue_deg <= 150
                    and sat_pct >= 50
                    and val >= 150
                ):
                    green_count += 1

            dark_ratio = dark_blue_count / total
            green_ratio = green_count / total

            logger.debug(
                "Цвет: dark_blue=%.2f%%, green=%.2f%%",
                dark_ratio * 100, green_ratio * 100,
            )

            # Казино-сайт: много тёмно-синего + есть зелёный
            if dark_ratio >= 0.35 and green_ratio >= 0.02:
                confidence = min(dark_ratio * 1.5, 0.90)
                return True, confidence

            # Преимущественно тёмно-синий (Withdrawal Success сцена)
            if dark_ratio >= 0.45:
                confidence = min(dark_ratio * 1.3, 0.85)
                return True, confidence

            # Подозрительное
            return False, dark_ratio

        except Exception as e:
            logger.debug("Ошибка анализа цвета: %s", e)
            return False, 0.0

    # -----------------------------------------------------------------------
    # Метод 4: OCR
    # -----------------------------------------------------------------------

    def _predict_ocr(self, image_data: bytes) -> tuple[bool, str]:
        """
        Ищет ключевые слова казино/гемблинга в тексте на изображении.

        Returns:
            (is_spam, found_keyword)
        """
        try:
            import pytesseract
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            text = pytesseract.image_to_string(image, lang="rus+eng").lower()

            for keyword in SPAM_KEYWORDS:
                if keyword.lower() in text:
                    logger.debug("OCR нашёл ключевое слово: %r", keyword)
                    return True, keyword

            if URL_PATTERN.search(text):
                logger.debug("OCR нашёл URL в тексте изображения")
                return True, "url"

        except ImportError:
            logger.debug("pytesseract не установлен, OCR пропущен.")
        except Exception as e:
            logger.debug("Ошибка OCR: %s", e)

        return False, ""

