# =============================================================================
# train.py — Обучение EfficientNet-B0 на датасете казино/спам изображений
# =============================================================================
#
# Запуск:
#   python train.py
#   python train.py --epochs 30 --batch-size 16 --lr 0.0001
#
# Перед запуском поместите изображения:
#   dataset/spam/   — скриншоты казино, рекламу, "Withdrawal Success" и т.д.
#   dataset/normal/ — обычные изображения (мемы, фото, скриншоты игр)
#
# Рекомендуемое кол-во изображений: от 50 на класс (лучше 200+).
# =============================================================================

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.transforms import v2 as transforms_v2

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("antispam.train")


# ---------------------------------------------------------------------------
# Аугментации
# ---------------------------------------------------------------------------

def get_train_transform():
    """Аугментации для обучающей выборки (расширяют датасет)."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.RandomGrayscale(p=0.05),
        # Имитируем скриншоты низкого качества
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transform():
    """Трансформации для валидационной выборки (без аугментаций)."""
    return transforms.Compose([
        transforms.Resize(config.TRAIN_IMAGE_SIZE),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------------------------------------------------------------------
# Построение модели
# ---------------------------------------------------------------------------

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Создаёт EfficientNet-B0 с предобученными весами ImageNet.

    Transfer Learning стратегия:
      - Замораживаем backbone (features) на первые эпохи
      - Обучаем только классификатор
      - Затем размораживаем всю сеть (fine-tuning)

    Args:
        num_classes: Количество классов (2: normal/spam).
        pretrained: Загружать предобученные веса ImageNet.

    Returns:
        Модель PyTorch.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Замораживаем backbone для начального обучения
    for param in model.features.parameters():
        param.requires_grad = False

    # Заменяем классификатор
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """
    Один эпох обучения.

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping для стабильности
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                "  Epoch %d | Batch %d/%d | Loss: %.4f | Acc: %.1f%%",
                epoch, batch_idx + 1, len(loader),
                total_loss / (batch_idx + 1), 100.0 * correct / total,
            )

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> tuple[float, float, dict]:
    """
    Валидация модели.

    Returns:
        (avg_loss, accuracy, per_class_accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for label, pred in zip(labels, predicted):
            name = class_names[label.item()]
            class_total[name] += 1
            if label == pred:
                class_correct[name] += 1

    per_class = {
        name: 100.0 * class_correct[name] / max(class_total[name], 1)
        for name in class_names
    }
    return total_loss / len(loader), 100.0 * correct / total, per_class


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение детектора спам-изображений (казино/гемблинг)"
    )
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS,
                        help=f"Количество эпох (по умолчанию: {config.TRAIN_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE,
                        help=f"Размер батча (по умолчанию: {config.TRAIN_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=config.TRAIN_LEARNING_RATE,
                        help=f"Learning rate (по умолчанию: {config.TRAIN_LEARNING_RATE})")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Не использовать предобученные веса ImageNet")
    parser.add_argument("--unfreeze-epoch", type=int, default=5,
                        help="С какой эпохи размораживать backbone (по умолчанию: 5)")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Проверка датасета
    # -----------------------------------------------------------------------
    dataset_dir = Path(config.DATASET_DIR)
    spam_dir = Path(config.DATASET_SPAM_DIR)
    normal_dir = Path(config.DATASET_NORMAL_DIR)

    if not spam_dir.exists() or not normal_dir.exists():
        logger.error(
            "Папки датасета не найдены!\n"
            "Создайте структуру:\n"
            "  dataset/spam/   — изображения казино, рекламы, скриншоты вывода\n"
            "  dataset/normal/ — обычные изображения (мемы, фото, скрины игр)\n"
        )
        sys.exit(1)

    spam_count = len(list(spam_dir.glob("*.*")))
    normal_count = len(list(normal_dir.glob("*.*")))

    if spam_count == 0 or normal_count == 0:
        logger.error(
            "Датасет пуст!\n"
            "spam/: %d изображений\n"
            "normal/: %d изображений\n"
            "Добавьте изображения и запустите train.py снова.",
            spam_count, normal_count,
        )
        sys.exit(1)

    logger.info("Датасет: spam=%d, normal=%d изображений", spam_count, normal_count)

    if spam_count < 20 or normal_count < 20:
        logger.warning(
            "⚠️  Мало изображений! Для хорошего качества рекомендуется "
            "минимум 50 изображений на класс (лучше 200+)."
        )

    # -----------------------------------------------------------------------
    # Загрузка датасета
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Устройство: %s", device)

    # Полный датасет (с трансформациями обучения)
    full_dataset = datasets.ImageFolder(
        root=str(dataset_dir),
        transform=get_train_transform(),
    )
    class_names = full_dataset.classes
    logger.info("Классы: %s", class_names)

    # Убеждаемся что порядок классов совпадает с config.MODEL_LABELS
    expected_labels = sorted(config.MODEL_LABELS)
    if sorted(class_names) != expected_labels:
        logger.error(
            "Несоответствие классов!\n"
            "Найдено: %s\n"
            "Ожидается (из config.MODEL_LABELS): %s\n"
            "Переименуйте папки в dataset/",
            class_names, config.MODEL_LABELS,
        )
        sys.exit(1)

    # Разбивка train/val
    val_size = max(int(len(full_dataset) * config.TRAIN_VAL_SPLIT), 1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Для валидации — без аугментаций
    val_dataset.dataset = datasets.ImageFolder(
        root=str(dataset_dir),
        transform=get_val_transform(),
    )

    logger.info("Train: %d | Val: %d", train_size, val_size)

    # Weighted sampler для балансировки классов
    class_counts = [spam_count, normal_count]
    weights = [1.0 / class_counts[label] for _, label in full_dataset.samples[:train_size]]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=train_size, replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # -----------------------------------------------------------------------
    # Модель, функция потерь, оптимизатор
    # -----------------------------------------------------------------------
    model = build_model(
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
    ).to(device)

    # Взвешенный CrossEntropy для несбалансированных датасетов
    class_weights = torch.tensor(
        [normal_count / spam_count, 1.0], dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Оптимизируем только незамороженные параметры
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # -----------------------------------------------------------------------
    # Создание папки для модели
    # -----------------------------------------------------------------------
    Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    best_model_path = Path(config.MODEL_PATH)
    best_val_acc = 0.0

    # -----------------------------------------------------------------------
    # Цикл обучения
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Начало обучения | epochs=%d | batch=%d | lr=%g",
                args.epochs, args.batch_size, args.lr)
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):

        # Размораживаем backbone после unfreeze_epoch
        if epoch == args.unfreeze_epoch:
            logger.info("Размораживаем backbone EfficientNet (full fine-tuning)...")
            for param in model.features.parameters():
                param.requires_grad = True
            # Пересоздаём оптимизатор с меньшим lr для backbone
            optimizer = AdamW([
                {"params": model.features.parameters(), "lr": args.lr * 0.1},
                {"params": model.classifier.parameters(), "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=1e-6
            )

        # Обучение
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Валидация
        val_loss, val_acc, per_class = validate(
            model, val_loader, criterion, device, class_names
        )

        scheduler.step()

        logger.info(
            "Epoch %2d/%d | "
            "Train Loss: %.4f Acc: %.1f%% | "
            "Val Loss: %.4f Acc: %.1f%% | "
            "Per-class: %s",
            epoch, args.epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            {k: f"{v:.1f}%" for k, v in per_class.items()},
        )

        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                "  ✓ Лучшая модель сохранена (val_acc=%.1f%%): %s",
                best_val_acc, best_model_path,
            )

    # -----------------------------------------------------------------------
    # Итог
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Обучение завершено!")
    logger.info("Лучшая точность на валидации: %.1f%%", best_val_acc)
    logger.info("Модель сохранена: %s", best_model_path)
    logger.info("Для запуска бота: python bot.py")
    logger.info("=" * 60)

    # Сохраняем метаданные обучения
    meta_path = Path(config.MODEL_DIR) / "training_meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Best val accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Train samples: {train_size}\n")
        f.write(f"Val samples: {val_size}\n")
    logger.info("Метаданные сохранены: %s", meta_path)


if __name__ == "__main__":
    main()
