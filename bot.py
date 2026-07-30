# =============================================================================
# bot.py — Основной файл Discord-бота
# =============================================================================

import asyncio
import logging
import logging.handlers
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import discord
from discord.ext import commands

import config
import database as db
from detector import ImageDetector

# ---------------------------------------------------------------------------
# Настройка логирования
# ---------------------------------------------------------------------------
Path(config.LOGS_DIR).mkdir(parents=True, exist_ok=True)
Path(config.REVIEW_DIR).mkdir(parents=True, exist_ok=True)
Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)

log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Файловый хендлер (ротация каждые 7 дней, хранение 30 дней)
file_handler = logging.handlers.TimedRotatingFileHandler(
    config.LOG_FILE, when="D", interval=7, backupCount=4, encoding="utf-8"
)
file_handler.setFormatter(log_formatter)

# Консольный хендлер
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
)
logger = logging.getLogger("antispam.bot")


# ---------------------------------------------------------------------------
# Инициализация бота
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
    help_command=None,  # отключаем стандартную !help
)

# Глобальный экземпляр детектора
detector = ImageDetector()


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

async def send_log(guild: discord.Guild, message: str) -> None:
    """
    Отправляет сообщение в лог-канал сервера.
    Каждый сервер настраивает свой канал через !setlogchannel.
    """
    # 1. Проверяем базу данных (настройка per-guild)
    channel_id = db.get_log_channel(guild.id)
    # 2. Если не настроен в БД — берём глобальный из config.py
    if not channel_id:
        channel_id = config.LOG_CHANNEL_ID
    if not channel_id:
        return
    channel = guild.get_channel(channel_id)
    if channel:
        try:
            await channel.send(message)
        except discord.Forbidden:
            logger.warning("Нет доступа к лог-каналу %s на сервере %s",
                             channel_id, guild.name)


async def download_attachment(attachment: discord.Attachment) -> bytes | None:
    """
    Скачивает вложение Discord и возвращает бинарные данные.

    Returns:
        Байты изображения или None при ошибке.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as resp:
                if resp.status == 200:
                    return await resp.read()
    except Exception as e:
        logger.error("Ошибка скачивания вложения %s: %s", attachment.filename, e)
    return None


def save_for_review(image_data: bytes, filename: str) -> Path:
    """Сохраняет изображение в папку logs/review/ для ручной проверки."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(filename).name
    out_path = Path(config.REVIEW_DIR) / f"{timestamp}_{safe_name}"
    out_path.write_bytes(image_data)
    return out_path


def is_image(filename: str) -> bool:
    """Проверяет, является ли файл изображением по расширению."""
    return Path(filename).suffix.lower() in config.SUPPORTED_FORMATS


async def apply_punishment(
    message: discord.Message,
    image_data: bytes,
    filename: str,
    confidence: float,
    method: str,
) -> None:
    """
    Применяет систему наказаний на основе количества предупреждений.

    1-е нарушение: удалить + предупреждение
    2-е нарушение: удалить + второе предупреждение
    3-е нарушение: удалить + мут 1 час + сброс счётчика
    """
    user = message.author
    guild = message.guild

    # Добавляем предупреждение и получаем новый счётчик
    warn_count = db.add_warning(user.id, guild.id)

    # Логируем нарушение в БД
    db.log_violation(
        user_id=user.id,
        guild_id=guild.id,
        username=str(user),
        filename=filename,
        confidence=confidence,
        method=method,
        action=f"warn_{warn_count}" if warn_count < 3 else "mute",
    )
    db.increment_stat("violations_found")

    # Удаляем сообщение
    try:
        await message.delete()
        logger.info(
            "Удалено сообщение от %s (нарушение #%d, уверенность=%.0f%%)",
            user, warn_count, confidence * 100,
        )
    except discord.Forbidden:
        logger.warning("Нет прав на удаление сообщения от %s", user)
    except discord.NotFound:
        pass  # сообщение уже удалено

    # Применяем наказание в зависимости от счётчика
    if warn_count == 1:
        msg = config.MSG_WARN_1.format(mention=user.mention)
        await message.channel.send(msg)
        await send_log(guild, f"⚠️ **Предупреждение 1/3** | {user} | {filename}")

    elif warn_count == 2:
        msg = config.MSG_WARN_2.format(mention=user.mention)
        await message.channel.send(msg)
        await send_log(guild, f"⚠️ **Предупреждение 2/3** | {user} | {filename}")

    else:
        # 3-е нарушение — мут
        mute_until = datetime.now(timezone.utc) + timedelta(seconds=config.MUTE_DURATION)
        try:
            await user.timeout(mute_until, reason="Повторная отправка запрещённых изображений")
            logger.info("Мут выдан пользователю %s до %s", user, mute_until)
        except discord.Forbidden:
            logger.warning("Нет прав на выдачу мута пользователю %s", user)
        except Exception as e:
            logger.error("Ошибка выдачи мута: %s", e)

        msg = config.MSG_MUTED.format(mention=user.mention)
        await message.channel.send(msg)
        await send_log(
            guild,
            f"🔇 **Мут 1 час** | {user} | нарушений: {warn_count} | {filename}",
        )
        db.increment_stat("users_muted")

        # Сбрасываем счётчик после мута
        db.reset_warnings(user.id, guild.id)

        # Запускаем задачу авто-сброса (на случай если бот перезапустится раньше)
        # В этом случае мут Discord сам снимется, а warnings уже сброшены.


# ---------------------------------------------------------------------------
# События
# ---------------------------------------------------------------------------

@bot.event
async def on_ready() -> None:
    """Вызывается когда бот подключился к Discord."""
    db.init_db()
    logger.info(
        "Бот запущен: %s (ID: %s) | Серверов: %d",
        bot.user, bot.user.id, len(bot.guilds),
    )
    logger.info("Сервера: %s", [g.name for g in bot.guilds])
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name=f"за изображениями в {len(bot.guilds)} серверах 🔍",
        )
    )


@bot.event
async def on_guild_join(guild: discord.Guild) -> None:
    """Бот добавлен на новый сервер — отправляет приветственное сообщение."""
    logger.info("Бот добавлен на сервер: %s (ID: %s)", guild.name, guild.id)
    # Находим первый доступный канал
    welcome_channel = None
    for channel in guild.text_channels:
        if channel.permissions_for(guild.me).send_messages:
            welcome_channel = channel
            break
    if welcome_channel:
        embed = discord.Embed(
            title="🔐 Anti-Spam Бот подключён!",
            description=(
                "Я буду автоматически удалять рекламу казино и гемблинга.\n\n"
                "⭐ **Настройка:**\n"
                "▸ `!setlogchannel #канал` — канал для уведомлений (необязательно)\n"
                "▸ `!help` — список всех команд\n\n"
                "⚠️ Боту нужны права: **Manage Messages** + **Moderate Members**"
            ),
            color=discord.Color.blurple(),
        )
        await welcome_channel.send(embed=embed)
    # Обновляем статус бота
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name=f"за изображениями в {len(bot.guilds)} серверах 🔍",
        )
    )


@bot.event
async def on_message(message: discord.Message) -> None:
    """Обрабатывает каждое новое сообщение."""
    # Игнорируем сообщения самого бота
    if message.author.bot:
        return

    # Игнорируем сообщения вне серверов (DM)
    if not message.guild:
        await bot.process_commands(message)
        return

    # Обрабатываем изображения в вложениях
    for attachment in message.attachments:
        if not is_image(attachment.filename):
            continue

        db.increment_stat("images_checked")
        logger.info(
            "Проверяю изображение от %s: %s",
            message.author, attachment.filename,
        )

        # Скачиваем изображение
        image_data = await download_attachment(attachment)
        if image_data is None:
            continue

        # Классифицируем
        is_spam, confidence, method = detector.predict(image_data)
        logger.info(
            "Результат: is_spam=%s, confidence=%.2f, method=%s | файл: %s",
            is_spam, confidence, method, attachment.filename,
        )

        if is_spam:
            await apply_punishment(
                message=message,
                image_data=image_data,
                filename=attachment.filename,
                confidence=confidence,
                method=method,
            )
            # Прекращаем проверку остальных вложений этого сообщения
            return

        elif confidence >= (config.CONFIDENCE_THRESHOLD * 0.7):
            # Уверенность в "серой зоне" — отправляем на ручную проверку
            review_path = save_for_review(image_data, attachment.filename)
            review_msg = config.MSG_REVIEW.format(
                mention=message.author.mention,
                confidence=confidence,
            )
            logger.info("На ручную проверку: %s → %s", attachment.filename, review_path)
            await send_log(message.guild, review_msg)

    # Обрабатываем команды (если сообщение содержит команду)
    await bot.process_commands(message)


# ---------------------------------------------------------------------------
# Команды администратора
# ---------------------------------------------------------------------------

def is_admin():
    """Декоратор: проверяет что пользователь администратор или имеет право manage_messages."""
    async def predicate(ctx: commands.Context) -> bool:
        return (
            ctx.author.guild_permissions.administrator
            or ctx.author.guild_permissions.manage_messages
        )
    return commands.check(predicate)


@bot.command(name="warnings")
@is_admin()
async def cmd_warnings(ctx: commands.Context, member: discord.Member) -> None:
    """
    !warnings @user — показывает количество предупреждений пользователя.
    """
    count = db.get_warnings(member.id, ctx.guild.id)
    embed = discord.Embed(
        title="📋 Предупреждения",
        color=discord.Color.orange(),
    )
    embed.add_field(name="Пользователь", value=member.mention)
    embed.add_field(name="Предупреждений", value=f"**{count}** / 3")
    embed.set_footer(text=f"Запросил: {ctx.author}")
    await ctx.send(embed=embed)


@bot.command(name="clearwarnings")
@is_admin()
async def cmd_clearwarnings(ctx: commands.Context, member: discord.Member) -> None:
    """
    !clearwarnings @user — сбрасывает предупреждения пользователя.
    """
    db.reset_warnings(member.id, ctx.guild.id)
    embed = discord.Embed(
        title="✅ Предупреждения сброшены",
        description=f"Предупреждения {member.mention} очищены.",
        color=discord.Color.green(),
    )
    await ctx.send(embed=embed)
    logger.info("%s сбросил предупреждения для %s", ctx.author, member)


@bot.command(name="reloadmodel")
@is_admin()
async def cmd_reloadmodel(ctx: commands.Context) -> None:
    """
    !reloadmodel — перезагружает модель с диска без перезапуска бота.
    """
    msg = await ctx.send("⏳ Перезагрузка модели...")
    success = detector.reload()
    if success:
        embed = discord.Embed(
            title="✅ Модель перезагружена",
            description=f"Путь: `{config.MODEL_PATH}`",
            color=discord.Color.green(),
        )
    else:
        embed = discord.Embed(
            title="❌ Ошибка загрузки модели",
            description=(
                f"Файл `{config.MODEL_PATH}` не найден или повреждён. "
                "Убедитесь что обучение завершено (`python train.py`)."
            ),
            color=discord.Color.red(),
        )
    await msg.edit(content=None, embed=embed)


@bot.command(name="stats")
@is_admin()
async def cmd_stats(ctx: commands.Context) -> None:
    """
    !stats — показывает глобальную статистику бота.
    """
    stats = db.get_stats()
    model_status = "✅ Загружена" if detector.model is not None else "❌ Не загружена"

    embed = discord.Embed(
        title="📊 Статистика Anti-Spam бота",
        color=discord.Color.blurple(),
        timestamp=datetime.utcnow(),
    )
    embed.add_field(name="🖼️ Проверено изображений", value=stats.get("images_checked", 0))
    embed.add_field(name="🚫 Найдено нарушений", value=stats.get("violations_found", 0))
    embed.add_field(name="🔇 Пользователей замучено", value=stats.get("users_muted", 0))
    embed.add_field(name="🤖 Статус модели", value=model_status, inline=False)
    embed.add_field(
        name="⚙️ Порог уверенности",
        value=f"{config.CONFIDENCE_THRESHOLD:.0%}",
    )
    embed.set_footer(text=f"Запросил: {ctx.author}")
    await ctx.send(embed=embed)


@bot.command(name="setlogchannel")
@is_admin()
async def cmd_setlogchannel(ctx: commands.Context, channel: discord.TextChannel) -> None:
    """
    !setlogchannel #канал — настраивает канал для логов на этом сервере.
    Каждый сервер настраивает СВОЙ канал независимо!
    """
    db.set_log_channel(ctx.guild.id, channel.id)
    embed = discord.Embed(
        title="✅ Лог-канал настроен",
        description=f"Уведомления будут отправляться в {channel.mention}",
        color=discord.Color.green(),
    )
    await ctx.send(embed=embed)
    # Тестовое сообщение в выбранный канал
    await channel.send("🔍 Этот канал выбран для логов Anti-Spam бота.")
    logger.info("%s настроил лог-канал %s на %s", ctx.author, channel, ctx.guild)


@bot.command(name="help")
async def cmd_help(ctx: commands.Context) -> None:
    """
    !help — показывает список команд бота.
    """
    embed = discord.Embed(
        title="🤖 Anti-Spam Bot — Команды",
        color=discord.Color.blurple(),
    )
    embed.add_field(
        name="!warnings @user",
        value="Показать количество предупреждений пользователя",
        inline=False,
    )
    embed.add_field(
        name="!clearwarnings @user",
        value="Сбросить предупреждения пользователя",
        inline=False,
    )
    embed.add_field(
        name="!setlogchannel #канал",
        value="Настроить канал для уведомлений (свой для каждого сервера)",
        inline=False,
    )
    embed.add_field(
        name="!reloadmodel",
        value="Перезагрузить модель без перезапуска бота",
        inline=False,
    )
    embed.add_field(
        name="!stats",
        value="Показать статистику бота",
        inline=False,
    )
    embed.set_footer(text="Только для администраторов (manage_messages)")
    await ctx.send(embed=embed)


# ---------------------------------------------------------------------------
# Обработка ошибок
# ---------------------------------------------------------------------------

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
    """Глобальный обработчик ошибок команд."""
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ Не хватает аргумента: `{error.param.name}`. Используйте `!help`.")
    elif isinstance(error, commands.MemberNotFound):
        await ctx.send("❌ Пользователь не найден.")
    elif isinstance(error, commands.CheckFailure):
        await ctx.send("❌ У вас нет прав для использования этой команды.")
    elif isinstance(error, commands.CommandNotFound):
        pass  # Игнорируем неизвестные команды
    else:
        logger.error("Ошибка команды %s: %s", ctx.command, error, exc_info=True)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if config.TOKEN == "YOUR_BOT_TOKEN_HERE":
        logger.error(
            "Токен бота не настроен! Установите переменную DISCORD_TOKEN "
            "или замените YOUR_BOT_TOKEN_HERE в config.py"
        )
        raise SystemExit(1)

    logger.info("Запуск бота...")
    bot.run(config.TOKEN, log_handler=None)
