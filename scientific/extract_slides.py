#!/usr/bin/env python3
"""
extract_slides.py

Извлекает кадры в моменты смены слайда/графика из YouTube-видео, и
(опционально) транскрипт из субтитров/автоматических подписей.

Пайплайн:
  1. Скачивает видео через yt-dlp (и в том же вызове — субтитры, если
     они есть: ручные или автоматические)
  2. Находит моменты смены сцены через PySceneDetect (ContentDetector)
  3. Сохраняет кадр на каждой смене сцены, с таймстампом в имени файла
  4. Убирает похожие кадры подряд через перцептивный хэш (imagehash)
  5. Если были субтитры — разворачивает их в чистый транскрипт (без
     повторов, которыми пестрят автоматические подписи YouTube) и
     сопоставляет каждый кадр с тем, что говорилось, пока слайд был
     на экране (manifest.json)

Транскрипт вытягивается тем же yt-dlp, которым скачивается видео -
никаких дополнительных зависимостей для этого не требуется.

Установка зависимостей:
    pip install yt-dlp scenedetect[opencv] imagehash pillow

Использование:
    python scientific/extract_slides.py "https://youtu.be/uDKEIoBG5w0" -o slides/
    python scientific/extract_slides.py URL -o out/ --langs en,ru
    python scientific/extract_slides.py URL -o out/ --transcript-only
    python scientific/extract_slides.py URL -o out/ --no-transcript

Опции:
    -o, --output       папка для результатов (по умолчанию ./slides)
    --threshold        порог детекции смены сцены (по умолчанию 27.0 —
                        стандартная чувствительность для резкой смены
                        слайда; меньше значение = чувствительнее к
                        плавным переходам)
    --hash-distance    порог похожести кадров для дедупликации, 0-64
                        (по умолчанию 5; больше = более строгая
                        дедупликация, т.е. будет выкидывать больше
                        похожих кадров)
    --langs            предпочитаемые языки субтитров через запятую,
                        в порядке приоритета (по умолчанию "en")
    --no-transcript    не вытягивать субтитры/транскрипт вообще
    --transcript-only  только транскрипт: видео не скачивается и
                        кадры не извлекаются (быстро - субтитры не
                        требуют скачивания самого видео)
    -q, --quiet        не выводить собственный лог yt-dlp
    --keep-video       не удалять скачанное видео после обработки

Результат в папке вывода:
    scene_XXX_<ms>.jpg     - кадры, имя содержит таймстамп в мс
    transcript.txt          - чистый транскрипт, одним текстом
    transcript_timestamped.txt - тот же транскрипт, с меткой времени на реплику
    manifest.json            - [{frame, timestamp, timestamp_hms, transcript}, ...] -
                                для каждого кадра то, что говорилось, пока он был на экране
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def check_dependencies(need_slides: bool = True):
    missing = []
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        missing.append("yt-dlp")

    if need_slides:
        try:
            from scenedetect import open_video  # noqa: F401
        except ImportError:
            missing.append("scenedetect[opencv]")
        try:
            import imagehash  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            missing.append("imagehash pillow")

    if missing:
        print("Не хватает зависимостей. Установите:")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)


def fetch(url: str, workdir: Path, *, want_video: bool, want_subs: bool,
          langs: list[str], quiet: bool = False) -> tuple[Path | None, dict[str, Path]]:
    """One yt-dlp call for whichever combination of video/subtitles is
    needed - fetching them in separate calls would mean resolving the
    same video listing twice for no benefit."""
    import yt_dlp

    outtmpl = str(workdir / "source.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "quiet": quiet,
        "noplaylist": True,
    }
    if want_video:
        # 720p достаточно для детекции слайдов, качать быстрее
        ydl_opts["format"] = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best"
    else:
        ydl_opts["skip_download"] = True

    if want_subs:
        ydl_opts.update({
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": langs,
            "subtitlesformat": "vtt",
        })

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        video_path = None
        if want_video:
            # prepare_filename() can report the wrong extension once
            # yt-dlp merges separate video+audio streams (yt-dlp issue
            # #5517) - requested_downloads[]['filepath'] is the path the
            # merger actually renamed its output to, so prefer that when
            # present.
            requested = info.get("requested_downloads") or []
            filename = requested[0]["filepath"] if requested else ydl.prepare_filename(info)
            video_path = Path(filename)

        subtitle_paths = {
            lang: Path(sub["filepath"])
            for lang, sub in (info.get("requested_subtitles") or {}).items()
            if sub.get("filepath") and Path(sub["filepath"]).exists()
        }

    return video_path, subtitle_paths


def pick_subtitle(subtitle_paths: dict[str, Path], langs: list[str]) -> Path | None:
    """First match in preference order. yt-dlp's language matching is
    regex-based, so a request for "en" can come back keyed as "en-US" -
    fall back to a prefix match before giving up and taking whatever was
    actually downloaded."""
    for lang in langs:
        if lang in subtitle_paths:
            return subtitle_paths[lang]
    for lang, path in subtitle_paths.items():
        if any(lang.startswith(pref) for pref in langs):
            return path
    return next(iter(subtitle_paths.values()), None)


# ---------------------------------------------------------------------------
# Subtitles -> transcript
# ---------------------------------------------------------------------------

_VTT_TAG_RE = re.compile(r"<[^>]+>")
_VTT_TIMING_RE = re.compile(r"(?:\d{2}:)?\d{2}:\d{2}[.,]\d{3}")


def _parse_timecode(tc: str) -> float:
    tc = tc.replace(",", ".")
    parts = tc.split(":")
    if len(parts) == 2:
        parts = ["00", *parts]
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_vtt(path: Path) -> list[tuple[float, str]]:
    """Parse a WebVTT file into (start_seconds, text) cues, one per block.
    Cue-settings (align/position) and inline tags (<c>, per-word
    timestamps) are stripped - they're rendering hints, not content."""
    cues = []
    raw = path.read_text(encoding="utf-8", errors="replace")

    for block in re.split(r"\r?\n\r?\n", raw):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        timing_idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_idx is None:
            continue  # header/metadata block (WEBVTT, Kind:, Language:, ...)

        start_str = _VTT_TIMING_RE.search(lines[timing_idx].split("-->")[0])
        if not start_str:
            continue
        start = _parse_timecode(start_str.group())

        text = _VTT_TAG_RE.sub("", " ".join(lines[timing_idx + 1:]))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            cues.append((start, text))

    return cues


def dedupe_rolling_captions(cues: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """YouTube's auto-generated captions render as a rolling window: each
    cue re-shows a tail of previously-seen words before scrolling in new
    ones, so flattening cues verbatim repeats most of the transcript
    several times over. Keep only the words each cue adds beyond its
    longest word-level overlap with the previous cue.

    A plain prefix/substring check only covers a window that keeps
    growing; YouTube's actually shifts (drops old words as new ones
    scroll in), so the previous cue's text is neither a prefix of the
    next one nor contained in it - only the overlap at the boundary is
    shared. Comparing word-by-word instead of finding one containment
    relationship handles both cases the same way.
    """
    result = []
    prev_words: list[str] = []

    for start, text in cues:
        words = text.split()
        if not words:
            continue

        overlap = 0
        for k in range(min(len(prev_words), len(words)), 0, -1):
            if prev_words[-k:] == words[:k]:
                overlap = k
                break

        new_words = words[overlap:]
        if new_words:
            result.append((start, " ".join(new_words)))
        prev_words = words

    return result


def _format_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_transcript(lines: list[tuple[float, str]], *, timestamps: bool) -> str:
    if not timestamps:
        return " ".join(text for _, text in lines)
    return "\n".join(f"[{_format_hms(start)}] {text}" for start, text in lines)


def build_manifest(frames: list[tuple[Path, float]],
                    transcript_lines: list[tuple[float, str]]) -> list[dict]:
    """Pair each kept frame with whatever was said while it was on
    screen - from this frame's timestamp up to the next frame's (or to
    the end of the transcript for the last one). No tunable window: the
    frame timestamps themselves are the natural segmentation.

    The first frame's window starts at 0 rather than its own timestamp:
    scenedetect samples a representative frame from partway into each
    scene (not necessarily its very first frame), so anything said
    between the video's start and that first sampled timestamp would
    otherwise belong to no frame at all and silently disappear.
    """
    manifest = []
    for i, (frame, start) in enumerate(frames):
        window_start = 0.0 if i == 0 else start
        window_end = frames[i + 1][1] if i + 1 < len(frames) else float("inf")
        text = " ".join(t for ts, t in transcript_lines if window_start <= ts < window_end)
        manifest.append({
            "frame": frame.name,
            "timestamp": round(start, 2),
            "timestamp_hms": _format_hms(start),
            "transcript": text,
        })
    return manifest


# ---------------------------------------------------------------------------
# Video -> frames
# ---------------------------------------------------------------------------

_FRAME_TIMESTAMP_MS_RE = re.compile(r"_(\d+)\.\w+$")


def _frame_timestamp_seconds(filename: str) -> float:
    """Recover the timestamp baked into a frame's filename by the
    $TIMESTAMP_MS template variable in detect_scenes_and_save_frames()."""
    match = _FRAME_TIMESTAMP_MS_RE.search(filename)
    if not match:
        raise ValueError(f"couldn't read a timestamp out of frame filename: {filename!r}")
    return int(match.group(1)) / 1000.0


def detect_scenes_and_save_frames(video_path: Path, out_dir: Path,
                                   threshold: float) -> list[tuple[Path, float]]:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    from scenedetect.scene_manager import save_images

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        print("Сцены не найдены — возможно, видео без резких смен кадра. "
              "Попробуйте уменьшить --threshold.")
        return []

    print(f"Найдено {len(scene_list)} смен сцены")

    image_filenames = save_images(
        scene_list,
        video,
        num_images=1,          # один кадр на сцену (в начале)
        output_dir=str(out_dir),
        # $TIMESTAMP_MS печёт время сохранённого кадра прямо в имя файла.
        # Это единственный надёжный источник таймстампа: ключи словаря,
        # который save_images() возвращает (scene_num), в установленной
        # версии scenedetect оказались 0-индексированными вопреки
        # собственному докстрингу ("starting from 1") - подставлять их
        # напрямую в scene_list[scene_num - 1] давало таймкод СОСЕДНЕЙ
        # сцены на каждом кадре. Читать значение из имени файла надёжнее,
        # чем полагаться на конвенцию индексации, которая, как выяснилось,
        # может разойтись с документацией. Скобки вокруг имён переменных
        # обязательны: это Python string.Template, а $SCENE_NUMBER_ без
        # них жадно поглощает подчёркивание как часть имени переменной и
        # остаётся неподставленным.
        image_name_template="scene_${SCENE_NUMBER}_${TIMESTAMP_MS}",
    )

    frames = []
    for files in image_filenames.values():
        for f in files:
            frames.append((out_dir / f, _frame_timestamp_seconds(f)))
    frames.sort(key=lambda item: item[0])
    return frames


def dedupe_frames(frames: list[tuple[Path, float]], hash_distance: int) -> list[tuple[Path, float]]:
    import imagehash
    from PIL import Image

    kept = []
    last_hash = None

    for frame, timestamp in frames:
        try:
            h = imagehash.phash(Image.open(frame))
        except Exception as e:
            print(f"Пропускаю {frame.name}: не удалось открыть ({e})")
            continue

        if last_hash is None or (h - last_hash) > hash_distance:
            kept.append((frame, timestamp))
            last_hash = h
        else:
            frame.unlink()  # удаляем дубликат

    return kept


def main():
    parser = argparse.ArgumentParser(
        description="Извлечение кадров смены слайда/графика и транскрипта из YouTube-видео"
    )
    parser.add_argument("url", help="Ссылка на YouTube-видео")
    parser.add_argument("-o", "--output", default="slides", help="Папка для результатов")
    parser.add_argument("--threshold", type=float, default=27.0,
                         help="Порог детекции смены сцены (по умолчанию 27.0)")
    parser.add_argument("--hash-distance", type=int, default=5,
                         help="Порог похожести для дедупликации, 0-64 (по умолчанию 5)")
    parser.add_argument("--langs", default="en",
                         help="Предпочитаемые языки субтитров через запятую, по приоритету (по умолчанию en)")
    parser.add_argument("--no-transcript", action="store_true",
                         help="Не вытягивать субтитры/транскрипт")
    parser.add_argument("--transcript-only", action="store_true",
                         help="Только транскрипт - без скачивания видео и извлечения кадров")
    parser.add_argument("-q", "--quiet", action="store_true",
                         help="Не выводить собственный лог yt-dlp")
    parser.add_argument("--keep-video", action="store_true",
                         help="Не удалять скачанное видео после обработки")
    args = parser.parse_args()

    if args.transcript_only and args.no_transcript:
        parser.error("--transcript-only и --no-transcript вместе не имеют смысла")

    want_video = not args.transcript_only
    want_subs = not args.no_transcript
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]

    check_dependencies(need_slides=want_video)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    workdir = out_dir / "_download"
    workdir.mkdir(exist_ok=True)

    print("Скачиваю..." if want_video else "Скачиваю субтитры...")
    video_path, subtitle_paths = fetch(
        args.url, workdir, want_video=want_video, want_subs=want_subs,
        langs=langs, quiet=args.quiet,
    )
    if video_path:
        print(f"Видео скачано: {video_path}")

    transcript_lines: list[tuple[float, str]] = []
    if want_subs:
        subtitle_path = pick_subtitle(subtitle_paths, langs)
        if subtitle_path is None:
            print("Субтитры не найдены (ни ручные, ни автоматические) — транскрипт пропущен.")
        else:
            cues = parse_vtt(subtitle_path)
            transcript_lines = dedupe_rolling_captions(cues)
            (out_dir / "transcript.txt").write_text(
                format_transcript(transcript_lines, timestamps=False), encoding="utf-8")
            (out_dir / "transcript_timestamped.txt").write_text(
                format_transcript(transcript_lines, timestamps=True), encoding="utf-8")
            print(f"Транскрипт сохранён ({len(transcript_lines)} реплик): {out_dir / 'transcript.txt'}")

    frames: list[tuple[Path, float]] = []
    if want_video:
        print("Ищу смены слайдов/графиков...")
        frames = detect_scenes_and_save_frames(video_path, out_dir, args.threshold)

        if frames:
            print(f"Сохранено {len(frames)} кадров, убираю похожие дубликаты...")
            frames = dedupe_frames(frames, args.hash_distance)
            print(f"Осталось {len(frames)} уникальных кадров в {out_dir}")

    if frames and transcript_lines:
        manifest = build_manifest(frames, transcript_lines)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Манифест кадр→транскрипт сохранён: {out_dir / 'manifest.json'}")

    if want_video and args.keep_video:
        print(f"Видео оставлено в {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)

    print("Готово.")


if __name__ == "__main__":
    main()
