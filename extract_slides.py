#!/usr/bin/env python3
"""
extract_slides.py

Извлекает кадры в моменты смены слайда/графика из YouTube-видео.

Пайплайн:
  1. Скачивает видео через yt-dlp
  2. Находит моменты смены сцены через PySceneDetect (ContentDetector)
  3. Сохраняет кадр на каждой смене сцены
  4. Убирает похожие кадры подряд через перцептивный хэш (imagehash)

Установка зависимостей:
    pip install yt-dlp scenedetect[opencv] imagehash pillow

Использование:
    python extract_slides.py "https://youtu.be/uDKEIoBG5w0" -o slides/

Опции:
    -o, --output       папка для результатов (по умолчанию ./slides)
    --threshold        порог детекции смены сцены (по умолчанию 27.0 —
                        стандартная чувствительность для резкой смены слайда;
                        меньше значение = чувствительнее к плавным переходам)
    --hash-distance     порог похожести кадров для дедупликации, 0-64
                        (по умолчанию 5; больше = более строгая дедупликация,
                        т.е. будет выкидывать больше похожих кадров)
    --keep-video        не удалять скачанное видео после обработки
"""

import argparse
import shutil
import sys
from pathlib import Path


def check_dependencies():
    missing = []
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        missing.append("yt-dlp")
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


def download_video(url: str, workdir: Path) -> Path:
    import yt_dlp

    outtmpl = str(workdir / "source.%(ext)s")
    ydl_opts = {
        # 720p достаточно для детекции слайдов, качать быстрее
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best",
        "outtmpl": outtmpl,
        "quiet": False,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    return Path(filename)


def detect_scenes_and_save_frames(video_path: Path, out_dir: Path, threshold: float) -> list[Path]:
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
        image_name_template="scene_$SCENE_NUMBER",
    )

    frames = []
    for scene_num, files in image_filenames.items():
        for f in files:
            frames.append(out_dir / f)
    frames.sort()
    return frames


def dedupe_frames(frames: list[Path], hash_distance: int) -> list[Path]:
    import imagehash
    from PIL import Image

    kept = []
    last_hash = None

    for frame in frames:
        try:
            h = imagehash.phash(Image.open(frame))
        except Exception as e:
            print(f"Пропускаю {frame.name}: не удалось открыть ({e})")
            continue

        if last_hash is None or (h - last_hash) > hash_distance:
            kept.append(frame)
            last_hash = h
        else:
            frame.unlink()  # удаляем дубликат

    return kept


def main():
    parser = argparse.ArgumentParser(description="Извлечение кадров смены слайда/графика из YouTube-видео")
    parser.add_argument("url", help="Ссылка на YouTube-видео")
    parser.add_argument("-o", "--output", default="slides", help="Папка для результатов")
    parser.add_argument("--threshold", type=float, default=27.0,
                         help="Порог детекции смены сцены (по умолчанию 27.0)")
    parser.add_argument("--hash-distance", type=int, default=5,
                         help="Порог похожести для дедупликации, 0-64 (по умолчанию 5)")
    parser.add_argument("--keep-video", action="store_true",
                         help="Не удалять скачанное видео после обработки")
    args = parser.parse_args()

    check_dependencies()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    workdir = out_dir / "_download"
    workdir.mkdir(exist_ok=True)

    print("Скачиваю видео...")
    video_path = download_video(args.url, workdir)
    print(f"Видео скачано: {video_path}")

    print("Ищу смены слайдов/графиков...")
    frames = detect_scenes_and_save_frames(video_path, out_dir, args.threshold)

    if frames:
        print(f"Сохранено {len(frames)} кадров, убираю похожие дубликаты...")
        kept = dedupe_frames(frames, args.hash_distance)
        print(f"Осталось {len(kept)} уникальных кадров в {out_dir}")

    if not args.keep_video:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"Видео оставлено в {workdir}")

    print("Готово.")


if __name__ == "__main__":
    main()
