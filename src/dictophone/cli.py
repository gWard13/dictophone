from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .core import validate_transcript, wer


def build_parser():
    p = argparse.ArgumentParser(description="Локальный конвейер MP3 → JSON")
    commands = p.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Полная обработка локального MP3")
    run.add_argument("audio", type=Path)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path)
    run.add_argument("--language", choices=["ru", "en"], default="ru")
    run.add_argument("--session", type=Path, help="Явно сохранить приватные данные для повтора LLM")
    run.add_argument("--report", type=Path, help="Метрики и предупреждения без текста")
    retry = commands.add_parser("retry-llm", help="Повторить LLM без ASR и диаризации")
    retry.add_argument("session", type=Path)
    retry.add_argument("--config", type=Path, required=True)
    retry.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="Проверка экспортированного JSON")
    validate.add_argument("json_file", type=Path)
    validate.add_argument("--duration-ms", type=int)
    evaluate = commands.add_parser("evaluate", help="WER по двум JSON-расшифровкам")
    evaluate.add_argument("reference", type=Path)
    evaluate.add_argument("hypothesis", type=Path)
    doctor = commands.add_parser("doctor", help="Проверка библиотек без загрузки моделей")
    doctor.add_argument("--config", type=Path, help="Также проверить наличие локальных моделей")
    cut = commands.add_parser("cut", help="Вырезать точный фрагмент локального MP3")
    cut.add_argument("audio", type=Path)
    cut.add_argument("--start-ms", type=int, required=True)
    cut.add_argument("--end-ms", type=int, required=True)
    cut.add_argument("--output", type=Path, required=True)
    clips = commands.add_parser("prepare-clips", help="Нарезать подкаст по локальному JSON-плану")
    clips.add_argument("manifest", type=Path)
    return p


def check_paths(paths: list[Path | None], protected: list[Path]):
    destinations = [p.resolve() for p in paths if p is not None]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Output, session and report paths must differ")
    if set(destinations).intersection(p.resolve() for p in protected):
        raise ValueError("Output paths must not overwrite input files")


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    # Protect against accidental hub access by dependencies; local paths remain mandatory.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        if args.command == "cut":
            from .clips import cut_mp3
            print(json.dumps(cut_mp3(args.audio, args.output, args.start_ms, args.end_ms), indent=2))
            return 0
        if args.command == "prepare-clips":
            from .clips import prepare_clips
            print(prepare_clips(args.manifest))
            return 0
        if args.command == "doctor":
            import importlib
            missing = []
            for name in ("numpy", "av", "faster_whisper", "sherpa_onnx", "llama_cpp"):
                try:
                    module = importlib.import_module(name)
                    print(f"{name}: OK ({getattr(module, '__version__', 'version unknown')})")
                except Exception:
                    missing.append(name)
                    print(f"{name}: NOT AVAILABLE")
            if args.config:
                from .pipeline import load_config
                from .backends import require_file
                cfg = load_config(args.config)
                paths = [Path(cfg["asr"]["model_dir"]) / name
                         for name in ("model.bin", "config.json", "tokenizer.json")]
                paths += [Path(cfg["diarization"]["segmentation_model"]),
                          Path(cfg["diarization"]["embedding_model"]), Path(cfg["llm"]["model_path"])]
                for path in paths:
                    try:
                        require_file(path)
                        print(f"Model {path.name}: OK")
                    except FileNotFoundError:
                        missing.append(str(path))
                        print(f"Model {path.name}: MISSING")
            return 1 if missing else 0
        if args.command == "validate":
            data = json.loads(args.json_file.read_text(encoding="utf-8-sig"))
            validate_transcript(data, args.duration_ms)
            print("JSON корректен")
            return 0
        if args.command == "evaluate":
            texts = []
            for path in (args.reference, args.hypothesis):
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                validate_transcript(data)
                texts.append(" ".join(row["text"] for row in data["transcript"]))
            print(json.dumps(wer(*texts), ensure_ascii=False, indent=2))
            return 0
        from .pipeline import load_config, retry_llm, run
        cfg = load_config(args.config)
        if args.command == "run":
            output = args.output or args.audio.with_suffix(".json")
            check_paths([output, args.session, args.report], [args.audio, args.config])
            metadata = run(args.audio, output, cfg, args.language, args.session, args.report,
                           progress=lambda stage: print(f"Этап: {stage}", file=sys.stderr))
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
            if not metadata["speech_detected"]:
                print("Речь не обнаружена", file=sys.stderr)
        else:
            check_paths([args.output], [args.session, args.config])
            print(json.dumps(retry_llm(args.session, args.output, cfg), ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Обработка прервана", file=sys.stderr)
        return 130
    except MemoryError:
        print("Ошибка: недостаточно оперативной памяти", file=sys.stderr)
        return 2
    except Exception as exc:
        # No traceback or raw native message: either may contain private text.
        category = type(exc).__name__
        messages = {"FileNotFoundError": "файл или модель отсутствует",
                    "PermissionError": "нет доступа к файлу",
                    "ModuleNotFoundError": "не установлена необходимая библиотека",
                    "JSONDecodeError": "повреждён JSON конфигурации или входных данных",
                    "ValueError": "неверный файл, конфигурация или данные",
                    "OSError": "ошибка чтения, записи или недостаточно ресурсов"}
        print(f"Ошибка [{category}]: {messages.get(category, 'сбой обработки; проверьте модели и файл')}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
