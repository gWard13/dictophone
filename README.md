# Диктофон — локальная расшифровка подкастов и разговоров

Python-библиотека и CLI: **MP3 → диаризация → faster-whisper → согласование реплик → локальная LLM → JSON**.

Сейчас разрабатывается только Python-логика. Android-интеграция в текущую работу не входит.

## Возможности и модели

- Чтение локального MP3, преобразование в mono PCM 16 кГц.
- Русский/английский ASR: faster-whisper / Whisper small, CPU INT8, метки слов.
- Автоматическое определение участников: sherpa-onnx, сегментация pyannote, голосовые эмбеддинги 3D-Speaker, кластеризация.
- Согласование слов с говорящими по времени; `speaker_N`, `unknown`, `overlap`, `[Неразборчиво]`.
- Пунктуация и регистр: локальная Qwen 0.5B GGUF. Изменение слов запрещено проверкой; при отказе LLM сохраняется ASR.
- JSON-экспорт, повтор только LLM, нарезка подкастов, WER по ручному эталону и отдельный отчёт RTF.

Интернет нужен для установки зависимостей и первоначальной загрузки весов. Обработка использует локальные модели, аудио во внешние сервисы не передаётся.

## Структура

Каталоги организованы по [шаблону хакатона](https://github.com/grigorevmp/hacathon_template); его код не копировался.

```text
src/dictophone/        конвейер, модельные адаптеры, согласование, CLI
config/               примеры конфигураций моделей и нарезки
docs/                 архитектура, ограничения, протокол и результаты тестов
artifacts/test-results/ обезличенные метрики проверок
artifacts/runs/       новые расшифровки и отчёты, вне Git
scripts/              первоначальная загрузка моделей
tests/                логика, ошибки и настоящий MP3-кодек
data/raw/             исходные MP3, вне Git
data/clips/           фрагменты и их manifest.json, вне Git
data/references/      ручные эталоны, вне Git
models/               локальные веса, вне Git
```

## Установка и запуск

Рекомендуется **Python 3.11 x64**, проверено на Windows. Из корня репозитория:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe -m pip install "llama-cpp-python>=0.3,<0.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
.venv/Scripts/python.exe scripts/prepare_models.py
Copy-Item config/pipeline.example.json config/pipeline.local.json
.venv/Scripts/python.exe -m dictophone doctor --config config/pipeline.local.json
```

Проверенные версии Windows/Python 3.11 — `requirements-tested.txt`. На других платформах LLM может потребовать CMake и C++-компилятор. Если готового wheel LLM нет, понадобятся CMake и Visual Studio Build Tools с C++ на Windows. Подробности работы моделей: [архитектура](docs/ARCHITECTURE.md).

```powershell
.venv/Scripts/python.exe -m dictophone run "C:/audio/podcast.mp3" --config config/pipeline.local.json --language ru --output artifacts/runs/podcast.json --report artifacts/runs/podcast.metrics.json
```

Модели работают последовательно на CPU. Время отсчитывается от начала импортированного файла, включая паузы. Язык задаётся явно, по умолчанию русский. Длительность программно не ограничена, возможность обработки зависит от ресурсов.

Для проверки записей с музыкальной заставкой доступен экспериментальный `config/pipeline.speech-guided.json`: ASR работает в контекстных окнах вокруг речевых интервалов, порог кластеризации равен 0,8. Он проверен на одном пользовательском примере и не является универсальной настройкой. См. [результаты эксперимента](docs/TEST_RESULTS.md).

`config/pipeline.short-turns.json` добавляет автоматическое сравнение коротких реплик с голосовыми образцами остальных интервалов. Это частичное улучшение, а не подтверждённое решение идентификации: [проверка коротких реплик](docs/TEST_RESULTS.md).

## Нарезка подкаста

Один фрагмент от первой до второй минуты:

```powershell
.venv/Scripts/python.exe -m dictophone cut "C:/audio/podcast.mp3" --start-ms 60000 --end-ms 120000 --output data/clips/podcast_01.mp3
```

Для нескольких фрагментов заполните `config/podcast_clips.local.json` по примеру и выполните:

```powershell
.venv/Scripts/python.exe -m dictophone prepare-clips config/podcast_clips.local.json
```

Интервалы в примере условные. Метки расшифровки относятся к клипу; смещение относительно выпуска хранится отдельно. Выбор записей и правила разметки: [тестовые подкасты](data/README.md).

## Экспорт и проверки

```json
{
  "transcript": [
    {
      "speaker_id": "speaker_1",
      "start_ms": 1250,
      "end_ms": 4380,
      "text": "Добрый день!"
    }
  ]
}
```

Это пример схемы, а не модельный результат. Файл: UTF-8 без BOM, LF, отступ 2. При отсутствии речи массив пустой.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m dictophone validate artifacts/runs/podcast.json
.venv/Scripts/python.exe -m dictophone evaluate data/references/podcast.json artifacts/runs/podcast.json
```

Реальные прогоны и выявленные ошибки: [отчёт](docs/TEST_RESULTS.md). GitHub Actions проверяет логику и кодек без загрузки весов; точность моделей в CI не оценивается.

Диаризация может дробить один голос; короткие реплики и перекрывающаяся речь сложны. Смешанные голоса не разделяются. VAD по умолчанию выключен после обнаруженного пропуска короткой русской фразы. Качество на подкастах требует ручных эталонов; WER/DER без них не заявляются.

## Репозиторий

Аудио, веса, приватные расшифровки и локальные настройки исключены через `.gitignore`. Основной репозиторий: https://github.com/gWard13/dictophone (ветка `main`). У весов и сторонних записей собственные лицензии; лицензия исходников остаётся решением владельца проекта.

## Повтор только LLM

Для явного сохранения приватной сессии:

```powershell
.venv/Scripts/python.exe -m dictophone run "C:/audio/podcast.mp3" --config config/pipeline.local.json --session artifacts/runs/private.session.json --output artifacts/runs/podcast.json
.venv/Scripts/python.exe -m dictophone retry-llm artifacts/runs/private.session.json --config config/pipeline.local.json --output artifacts/runs/corrected.json
```

Сессия содержит приватный текст и не включается в Git. Обычный запуск без `--session` её не сохраняет.

## Воспроизводимая подготовка

При скачивании моделей можно задать `--asr-revision` и `--llm-revision` конкретными commit hash. Сохраните полученный model manifest и окружение. По умолчанию используются ревизии `main`; ограничения воспроизводимости описаны отдельно.

## Документация

- [Архитектура и модели](docs/ARCHITECTURE.md)
- [Ограничения и нерешённые ошибки](docs/LIMITATIONS.md)
- [Формат входа, выхода и промежуточных данных](docs/PROTOCOL.md)
- [Результаты тестов и экспериментов](docs/TEST_RESULTS.md)

## Работа с Git

```powershell
git clone https://github.com/gWard13/dictophone.git
cd dictophone
```

Для приватного репозитория требуется авторизация GitHub. ZIP не содержит `.git`; после его распаковки при необходимости выполните `git init -b main`.

Перед публикацией изменений проверьте `git status --short` и `git diff --cached --stat`. В коммит не должны попадать модели, аудио, приватные сессии и `config/*.local.json`. Используйте собственные настроенные имя и email Git.
