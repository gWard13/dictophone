# Работа с репозиторием

Основной репозиторий: https://github.com/gWard13/dictophone, ветка `main`.

Организация каталогов следует https://github.com/grigorevmp/hacathon_template: `src`, `config`, `docs`, `artifacts`. Шаблон не клонировался, его файлы не включались в проект.

## Получение исходников

```powershell
git clone https://github.com/gWard13/dictophone.git
cd dictophone
```

Для приватного репозитория потребуется авторизация GitHub с доступом к нему. Модели, окружение и аудио не включены в Git: их подготовка описана в README.md.

ZIP содержит исходники без служебного `.git`. После распаковки ZIP создайте локальный репозиторий командой `git init -b main`, если собираетесь работать через Git.

## Последующие изменения

Используйте собственные настроенные имя и email Git. После изменений:

```powershell
git status --short
git add .
git diff --cached --stat
git commit -m "Describe the change"
git push origin main
```

В индексе не должно быть весов, аудио, приватных сессий и `config/*.local.json`. Обезличенные отчёты находятся в `artifacts/validation/`.

GitHub Actions запускает тесты на Linux/Python 3.11 без модельных весов. Для реального инференса нужны локальные модели и аудио.
