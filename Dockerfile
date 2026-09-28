# Помощник диспетчера (ЛЦТ2026, задача 3 «Билайн Бизнес») — веб-приложение в контейнере.
#
#   docker build -t dispatcher .
#   docker run --rm -p 8000:8000 dispatcher          # → http://127.0.0.1:8000
#
# torch — сборка для CPU (без CUDA, образ меньше в разы). Данные дня, кэши геокодера, матриц
# и геометрии — из репозитория: на выданных данных расчёт не ходит в интернет, из сети грузится
# только подложка карты. Портфель по умолчанию — 4 процесса поиска: контейнеру нужно 4 ядра и ~4 ГБ.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

WORKDIR /app
COPY requirements.txt requirements.txt
# Та же версия, что в requirements.txt: иначе pip заменит сборку CPU на сборку с CUDA с PyPI (гигабайты).
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0" \
 && pip install -r requirements.txt

COPY data/raw/obezlichivanie data/raw/obezlichivanie
COPY data/interim data/interim
COPY data/traffic data/traffic
COPY data/samples data/samples
COPY app.py run.py ./
COPY docs/DOCUMENTATION.html docs/ENGINE.html docs/
COPY src src
COPY config config
COPY web web
# Короткий расчёт при сборке: numba компилирует ядра поиска в кэш образа (первый запуск не ждёт
# компиляции), и сборка падает, если приложение не считает план.
RUN python run.py --region Югоцентр --seconds 3

EXPOSE 8000
CMD ["uvicorn", "app:api", "--host", "0.0.0.0", "--port", "8000"]
