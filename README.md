# computer-vision

Видео → боксы (L1) → журнал действий (L2) → события со статусами (L3, движок `data_prepare`). Temporal-воркер `cv-worker` на task queue `cv-gpu` берёт одну задачу за раз (GTX 1650, 4 ГБ).

- `cv` — L1: детекторы, постобработка стекла, кадры-доказательства, воркер, обучение и предразметка.
- `decision` — L2 по схеме NVIDIA SOP Monitoring: треки и роли, отрезки, шаги процедуры, оценка.

## Ограничения на модели

- Только open source с лицензией, разрешающей коммерческое использование: права на решение уходят ForteBank.
- Ultralytics YOLO (AGPL-3.0) — только для сравнения; в прод идёт RF-DETR (Apache-2.0).
- Никакого распознавания лиц. Роль человека (кассир, клиент) — по поведению трека.
- После загрузки весов работа без интернета.
- Никаких зон в абсолютных координатах кадра: финальное видео снято под другим углом.

| Задача | Модель | Лицензия |
|---|---|---|
| Люди, деньги, предметы (14 классов `data_prepare/cvat/labels.json`) | RF-DETR small, YOLO26s для сравнения | Apache-2.0 / AGPL-3.0 |
| Предразметка для CVAT | Grounding DINO tiny | Apache-2.0 |
| Трекинг людей | свой: IoU + близость центров (`decision/scene.py`) | — |

## Детектор в сервисе

Переменные cv-worker (`infra/.env`):

| Переменная | Значение |
|---|---|
| `DETECTOR` | `rfdetr` (по умолчанию для прода), `yolo`, `stub` — заглушка по датасету |
| `DETECTOR_WEIGHTS` | пусто — `data/runs/rfdetr/det_l1-rfdetr-small/checkpoint_best_total.pth` или `data/runs/yolo/det_l1-yolo26s/weights/best.pt` |
| `SAMPLE_FPS` | кадров в секунду на анализ, 4 |
| `DETECTOR_CONF` | порог уверенности, 0,35 |
| `DETECTOR_DEVICE` | `cpu` или номер GPU; пусто — GPU, если есть |
| `STEP_CLASSIFIER` | `rules` |

## L2 по схеме NVIDIA SOP Monitoring

У NVIDIA ([sop-monitoring-blueprints](https://github.com/NVIDIA/sop-monitoring-blueprints), Apache-2.0) три шага:
1. DDM режет поток на отрезки по границам событий: `threshold`, `min_length_sec`, `max_length_sec`.
2. VLM Cosmos-Reason2 выбирает для отрезка номер шага из `actions.json`.
3. SOP-чекер ищет пропущенные и переставленные шаги.

Качество меряют метрикой `seq_accuracy`. Две модели (DDM и VLM на 2B параметров) не укладываются в GTX 1650 и 5 минут на видео, а данных для обучения DDM нет. Поэтому части заменены так:

| NVIDIA | Здесь |
|---|---|
| DDM: границы событий | `decision/scene.py`: сигналы состояния сцены по кадрам из боксов L1 — клиент есть, деньги на столе, в машинке, у кассира (сколько пачек), вещь или сумка на столе, удостоверение, телефон в руке. `decision/steps.py`: сглаживание (`SMOOTH_SEC`), минимальная длина (`MIN_LENGTH_SEC`, `HOLD_SEC`), граница отрезка — смена состояния, длинный отрезок режется (`MAX_LENGTH_SEC`) |
| VLM: отрезок → шаг | `RulesClassifier`: переход между отрезками → действие журнала (`put banknotes desk`, `person_leave client`, `split_bundle`, `phone_call` и т. д.). Интерфейс — под будущий VLM (`STEP_CLASSIFIER`) |
| `actions.json` | `python -m decision.sop` → `data_prepare/reference/actions.json` из `scenarios.yaml`: пронумерованные шаги, «действие вне процедуры», `actions_can_be_skipped` |
| SOP-чекер | L3 `data_prepare`: порядок и пропуски шагов → предупреждения, нормативные инварианты → нарушения |
| `seq_accuracy` | `decision/sop.py`: 1 − редакционное расстояние / длина эталона по видимым L2 шагам |

**Роли.** Кассир — самый долгий трек и любой трек у его места (вернувшийся кассир — новый трек). Клиент — остальные треки дольше секунды. Отражения отсеяны в L1.

**Экземпляры сценариев.**
- Новый `client_service` заводится на каждый приход клиента. Иначе L3 спутает деньги двух клиентов: будут ложные «пересчёт после ухода» и пропущенный `desk_carryover`.
- Уход кассира без клиента — `leave_workplace`.
- Остальное без клиента — `cash_handling`.

Все экземпляры `partial`: L2 видит не все шаги (осмотр удостоверения, бумаги), и пропущенный шаг не становится замечанием.

**Что лежит на столе в первом кадре**, считается положенным до начала видео. Так ловятся деньги прошлого клиента в начале ролика.

## Оценка

```sh
cd ../infra
docker compose --profile tools run --rm cv python -m cv.infer --detector rfdetr     # кэш L1: data/features/<sha>/l1_rfdetr.json
docker compose --profile tools run --rm cv python -m decision.eval --detector rfdetr # вердикт, таймкод ±3 с, seq_accuracy
```

- **dev** — видео, кадры которых в train детектора, и X-1 (по его журналу писались правила).
- **honest** — val детектора (X-19, X-27, X-37, X-45) и 16 видео с нарушениями, тип которых определён по кадрам: `data_prepare/reference/violations_unreviewed.yaml`, `confirmed: false`.

## Обучение

```sh
cd ../infra
docker compose --profile tools run --rm data-prepare build-datasets --round r1 r1b --name det_l1
docker compose --profile tools run --rm cv python -m cv.train_yolo --dataset det_l1 --run det_l1-yolo26s --imgsz 640 --epochs 50 --patience 15
docker compose --profile tools run --rm cv python -m cv.train_rfdetr --dataset det_l1 --run det_l1-rfdetr-small --resolution 512 --batch 1 --accum 16 --epochs 30
docker compose --profile tools run --rm cv python -m cv.eval_detector --dataset det_l1 --yolo <best.pt> --rfdetr <checkpoint.pth>
```

`det_l1` собирается из выгрузок CVAT r1 и r1b. Предразметка пока не проверена человеком и принята как есть. Ручные правки попадут в датасет повторным `cvat-io export`.

На GTX 1650 рабочий стол занимает ~1,25 ГБ памяти:
- RF-DETR small при 640 px и batch 2 не помещается, работает 512 px и batch 1;
- Grounding DINO — только fp32 (в fp16 выдаёт NaN), вход 640 px.

## Стекло и отражения

В кадре стеклянная перегородка с дверью: в ней отражаются кассир и клиенты, сквозь неё видны люди в соседнем помещении. Детектор учит класс `ignore_zone` (стекло) и `reflection` (человек в стекле).

Постобработка (`cv/geometry.py`, общая с предразметкой) работает так:
- **Зона стекла** — медиана предсказанных `ignore_zone` по видео.
- **Мелкие предметы.** Если центр внутри зоны и бокс не пересекает `table`, предмет отбрасывается.
- **Люди.** `person` целиком внутри зоны, не в двери и не у её низа — это `reflection`. Если на месте одного человека есть и `person`, и `reflection`, остаётся более уверенный бокс.
