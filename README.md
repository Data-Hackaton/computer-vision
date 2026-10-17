# computer-vision

Два пакета и Temporal-воркер на task queue `cv-gpu`:

- `cv` — видео → `timeline.json`: люди, руки, объекты, треки, примитивные события.
- `decision` — `timeline.json` + `rules.yaml` → нарушения в схеме API: тип, таймкод, кадры, описание, норматив.

Воркер берёт одну активность за раз: на GTX 1650 4 ГБ модели грузятся по очереди. Цель — не больше 60 секунд на видео длиной 145 секунд.

## Ограничения на модели

- Только open source с лицензией, разрешающей коммерческое использование: права на решение уходят ForteBank. Лицензию каждой модели записываем в таблицу ниже.
- Ultralytics YOLO не берём: AGPL-3.0.
- Никакого распознавания и идентификации лиц. Роль человека (кассир, клиент, инкассатор) определяется по положению и поведению трека.
- После загрузки весов работа без интернета (`HF_HUB_OFFLINE=1`).
- Никаких зон в абсолютных координатах кадра: финальное видео снято под другим углом.

| Задача | Кандидат | Лицензия |
|---|---|---|
| Люди, объекты | RT-DETR / D-FINE | Apache-2.0 |
| Open-vocab объекты (деньги, сумка, ящик) | OWLv2 / Grounding DINO | Apache-2.0 |
| Поза, запястья | ViTPose / RTMPose | Apache-2.0 |
| Трекинг | ByteTrack (`supervision`) | MIT |

## Контракт `timeline.json` (черновик)

Словарь `events[].kind` фиксируется после каталога нарушений (фаза 0 roadmap). Время везде в секундах от начала видео.

```json
{
  "video": {"duration_sec": 25.93, "fps": 30, "width": 464, "height": 848, "sample_fps": 5},
  "tracks": [
    {
      "id": 3, "cls": "person", "role": "cashier",
      "frames": [{"t": 0.2, "bbox": [x1, y1, x2, y2], "conf": 0.91, "wrists": [[x, y], [x, y]]}]
    },
    {"id": 7, "cls": "cash", "frames": [{"t": 4.0, "bbox": [x1, y1, x2, y2], "conf": 0.66}]}
  ],
  "events": [
    {"kind": "hand_over", "t_start": 21.4, "t_peak": 22.0, "t_end": 22.8,
     "actors": [3, 5], "objects": [7], "bbox": [x1, y1, x2, y2], "conf": 0.74}
  ]
}
```

## `rules.yaml` (черновик)

Одно правило на тип нарушения. `norm` заполняется только по тексту документа на adilet.zan.kz: номер пункта по памяти не пишем, за это снимают баллы.

```yaml
- id: cash_handover_outside_tray
  type: "Передача наличных в обход установленного порядка"
  predicate: hand_over            # функция в decision/predicates.py
  params: {min_conf: 0.5}
  norm:
    document: "<название акта НБРК>"
    clause: "<пункт>"
    url: "https://adilet.zan.kz/rus/docs/<id>"
    quote: "<цитата из пункта>"
  description: "{actor_from} передаёт {object} {actor_to} на {t:.1f} с."
  evidence: [before, peak, after]
```

Описание составляется только из того, что видно на видео; выводы о данных, которых на видео нет (проводки, сверка), запрещены п. 7 ТЗ.

## Запуск

Через docker compose из `infra` (образ `cv`, GPU через CDI, данные — `data_prepare/data` → `/data`; `src/` смонтирован, правки кода без пересборки):

```sh
cd ../infra
docker compose --profile tools build cv
docker compose --profile tools run --rm cv python -m cv.prelabel --round r1          # предразметка Grounding DINO
docker compose --profile tools run --rm cv python -m cv.train_yolo --dataset det_r1   # YOLO26s
docker compose --profile tools run --rm cv python -m cv.train_rfdetr --dataset det_r1 # RF-DETR small
docker compose --profile tools run --rm cv python -m cv.eval_detector --dataset det_r1 --yolo <best.pt> --rfdetr <checkpoint.pth>
```

На GTX 1650 (4 ГБ): Grounding DINO только в fp32 (в fp16 — NaN), вход 640 px; Ultralytics сам выключает AMP; YOLO26s при 864 px влезает только с batch 1 — для обучения уменьшать imgsz или брать yolo26n.

## Стекло и отражения

В кадре стеклянная перегородка с дверью: в ней отражаются кассир и клиенты, сквозь неё видны люди в соседнем помещении — это главный источник ложных срабатываний. Детектор учит класс `ignore_zone` (бокс по стеклянной области) и класс `reflection` (человек в стекле). Постобработка детекций кадра:
- мелкие предметы (`hand`, `banknotes`, `id_card`, `phone`, `bag`, `foreign_object`, `computer`), у которых центр внутри `ignore_zone` и бокс не пересекает `table`, отбрасываются;
- `reflection` в журнал действий не попадает;
- `person` целиком внутри зоны, не в двери и не у её низа — считается отражением.

Та же логика уже работает в предразметке (`src/cv/prelabel.py`: `video_zone`, `SMALL_IN_ZONE`).
