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
| `SAMPLE_FPS` | кадров в секунду на анализ, 4; `native` — каждый кадр видео (не больше `FRAME_BUDGET` = 9000 кадров на видео) |
| `DETECTOR_CONF` | порог уверенности, 0,35 |
| `DETECTOR_DEVICE` | `cpu` или номер GPU; пусто — GPU, если есть |
| `DETECTOR_VIEWS` | пусто — один проход; JSON `[{"name":"orig","weight":0.76},{"name":"unsharp",…},{"name":"clahe",…}]` — ансамбль видов кадра с WBF (`cv.views`) |
| `TRACKER` | `greedy` — жадный трекер людей; `bytetrack`, `botsort` — трекеры Ultralytics (`decision.tracking`) |
| `POSE` | `off`; `lite`, `full`, `heavy` — позы людей MediaPipe в `attributes.pose` (`cv.pose`), сигналы из них — `decision.posture`; `POSE_EVERY` — каждый n-й кадр |
| `STEP_CLASSIFIER` | `hybrid` — правила + обученный классификатор шагов (нет модели — `rules`); `rules` |
| `STEP_MODEL` | пусто — `data/runs/sop/steps_<DETECTOR>.joblib` |
| `VLM_URL` | пусто — без VLM (по умолчанию); `http://vlm:8080` — включить |
| `VLM_SAFE` | `1` — вопрос VLM про сейф (по умолчанию выключен, см. ниже) |

## L2 по схеме NVIDIA SOP Monitoring

У NVIDIA ([sop-monitoring-blueprints](https://github.com/NVIDIA/sop-monitoring-blueprints), Apache-2.0) четыре части:
1. DDM, обученный на размеченных таймкодах, режет поток на отрезки: `threshold`, `min_length_sec`, `max_length_sec`.
2. VLM Cosmos-Reason2 выбирает для отрезка номер шага из `actions.json`.
3. SOP-чекер (`MissingNumberDetector`) ищет по циклам пропущенные и переставленные шаги.
4. Качество меряют метрикой `seq_accuracy`.

Здесь взято всё, с поправкой на GTX 1650 и 5 минут на видео:

| NVIDIA | Здесь |
|---|---|
| DDM: границы событий | `decision/scene.py` — сигналы состояния сцены по кадрам из боксов L1 (клиент есть, деньги на столе и в машинке, экран, двое у места кассира…). `decision/steps.py` — сглаживание `SMOOTH_SEC`, минимальная длина `MIN_LENGTH_SEC`/`HOLD_SEC`, граница — смена состояния, длинный отрезок режется `MAX_LENGTH_SEC` |
| DDM, обученный на таймкодах | `decision/learned.py --target steps` — градиентный бустинг по окнам 4 с, метки — действия эталонного журнала (24 видео, 141 действие). В `STEP_CLASSIFIER=hybrid` добавляет к правилам шаги, которых правила не видят: сейф, удостоверение, передача денег клиентом, раскладка |
| VLM: отрезок → номер шага | `RulesClassifier` — переход между отрезками → действие журнала. `decision/vlm.py` — Qwen3-VL-2B (Cosmos-Reason2 — её дообучение) в сервисе `vlm` (llama.cpp, CPU) отвечает номером варианта на вопросы в спорных моментах |
| `actions.json` | `python -m decision.sop` → `data_prepare/reference/actions.json` из `scenarios.yaml` |
| SOP-чекер | `decision/sop_checker.py` — их `MissingNumberDetector` без изменений (Apache-2.0). Обязательные шаги цикла — из эталона: 75% чистых циклов обслуживания, это «пришёл клиент → деньги в машинку». Пропуски и перестановки — предупреждения «Пропущен шаг процедуры». Нормативные правила — L3 `data_prepare` |
| `seq_accuracy` | `decision/sop.py` — 1 − редакционное расстояние / длина эталона |

**Автоматы (pytransitions).** L2 и L3 написаны на конечных автоматах `transitions` (MIT). Схемы в mermaid — [`data_prepare/docs/procedures.md`](../data_prepare/docs/procedures.md), их генерирует `python -m decision.fsm --diagrams`.

| Уровень | Модуль | Автоматы |
|---|---|---|
| L2 | `decision/fsm.py` | `SceneModel` — по автомату на фасет сцены (клиент, посетители, кассир, стол, машинка, пачки, экран, телефон, двое у места кассира), действия выдают колбэки переходов; `InstanceMachine` — экземпляры процедуры |
| L3 | `data_prepare/src/data_prepare/sop_fsm.py` | `ProcedureMachine` — указатель по шагам шаблона; инварианты `Forbidden`, `ForbiddenAfter`, `Before`, `MaxDuration`; `DeskCarryover` — на видео |

Правила, которые смотрят на экземпляр целиком (обязательные шаги, `before`, `max_duration`), закрываются в `finalize()` экземпляра. В потоке для камер это момент, когда экземпляр закрылся.

Результат совпадает со старыми ручными реализациями до каждого срабатывания:
- L3 — на всех 24 журналах эталона и синтетике (`data_prepare/tests/test_sop_fsm.py`);
- L2 — на кэше всех 40 видео и синтетике (`tests/test_decision.py`).

Оценка `decision.eval` не изменилась.

**Роли.**
- Кассир — самый долгий трек и любой трек у его места; место — по первой четверти трека.
- Клиент — трек дольше секунды, хоть раз подходивший к столешнице.
- Двое у места кассира одновременно — посторонний за кассовым столом.

**Уход кассира.**
- Кассира нет дольше 3 с, или он пропал в конце ролика.
- Момент ухода — когда он последний раз был у своего места, но не раньше его последнего действия со столом.

**Экземпляры сценариев.**
- Новый `client_service` заводится на каждый приход клиента. Иначе L3 спутает деньги двух клиентов: будут ложные «пересчёт после ухода» и пропущенный `desk_carryover`.
- Уход кассира без клиента превращает текущий экземпляр в `leave_workplace`.
- Остальное без клиента — `cash_handling`.

Все экземпляры `partial`: L2 видит не все шаги, и пропущенный шаг не становится нарушением.

**Экран.** Яркость бокса ноутбука (`computer.bright` из L1), порог 0,35: рабочий экран 0,40–0,50, заставка 0,22–0,29. Падение яркости перед уходом — `screen_lock`.

### ECL: достоверность действий

`ECL=on` и `ECL_MODEL=<joblib>` включают оценку кандидатов L2 перед L3. По умолчанию ECL выключен.
Действие с низкой оценкой по-прежнему засчитывает шаг процедуры, но связанное именно с ним нарушение
становится предупреждением. Оценка и пояснение сохраняются в описании события. Неполностью размеченные
действия со столом не калибруются: отсутствие записи в журнале не означает, что действия не было.

Обучение: `python -m decision.evidence --detector yolo --cache-tag <L1-тег> --fit --out <модель.joblib>`.
Рядом сохраняется `.metrics.json`: AUC, Brier score, объём выборки по классам и пропущенные разбиения.
Оценка использует внешний leave-one-video-out; пороги для него подбираются по внутреннему разбиению
оставшихся видео. Непригодное разбиение не заменяется моделью, видевшей проверяемое видео.
L1 и классификатор шагов при этом фиксированы — это проверка слоя ECL, не всего обучения на новых камерах.

Формат артефакта — 2; старую экспериментальную модель нужно переобучить. `publish-model.sh --ecl <joblib>`
добавляет `model.json.ecl = {file, sha256}`. Релиз без поля `ecl` выключает ECL и сбрасывает путь к модели;
выбранный артефакт проверяется по SHA256, формату и признакам до первой задачи. Повреждённая или отсутствующая
модель при явно включённом ECL завершает запуск с ошибкой. `DATA_REPO` позволяет указать репозиторий разметки
при нестандартном расположении каталогов; обычные worktree правильно записывают `data_commit` автоматически.

Проверка 18.10: одинаковые боксы YOLO26s, ByteTrack, hybrid L2, 4 кадра/с; базовый вариант получает те же боксы
без `attributes.pose`. [Полный отчёт](docs/validation/ecl-20261018.json).

| Вариант | Верный вердикт | Таймкод ±3 с | Ложные |
|---|---:|---:|---|
| База | 24/40 | 3/21 | X-31 |
| MediaPipe lite | 24/40 | 3/21 | X-31 |
| ECL | 23/40 | 2/21 | X-31 |
| MediaPipe lite + ECL | 23/40 | 2/21 | X-31 |

ECL понижает верное нарушение X-30 до предупреждения. AUC ≈0,73, Brier ≈0,206 на 110 кандидатах;
прироста качества нет, ECL и поза остаются выключены. Поза: 280,4 с суммарно на 40 видео, максимум 34,3 с
на X-14 при ограничении контейнера 8 CPU, ниже существующего порога L1 100 с. Это замер L1 одного процесса,
не полный SLA очереди. При выключенных возможностях роли, сегменты, действия и события совпадают с `main`
на всех 40 видео (сравнение исключает только новые служебные поля действий `p_learned` и `t_seen`).

Воспроизведение из `infra` после сборки образа `cv`:

```sh
docker compose --profile tools run --rm --no-deps -T \
  -e POSE=lite -e SAMPLE_FPS=4 -e DETECTOR_DEVICE=cpu -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 \
  cv python -m cv.infer --detector yolo --cache-tag ecl-pose-check
docker compose --profile tools run --rm --no-deps -T \
  -v ../computer-vision/scripts:/app/scripts:ro -e TRACKER=bytetrack -e STEP_CLASSIFIER=hybrid \
  -e STEP_MODEL=/data/runs/sop/steps_yolo.joblib \
  cv python /app/scripts/validate-ecl.py --pose-cache-tag ecl-pose-check --out /data/runs/ecl-check
```

Скрипт создаёт отдельный набор кэшей, переобучает два калибратора и сохраняет четыре оценки в `comparison.json`.
Исходные кэши не изменяет. Для включения времени L1 в отчёт передайте `--l1-log` с журналом первого прогона.

### VLM

```sh
cd ../data_prepare/data/cache && mkdir -p vlm && cd vlm   # 1,8 ГБ + 0,8 ГБ, Apache-2.0
curl -LO https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q8_0.gguf
curl -LO https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-F16.gguf
cd ../../../../infra && docker compose --profile app up -d vlm                          # localhost:48091
```

Вопросы — пронумерованные варианты, как в `vlm_prompts` NVIDIA. Кадр — увеличенный фрагмент вокруг бокса L1 (сейф, столешница). На целом кадре в 448 px мелкие предметы занимают десяток пикселей.

**Шлюз (17.10).**
- 13 известных случаев: целый кадр — 7 из 13 при 9 с на вопрос; фрагмент — 10 из 13 при 2 с.
- Сейф — 4 из 4, люди на стороне кассира — 2 из 2.
- Стол — модель почти всегда отвечает «штампы и документы».
- Экран — заставку блокировки (X-29) называет активным экраном.

**На 40 видео** в реальных моментах:
- «Сейф открыт» модель отвечает и тогда, когда кассир сам работает у сейфа, который стоит вплотную к креслу (X-12, X-23): +1 нарушение, +2–3 ложных.
- «Двое на стороне кассира» отвечает почти при каждом приходе клиента.

Итог: VLM по умолчанию выключена. При `VLM_URL` она только подтверждает признак L1 «двое у места кассира», а вопрос про сейф включается отдельно (`VLM_SAFE=1`). Ответы кэшируются в `data/features/<sha>/vlm_*.json`.

## Оценка

```sh
cd ../infra
docker compose --profile tools run --rm cv python -m cv.infer --detector yolo         # кэш L1: data/features/<sha>/l1_yolo.json
docker compose --profile tools run --rm cv python -m decision.eval --detector yolo     # вердикт, таймкод ±3 с, seq_accuracy
docker compose --profile tools run --rm -e VLM_URL=http://vlm:8080 cv python -m decision.eval --detector yolo --vlm
docker compose --profile tools run --rm cv python -m decision.learned --detector yolo --target steps        # LOO шагов
docker compose --profile tools run --rm cv python -m decision.learned --detector yolo --target steps --fit  # модель hybrid
```

- **dev** — видео, кадры которых в train детектора, и X-1 (по его журналу писались правила).
- **honest** — val детектора (X-19, X-27, X-37, X-45) и 16 видео с нарушениями, тип которых определён по кадрам: `data_prepare/reference/violations_unreviewed.yaml`, `confirmed: false`.

Результаты на 40 видео (21 с нарушением, 19 чистых), YOLO26s итоговая:

| Режим | Точность вердикта | Нарушения с верным таймкодом ±3 с | Ложные на чистых |
|---|---|---|---|
| правила | 0,57 | 3 из 21 (+3 с другим временем) | 2 (X-31, X-35 — отражение в стекле посчитано вторым человеком) |
| правила + VLM | 0,60 на весах эпохи 7 | +1 (X-15) | +1–2 (сейф у кресла) |

`seq_accuracy` на 24 видео с журналом, проверка без одного видео: правила 0,19, обученный классификатор 0,22, гибрид 0,23.

Лучше всего ловятся:
- уход с незаблокированным компьютером (X-28);
- деньги предыдущего клиента на столе (X-38);
- посторонний за кассовым столом (X-30);
- деньги на столе при уходе кассира (X-26 — на весах эпохи 7).

Слабее всего — то, чего не видно в боксах:
- купюры в руках (полнота класса `banknotes` 0,07): X-1, X-34;
- состояние сейфа: X-13, X-20;
- штампы на столе: X-24;
- документы инкассатора: X-17.

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

## Выкатка модели

Модель на сервере — релиз GitHub `model-*` этого репозитория: веса, классификатор шагов, обученный на L1 этих весов, и `model.json` (детектор, порог, sha256, коммиты кода и данных, метрики обучения).

```sh
scripts/publish-model.sh yolo ../data_prepare/data/runs/yolo/det_l1-yolo26s \
    --steps ../data_prepare/data/runs/sop/steps_yolo.joblib     # --draft — без выкатки, --out <каталог> — только собрать
```

Публикация запускает workflow `model-deploy` на runner k3s-lab, он вызывает `infra/model.sh deploy <тег>`:
1. Скачивает релиз и сверяет sha256.
2. Прогоняет L1 на CPU по 40 видео и оценивает (`decision.eval --summary`) вместе с текущей моделью.
3. Не выкатывает, если верных вердиктов или попаданий ±3 с меньше, чем у текущей, больше чем на одно.
4. Ждёт пустой очереди, переключает cv-worker и проверяет, что все реплики поднялись с тегом в identity (`GET /api/v1/queue`) и контрольное видео анализируется.
5. При сбое возвращает прежнюю модель.

Вручную (Actions → model-deploy → Run workflow) — выкатить любой тег, `force` — без сравнения, `rollback` — вернуть предыдущую. Автовыкатку останавливает переменная организации `DEPLOY_FREEZE=1`.

С релизом cv-worker читает всё из `MODEL_DIR` (`cv.detectors.apply_model_dir`): `DETECTOR`, веса, `DETECTOR_CONF`, `STEP_MODEL`; без классификатора шагов в релизе — только правила. Кэш L1 релиза — `features/<sha>/l1_<детектор>@<тег>.json`, обычный кэш для обучения он не трогает. Тег модели — в `jobs.model` и `GET /api/v1/pipeline` (`model_version`).

## Стекло и отражения

В кадре стеклянная перегородка с дверью: в ней отражаются кассир и клиенты, сквозь неё видны люди в соседнем помещении. Детектор учит класс `ignore_zone` (стекло) и `reflection` (человек в стекле).

Постобработка (`cv/geometry.py`, общая с предразметкой) работает так:
- **Зона стекла** — медиана предсказанных `ignore_zone` по видео.
- **Мелкие предметы.** Если центр внутри зоны и бокс не пересекает `table`, предмет отбрасывается.
- **Люди.** `person` целиком внутри зоны, не в двери и не у её низа — это `reflection`. Если на месте одного человека есть и `person`, и `reflection`, остаётся более уверенный бокс.
