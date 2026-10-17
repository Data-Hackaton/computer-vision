"""Предразметка целых видео для CVAT из предсказаний детектора: «CVAT for video 1.1», треки с ключевыми кадрами.

Источник — кэш L1 (python -m cv.infer): боксы 4 кадра/с. Номер кадра — исходный (30 кадров/с), CVAT интерполирует
между ключевыми кадрами, разметчик правит, а не рисует. Люди — треки и роли (кассир / клиент) из L2
(decision.scene); остальные классы связываются между соседними кадрами по классу и перекрытию (разрыв до
LINK_GAP_SEC). Задача в CVAT — из того же .mov, шаг кадров 1, метки — data_prepare/cvat/labels.json.
  python -m cv.export_cvat --detector yolo   → data/cvat/prelabels-video-<детектор>.zip (по файлу X-n.xml на видео)
"""

import argparse
import csv
import io
import json
import zipfile
from xml.sax.saxutils import escape

from cv.detectors import DATA_DIR
from cv.geometry import iou
from cv.infer import cache_path
from decision.scene import assign_roles, frames_of, track_people

LINK_GAP_SEC = 1.0  # бокс того же класса не дальше этого во времени и с перекрытием > LINK_IOU — тот же объект
LINK_IOU = 0.3
LABELS = DATA_DIR.parent / "cvat" / "labels.json"  # data_prepare/cvat/labels.json рядом с data/


def link(frames: list[tuple[float, list[dict]]]) -> dict[str, list[dict]]:
    """Треки: люди — по track_id из L2, остальные — жадно по классу и перекрытию с последним боксом трека."""
    tracks: dict[str, list[dict]] = {}
    alive: dict[str, tuple[float, list[float], str]] = {}  # трек → (время, бокс, класс)
    n = 0
    for t, boxes in frames:
        for b in boxes:
            if b["cls"] in ("person", "reflection") and "track_id" in b:
                tracks.setdefault(f"p{b['track_id']}", []).append({**b, "cls": "person" if b["cls"] == "reflection" else b["cls"]})
                continue
            best, score = None, LINK_IOU
            for key, (seen, last, cls) in alive.items():
                if cls == b["cls"] and 0 < t - seen <= LINK_GAP_SEC and (s := iou(b["bbox"], last)) > score:
                    best, score = key, s
            if best is None:
                n += 1
                best = f"o{n}"
            alive[best] = (t, b["bbox"], b["cls"])
            tracks.setdefault(best, []).append(b)
    return tracks


def attributes(b: dict) -> dict[str, str]:
    attrs = {}
    if b["cls"] == "person":
        attrs["role"] = (b.get("attributes") or {}).get("role", "unknown")
    elif b["cls"] == "ignore_zone":
        attrs["kind"] = "glass"
    return attrs


def video_xml(name: str, boxes: list[dict], fps: float, size: int, detector: str) -> str:
    frames = frames_of(boxes)
    track_people(frames, fps=4)
    assign_roles(frames, fps=4)
    labels = json.loads(LABELS.read_text())
    out = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>", "  <meta>", "    <task>",
           f"      <name>{escape(name)}</name>", f"      <size>{size}</size>", "      <mode>interpolation</mode>",
           "      <start_frame>0</start_frame>", f"      <stop_frame>{size - 1}</stop_frame>", "      <labels>"]
    for label in labels:
        out.append(f"        <label><name>{label['name']}</name><type>rectangle</type><attributes>")
        for a in label.get("attributes", []):
            out.append(f"          <attribute><name>{a['name']}</name><mutable>{str(a.get('mutable', False)).lower()}</mutable>"
                       f"<input_type>{a['input_type']}</input_type><default_value>{a.get('default_value', '')}</default_value>"
                       f"<values>{chr(10).join(a.get('values', []))}</values></attribute>")
        out.append("        </attributes></label>")
    out += ["      </labels>", "    </task>", f"    <source>{escape(name)}.mov, детектор {detector}</source>", "  </meta>"]
    for tid, (key, track) in enumerate(sorted(link(frames).items(), key=lambda kv: kv[1][0]["t"])):
        out.append(f'  <track id="{tid}" label="{track[0]["cls"]}" source="auto">')
        last = None
        for b in track:
            frame = min(size - 1, round(b["t"] * fps))
            if frame == last:
                continue
            x1, y1, x2, y2 = b["bbox"]
            attrs = "".join(f'<attribute name="{k}">{v}</attribute>' for k, v in attributes(b).items())
            out.append(f'    <box frame="{frame}" outside="0" occluded="0" keyframe="1" xtl="{x1:.2f}" ytl="{y1:.2f}" '
                       f'xbr="{x2:.2f}" ybr="{y2:.2f}" z_order="0">{attrs}</box>')
            last = frame
        end = min(size - 1, last + max(1, round(fps / 4)))  # трек закрывается на следующем опорном кадре
        if end > last:
            x1, y1, x2, y2 = track[-1]["bbox"]
            out.append(f'    <box frame="{end}" outside="1" occluded="0" keyframe="1" xtl="{x1:.2f}" ytl="{y1:.2f}" '
                       f'xbr="{x2:.2f}" ybr="{y2:.2f}" z_order="0"></box>')
        out.append("  </track>")
    out.append("</annotations>")
    return "\n".join(out) + "\n"


README = """Предразметка целых видео для CVAT (детектор {detector}, 4 кадра/с → треки с ключевыми кадрами).
Импорт: CVAT → задача из того же X-n.mov (шаг кадров 1, начальный кадр 0, метки из data_prepare/cvat/labels.json)
→ Actions → Upload annotations → формат «CVAT 1.1» → X-n.xml. Между ключевыми кадрами CVAT интерполирует.
Люди — треки с ролью (cashier / client), остальное — по классу и перекрытию. Разметчик правит: двигает ключевые кадры,
удаляет лишнее, дорисовывает пропущенное. Самые слабые классы детектора: banknotes, id_card, phone, bag.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default="yolo", choices=["yolo", "rfdetr"])
    args = ap.parse_args()
    with (DATA_DIR / "labels.csv").open() as f:
        rows = list(csv.DictReader(f))
    out = DATA_DIR / "cvat" / f"prelabels-video-{args.detector}.zip"
    buf, stats = io.BytesIO(), []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", README.format(detector=args.detector))
        for r in rows:  # копии (X-47 = X-4, X-48 = X-5) — свой файл под своим именем
            path = cache_path(r["sha256"], args.detector)
            if not path.exists():
                print(f"{r['video']}: нет кэша L1 ({path}) — python -m cv.infer --detector {args.detector}")
                continue
            fps = float(r["fps"])
            size = round(float(r["duration_sec"]) * fps)
            xml = video_xml(r["video"], json.loads(path.read_text())["boxes"], fps, size, args.detector)
            zf.writestr(f"{r['video']}.xml", xml)
            stats.append((r["video"], xml.count("<track "), xml.count('outside="0"')))
    out.write_bytes(buf.getvalue())
    print(f"→ {out}: {len(stats)} видео, треков {sum(s[1] for s in stats)}, ключевых кадров {sum(s[2] for s in stats)}, "
          f"{len(buf.getvalue()) / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
