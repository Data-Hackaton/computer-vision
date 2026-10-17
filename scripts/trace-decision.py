"""Inspect L2 tracks and oracle L3 results without running L1."""
import argparse
import csv
import json
from collections import Counter

import yaml

from cv.detectors import DATA_DIR, REFERENCE_DIR, Detection
from cv.geometry import median_box
from cv.infer import cache_path
from cv.pipeline import build_video_events, load_spec
from decision.steps import analyze

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--videos', nargs='*')
ap.add_argument('--cache-tag', default='baseline')
args = ap.parse_args()
spec = load_spec()
journals = {**yaml.safe_load((REFERENCE_DIR/'actions.yaml').read_text()),
            **yaml.safe_load((REFERENCE_DIR/'violations.yaml').read_text())}
for row in csv.DictReader((DATA_DIR/'labels.csv').open()):
    if row['duplicate_of'] or args.videos and row['video'] not in args.videos:
        continue
    cache = json.loads(cache_path(row['sha256'], 'yolo', args.cache_tag).read_text())
    res = analyze(cache['boxes'], cache['fps'])
    tracks = []
    for tid, role in res['roles'].items():
        bs = [b for b in cache['boxes'] if b.get('track_id') == tid]
        tracks.append({'id':tid, 'role':role, 'classes':dict(Counter(b['cls'] for b in bs)),
                       't':[bs[0]['t'],bs[-1]['t']], 'box':median_box([b['bbox'] for b in bs])})
    oracle = None
    if row['video'] in journals:
        oracle = [e for e in build_video_events(Detection('reference', instances=journals[row['video']]['instances']), spec)
                  if e['status'] == 'violation']
    print(json.dumps({'video':row['video'], 'tracks':tracks,'zones':res['zones'],
                      'learned':[a for a in res['actions'] if a.get('source') == 'learned'],
                      'classes':dict(Counter(b['cls'] for b in cache['boxes'])),
                      'oracle':oracle},ensure_ascii=False))
