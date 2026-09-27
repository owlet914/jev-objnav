"""Content-based source/data fingerprints; never reads credential files."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE_ROOTS = ('nav_jev_bridge/', 'jev_obj/src/', 'jev_obj/config/', 'jev_obj/vlm/',
                'jev_obj/habitat2ros/', 'jev_obj/basic_utils/', 'jev_obj/llm/', 'tools/')
SOURCE_FILES = {'jev_obj/habitat_evaluation.py','jev_obj/region_evaluation_audit.py','jev_obj/params.py','pyproject.toml'}
SUFFIXES = {'.py','.cpp','.h','.hpp','.xml','.launch','.yaml','.yml','.sh','.toml','.txt'}

def source_fingerprint(root: Path):
    names = subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=root).decode('utf-8').split('\0')
    rows={}
    for name in sorted(set(names)):
        if not name or not (name in SOURCE_FILES or name.startswith(SOURCE_ROOTS)): continue
        p=root/name
        if p.suffix not in SUFFIXES: continue
        if any(part in ('data','.deps','build','devel','__pycache__') for part in p.parts): continue
        rows[name]=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'DELETED'
    raw=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    return {'sha256':hashlib.sha256(raw).hexdigest(),'files':rows,'scope':'working tree bytes including staged and untracked source/config; excludes credentials/data/logs'}

def pilot_manifest(root: Path):
    import gzip
    base=root/'jev_obj/data/datasets/objectnav/ovon_small/pilot'
    files=[base/'pilot.json.gz']+sorted((base/'content').glob('*.json.gz'))
    hashes={}; episodes=[]
    for p in files:
        hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
        data=json.loads(gzip.decompress(p.read_bytes()))
        for ep in data.get('episodes',[]):
            episodes.append({'scene_id':ep['scene_id'],'episode_id':ep['episode_id'],
                             'source_episode_id':ep.get('info',{}).get('ovon_episode_id',ep['episode_id']),
                             'target':ep['object_category']})
    if len(episodes)!=24: raise ValueError(f'pilot manifest expected 24 episodes, got {len(episodes)}')
    return {'split':'pilot','count':len(episodes),'files':hashes,'episodes':episodes,
            'sha256':hashlib.sha256(json.dumps({'files':hashes,'episodes':episodes},sort_keys=True).encode()).hexdigest()}
