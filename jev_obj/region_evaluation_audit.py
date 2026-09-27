"""Local audit events and run invariants for real region-graph evaluation."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.request import urlopen

class RegionEvaluationAudit:
    def __init__(self, cfg, env):
        self.root = Path(os.environ['JEV_REGION_RUN_DIR'])
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if cfg.habitat.dataset.split != 'pilot' or env.number_of_episodes != 24:
            raise RuntimeError('REGION_EVAL_DATASET_MISMATCH')
        if bool(cfg.habitat.environment.iterator_options.shuffle) or bool(cfg.habitat.environment.iterator_options.cycle):
            raise RuntimeError('REGION_EVAL_ITERATOR_MISMATCH')
        expected_single=int(self.manifest.get('smoke_episode_index',0)) if self.manifest['mode']=='smoke' else -1
        if int(cfg.test_epi_num)!=expected_single or Path(cfg.video_output_path).resolve()!=(self.root/'habitat_output').resolve():
            raise RuntimeError('REGION_EVAL_OUTPUT_OR_MODE_MISMATCH')
        self.check_bridge()
        import rospy
        parameters={name:rospy.get_param('/exploration_node/jev/'+name) for name in ('enabled','url','state_profile','timeout_ms','region_tile_size_m')}
        if parameters['enabled'] is not True or parameters['state_profile']!='region_graph' or parameters['url']!='http://127.0.0.1:8766/decide':
            raise RuntimeError('REGION_ROS_CONFIG_MISMATCH')
        self.manifest['effective_ros_parameters']=parameters
        self.manifest['region_contract']['region_tile_size_m']=parameters['region_tile_size_m']
        self.manifest['timeouts']['ros_http_s']=parameters['timeout_ms']/1000
        from omegaconf import OmegaConf
        (self.root/'effective_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True),encoding='utf-8')
        ordered=[]
        for ep in env.episodes:
            source_episode_id=(ep.info or {}).get('ovon_episode_id',ep.episode_id)
            ordered.append({
                'scene_id':ep.scene_id,
                'episode_id':ep.episode_id,
                'source_episode_id':source_episode_id,
                'target':ep.object_category,
            })
        expected=[
            (Path(ep['scene_id']).name,str(ep['source_episode_id']),ep['target'])
            for ep in self.manifest['dataset_manifest']['episodes']
        ]
        actual=[
            (Path(ep['scene_id']).name,str(ep['source_episode_id']),ep['target'])
            for ep in ordered
        ]
        (self.root/'effective_episode_order.json').write_text(json.dumps(ordered,indent=2),encoding='utf-8')
        if actual != expected:
            raise RuntimeError('REGION_EVAL_EPISODE_LIST_MISMATCH')
        self.manifest['effective_config_sha256']=hashlib.sha256((self.root/'effective_config.yaml').read_bytes()).hexdigest()
        self.manifest['seed']=int(cfg.habitat.seed)
        (self.root/'manifest.json').write_text(json.dumps(self.manifest,ensure_ascii=False,indent=2),encoding='utf-8')

    def check_bridge(self):
        with urlopen('http://127.0.0.1:8766/health',timeout=5) as response: health=json.load(response)
        old=self.manifest['bridge_health']
        if health['provider']['instance_id'] != old['provider']['instance_id'] or health['state_profile']!='region_graph' or health['provider']['name']!='openrouter':
            raise RuntimeError('REGION_BRIDGE_INSTANCE_CHANGED')

    def event(self, kind, **fields):
        row={'event':kind,'timestamp_unix_s':time.time(),'run_id':self.root.name,**fields}
        with (self.root/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush()
        return row

    def result(self, **fields):
        row=self.event('episode_result',**fields)
        with (self.root/'episodes.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
