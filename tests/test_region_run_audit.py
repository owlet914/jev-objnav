import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from region_run_fingerprint import source_fingerprint
from verify_region_run import verify
from preflight_jev_region import validate_health, implementation_hash

class RegionRunAuditTests(unittest.TestCase):
    def test_fingerprint_changes_for_untracked_and_staged_file_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            subprocess.run(['git','init','-q',directory],check=True,capture_output=True)
            (root/'nav_jev_bridge').mkdir()
            source=root/'nav_jev_bridge/region_graph.py'
            source.write_text('version1')
            first=source_fingerprint(root)['sha256']
            source.write_text('version2')
            second=source_fingerprint(root)['sha256']
            self.assertNotEqual(first,second)
            subprocess.run(['git','add','.'],cwd=root,check=True,capture_output=True)
            self.assertEqual(second,source_fingerprint(root)['sha256'])
            source.write_text('version3')
            self.assertNotEqual(second,source_fingerprint(root)['sha256'])

    def test_health_rejects_demo_old_profile_and_old_code(self):
        valid={'status':'ok','provider':{'name':'openrouter','model':'typesafe/jev-1.13','diagnostics_version':'region-audit-v2','implementation_sha256':implementation_hash()},
               'state_profile':'region_graph','schema_version':'jev-region-graph/1.1','trace_enabled':True,'trace_dir':'local'}
        validate_health(valid)
        for field,value in [('name','demo'),('diagnostics_version','upstream-errors-v1')]:
            row=copy.deepcopy(valid); row['provider'][field]=value
            with self.assertRaises(ValueError): validate_health(row)
        row=copy.deepcopy(valid); row['state_profile']='full'
        with self.assertRaises(ValueError): validate_health(row)

    def test_exit_zero_without_results_or_traces_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'traces').mkdir()
            manifest={'run_id':'audit','expected_episode_count':24,'source_fingerprint':{'sha256':'abc'},
                      'bridge_health':{'trace_dir':str(root/'traces'),'provider':{'instance_id':'instance'}}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'habitat_exit_code.txt').write_text('0')
            with patch('verify_region_run.source_fingerprint',return_value={'sha256':'abc'}): report=verify(root)
            self.assertFalse(report['accepted'])
            self.assertIn('episode_count_mismatch',report['issues'])
            self.assertIn('no_production_api_trace',report['issues'])

    def test_technical_failure_and_missing_execution_do_not_pass(self):
        from nav_jev_bridge.api_diagnostics import request_facts
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); traces=root/'traces'; traces.mkdir()
            manifest={'run_id':'smoke_audit','expected_episode_count':1,'source_fingerprint':{'sha256':'abc'},
                      'bridge_health':{'trace_dir':str(traces),'provider':{'instance_id':'instance'}}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            episode={'episode_id':'ep','success':0,'spl':0.0,'completed':True,'technical_failure':True}
            (root/'episodes.jsonl').write_text(json.dumps(episode)+'\n')
            request={'state':{},'questions':{'next_goal':{'criteria':{'c':'choice'}}}}
            trace={'request_id':'smoke_audit:ep:obs-4:map-5','episode_id':'ep','observation_id':4,
                   'provider':{'name':'openrouter','instance_id':'instance'},'state_profile':'region_graph',
                   'model_request':request,'wire':request_facts(request),'decision':{'status':'GOAL','action_type':'explore'},
                   'upstream':{'raw_response':{'id':'synthetic'},'usage':{'input_tokens':10}}}
            (traces/'synthetic.json').write_text(json.dumps(trace))
            with patch('verify_region_run.source_fingerprint',return_value={'sha256':'abc'}): report=verify(root)
            self.assertIn('technical_or_incomplete_episode',report['issues'])
            self.assertIn('missing_execution_chain:ep',report['issues'])
            self.assertFalse(report['accepted'])

if __name__=='__main__': unittest.main()
