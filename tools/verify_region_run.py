"""Validate and summarize a real region_graph run; never infer API use from health alone."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from region_run_fingerprint import source_fingerprint
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nav_jev_bridge.api_diagnostics import request_facts

ROOT=Path(__file__).resolve().parents[1]

def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []

def verify(run_dir):
    manifest=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    episodes=read_lines(run_dir/'episodes.jsonl'); events=read_lines(run_dir/'events.jsonl')
    issues=[]
    if len({row['episode_id'] for row in episodes})!=len(episodes): issues.append('duplicate_episode_results')
    if len(episodes)!=manifest['expected_episode_count']: issues.append('episode_count_mismatch')
    if any(row.get('technical_failure') or row.get('completed') is not True for row in episodes): issues.append('technical_or_incomplete_episode')
    if source_fingerprint(ROOT)['sha256']!=manifest['source_fingerprint']['sha256']: issues.append('source_changed')
    binary=ROOT/'jev_obj/devel/lib/exploration_manager/exploration_node'
    if manifest.get('planner_binary_sha256') and hashlib.sha256(binary.read_bytes()).hexdigest()!=manifest['planner_binary_sha256']:
        issues.append('planner_binary_changed')
    traces=[]
    health=manifest['bridge_health']
    for path in sorted(Path(health['trace_dir']).glob('*.json')):
        trace=json.loads(path.read_text(encoding='utf-8'))
        if not str(trace.get('request_id','')).startswith(manifest['run_id']+':'): continue
        if trace.get('provider',{}).get('instance_id')!=health['provider']['instance_id']: issues.append('provider_instance_mismatch')
        if trace.get('provider',{}).get('name')!='openrouter' or trace.get('state_profile')!='region_graph': issues.append('non_real_provider')
        request=trace.get('model_request')
        if request and trace.get('wire')!=request_facts(request): issues.append('wire_mismatch')
        if trace.get('decision',{}).get('status')=='GOAL' and not trace.get('upstream',{}).get('raw_response'): issues.append('missing_raw_upstream')
        traces.append((path,trace))
    if not traces: issues.append('no_production_api_trace')
    results=[]
    for row in episodes:
        decisions=[t for _,t in traces if t['episode_id']==row['episode_id']]
        valid=[t for t in decisions if t.get('decision',{}).get('status')=='GOAL']
        completions=[e for e in events if e['event']=='action_completion' and e.get('episode_id')==row['episode_id']]
        if not valid: issues.append('episode_without_real_decision:'+row['episode_id'])
        if valid and not any(e.get('request_id')==t['request_id'] and e.get('next_observation_id',-1)>t['observation_id'] for e in completions for t in valid):
            issues.append('missing_execution_chain:'+row['episode_id'])
        tokens=[t.get('upstream',{}).get('usage',{}).get('input_tokens') for t in valid if isinstance(t.get('upstream',{}).get('usage'),dict)]
        tokens=[v for v in tokens if isinstance(v,int)]
        modes={mode:sum(t['decision'].get('action_type')==mode for t in valid) for mode in ('explore','approach_object')}
        modes['observe_rotation']=sum(t['decision'].get('kind')=='observe_rotation' for t in valid)
        results.append({**row,'api_attempts':len(decisions),'api_valid_responses':len(valid),'api_errors':len(decisions)-len(valid),
                        'input_tokens_total':sum(tokens) if len(tokens)==len(valid) and tokens else None,
                        'max_input_tokens':max(tokens) if tokens else None,'modes':modes,
                        'api_latency_ms':[t.get('latency_ms') for t in decisions]})
    distinct={}
    for p,t in traces:
        if t.get('decision',{}).get('status')=='GOAL' and isinstance(t.get('wire',{}).get('request_bytes'),int):
            distinct[(t['request_id'],t['wire']['request_bytes'])]=(p,t)
    ordered=sorted(distinct.values(), key=lambda row:row[1]['wire']['request_bytes'])
    probes=[]
    if len(ordered)>=3 and len({t['wire']['request_bytes'] for _,t in ordered})>=3:
        chosen=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
        for size,(p,t) in zip(('small','medium','large'),chosen):
            probes.append({'size_relative_to_this_run':size,'trace_file':str(p),'request_id':t['request_id'],
                           'request_bytes':t['wire']['request_bytes'],'usage':t.get('upstream',{}).get('usage'),
                           'actual_model':t.get('upstream',{}).get('actual_model'),'generation_id':t.get('upstream',{}).get('generation_id'),
                           'counts':t.get('input_counts'), 'executed_in_navigation':True})
    report={'run_id':manifest['run_id'],'accepted':not issues,'issues':sorted(set(issues)),
            'completed_episodes':len(episodes),'expected_episodes':manifest['expected_episode_count'],
            'success_rate':sum(r['success'] for r in episodes)/len(episodes) if episodes else None,
            'mean_spl':sum(r['spl'] for r in episodes)/len(episodes) if episodes else None,
            'production_size_samples':probes,'episodes':results}
    (run_dir/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--admit-pilot',action='store_true')
    args=parser.parse_args(); report=verify(args.run_dir)
    if args.admit_pilot and not args.run_dir.name.startswith('smoke_'):
        raise SystemExit('Pilot refused: evidence must come from a smoke run')
    if args.admit_pilot and len(report['production_size_samples'])<3:
        raise SystemExit('Pilot refused: need three production states accepted by real Jev; run additional probes first')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['accepted']: raise SystemExit(1)
if __name__=='__main__': main()
