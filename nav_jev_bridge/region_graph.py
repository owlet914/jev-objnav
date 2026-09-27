"""Validation and model projection for the lossy Jev region-graph contract."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "jev-region-graph/1.1"
FORBIDDEN_MODEL_KEYS = {
    "cells", "good_cells", "points", "path_points", "raw_neighborhood",
    "occupancy_rows", "semantic_value_rows", "cost_matrix", "point_cloud",
}


class RegionGraphError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RegionGraphProjection:
    state: dict[str, Any]
    criteria: dict[str, str]
    option_to_candidate: dict[str, str]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _walk_forbidden(value: Any, path: str = "state") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RegionGraphError("REGION_GRAPH_NONFINITE_VALUE", f"{path} must be finite or null")
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_MODEL_KEYS:
                raise RegionGraphError("REGION_GRAPH_RAW_DETAIL_LEAK", f"{path}.{key} is forbidden")
            _walk_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, f"{path}[{index}]")


# Versioned field contract. Unknown fields fail before projection/API calls.
FIELDS = {
 'state': 'schema_version representation identity task robot perception jev_obj_policy object_confirmation regions connections observation_actions options history coverage unassigned_entities selection_parameters semantic_alignment projection_summary',
 'representation': 'kind lossy region_definition frame_id units precision parameters',
 'units': 'position yaw score',
 'precision': 'position_decimals score_decimals local_execution_uses_full_precision',
 'parameters': 'region_tile_size_m tile_origin_xy native_resolution_m connectivity localization_tolerance_m history_window',
 'identity': 'episode_id request_id observation_id map_revision region_graph_revision',
 'task': 'target_object room_prior related_objects prior_source',
 'robot': 'region_id region_assignment pose', 'pose': 'x y yaw',
 'perception': 'image_text_match target_detection related_detection recent_observations',
 'image_text_match': 'query_text score_type raw_score valid available fallback backend observation_id synchronized_with_map invalid_reason',
 'target_detection': 'available fallback backend matched_boxes valid_masks observation_id',
 'related_detection': 'requested available',
 'recent_observation': 'observation_id image_text_match_valid image_text_match_raw_score target_match_count valid_mask_count',
 'jev_obj_policy': 'mode mode_id decision_order semantic_thresholds active_frontier_statistics dormant_frontier_statistics high_semantic_subset hybrid_branch available_stages earliest_available_stage earliest_available_priority interpretation',
 'semantic_thresholds': 'std_dev max_to_mean max_to_mean_percentage',
 'frontier_statistics': 'total_count reachable_count mean std_dev maximum max_to_mean zero_mean_ratio_definition adaptive_max_to_mean_threshold',
 'high_semantic_item': 'frontier_id selected value_to_mean',
 'object_confirmation': 'status pending_candidate_id pending_target_id start_observation_id last_observation_id attempts valid_fresh_hits required_fresh_hits max_attempts min_confidence_margin min_unique_viewpoints min_viewpoint_baseline_m last_outcome blocked_candidate_id blocked_target_id blocked_until_observation_id blocked_unique_viewpoints blocked_viewpoint_baseline_m stop_rule',
 'region': 'id bbox_xy anchor_xy free_area_m2 visited last_robot_visit_step visit_count lineage semantic objects frontiers',
 'bbox_xy': 'min max', 'lineage': 'parent_ids',
 'semantic': 'valid mean p90 max valid_area_fraction aggregation',
 'object': 'id center_xy bbox_xy region_assignment target_confidence target_observations target_point_count target_unique_viewpoints target_viewpoint_baseline_m target_viewpoint_bearing_span_rad extent_xyz_m stored_best_label computed_best_label high_confidence_eligible relaxed_eligible post_arrival_confirmation_status class_evidence class_evidence_omitted_count selection_status not_selectable_reason last_observed_id observations_since_seen updated_in_current_observation provisional selectable confidence_context verification_need',
 'class': 'class_index label fused_confidence observation_count observation_point_sum integer_function_score',
 'frontier': 'id state position_xy approach_region_id semantic_score semantic_valid semantic_method fusion_weight cluster_cells unknown_boundary_length_m information_gain reachability selectable not_selectable_reason',
 'connection': 'id from to bidirectional status anchor_path_length_m portal_width_estimate_m min_clearance_m verified_map_revision',
 'confidence_context': 'score_kind operational_gate delta_to_gate ratio_to_gate tier percentile_among_current_objects target_rank_among_queried_labels queried_runner_up_label margin_to_queried_runner_up queried_label_scope open_world_identity_verified',
 'verification_need': 'level reasons independent_view_recommended',
 'observation': 'id region_id direction rotation_rad expected_yaw selectable focus_kind focus_target_id focus_region_id focus_bearing_rad focus_confidence evidence_gaps purpose expected_effect',
 'option': 'id action_type exploration_kind target_id region_id distance_m route_region_ids candidate_id local_path_verified repeated_selection_count source_stage source_priority original_policy_stage_active safety_mode semantic_score information_gain focus_context',
 'focus_context': 'action_role focus_kind focus_target_id focus_region_id evidence_goal evidence_gaps expected_resolution confidence_tier confidence_delta_to_gate confidence_percentile semantic_percentile region_visited region_visit_count',
 'history': 'step_count action_origin last_mode last_target_id low_level_action_completed goal_reached collision_count latest_collision last_step_progress_m cumulative_progress_m window_size omitted_selections',
 'coverage': 'raw_map_complete high_level_evidence_complete decision_evidence_complete summary_contract_valid source_profile regions_total regions_exported objects_total objects_exported frontiers_total frontiers_exported executable_candidates_total executable_candidates_exported omitted_entities detail_policy edge_absence_meaning',
 'unassigned_entities': 'objects frontiers',
 'selection_parameters': 'min_target_confidence min_target_observations min_independent_viewpoints min_viewpoint_baseline_m eligibility_evaluated_locally_at_full_precision class_function_score_definition',
 'semantic_alignment': 'object_confidence queried_class_comparison image_text_match frontier_semantics reference_policy_examples',
 'score_alignment': 'meaning calibrated_probability operational_gate tiers current_distribution interpretation',
 'distribution': 'count minimum median p90 maximum',
 'class_alignment': 'scope top_rank_meaning missing_competitors_meaning',
 'examples': 'borderline_object supported_object uncertain_region',
 'projection_summary': 'view all_safe_options_retained regions_total regions_detailed regions_summarized objects_total objects_detailed frontiers_total frontiers_detailed detail_selection local_execution_retains_full_state',
}


def _keys(value, kind):
    if not isinstance(value, dict):
        raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR', f'{kind} must be an object')
    unknown = set(value) - set(FIELDS[kind].split())
    if unknown:
        raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR', f'{kind} unexpected fields: {sorted(unknown)}')


def _xy(value):
    return isinstance(value, list) and len(value) == 2 and all(_finite(x) for x in value)


def _bbox(value):
    _keys(value, 'bbox_xy')
    if not _xy(value.get('min')) or not _xy(value.get('max')) or any(a>b for a,b in zip(value['min'],value['max'])):
        raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR', 'invalid bbox')


def _validate_fields(state):
    _walk_forbidden(state)
    _keys(state, 'state')
    for kind in ('representation','identity','task','robot','perception','jev_obj_policy','object_confirmation','history','coverage','unassigned_entities','selection_parameters','semantic_alignment','projection_summary'):
        if kind in state: _keys(state[kind], kind)
    if 'semantic_alignment' in state:
        for key in ('object_confidence', 'image_text_match', 'frontier_semantics'):
            _keys(state['semantic_alignment'][key], 'score_alignment')
            _keys(state['semantic_alignment'][key]['current_distribution'], 'distribution')
        _keys(state['semantic_alignment']['queried_class_comparison'], 'class_alignment')
        _keys(state['semantic_alignment']['reference_policy_examples'], 'examples')
    rep = state['representation']
    for kind in ('units','precision','parameters'):
        if kind in rep: _keys(rep[kind],kind)
    if not isinstance(rep.get('frame_id'),str) or not rep['frame_id']:
        raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','frame_id is required')
    if not state.get('task',{}).get('target_object'):
        raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','target_object is required')
    pose = state.get('robot',{}).get('pose')
    _keys(pose,'pose')
    if not all(_finite(pose.get(k)) for k in ('x','y','yaw')):
        raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','pose must be finite')
    for kind,value in state.get('perception',{}).items():
        if kind == 'recent_observations':
            if not isinstance(value, list):
                raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','recent_observations must be a list')
            for row in value:
                _keys(row, 'recent_observation')
            continue
        _keys(value,kind)
        if kind in ('image_text_match','target_detection'):
            valid = value.get('valid') is True if kind=='image_text_match' else value.get('available') is True
            if valid and (value.get('fallback') is not False or value.get('observation_id') != state['identity']['observation_id']):
                raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','perception validity/version mismatch')
            if kind=='image_text_match' and (valid != _finite(value.get('raw_score'))):
                raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','ITM validity/score mismatch')
    policy = state.get('jev_obj_policy')
    if not isinstance(policy, dict) or not isinstance(policy.get('decision_order'), list) or not policy['decision_order']:
        raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','jev_obj_policy decision order is required')
    _keys(policy.get('semantic_thresholds'), 'semantic_thresholds')
    for name in ('active_frontier_statistics', 'dormant_frontier_statistics'):
        stats = policy.get(name)
        _keys(stats, 'frontier_statistics')
        if any(not _finite(stats.get(key)) for key in ('mean','std_dev','maximum','max_to_mean','adaptive_max_to_mean_threshold')):
            raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR',f'{name} contains invalid statistics')
    for item in policy.get('high_semantic_subset', []):
        _keys(item, 'high_semantic_item')
    if not isinstance(policy.get('available_stages'), list) or not policy['available_stages']:
        raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','jev_obj_policy available stages are required')
    confirmation = state.get('object_confirmation')
    if not isinstance(confirmation, dict) or confirmation.get('status') not in ('idle','pending','rejected_cooldown'):
        raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','invalid object confirmation status')
    for key in ('attempts','valid_fresh_hits','required_fresh_hits','max_attempts'):
        if type(confirmation.get(key)) is not int or confirmation[key] < 0:
            raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','invalid object confirmation counters')
    if confirmation['required_fresh_hits'] < 1 or confirmation['max_attempts'] < confirmation['required_fresh_hits']:
        raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','invalid object confirmation policy')
    for region in state.get('regions',[]):
        _keys(region,'region'); _bbox(region.get('bbox_xy'))
        if not _xy(region.get('anchor_xy')) or not _finite(region.get('free_area_m2')) or region['free_area_m2'] <= 0:
            raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','invalid region geometry')
        if any(x<lo or x>hi for x,lo,hi in zip(region['anchor_xy'],region['bbox_xy']['min'],region['bbox_xy']['max'])):
            raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','anchor outside bbox')
        if 'lineage' in region: _keys(region['lineage'],'lineage')
        sem=region.get('semantic'); _keys(sem,'semantic')
        fraction=sem.get('valid_area_fraction')
        if not _finite(fraction) or not 0<=fraction<=1 or type(sem.get('valid')) is not bool:
            raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','invalid semantic validity')
        if sem['valid']:
            if fraction<=0 or not all(_finite(sem.get(k)) for k in ('mean','p90','max')) or sem['p90']>sem['max'] or sem['mean']>sem['max']+0.0001:
                raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','invalid semantic statistics')
        elif fraction!=0 or any(sem.get(k) is not None for k in ('mean','p90','max')):
            raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','invalid semantic values must be null')
    for container in list(state.get('regions',[]))+[state.get('unassigned_entities',{})]:
        for plural,kind in (('objects','object'),('frontiers','frontier')):
            for entity in container.get(plural,[]):
                _keys(entity,kind)
                if kind=='object':
                    if entity.get('bbox_xy') is not None: _bbox(entity['bbox_xy'])
                    for label in entity.get('class_evidence',[]): _keys(label,'class')
                    if 'confidence_context' in entity: _keys(entity['confidence_context'],'confidence_context')
                    if 'verification_need' in entity: _keys(entity['verification_need'],'verification_need')
                else:
                    if entity.get('semantic_valid') is True:
                        if not _finite(entity.get('semantic_score')) or not _finite(entity.get('fusion_weight')) or entity['fusion_weight']<=0:
                            raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','frontier has no semantic support')
                    elif entity.get('semantic_score') is not None:
                        raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','invalid frontier score must be null')
    for group,kind in (('connections','connection'),('observation_actions','observation'),('options','option')):
        for row in state.get(group,[]):
            _keys(row,kind)
            if kind == 'option' and 'focus_context' in row:
                _keys(row['focus_context'],'focus_context')


def _entity_index(
    state: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str], int, int]:
    regions: dict[str, dict[str, Any]] = {}
    owners: dict[str, str] = {}
    entity_types: dict[str, str] = {}
    object_count = 0
    frontier_count = 0
    for region in state.get("regions", []):
        if not isinstance(region, dict) or not isinstance(region.get("id"), str):
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "every region requires a string id")
        region_id = region["id"]
        if region_id in regions:
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"duplicate region {region_id}")
        regions[region_id] = region
        for group in ("objects", "frontiers"):
            if not isinstance(region.get(group), list):
                raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"{region_id}.{group} must be a list")
            for entity in region[group]:
                entity_id = entity.get("id") if isinstance(entity, dict) else None
                if not isinstance(entity_id, str) or not entity_id or entity_id in owners:
                    raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "entity ids must be unique strings")
                owners[entity_id] = region_id
                entity_types[entity_id] = group[:-1]
                object_count += group == "objects"
                frontier_count += group == "frontiers"
    unassigned = state.get("unassigned_entities", {})
    if not isinstance(unassigned, dict):
        raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "unassigned_entities must be an object")
    for group in ("objects", "frontiers"):
        rows = unassigned.get(group, [])
        if not isinstance(rows, list):
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"unassigned_entities.{group} must be a list")
        for entity in rows:
            entity_id = entity.get("id") if isinstance(entity, dict) else None
            if not isinstance(entity_id, str) or not entity_id or entity_id in owners:
                raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "unassigned entity ids must be unique")
            owners[entity_id] = "unassigned"
            entity_types[entity_id] = group[:-1]
            object_count += group == "objects"
            frontier_count += group == "frontiers"
    observations = state.get("observation_actions", [])
    if not isinstance(observations, list):
        raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "observation_actions must be a list")
    for observation in observations:
        entity_id = observation.get("id") if isinstance(observation, dict) else None
        region_id = observation.get("region_id") if isinstance(observation, dict) else None
        if (not isinstance(entity_id, str) or not entity_id or entity_id in owners or
                region_id not in regions):
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "observation action identity is invalid")
        owners[entity_id] = region_id
        entity_types[entity_id] = "observation_action"
    return regions, owners, entity_types, object_count, frontier_count


def validate_region_graph_state(state: Any) -> None:
    if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
        raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"schema_version must be {SCHEMA_VERSION}")
    _validate_fields(state)
    representation = state.get("representation")
    if not isinstance(representation, dict) or representation.get("kind") != "region_graph":
        raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "representation.kind must be region_graph")
    if representation.get("lossy") is not True:
        raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "region graph must declare lossy=true")
    identity = state.get("identity")
    required_identity = ("episode_id", "request_id", "observation_id", "map_revision", "region_graph_revision")
    if not isinstance(identity, dict) or any(key not in identity for key in required_identity):
        raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "identity is incomplete")
    if not isinstance(identity["episode_id"], str) or not identity["episode_id"] or not isinstance(identity["request_id"], str) or not identity["request_id"]:
        raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "identity strings are invalid")
    if any(type(identity[key]) is not int or identity[key] < 0 for key in required_identity[2:]):
        raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "identity revisions must be nonnegative integers")
    if not isinstance(state.get("regions"), list) or not isinstance(state.get("connections"), list):
        raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "regions and connections must be lists")
    regions, owners, entity_types, object_count, frontier_count = _entity_index(state)
    robot = state.get("robot")
    if not isinstance(robot, dict) or robot.get("region_id") not in regions:
        raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "robot.region_id must reference a region")
    directed_edges: set[tuple[str, str]] = set()
    for edge in state["connections"]:
        if not isinstance(edge, dict) or edge.get("from") not in regions or edge.get("to") not in regions:
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "connection endpoints must reference regions")
        if edge.get("status") != "verified_reachable":
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "only verified_reachable edges may be exported")
        if edge.get("verified_map_revision") != identity["map_revision"] or edge["from"]==edge["to"]:
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "stale or self connection")
        directed_edges.add((edge["from"], edge["to"]))
        if edge.get("bidirectional") is True:
            directed_edges.add((edge["to"], edge["from"]))
    options = state.get("options")
    if not isinstance(options, list) or not options:
        raise RegionGraphError("NO_EXECUTABLE_CANDIDATE", "region graph has no executable option")
    seen: set[str] = set()
    candidate_ids: set[str] = set()
    for option in options:
        option_id = option.get("id") if isinstance(option, dict) else None
        if not isinstance(option_id, str) or not option_id or option_id in seen:
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "option ids must be unique strings")
        seen.add(option_id)
        if option.get("action_type") not in ("explore", "approach_object"):
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"invalid action_type for {option_id}")
        if option.get("target_id") not in owners:
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"unknown target for {option_id}")
        target_type = entity_types[option["target_id"]]
        if option.get("action_type") == "approach_object" and target_type != "object":
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"approach target is not an object for {option_id}")
        if option.get("action_type") == "explore" and target_type not in ("frontier", "observation_action"):
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"explore target type is invalid for {option_id}")
        if option.get("region_id") not in regions:
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"unknown region for {option_id}")
        if target_type != "object" and owners[option["target_id"]] != option["region_id"]:
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"target region mismatch for {option_id}")
        if option.get("local_path_verified") is not True or not _finite(option.get("distance_m")) or option["distance_m"] < 0:
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"unverified path for {option_id}")
        route = option.get("route_region_ids")
        if not isinstance(route, list) or not route or any(item not in regions for item in route):
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"invalid route for {option_id}")
        if route[0] != robot["region_id"] or route[-1] != option["region_id"]:
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"route endpoints mismatch for {option_id}")
        if any((left, right) not in directed_edges for left, right in zip(route, route[1:])):
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"route uses an unverified edge for {option_id}")
        if (not isinstance(option.get("candidate_id"), str) or not option["candidate_id"] or
                option["candidate_id"] in candidate_ids):
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"candidate_id missing for {option_id}")
        if (not isinstance(option.get('source_stage'), str) or not option['source_stage'] or
                type(option.get('source_priority')) is not int or option['source_priority'] < 1 or
                type(option.get('original_policy_stage_active')) is not bool or
                not isinstance(option.get('safety_mode'), str)):
            raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR', f'policy provenance missing for {option_id}')
        for score_name in ('semantic_score','information_gain'):
            if option.get(score_name) is not None and not _finite(option[score_name]):
                raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR', f'invalid {score_name} for {option_id}')
        candidate_ids.add(option["candidate_id"])
    option_targets = {row['target_id'] for row in options}
    for container in list(state['regions'])+[state.get('unassigned_entities',{})]:
        for plural in ('objects','frontiers'):
            for entity in container.get(plural,[]):
                if type(entity.get('selectable')) is not bool or entity['selectable'] != (entity['id'] in option_targets):
                    raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','selectable/option mismatch')
    for entity in state.get('observation_actions',[]):
        if entity.get('selectable') is not True or entity['id'] not in option_targets:
            raise RegionGraphError('REGION_GRAPH_SCHEMA_ERROR','observation/option mismatch')
    coverage = state.get("coverage")
    if (not isinstance(coverage, dict) or coverage.get("raw_map_complete") is not False or
            coverage.get("high_level_evidence_complete") is not False or
            coverage.get("decision_evidence_complete") is not True):
        raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "decision-evidence coverage flags are invalid")
    if type(coverage.get("summary_contract_valid")) is not bool or coverage.get("omitted_entities") != 0:
        raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", "summary coverage is incomplete")
    expected = {
        "regions_exported": len(regions),
        "objects_exported": object_count,
        "frontiers_exported": frontier_count,
        "executable_candidates_exported": len(options),
    }
    for key, value in expected.items():
        total_key = key.replace("_exported", "_total")
        if type(coverage.get(key)) is not int or coverage.get(key) != value or type(coverage.get(total_key)) is not int or coverage[total_key] != value:
            raise RegionGraphError("REGION_GRAPH_SCHEMA_ERROR", f"coverage.{key} does not match exported data")
    _walk_forbidden(state)


def instructions() -> str:
    return (
        "Choose exactly one listed option ID to find the target object. An explore option gathers "
        "evidence at a frontier or by an in-place observation; targeted observation options inspect "
        "one named object or region from the current pose. An approach_object option commits travel "
        "toward an observed hypothesis and never declares task success. Use jev_obj_policy as the reference "
        "rule gate: source_priority and original_policy_stage_active explain what the original "
        "JEV-ObjNav selector would consider first, but you may override it when current evidence, "
        "progress, or confirmation history supports a safer or more informative choice. Region IDs are geometric free-space "
        "components, not room labels. Read semantic_alignment before comparing scores. Detector, image-text, "
        "and frontier scores are not probabilities; a detector score just above its operational gate is "
        "borderline rather than strong, and top-ranked means top only among the queried labels, not open-world identity. "
        "Use verified connectivity, route regions, target evidence, visits, distance and progress. "
        "Treat rejected_cooldown as negative post-arrival evidence and do not infer success from "
        "proximity. A rejected object remains unavailable after the minimum cooldown until its physical "
        "viewpoint count and translation baseline improve. A selectable object labelled "
        "deferred_weak_target_hypothesis is executable but "
        "is not currently endorsed: prefer an original_policy_stage_active frontier unless fresh "
        "synchronized target evidence provides a concrete reason to override the reference gate. "
        "Detector observation count means frames, not independent physical viewpoints. Repeated turns "
        "from one position are correlated evidence; use target_unique_viewpoints, translation baseline, "
        "bearing span and object extent to judge geometric support. In-place inspections are bounded and "
        "cannot by themselves satisfy the translated-view STOP requirement. Adapted reference examples: "
        "(1) score 0.52 with gate 0.50, one physical viewpoint, or a narrow margin is a reason to inspect "
        "briefly and then seek translated-view evidence, not to treat it as 52% certainty; "
        "(2) a fresh multi-view object well above its local gate can justify approach, while local "
        "post-arrival confirmation still decides STOP; (3) when object identity is uncertain but a "
        "region has useful semantic/frontier evidence, inspect that region or explore its frontier. "
        "These examples are decision priors, not copied hard rules: use the current normalized context. "
        "Avoid repeating an option that made no progress unless evidence improved. The state is a "
        "lossy summary: never invent coordinates, targets, paths, hidden geometry or a success signal."
    )


def _criterion(option: dict[str, Any], entity: dict[str, Any] | None = None) -> str:
    """Return a compact, human-readable choice label.

    Detailed evidence already lives once in ``state.regions`` and
    ``state.options``. Criteria are a decision index, not a second state copy.
    """
    mode = option["action_type"]
    target = option["target_id"]
    region = option["region_id"]
    distance = option["distance_m"]
    repeat = option.get("repeated_selection_count", 0)
    stage = option.get("source_stage")
    priority = option.get("source_priority")
    gate = option.get("original_policy_stage_active")
    semantic = option.get("semantic_score")
    if mode == "approach_object":
        entity = entity or {}
        context = entity.get('confidence_context', {})
        verification = entity.get('verification_need', {})
        return (
            f"Approach {target} in {region}; d={distance}m; stage={stage}/p{priority}; "
            f"reference_gate={gate}; score={semantic}; tier={context.get('tier')}; "
            f"delta_to_gate={context.get('delta_to_gate')}; "
            f"frames={entity.get('target_observations')}; "
            f"physical_views={entity.get('target_unique_viewpoints')}; "
            f"baseline={entity.get('target_viewpoint_baseline_m')}m; "
            f"fresh={entity.get('updated_in_current_observation')}; "
            f"verify={verification.get('level')}:{verification.get('reasons')}; "
            f"repeat={repeat}; approach is not success."
        )
    kind = option.get("exploration_kind", "frontier")
    entity = entity or {}
    focus_kind = entity.get("focus_kind") if kind == "observe_rotation" else "region_frontier"
    focus_target = entity.get("focus_target_id") if kind == "observe_rotation" else target
    gaps = entity.get("evidence_gaps") if kind == "observe_rotation" else ["target_identity_not_yet_reliable"]
    return (
        f"Explore {target} in {region}; kind={kind}; d={distance}m; "
        f"stage={stage}/p{priority}; reference_gate={gate}; score={semantic}; "
        f"focus={focus_kind}:{focus_target}; gaps={gaps}; repeat={repeat}."
    )


def _compact_class_evidence(
    rows: list[dict[str, Any]], target_label: str | None,
    stored_best_label: str | None, computed_best_label: str | None,
) -> tuple[list[dict[str, Any]], int]:
    """Keep target/best competitors while removing repeated low-value class detail."""
    wanted = {
        str(value).strip().lower()
        for value in (target_label, stored_best_label, computed_best_label)
        if isinstance(value, str) and value.strip()
    }
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("fused_confidence"))
            if _finite(row.get("fused_confidence")) else float("-inf")
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_labels: set[str] = set()
    for row in ranked:
        label = str(row.get("label", "")).strip().lower()
        if label in wanted or len(selected) < 2:
            if label in selected_labels:
                continue
            selected.append({
                key: row.get(key)
                for key in (
                    "label", "fused_confidence", "observation_count",
                    "integer_function_score",
                )
            })
            selected_labels.add(label)
    return selected, max(0, len(rows) - len(selected))


def _rounded(value: Any, digits: int = 4) -> float | None:
    return round(float(value), digits) if _finite(value) else None


def _distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values if _finite(value))
    if not ordered:
        return {"count": 0, "minimum": None, "median": None, "p90": None, "maximum": None}
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    p90_index = max(0, min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1))
    return {
        "count": len(ordered),
        "minimum": _rounded(ordered[0]),
        "median": _rounded(median),
        "p90": _rounded(ordered[p90_index]),
        "maximum": _rounded(ordered[-1]),
    }


def _percentile(value: Any, values: list[float]) -> float | None:
    if not _finite(value):
        return None
    ordered = [float(item) for item in values if _finite(item)]
    if not ordered:
        return None
    return round(sum(item <= float(value) for item in ordered) / len(ordered), 3)


def _confidence_tier(value: Any, gate: Any) -> str:
    if not _finite(value) or not _finite(gate):
        return "unknown"
    delta = float(value) - float(gate)
    if delta < 0:
        return "below_operational_gate"
    if delta < 0.1:
        return "borderline_above_gate"
    if delta < 0.25:
        return "supported_above_gate"
    return "strongly_above_gate"


def _object_confidence_context(
    entity: dict[str, Any], target_label: str | None,
    confidence_values: list[float], selection: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = entity.get("target_confidence")
    gate = selection.get("min_target_confidence")
    min_observations = selection.get("min_target_observations")
    rows = entity.get("class_evidence", [])
    target_key = str(target_label or "").strip().lower()
    ranked = sorted(
        [row for row in rows if _finite(row.get("fused_confidence"))],
        key=lambda row: float(row["fused_confidence"]), reverse=True,
    )
    target_row = next(
        (row for row in ranked if row.get("class_index") == 0 or
         str(row.get("label", "")).strip().lower() == target_key),
        None,
    )
    target_score = target_row.get("fused_confidence") if target_row else value
    competitors = [row for row in ranked if row is not target_row]
    runner_up = competitors[0] if competitors else None
    rank = None
    if _finite(target_score):
        rank = 1 + sum(float(row["fused_confidence"]) > float(target_score) for row in competitors)
    delta = float(value) - float(gate) if _finite(value) and _finite(gate) else None
    context = {
        "delta_to_gate": _rounded(delta),
        "tier": _confidence_tier(value, gate),
        "percentile_among_current_objects": _percentile(value, confidence_values),
        "target_rank_among_queried_labels": rank,
        "queried_runner_up_label": runner_up.get("label") if runner_up else None,
        "margin_to_queried_runner_up": _rounded(
            float(target_score) - float(runner_up["fused_confidence"])
            if _finite(target_score) and runner_up else None
        ),
    }
    reasons: list[str] = []
    if not _finite(value):
        reasons.append("target_score_missing")
    elif _finite(gate) and float(value) < float(gate):
        reasons.append("below_local_operational_gate")
    elif delta is not None and delta < 0.1:
        reasons.append("only_narrowly_above_local_gate")
    observations = entity.get("target_observations")
    if type(observations) is int and type(min_observations) is int and observations <= min_observations + 1:
        reasons.append("few_detector_observation_frames")
    unique_viewpoints = entity.get("target_unique_viewpoints")
    min_viewpoints = selection.get("min_independent_viewpoints")
    if type(unique_viewpoints) is int and type(min_viewpoints) is int and unique_viewpoints < min_viewpoints:
        reasons.append("too_few_physical_camera_positions")
    baseline = entity.get("target_viewpoint_baseline_m")
    min_baseline = selection.get("min_viewpoint_baseline_m")
    if _finite(baseline) and _finite(min_baseline) and float(baseline) < float(min_baseline):
        reasons.append("viewpoint_translation_baseline_too_small")
    if entity.get("updated_in_current_observation") is not True:
        reasons.append("not_supported_in_current_observation")
    if entity.get("post_arrival_confirmation_status") == "rejected_cooldown":
        reasons.append("failed_post_arrival_multiview_confirmation")
    if entity.get("provisional") is True:
        reasons.append("provisional_geometry")
    if rank is not None and rank > 1:
        reasons.append("target_not_top_among_queried_labels")
    if not reasons:
        reasons.append("open_world_identity_still_unverified")
    level = "high" if any(reason in reasons for reason in (
        "below_local_operational_gate", "not_supported_in_current_observation",
        "failed_post_arrival_multiview_confirmation", "target_not_top_among_queried_labels",
    )) else "medium" if any(reason in reasons for reason in (
        "only_narrowly_above_local_gate", "few_detector_observation_frames",
        "too_few_physical_camera_positions", "viewpoint_translation_baseline_too_small",
        "provisional_geometry",
    )) else "routine"
    return context, {"level": level, "reasons": reasons}


def _semantic_alignment(state: dict[str, Any], confidence_values: list[float],
                        frontier_values: list[float]) -> dict[str, Any]:
    gate = state.get("selection_parameters", {}).get("min_target_confidence")
    recent = state.get("perception", {}).get("recent_observations", [])
    itm_values = [
        row.get("image_text_match_raw_score") for row in recent
        if row.get("image_text_match_valid") is True
    ]
    return {
        "object_confidence": {
            "meaning": "fused target-detector similarity accumulated for a mapped object",
            "calibrated_probability": False,
            "operational_gate": _rounded(gate),
            "tiers": {
                "below_operational_gate": "locally executable only as a weak hypothesis, not endorsed",
                "borderline_above_gate": "0.00 to 0.10 above gate; inspect when practical",
                "supported_above_gate": "0.10 to 0.25 above gate; useful but not identity proof",
                "strongly_above_gate": "at least 0.25 above gate; still requires fresh local confirmation",
            },
            "current_distribution": _distribution(confidence_values),
            "interpretation": "compare delta-to-gate, rank, freshness and physical viewpoint diversity; detector frame count alone is correlated evidence and 0.5 never means 50% correctness",
        },
        "queried_class_comparison": {
            "scope": "target label plus task-provided related labels only",
            "top_rank_meaning": "best among queried labels, not best among every real-world class",
            "missing_competitors_meaning": "zero or absent competitor scores do not independently validate target identity",
        },
        "image_text_match": {
            "meaning": "raw image-text similarity for the current view",
            "calibrated_probability": False,
            "operational_gate": None,
            "tiers": None,
            "current_distribution": _distribution(itm_values),
            "interpretation": "use relative trend and synchronization; do not compare numerically with detector confidence",
        },
        "frontier_semantics": {
            "meaning": "target-related semantic value aggregated over navigable frontier/region cells",
            "calibrated_probability": False,
            "operational_gate": None,
            "tiers": None,
            "current_distribution": _distribution(frontier_values),
            "interpretation": "rank only against frontiers in this request and combine with reachability and visit history",
        },
        "reference_policy_examples": {
            "borderline_object": "slightly above gate plus one physical camera position -> use at most a small number of in-place checks, then seek translated-view evidence",
            "supported_object": "fresh support from translated physical viewpoints well above gate -> approach may be preferred, but approach is not STOP",
            "uncertain_region": "no reliable object identity plus informative frontier -> inspect the region or explore that frontier",
        },
    }


def _focus_context(option: dict[str, Any], entity: dict[str, Any] | None,
                   region: dict[str, Any] | None, frontier_values: list[float]) -> dict[str, Any]:
    entity = entity or {}
    region = region or {}
    if option.get("action_type") == "approach_object":
        confidence = entity.get("confidence_context", {})
        verification = entity.get("verification_need", {})
        return {
            "action_role": "navigate_near_hypothesis_not_success", "focus_kind": "object",
            "focus_target_id": option.get("target_id"), "focus_region_id": option.get("region_id"),
            "evidence_goal": "closer_view_for_local_confirmation",
            "evidence_gaps": verification.get("reasons", []),
            "expected_resolution": "proximity_only_then_local_confirmation",
            "confidence_tier": confidence.get("tier"),
            "confidence_delta_to_gate": confidence.get("delta_to_gate"),
            "confidence_percentile": confidence.get("percentile_among_current_objects"),
            "semantic_percentile": None, "region_visited": region.get("visited"),
            "region_visit_count": region.get("visit_count"),
        }
    if option.get("exploration_kind") == "observe_rotation":
        return {
            "action_role": "one_synchronized_view_without_translation",
            "focus_kind": entity.get("focus_kind", "general"),
            "focus_target_id": entity.get("focus_target_id"),
            "focus_region_id": entity.get("focus_region_id", option.get("region_id")),
            "evidence_goal": entity.get("purpose", "collect_fresh_observation"),
            "evidence_gaps": entity.get("evidence_gaps", []),
            "expected_resolution": entity.get("expected_effect", "refresh_evidence_from_changed_heading"),
            "confidence_tier": None, "confidence_delta_to_gate": None,
            "confidence_percentile": None, "semantic_percentile": None,
            "region_visited": region.get("visited"), "region_visit_count": region.get("visit_count"),
        }
    return {
        "action_role": "move_to_frontier_for_new_evidence", "focus_kind": "region_frontier",
        "focus_target_id": option.get("target_id"), "focus_region_id": option.get("region_id"),
        "evidence_goal": "observe_unknown_space_near_frontier",
        "evidence_gaps": ["target_identity_not_yet_reliable"],
        "expected_resolution": "new_free_space_and_semantic_evidence",
        "confidence_tier": None, "confidence_delta_to_gate": None,
        "confidence_percentile": None,
        "semantic_percentile": _percentile(entity.get("semantic_score"), frontier_values),
        "region_visited": region.get("visited"), "region_visit_count": region.get("visit_count"),
    }


def _round_model_value(value: Any, digits: int = 4) -> Any:
    """Round model-only numeric detail while preserving local full precision."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round_model_value(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: _round_model_value(item, digits) for key, item in value.items()}
    return value


def _compact_entity(entity: dict[str, Any], kind: str, target_label: str | None) -> dict[str, Any]:
    if kind == "object":
        compact = {
            key: copy.deepcopy(entity.get(key))
            for key in (
                "id", "center_xy", "bbox_xy", "target_confidence",
                "target_observations", "target_unique_viewpoints",
                "target_viewpoint_baseline_m", "target_viewpoint_bearing_span_rad",
                "extent_xyz_m", "high_confidence_eligible",
                "post_arrival_confirmation_status", "selection_status",
                "not_selectable_reason", "last_observed_id", "observations_since_seen",
                "updated_in_current_observation", "provisional", "selectable",
            )
        }
        evidence, omitted = _compact_class_evidence(
            entity.get("class_evidence", []), target_label,
            entity.get("stored_best_label"), entity.get("computed_best_label"),
        )
        compact["class_evidence"] = evidence
        compact["class_evidence_omitted_count"] = omitted
        compact = {
            key: value for key, value in compact.items()
            if value is not None and not (key == "provisional" and value is False)
        }
        return _round_model_value(compact)
    return _round_model_value({
        key: copy.deepcopy(entity.get(key))
        for key in (
            "id", "state", "position_xy", "semantic_score", "semantic_valid",
            "cluster_cells", "information_gain", "reachability", "selectable",
            "not_selectable_reason",
        )
    })


def _compact_model_state(state: dict[str, Any]) -> dict[str, Any]:
    """Create a bounded Jev view after validating the complete decision snapshot.

    Every executable option, its target evidence, its route and confirmation target
    remain represented. Regions and entities that cannot affect this decision are
    summarized instead of repeated in full. Local execution retains the complete
    snapshot and full-precision paths.
    """
    target_label = state.get("task", {}).get("target_object")
    source_objects: dict[str, dict[str, Any]] = {}
    source_frontiers: dict[str, dict[str, Any]] = {}
    entity_owner: dict[str, str] = {}
    for container in [*state.get("regions", []), state.get("unassigned_entities", {})]:
        owner = container.get("id", "unassigned")
        for entity in container.get("objects", []):
            source_objects[entity["id"]] = entity
            entity_owner[entity["id"]] = owner
        for entity in container.get("frontiers", []):
            source_frontiers[entity["id"]] = entity
            entity_owner[entity["id"]] = owner
    confidence_values = [
        float(entity["target_confidence"]) for entity in source_objects.values()
        if _finite(entity.get("target_confidence"))
    ]
    frontier_values = [
        float(entity["semantic_score"]) for entity in source_frontiers.values()
        if _finite(entity.get("semantic_score"))
    ]
    compact = {
        key: copy.deepcopy(state[key])
        for key in (
            "schema_version", "representation", "identity", "task", "robot",
            "perception", "jev_obj_policy", "object_confirmation", "history",
            "coverage", "selection_parameters",
        )
        if key in state
    }

    detailed_object_ids: set[str] = set()
    detailed_frontier_ids: set[str] = set()
    detailed_region_ids: set[str] = {state["robot"]["region_id"]}
    for option in state.get("options", []):
        detailed_region_ids.add(option["region_id"])
        detailed_region_ids.update(option.get("route_region_ids", []))
        target_id = option.get("target_id")
        if target_id in source_objects:
            detailed_object_ids.add(target_id)
        if target_id in source_frontiers:
            detailed_frontier_ids.add(target_id)
    region_ids = {row["id"] for row in state.get("regions", [])}
    for action in state.get("observation_actions", []):
        focus_id = action.get("focus_target_id")
        focus_region_id = action.get("focus_region_id")
        if focus_id in source_objects:
            detailed_object_ids.add(focus_id)
        if focus_id in source_frontiers:
            detailed_frontier_ids.add(focus_id)
        if focus_region_id in region_ids:
            detailed_region_ids.add(focus_region_id)
    confirmation = state.get("object_confirmation", {})
    for key in ("pending_target_id", "blocked_target_id"):
        target_id = confirmation.get(key)
        if target_id in source_objects:
            detailed_object_ids.add(target_id)

    option_object_ids = {
        option.get("target_id") for option in state.get("options", [])
        if option.get("target_id") in source_objects
    }
    ranked_objects = sorted(
        source_objects.values(),
        key=lambda row: float(row.get("target_confidence"))
        if _finite(row.get("target_confidence")) else float("-inf"),
        reverse=True,
    )
    for entity in ranked_objects:
        if len(detailed_object_ids - option_object_ids) >= 2:
            break
        detailed_object_ids.add(entity["id"])
    for entity_id in detailed_object_ids | detailed_frontier_ids:
        owner = entity_owner.get(entity_id)
        if owner and owner != "unassigned":
            detailed_region_ids.add(owner)

    compact["regions"] = []
    for region in state["regions"]:
        if region["id"] not in detailed_region_ids:
            continue
        compact["regions"].append(_round_model_value({
            "id": region["id"],
            "bbox_xy": copy.deepcopy(region["bbox_xy"]),
            "anchor_xy": copy.deepcopy(region["anchor_xy"]),
            "free_area_m2": region.get("free_area_m2"),
            "visited": region.get("visited"),
            "visit_count": region.get("visit_count"),
            "semantic": {
                key: copy.deepcopy(region.get("semantic", {}).get(key))
                for key in ("valid", "mean", "p90", "max", "valid_area_fraction")
            },
            "objects": [
                _compact_entity(entity, "object", target_label)
                for entity in region.get("objects", [])
                if entity["id"] in detailed_object_ids
            ],
            "frontiers": [
                _compact_entity(entity, "frontier", target_label)
                for entity in region.get("frontiers", [])
                if entity["id"] in detailed_frontier_ids
            ],
        }))
    compact["unassigned_entities"] = {
        "objects": [
            _compact_entity(entity, "object", target_label)
            for entity in state.get("unassigned_entities", {}).get("objects", [])
            if entity["id"] in detailed_object_ids
        ],
        "frontiers": [
            _compact_entity(entity, "frontier", target_label)
            for entity in state.get("unassigned_entities", {}).get("frontiers", [])
            if entity["id"] in detailed_frontier_ids
        ],
    }
    compact_objects: dict[str, dict[str, Any]] = {}
    compact_frontiers: dict[str, dict[str, Any]] = {}
    for container in [*compact["regions"], compact["unassigned_entities"]]:
        for entity in container.get("objects", []):
            context, verification = _object_confidence_context(
                source_objects[entity["id"]], target_label, confidence_values,
                state.get("selection_parameters", {}),
            )
            entity["confidence_context"] = context
            entity["verification_need"] = verification
            compact_objects[entity["id"]] = entity
        for entity in container.get("frontiers", []):
            compact_frontiers[entity["id"]] = entity
    compact["connections"] = [
        {
            key: copy.deepcopy(edge.get(key))
            for key in (
                "id", "from", "to", "bidirectional", "status",
                "min_clearance_m", "verified_map_revision",
            )
        }
        for edge in state.get("connections", [])
        if edge.get("from") in detailed_region_ids and edge.get("to") in detailed_region_ids
    ]
    compact["observation_actions"] = _round_model_value(
        copy.deepcopy(state.get("observation_actions", []))
    )
    observations = {row["id"]: row for row in compact["observation_actions"]}
    compact_regions = {row["id"]: row for row in compact["regions"]}
    compact["options"] = []
    for option in state.get("options", []):
        compact_option = _round_model_value({
            key: copy.deepcopy(option.get(key))
            for key in (
                "id", "action_type", "exploration_kind", "target_id", "region_id",
                "distance_m", "route_region_ids", "candidate_id", "local_path_verified",
                "repeated_selection_count", "source_stage", "source_priority",
                "original_policy_stage_active", "safety_mode", "semantic_score",
                "information_gain",
            )
            if option.get(key) is not None
        })
        entity = (
            compact_objects.get(option.get("target_id"))
            or compact_frontiers.get(option.get("target_id"))
            or observations.get(option.get("target_id"))
        )
        compact_option["focus_context"] = _focus_context(
            option, entity, compact_regions.get(option.get("region_id")), frontier_values
        )
        compact["options"].append(compact_option)
    compact["semantic_alignment"] = _semantic_alignment(
        state, confidence_values, frontier_values
    )

    detailed_object_count = len(compact_objects)
    detailed_frontier_count = len(compact_frontiers)
    compact["coverage"].update({
        "regions_exported": len(compact["regions"]),
        "objects_exported": detailed_object_count,
        "frontiers_exported": detailed_frontier_count,
        "executable_candidates_exported": len(compact["options"]),
        "omitted_entities": (
            len(source_objects) - detailed_object_count
            + len(source_frontiers) - detailed_frontier_count
        ),
        "detail_policy": (
            "decision-local view: every safe option, option target, route region, "
            "focused inspection and confirmation target is detailed; unrelated "
            "regions/entities are counted below and remain available to local execution"
        ),
    })
    compact["projection_summary"] = {
        "view": "decision_local_region_graph",
        "all_safe_options_retained": True,
        "regions_total": len(state.get("regions", [])),
        "regions_detailed": len(compact["regions"]),
        "regions_summarized": len(state.get("regions", [])) - len(compact["regions"]),
        "objects_total": len(source_objects),
        "objects_detailed": detailed_object_count,
        "frontiers_total": len(source_frontiers),
        "frontiers_detailed": detailed_frontier_count,
        "detail_selection": (
            "robot region + every option target/route + focused inspection + "
            "pending/rejected confirmation + two leading unavailable objects"
        ),
        "local_execution_retains_full_state": True,
    }
    return compact


def project_region_graph(snapshot: dict[str, Any]) -> RegionGraphProjection:
    state = copy.deepcopy(snapshot.get("region_graph"))
    validate_region_graph_state(state)
    state["coverage"]["summary_contract_valid"] = True
    model_state = _compact_model_state(state)
    entity_by_id: dict[str, dict[str, Any]] = {}
    for container in [*model_state["regions"], model_state.get("unassigned_entities", {})]:
        for plural in ("objects", "frontiers"):
            for entity in container.get(plural, []):
                entity_by_id[entity["id"]] = entity
    for entity in model_state.get("observation_actions", []):
        entity_by_id[entity["id"]] = entity
    criteria = {
        option["id"]: _criterion(option, entity_by_id.get(option["target_id"]))
        for option in model_state["options"]
    }
    mapping = {option["id"]: option["candidate_id"] for option in model_state["options"]}
    return RegionGraphProjection(
        state=model_state, criteria=criteria, option_to_candidate=mapping
    )
