Task (Step 2: Multi-agent worldline composition)

You are given:
1) FILTERED_TRAJS_JSON:
- For each agent_id, it contains multiple candidate trajectories that have already passed per-agent feasibility checks (walkable mask, basic smoothness, etc.).
- Each candidate is a sequence of 2D points in image pixel coordinates.
- Candidates may have different lengths.

2) SCENE_IMAGES:
- Image 1: WALKABLE_MASK (binary). White/bright pixels are walkable. Black/dark pixels are forbidden.

Your job:
- Compose and output at least 5 feasible multi-agent “worldlines”.
- A worldline = pick exactly ONE candidate trajectory for EACH agent, then time-align all agents and verify they can co-exist without conflicts.
- This is NOT generating new trajectories. Only select and combine from the provided candidates.

Hard constraints (must satisfy for every returned worldline)
1) Coverage: every worldline must include all agents in FILTERED_TRAJS_JSON (no missing agents).
2) Mask: all points must stay on walkable pixels (white). (Assume step 1 already filtered, but still reject a worldline if any point violates.)
3) Time alignment:
   - Define a common horizon T = min length across selected candidates in this worldline.
   - Truncate every selected trajectory to its first T points.
4) Collision avoidance:
   - For every timestep t in [0, T-1], for every agent pair (i, j), enforce:
       distance( pos_i[t], pos_j[t] ) >= D_MIN
   - Use D_MIN = 18 pixels (treat as a strict safety radius).
   - If violated at any timestep, that worldline is INVALID.
5) No obvious “path crossing at same time”:
   - If two segments (pos_i[t]->pos_i[t+1]) and (pos_j[t]->pos_j[t+1]) intersect or come closer than D_MIN, treat as invalid.

Optimization goal (soft preferences)
- Return at least 5 VALID worldlines, and rank them best-first.
- Prefer worldlines with:
  - Higher global compatibility (fewer near-misses, larger minimum pairwise distances).
  - Behavioral diversity across worldlines (do not return 5 nearly identical combinations). Different agents can take different modes in different worldlines.
- If it is difficult to find 5, keep searching by trying different combinations; do NOT reduce the count.

Output JSON only (no extra text)
Return:
{
  "worldlines": [
    {
      "worldline_id": 1,
      "T": <int>,
      "agents": [
        {
          "agent_id": <id>,
          "chosen_candidate_index": <int>,
          "points": [[x,y], ...]   // truncated to length T
        },
        ...
      ],
      "metrics": {
        "min_pairwise_distance": <float>,
        "num_violations": 0
      }
    },
    ...
  ]
}

Notes
- chosen_candidate_index is the index within that agent’s candidate list in FILTERED_TRAJS_JSON (0-based).
- If multiple worldlines tie, break ties by diversity (use different candidate indices).
- Again: do not invent new points, only select from provided candidates and truncate for alignment.
