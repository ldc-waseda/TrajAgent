Task
For each agent in SCENE_JSON:
- Generate 3 alternative world-lines by producing ONLY:
  - a random START point (walkable,  and safe)
  - a mid point that lies on a safe walkable pixel and serves as a smooth connector between start and end: it should be located along the natural corridor route so that the polyline start → mid → end forms a gentle, human-like turn (no sharp angle), preferably near corridor midlines / wide junction areas to keep clearance from boundaries.
  - an END point that within walkable area

Do NOT output trajectories or intermediate points. Output points only.

Inputs
1) SCENE_JSON: includes agent_id list and recorded traj for each agent.
2) WALKABLE_MASK: walkable pixels are bright/white, forbidden pixels are dark/black.
3) Current Window image: reference only for motion pattern (heading trend / speed level / turning tendency). Do NOT copy the exact path.

Coordinate system (OpenCV cv2, MUST follow)
- origin: (0,0) at top-left
- x increases right, y increases down
- point format: [x,y] as integer pixel coordinates
- bounds: 0 <= x < W, 0 <= y < H

Hard constraints (MUST NEVER violate)
1) Walkability
   Every output point (ALL starts and ALL ends) must be on walkable pixels in WALKABLE_MASK.

2) Safety preference (strong)
   Avoid boundary hugging (jagged/diagonal borders). Prefer corridor centerlines and wide junction areas.


Batch validation rule (MANDATORY)
- You MUST validate ALL points in ONE batch tool call:
  check_walkable_pixel(points=[[x,y], ...])
- This batch must include ALL starts and ALL ends for ALL agents (5 world-lines per agent).
- Do NOT call the tool one point at a time.
- If any invalid points are found, resample/repair and batch-check again until ALL points are valid.

World-line generation requirements
1) Per agent: exactly 3 world-lines.
2) Start diversity:
   - Starts must be mutually far apart (large pairwise distances) for the same agent.
3) Motion imitation (soft):
   - Infer each agent’s motion direction/turning style from the Current Window trajectories.
   - Sample starts and ends consistent with that style (do NOT copy the exact trajectory).

Spatial hints (walkable layout prior from WALKABLE_MASK)
- Walkable area is a large “I”-shaped corridor system.
- Top region: long horizontal corridor; safest near its middle.
- Middle region: wide vertical corridor (“stem”) downward from top junction; prefer centerline.
- Bottom region: widened plaza/foot near bottom, plus small extension bottom-right.
- Black regions dominate left/right: no shortcuts; prefer corridor/junction interior.

Output (STRICT: JSON only, no extra text)
{
  "agents": [
    {
      "agent_id": <id>,
      "candidates": [
        { "points": [x,y], "points": [x,y], "points": [x,y] },
        ...
        (total 3)
      ]
    },
    ...
  ]
}

Final reminder
- Only output JSON AFTER all starts and ends are batch-validated as walkable.
