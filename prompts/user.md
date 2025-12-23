Task: Explore future trajectories for all pedestrians mentioned in SCENE_JSON.

Inputs you will receive:
1) RECORDED_TRAJ:
- Contains observed trajectories (history) and any available context (agent ids, current positions, optional metadata).

2) SCENE_IMAGES:
You will see multiple images in a fixed order. Use them as follows:
- Image 1: WALKABLE_MASK (binary). White or bright pixels are walkable. Black or dark pixels are non-walkable.
- Image 2: ANNOTATED_TRAJECTORY_IMAGE (original scene with colorful trajectories or annotations).

Generation requirements:
- For each agent in RECORDED_TRAJ, generate exactly 4-5 alternative continuation trajectory candidates.
- Each candidate must be a smooth, continuous sequence of points [[x, y], ...] in the same coordinate system as RECORDED_TRAJ (typically image pixel coordinates). 
- TThe trajectory length is flexible. For each candidate, extend the future path as far as reasonably possible while remaining smooth and walkable. Across the 4 candidates, explore different feasible motion directions and smooth-curvature paths implied by the observed history (e.g., continuing forward, gentle left/right turns, alternative corridor choices), but keep every trajectory realistic and consistent with the agent’s recent motion
- The 4 candidates should provide reasonable diversity when multiple feasible options exist (e.g., straight, gentle turn, detour, slow-down/stop-and-go). Similar candidates are acceptable if the walkable space and observed motion strongly constrain the agent to a narrow corridor.
-Every future point must lie strictly inside the walkable region of the WALKABLE_MASK. Do not step onto non-walkable pixels. 

Diversity requirements:
- Do not simply add small noise to the same path.
- Include multiple plausible behaviors when possible: proceed forward, turn left, turn right.

Realism requirements:
- Keep step sizes consistent with the observed history (infer typical speed from the last several observed steps).
- Avoid sharp corners and unnatural oscillations.
- Avoid clearly colliding into visible obstacles or walls indicated by the images.

Output:
Return JSON only, following exactly:
{
  "agents": [
    {
      "agent_id": "<id from SCENE_JSON>",
      "candidates": [
        { "points": [[x,y], ... 19 pairs ...] },
        ...
      ]
    },
    ...
  ]
}
