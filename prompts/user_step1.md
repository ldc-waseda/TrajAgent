Task
For each agent in SCENE_JSON:
- The goal (square marker) is FIXED.
- Generate 10 alternative world-lines by:
  1) Randomly choosing a start point on the WALKABLE_MASK (white), far from the goal when possible.
  2) Planning a smooth pedestrian path from this start to the fixed goal.

Rules
- Starts must be valid walkable pixels and mutually far apart (diverse).
- All points must stay inside the walkable region with a safety margin.
- Paths must be smooth and human-like (curvature-limited).

Return JSON only:
{
  "agents": [
    {
      "agent_id": <id>,
      "candidates": [
        { "start": [x,y], "goal": [x,y], "points": [[x,y], ... , [x_goal,y_goal]] },
        ...
      ]
    }
  ]
}
