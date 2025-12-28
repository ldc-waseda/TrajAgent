Task
For each agent in SCENE_JSON:
- The goal (square marker) is FIXED.
- Generate 5 alternative world-lines by:
  1) Randomly choosing a start point on the WALKABLE_MASK (white) , far from the goal when possible.
  2) Planning a smooth pedestrian path from this start to the fixed goal.

Spatial hints (walkable layout prior, based on WALKABLE_MASK):
- The walkable area forms a large “I”-shaped corridor system.
- Top region: a long horizontal corridor spans across the upper part of the image (upper band). It has a wide central junction area, and two arms extending to the left and to the right. The safest walking area is near the middle of this horizontal band (avoid the jagged/diagonal borders on both sides).
- Middle region: from the top central junction, a wide vertical corridor goes downward through the center of the image (a straight “stem”). Prefer the centerline of this stem for safety margin.
- Bottom region: the vertical stem widens into a broader “foot/plaza” near the bottom, with an additional small extension towards the bottom-right. This bottom area is generally more spacious and safer than the narrow edges.
- Non-walkable space (black) occupies most of the left and right sides; do not attempt shortcuts through black regions. Most valid routes should flow through the top horizontal band, the central junction, and/or the central vertical stem.
- Waypoint suggestion: choose waypoints inside the wide top junction and along the midlines of the horizontal arms / vertical stem to maintain smooth, human-like paths with good boundary clearance.

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
