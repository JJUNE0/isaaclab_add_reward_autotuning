"""Ascending from a central lower platform; 40 cm is a target, not a result."""
from isaaclab.terrains import TerrainGeneratorCfg, MeshInvertedPyramidStairsTerrainCfg

STAIRS_CFG = TerrainGeneratorCfg(
    seed=42, size=(10.0, 10.0), border_width=5.0,
    num_rows=8, num_cols=4, curriculum=True, difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={"stairs_up": MeshInvertedPyramidStairsTerrainCfg(
        proportion=1.0, step_height_range=(0.05, 0.40), step_width=0.70,
        platform_width=3.0, border_width=1.0, holes=False,
    )},
)
