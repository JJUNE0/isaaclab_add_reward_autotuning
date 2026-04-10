# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains import FlatPatchSamplingCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    # [수정 1] size는 유지하되, border_width를 대폭 줄여야 합니다.
    size=(7.5, 7.5), 
    border_width=1.0,  # 기존 7.5 -> 1.0 (양쪽 합쳐 2.0m, 유효공간 7.75m 확보)
    
    num_rows=20,
    num_cols=10,
    color_scheme="random",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.5,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        "hf_pyramid_stair_inv_1": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            holes=False,
            proportion=0.3, # [수정 2] 합계 1.0을 맞추기 위해 조정 (예: 0.6 -> 0.7)
            step_height_range=(0.0, 0.3),
            step_width=0.5,
            platform_width=2.5,
            border_width=1.0,
        ),
        "hf_pyramid_stair_inv_2": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            holes=False,
            proportion=0.2, # [수정 2] 합계 1.0을 맞추기 위해 조정 (예: 0.6 -> 0.7)
            step_height_range=(0.0, 0.3),
            step_width=0.4,
            platform_width=2.5,
            border_width=1.0,
        ),
        "hf_pyramid_stair_inv_3": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            holes=False,
            proportion=0.2, # [수정 2] 합계 1.0을 맞추기 위해 조정 (예: 0.6 -> 0.7)
            step_height_range=(0.0, 0.3),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
        ),
        
        "cylinder_terrain_cfg": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=0.1,
            platform_width=1.5,
            platform_height=0.5,
            abs_height_noise=(0.0, 0.05),
            rel_height_noise=(0.95, 1.05),
            
            flat_patch_sampling={
                "root_spawn": FlatPatchSamplingCfg(
                    num_patches=5, patch_radius=0.2, 
                    x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), z_range=(-1.0, 1.0), 
                    max_height_diff=0.15
                ),
            },

            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                height=0.1, radius=0.6, max_yx_angle=0.0, degrees=True, num_objects=20
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                height=0.4, radius=0.25, max_yx_angle=20.0, degrees=True, num_objects=20
            )
        ),
        
        "repeated_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.1,
            platform_width=1.5,
            platform_height=0.5,
            abs_height_noise=(0.0, 0.1),
            rel_height_noise=(0.9, 1.1),
            
            # [추천] 로봇 스폰 위치 안전장치 추가
            flat_patch_sampling={
                "root_spawn": FlatPatchSamplingCfg(
                    num_patches=5, patch_radius=0.2, 
                    x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), z_range=(-1.0, 1.0), 
                    max_height_diff=0.15
                ),
            },

            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                height=0.05, size=(0.5, 0.5), max_yx_angle=0.0, degrees=True, num_objects=20 # 개수 조절 필요
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                height=0.2, size=(0.3, 0.3), max_yx_angle=20.0, degrees=True, num_objects=20
            )
        ),
        

        
        "pyramid_terrain_cfg": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
            proportion=0.1,
            platform_width=1.5,
            platform_height=0.5,
            abs_height_noise=(0.0, 0.1),
            
            flat_patch_sampling={
                "root_spawn": FlatPatchSamplingCfg(
                    num_patches=5, patch_radius=0.2, 
                    x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), z_range=(-1.0, 1.0), 
                    max_height_diff=0.15
                ),
            },

            object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                height=0.05, radius=1.0, max_yx_angle=0.0, degrees=True, num_objects=20
            ),
            object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                height=0.2, radius=0.2, max_yx_angle=10.0, degrees=True, num_objects=20
            )
        )                      
    }       
)