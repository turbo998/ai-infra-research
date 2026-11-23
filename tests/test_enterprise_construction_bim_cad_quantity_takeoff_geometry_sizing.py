from experiments.enterprise_construction_bim_cad_quantity_takeoff_geometry_sizing.run import build_summary


def test_enterprise_construction_bim_cad_quantity_takeoff_geometry_sizing() -> None:
    s = build_summary()
    assert s["turnaround_speedup_vs_manual"] == 493.71
    assert s["cost_reduction_pct_vs_manual"] == 99.8
    assert s["accuracy_gain_pts_vs_raster_vlm"] == 30.5
    assert s["schemes"]["vector_dxf_plus_tile_vlm_plus_shapely_geometry"]["takeoff_accuracy_pct"] == 99.1
