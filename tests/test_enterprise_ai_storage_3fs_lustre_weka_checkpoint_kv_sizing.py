from experiments.enterprise_ai_storage_3fs_lustre_weka_checkpoint_kv_sizing.run import evaluate_ai_storage


def test_evaluate_ai_storage():
    r = evaluate_ai_storage(32, 2400)
    assert r["sync_nfs_stall_sec"] == 240.0
    assert r["async_phase1_stall_sec"] == 2.8
    assert r["speedup_ratio"] == 85.7
    assert r["aggregate_3fs_bw_gbps"] == 1760
    assert r["three_fs_monthly_usd"] == 2800
