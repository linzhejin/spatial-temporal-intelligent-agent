import config


def test_auto_cpu_threads_caps_local_parallelism_at_four(tmp_path):
    assert config.resolve_vision_cpu_threads(
        "auto", cpu_count=18, cgroup_root=tmp_path,
    ) == 4


def test_auto_cpu_threads_respects_fractional_cgroup_quota(tmp_path):
    (tmp_path / "cpu.max").write_text("75000 100000\n", encoding="ascii")

    assert config.resolve_vision_cpu_threads(
        "auto", cpu_count=18, cgroup_root=tmp_path,
    ) == 1


def test_auto_cpu_threads_uses_cgroup_quota_when_it_is_over_one_core(tmp_path):
    (tmp_path / "cpu.max").write_text("150000 100000\n", encoding="ascii")

    assert config.resolve_vision_cpu_threads(
        "auto", cpu_count=18, cgroup_root=tmp_path,
    ) == 2


def test_explicit_cpu_thread_setting_overrides_auto_detection(tmp_path):
    (tmp_path / "cpu.max").write_text("50000 100000\n", encoding="ascii")

    assert config.resolve_vision_cpu_threads(
        "3", cpu_count=18, cgroup_root=tmp_path,
    ) == 3
