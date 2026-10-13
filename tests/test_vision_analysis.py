import pytest

from vision.analysis import analyze_observations


def frame_of_stopped_vehicles(count=8, offset=0):
    return [
        {"label": "car", "confidence": 0.92, "track_id": i,
         "box": [100 + i * 30 + offset, 120, 120 + i * 30 + offset, 140]}
        for i in range(count)
    ]


def test_stabilized_video_with_many_stopped_tracked_vehicles_proposes_congestion():
    frames = [frame_of_stopped_vehicles(offset=0) for _ in range(16)]
    roi = [{"id": "lane", "kind": "vehicle_lane", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    result = analyze_observations(
        frames, frame_width=640, frame_height=480, camera_stabilized=True, regions=roi,
    )
    assert result["metrics"]["frames_analyzed"] == 16
    candidate = next(item for item in result["candidates"] if item["kind"] == "possible_congestion")
    assert candidate["review_required"] is True
    assert candidate["evidence"]["summary"]["observed_duration_seconds"] == 15.0


def test_congestion_duration_does_not_bridge_missing_road_observations():
    frames = [frame_of_stopped_vehicles() for _ in range(18)]
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    region_transforms = [identity for _ in frames]
    region_transforms[8] = None
    pairwise_transforms = [None] + [identity for _ in frames[1:]]
    pairwise_transforms[8] = None
    roi = [{"id": "lane", "kind": "vehicle_lane",
            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]

    result = analyze_observations(
        frames, frame_width=640, frame_height=480,
        frame_transforms=pairwise_transforms,
        region_transforms=region_transforms, regions=roi,
    )

    assert all(item["kind"] != "possible_congestion" for item in result["candidates"])


def test_congestion_duration_does_not_bridge_large_decoder_timestamp_gap():
    frames = [frame_of_stopped_vehicles() for _ in range(90)]
    frame_times_s = [index * 0.2 for index in range(45)] + [
        20.0 + index * 0.2 for index in range(45)
    ]
    roi = [{"id": "lane", "kind": "vehicle_lane",
            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]

    result = analyze_observations(
        frames, frame_width=640, frame_height=480,
        sample_interval_s=0.2, frame_times_s=frame_times_s,
        camera_stabilized=True, regions=roi,
    )

    assert all(item["kind"] != "possible_congestion" for item in result["candidates"])


def test_short_red_light_like_stop_at_dense_sampling_does_not_propose_congestion():
    frames = [frame_of_stopped_vehicles() for _ in range(6)]
    roi = [{"id": "lane", "kind": "vehicle_lane", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]

    result = analyze_observations(
        frames, frame_width=640, frame_height=480, sample_interval_s=0.2,
        camera_stabilized=True, regions=roi,
    )

    assert all(item["kind"] != "possible_congestion" for item in result["candidates"])


def test_single_image_vehicle_count_never_claims_traffic_congestion():
    result = analyze_observations([frame_of_stopped_vehicles()], frame_width=640, frame_height=480)
    assert all(item["kind"] != "possible_congestion" for item in result["candidates"])
    assert result["metrics"]["motion_assessment"] == "insufficient_single_frame"


def test_video_without_stabilized_camera_never_claims_congestion_from_pixel_motion():
    frames = [frame_of_stopped_vehicles(offset=0) for _ in range(6)]
    result = analyze_observations(frames, frame_width=640, frame_height=480)
    assert result["metrics"]["motion_assessment"] == "camera_motion_uncompensated"
    assert all(item["kind"] != "possible_congestion" for item in result["candidates"])


def test_moving_camera_video_requires_roi_reassociation_before_congestion_candidate():
    frames = [frame_of_stopped_vehicles(offset=index * 20) for index in range(6)]
    translate_right = [[1, 0, 20], [0, 1, 0], [0, 0, 1]]
    roi = [{"id": "lane", "kind": "vehicle_lane", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]

    result = analyze_observations(
        frames, frame_width=640, frame_height=480,
        frame_transforms=[None] + [translate_right] * 5,
        regions=roi,
    )

    assert result["metrics"]["motion_assessment"] == "camera_motion_compensated"
    assert all(item["kind"] != "possible_congestion" for item in result["candidates"])
    assert result["safety"]["observation_region_reassociation_required"] is True
    assert result["safety"]["automatically_changes_routing"] is False
    assert result["safety"]["accident_recognition_supported"] is False


def test_moving_camera_projects_anchor_pedestrian_roi_before_crowd_candidate():
    frames = [
        [
            {"label": "pedestrian", "confidence": 0.9, "track_id": None,
             "box": [20 + index * 14 + frame_index * 20, 80,
                     30 + index * 14 + frame_index * 20, 115]}
            for index in range(16)
        ]
        for frame_index in range(6)
    ]
    pairwise = [[1, 0, 20], [0, 1, 0], [0, 0, 1]]
    roi_transforms = [
        [[1, 0, frame_index * 20], [0, 1, 0], [0, 0, 1]]
        for frame_index in range(6)
    ]
    roi = [{"id": "walkway", "kind": "pedestrian",
            "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}]

    result = analyze_observations(
        frames, frame_width=640, frame_height=480,
        frame_transforms=[None] + [pairwise] * 5,
        region_transforms=roi_transforms, regions=roi,
    )

    candidate = next(item for item in result["candidates"] if item["kind"] == "possible_crowding")
    assert candidate["evidence"]["region_id"] == "walkway"
    assert candidate["evidence"]["peak_person_count"] == 16
    assert candidate["evidence"]["region_mapping_basis"] == "homography_projected_from_video_anchor"
    assert result["safety"]["observation_region_reassociation_required"] is False


def test_moving_camera_does_not_emit_roi_candidate_when_anchor_mapping_is_incomplete():
    pedestrians = [
        {"label": "pedestrian", "confidence": 0.9, "track_id": None,
         "box": [30 + index * 14, 80, 40 + index * 14, 115]}
        for index in range(16)
    ]
    pairwise = [[1, 0, 20], [0, 1, 0], [0, 0, 1]]
    roi_transforms = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], None, None, None, None, None,
    ]
    roi = [{"id": "walkway", "kind": "pedestrian",
            "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}]

    result = analyze_observations(
        [pedestrians] * 6, frame_width=640, frame_height=480,
        frame_transforms=[None] + [pairwise] * 5,
        region_transforms=roi_transforms, regions=roi,
    )

    assert all(item["kind"] != "possible_crowding" for item in result["candidates"])
    assert result["safety"]["observation_region_reassociation_required"] is True


def test_roi_projection_rejects_a_region_that_is_only_partly_visible():
    from vision.analysis import project_regions_for_frame

    roi = [{"id": "walkway", "kind": "pedestrian",
            "polygon": [[0.5, 0.1], [0.9, 0.1], [0.9, 0.9], [0.5, 0.9]]}]
    translation_out_of_view = [[1, 0, 20], [0, 1, 0], [0, 0, 1]]

    assert project_regions_for_frame(roi, translation_out_of_view, 100, 100) == []


def test_region_transform_count_must_match_analyzed_frames():
    roi = [{"id": "walkway", "kind": "pedestrian",
            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]

    with pytest.raises(ValueError, match="match the analyzed frame count"):
        analyze_observations(
            [[], []], frame_width=640, frame_height=480,
            region_transforms=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], regions=roi,
        )


@pytest.mark.parametrize("cluster_size", [True, 1, 1.5, 0])
def test_minimum_image_space_cluster_size_must_be_an_integer_at_least_two(cluster_size):
    with pytest.raises(ValueError, match="minimum image-space crowd cluster size"):
        analyze_observations(
            [[]], frame_width=640, frame_height=480,
            min_crowding_cluster_size=cluster_size,
        )


def test_crowd_candidate_is_blocked_when_exclusion_roi_is_not_mapped_reliably():
    pedestrians = [
        {"label": "pedestrian", "confidence": 0.9, "track_id": None,
         "box": [8 + index * 2, 30, 10 + index * 2, 50]}
        for index in range(16)
    ]
    transforms = [
        [[1, 0, shift], [0, 1, 0], [0, 0, 1]]
        for shift in (0, 10, 20, 30, 40, 50)
    ]
    regions = [
        {"id": "walkway", "kind": "pedestrian",
         "polygon": [[0.05, 0.1], [0.5, 0.1], [0.5, 0.9], [0.05, 0.9]]},
        {"id": "private-yard", "kind": "exclude",
         "polygon": [[0.4, 0.1], [0.7, 0.1], [0.7, 0.9], [0.4, 0.9]]},
    ]

    result = analyze_observations(
        [pedestrians] * 6, frame_width=100, frame_height=100,
        frame_transforms=[None] + [transforms[1]] * 5,
        region_transforms=transforms, regions=regions,
    )

    assert all(item["kind"] != "possible_crowding" for item in result["candidates"])
    assert result["safety"]["observation_region_reassociation_required"] is True


def test_fixed_camera_does_not_require_observation_region_reassociation():
    frames = [frame_of_stopped_vehicles() for _ in range(3)]
    roi = [{"id": "lane", "kind": "vehicle_lane", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]

    result = analyze_observations(
        frames, frame_width=640, frame_height=480, camera_stabilized=True, regions=roi,
    )

    assert result["safety"]["observation_region_reassociation_required"] is False


def test_video_with_unreliable_camera_motion_compensation_only_reports_vehicle_counts():
    frames = [frame_of_stopped_vehicles(offset=index * 20) for index in range(6)]
    result = analyze_observations(
        frames, frame_width=640, frame_height=480,
        frame_transforms=[None, None, None, None, None, None],
    )

    assert result["metrics"]["motion_assessment"] == "camera_motion_uncompensated"
    assert all(item["kind"] != "possible_congestion" for item in result["candidates"])


def test_vehicle_detector_classes_never_become_accident_candidates():
    frame = frame_of_stopped_vehicles()
    result = analyze_observations([frame], frame_width=640, frame_height=480)
    assert all(item["kind"] != "possible_accident" for item in result["candidates"])


def test_accident_candidate_requires_explicit_detector_class_and_human_review():
    frame = [{"label": "accident", "confidence": 0.91, "track_id": None, "box": [10, 10, 80, 80]}]
    unsupported = analyze_observations([frame], frame_width=640, frame_height=480)
    assert all(item["kind"] != "possible_accident" for item in unsupported["candidates"])
    roi = [{"id": "lane", "kind": "vehicle_lane", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    result = analyze_observations(
        [frame], frame_width=640, frame_height=480, accident_model_enabled=True, regions=roi,
    )
    candidate = next(item for item in result["candidates"] if item["kind"] == "possible_accident")
    assert candidate["review_required"] is True
    assert candidate["auto_publish"] is False
    assert candidate["evidence"]["summary"]["region_id"] == "lane"


def test_accident_label_outside_annotated_vehicle_region_is_not_a_candidate():
    frame = [{"label": "accident", "confidence": 0.99, "track_id": None, "box": [500, 400, 600, 450]}]
    roi = [{"id": "lane", "kind": "vehicle_lane",
            "polygon": [[0, 0], [0.4, 0], [0.4, 1], [0, 1]]}]

    result = analyze_observations(
        [frame], frame_width=640, frame_height=480,
        accident_model_enabled=True, regions=roi,
    )

    assert all(item["kind"] != "possible_accident" for item in result["candidates"])


def test_vehicle_cluster_is_a_review_observation_not_a_road_condition():
    roi = [{"id": "lane", "kind": "vehicle_lane", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    result = analyze_observations(
        [frame_of_stopped_vehicles()], frame_width=640, frame_height=480, regions=roi,
    )
    assert any(item["kind"] == "vehicle_cluster_review" for item in result["candidates"])
    assert all(item["auto_publish"] is False for item in result["candidates"])
    cluster = next(item for item in result["candidates"] if item["kind"] == "vehicle_cluster_review")
    assert cluster["evidence"]["summary"]["region_id"] == "lane"


def test_crowd_candidate_requires_pedestrian_roi_and_sustained_video_evidence():
    pedestrians = [
        {"label": "pedestrian", "confidence": 0.9, "track_id": None,
         "box": [20 + index * 18, 80, 32 + index * 18, 120]}
        for index in range(16)
    ]
    roi = [{"id": "plaza", "kind": "pedestrian", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    result = analyze_observations(
        [pedestrians] * 6, frame_width=640, frame_height=480,
        camera_stabilized=True, regions=roi,
    )

    candidate = next(item for item in result["candidates"] if item["kind"] == "possible_crowding")
    assert candidate["review_required"] is True
    assert candidate["auto_publish"] is False
    assert candidate["evidence"]["region_id"] == "plaza"
    assert candidate["evidence"]["peak_person_count"] == 16
    assert "persons_per_square_meter" not in candidate["evidence"]
    assert candidate["evidence"]["peak_people_with_nearby_peer_count"] == 16
    assert candidate["evidence"]["movement_assessment"] == "mostly_stationary_in_image"
    assert candidate["evidence"]["observed_duration_seconds"] == 5.0
    assert candidate["evidence"]["sustained_cluster_frames"] == 6
    assert candidate["evidence"]["sustained_cluster_duration_seconds"] == 5.0
    assert candidate["evidence"]["minimum_cluster_size"] == 8


def test_brief_pedestrian_cluster_at_dense_sampling_does_not_propose_crowding():
    pedestrians = [
        {"label": "pedestrian", "confidence": 0.9, "track_id": None,
         "box": [20 + index * 18, 80, 32 + index * 18, 120]}
        for index in range(16)
    ]
    roi = [{"id": "plaza", "kind": "pedestrian", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]

    result = analyze_observations(
        [pedestrians] * 3, frame_width=640, frame_height=480, sample_interval_s=0.2,
        camera_stabilized=True, regions=roi,
    )

    assert all(item["kind"] != "possible_crowding" for item in result["candidates"])


def test_crowd_candidate_identifies_the_roi_that_contains_the_people():
    pedestrians = [
        {"label": "pedestrian", "confidence": 0.9, "track_id": None,
         "box": [20 + index * 14, 80, 32 + index * 14, 120]}
        for index in range(16)
    ]
    regions = [
        {"id": "empty-plaza", "kind": "pedestrian", "polygon": [[0.72, 0], [1, 0], [1, 1], [0.72, 1]]},
        {"id": "crowded-walkway", "kind": "pedestrian", "polygon": [[0, 0], [0.55, 0], [0.55, 1], [0, 1]]},
    ]

    result = analyze_observations(
        [pedestrians] * 6, frame_width=640, frame_height=480,
        camera_stabilized=True, regions=regions,
    )

    candidate = next(item for item in result["candidates"] if item["kind"] == "possible_crowding")
    assert candidate["evidence"]["region_id"] == "crowded-walkway"


def test_congestion_candidate_is_bound_to_its_vehicle_lane_roi():
    frames = [frame_of_stopped_vehicles(offset=0) for _ in range(16)]
    roi = [{"id": "east-lane", "kind": "vehicle_lane", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]

    result = analyze_observations(
        frames, frame_width=640, frame_height=480, camera_stabilized=True, regions=roi,
    )

    candidate = next(item for item in result["candidates"] if item["kind"] == "possible_congestion")
    assert candidate["evidence"]["summary"]["region_id"] == "east-lane"


def test_pedestrians_outside_observed_region_are_not_counted_as_crowding():
    pedestrians = [
        {"label": "pedestrian", "confidence": 0.9, "track_id": None,
         "box": [400 + index, 400, 410 + index, 420]}
        for index in range(20)
    ]
    roi = [{"id": "walkway", "kind": "pedestrian", "polygon": [[0, 0], [0.3125, 0], [0.3125, 0.4166667], [0, 0.4166667]]}]
    result = analyze_observations(
        [pedestrians] * 4, frame_width=640, frame_height=480,
        camera_stabilized=True, regions=roi,
    )

    assert result["metrics"]["peak_pedestrian_count"] == 0
    assert all(item["kind"] != "possible_crowding" for item in result["candidates"])


def test_distributed_people_do_not_produce_a_crowding_candidate():
    pedestrians = [
        {"label": "pedestrian", "confidence": 0.9, "track_id": None,
         "box": [40 + (index % 5) * 110, 40 + (index // 5) * 130,
                 50 + (index % 5) * 110, 60 + (index // 5) * 130]}
        for index in range(15)
    ]
    roi = [{"id": "plaza", "kind": "pedestrian", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]

    result = analyze_observations(
        [pedestrians] * 6, frame_width=640, frame_height=480,
        camera_stabilized=True, regions=roi,
    )

    assert all(item["kind"] != "possible_crowding" for item in result["candidates"])


def test_transient_image_space_cluster_does_not_produce_a_crowding_candidate():
    clustered = [
        {"label": "pedestrian", "confidence": 0.9, "track_id": None,
         "box": [20 + index * 12, 80, 32 + index * 12, 120]}
        for index in range(16)
    ]
    distributed = [
        {"label": "pedestrian", "confidence": 0.9, "track_id": None,
         "box": [40 + (index % 5) * 110, 40 + (index // 5) * 130,
                 50 + (index % 5) * 110, 60 + (index // 5) * 130]}
        for index in range(15)
    ]
    roi = [{"id": "plaza", "kind": "pedestrian", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]

    result = analyze_observations(
        [clustered, clustered, distributed, distributed, distributed, distributed],
        frame_width=640, frame_height=480, camera_stabilized=True, regions=roi,
    )

    assert all(item["kind"] != "possible_crowding" for item in result["candidates"])


def test_separate_small_groups_do_not_pass_the_single_cluster_threshold():
    pedestrians = [
        {"label": "pedestrian", "confidence": 0.9, "track_id": None,
         "box": [start + index * 12, 80, start + index * 12 + 10, 120]}
        for start in (30, 300)
        for index in range(4)
    ] + [
        {"label": "pedestrian", "confidence": 0.9, "track_id": None,
         "box": [500, 400, 510, 420]}
    ]
    roi = [{"id": "plaza", "kind": "pedestrian", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]

    result = analyze_observations(
        [pedestrians] * 6, frame_width=640, frame_height=480,
        camera_stabilized=True, regions=roi,
    )

    assert all(item["kind"] != "possible_crowding" for item in result["candidates"])


def test_crowd_candidate_reports_observed_image_space_movement():
    frames = [
        [
            {"label": "pedestrian", "confidence": 0.9, "track_id": None,
             "box": [20 + index * 22 + frame_index * 4, 80,
                     32 + index * 22 + frame_index * 4, 120]}
            for index in range(16)
        ]
        for frame_index in range(6)
    ]
    roi = [{"id": "walkway", "kind": "pedestrian", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]

    result = analyze_observations(
        frames, frame_width=640, frame_height=480,
        camera_stabilized=True, regions=roi,
    )

    candidate = next(item for item in result["candidates"] if item["kind"] == "possible_crowding")
    assert candidate["evidence"]["movement_assessment"] == "mostly_moving_in_image"
    assert candidate["evidence"]["moving_person_track_count"] > 0


def test_vehicle_detections_inside_marked_parking_region_are_excluded_from_road_counts():
    frames = [frame_of_stopped_vehicles()]
    parking = [{"id": "parking", "kind": "parking", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    result = analyze_observations(frames, frame_width=640, frame_height=480, regions=parking)
    assert result["metrics"]["peak_vehicle_count"] == 0
    assert all(item["kind"] != "vehicle_cluster_review" for item in result["candidates"])
