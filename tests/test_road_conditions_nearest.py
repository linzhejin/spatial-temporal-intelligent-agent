import networkx as nx
from shapely.geometry import LineString

from spatial.coord_transform import gcj02_to_wgs84
from spatial.road_conditions import nearest_edges

CLICK_GCJ = (114.36, 30.54)
_CLICK_WGS = gcj02_to_wgs84(*CLICK_GCJ)  # (lng, lat)


def _road_graph():
    G = nx.MultiDiGraph()
    base_lng, base_lat = _CLICK_WGS

    def add(u, v, lat, name):
        geom = LineString([(base_lng - 0.01, lat), (base_lng + 0.01, lat)])
        G.add_edge(u, v, key=0, geometry=geom, length=220.0, name=name)
        G.add_edge(v, u, key=0,
                   geometry=LineString([(base_lng + 0.01, lat), (base_lng - 0.01, lat)]),
                   length=220.0, name=name)

    add(1, 2, base_lat + 0.0001, "近路")  # ~11 m 高于点击点
    add(3, 4, base_lat + 0.0006, "中路")  # ~67 m
    add(5, 6, base_lat + 0.0012, "远路")  # ~133 m
    return G


def test_nearest_edges_ranks_by_distance_and_limits():
    G = _road_graph()
    results = nearest_edges(G, *CLICK_GCJ, max_dist_m=500.0, limit=2)
    assert len(results) == 2
    assert [r["road_name"] for r in results] == ["近路", "中路"]
    assert all(r["dist_m"] > 0 for r in results)
    assert results[0]["dist_m"] < results[1]["dist_m"]


def test_nearest_edges_respects_radius():
    G = _road_graph()
    results = nearest_edges(G, *CLICK_GCJ, max_dist_m=60.0)
    assert len(results) == 1
    assert results[0]["road_name"] == "近路"


def test_nearest_edges_returns_snap_geometry():
    G = _road_graph()
    results = nearest_edges(G, *CLICK_GCJ)
    assert results
    for r in results:
        assert set(r) >= {"u", "v", "key", "edges", "chain_length_m", "geometry_gcj",
                          "snap_lng_gcj", "snap_lat_gcj", "dist_m", "road_name"}
        assert isinstance(r["geometry_gcj"], list) and r["geometry_gcj"]