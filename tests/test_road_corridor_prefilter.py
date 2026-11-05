"""Road closures must still cover parallel geometry without scanning distant roads."""

import networkx as nx

from spatial import road_conditions


def test_corridor_keeps_parallel_and_curved_edges_but_skips_distant_roads(monkeypatch):
    graph = nx.MultiDiGraph()
    lon, lat = 114.35, 30.54

    def add_edge(u, v, start, end, geometry=None):
        graph.add_node(u, x=start[0], y=start[1])
        graph.add_node(v, x=end[0], y=end[1])
        attrs = {} if geometry is None else {"geometry": geometry}
        graph.add_edge(u, v, key=0, **attrs)

    # The bound road and its nearby parallel duplicate must both remain closed.
    add_edge(1, 2, (lon, lat), (lon + 0.001, lat))
    add_edge(3, 4, (lon, lat + 0.00005), (lon + 0.001, lat + 0.00005))
    # The endpoints of this WKT edge lie outside the buffer, but the middle
    # follows the corridor. Endpoint-only filtering would wrongly miss it.
    add_edge(
        5, 6, (lon - 0.0001, lat + 0.0002), (lon + 0.0011, lat + 0.0002),
        f"LINESTRING ({lon - 0.0001} {lat + 0.0002}, {lon} {lat}, "
        f"{lon + 0.001} {lat}, {lon + 0.0011} {lat + 0.0002})",
    )
    # A crossing only touching the corridor is not the same physical road.
    add_edge(7, 8, (lon + 0.0005, lat - 0.0002), (lon + 0.0005, lat + 0.0002))
    # Invalid WKT follows the existing endpoint fallback and must not be lost.
    add_edge(9, 10, (lon, lat + 0.00004), (lon + 0.001, lat + 0.00004), "bad geometry")

    for i in range(300):
        base = 100 + 2 * i
        x = lon + 0.02 + i * 0.00001
        geometry = None if i % 2 else f"LINESTRING ({x} {lat}, {x + 0.0001} {lat})"
        add_edge(base, base + 1, (x, lat), (x + 0.0001, lat), geometry)

    original = road_conditions._edge_line
    calls = []

    def tracked_line(g, u, v, data):
        calls.append((u, v))
        return original(g, u, v, data)

    monkeypatch.setattr(road_conditions, "_edge_line", tracked_line)
    result = road_conditions._expand_coincident_corridor_edges(graph, {(1, 2, 0)}, {})

    assert {(1, 2, 0), (3, 4, 0), (5, 6, 0), (9, 10, 0)} <= result
    assert (7, 8, 0) not in result
    assert not any(u >= 100 for u, _, _ in result)
    assert len(calls) < 20, "distant roads should be rejected before geometry construction"
