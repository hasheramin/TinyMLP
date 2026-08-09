def topological_sort(root):
    """
    Returns a topologically sorted list of nodes from the computational graph rooted at `root`.
    """

    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(root)
    return topo


def trace(root):
    """
    Traces all nodes and edges in the computational graph ending at `root`.
    Returns (nodes, edges).
    """

    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges
