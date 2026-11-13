"""Original GNN dependency-closure and fanout/transfer accounting model."""

import argparse
from dataclasses import asdict, dataclass, replace
import itertools
import json
import math
from pathlib import Path


GRAPH=[[1,2,3],[0,4],[0,3,5],[1,6],[2,3,5],[0,6],[]]
FEATURES=[[-2.0,0.5],[1.0,-0.8],[0.2,1.3],[-0.4,-1.1],[1.7,0.3],[-1.2,0.9],[0.6,-0.2]]
SEEDS=[4,1]
LABELS=[0.2,-0.5]
SELF=[[0.6,-0.2],[0.1,0.4]]
NEIGHBOR=[[0.3,0.5],[-0.6,0.2]]
BIAS=[0.15,0.12]


def validate(graph: list, features: list):
    if not graph or len(graph)!=len(features) or any(len(row)!=2 for row in features):
        raise ValueError("graph and two-dimensional node features must match")
    if any(not math.isfinite(value) for row in features for value in row):
        raise ValueError("features must be finite")
    if any(type(node) is not int or not 0<=node<len(graph) for row in graph for node in row):
        raise ValueError("neighbor id is outside the graph")


def forward(graph: list, features: list, layers: int=2, divisors: list | None=None) -> tuple[list,list]:
    validate(graph,features)
    if type(layers) is not int or layers<1:
        raise ValueError("layers must be a positive integer")
    if divisors is not None and (len(divisors)!=len(graph) or any(type(n) is not int or n<0 for n in divisors)):
        raise ValueError("divisors must be nonnegative node counts")
    history=[[row[:] for row in features]]
    preactivations=[]
    for _ in range(layers):
        previous=history[-1]
        raw=[]
        for node,neighbors in enumerate(graph):
            divisor=divisors[node] if divisors is not None else len(neighbors)
            if neighbors and not divisor:
                raise ValueError("nonempty neighborhood needs a positive divisor")
            pooled=[sum(previous[j][c] for j in neighbors)/divisor if neighbors else 0.0 for c in range(2)]
            raw.append([BIAS[c]+sum(SELF[c][k]*previous[node][k]+NEIGHBOR[c][k]*pooled[k] for k in range(2))
                        for c in range(2)])
        preactivations.append(raw)
        history.append([[max(value,0.0) for value in row] for row in raw])
    return history,preactivations


def seed_loss(output: list, seeds: list[int], labels: list[float]) -> float:
    if not seeds or len(seeds)!=len(labels) or any(not 0<=node<len(output) for node in seeds):
        raise ValueError("loss needs valid seeds with matching labels")
    return sum((output[node][0]+0.3*output[node][1]-label)**2 for node,label in zip(seeds,labels))/len(seeds)


def feature_gradient(graph: list, features: list, seeds: list[int], labels: list[float], layers: int=2) -> list:
    history,raw=forward(graph,features,layers)
    seed_loss(history[-1],seeds,labels)
    gradient=[[0.0,0.0] for _ in graph]
    for node,label in zip(seeds,labels):
        coefficient=2*(history[-1][node][0]+0.3*history[-1][node][1]-label)/len(seeds)
        gradient[node][0]+=coefficient
        gradient[node][1]+=0.3*coefficient
    for layer in reversed(range(layers)):
        previous_gradient=[[0.0,0.0] for _ in graph]
        for node,neighbors in enumerate(graph):
            dz=[gradient[node][c] if raw[layer][node][c]>0 else 0.0 for c in range(2)]
            for k in range(2):
                previous_gradient[node][k]+=sum(SELF[c][k]*dz[c] for c in range(2))
                for neighbor in neighbors:
                    previous_gradient[neighbor][k]+=sum(NEIGHBOR[c][k]*dz[c] for c in range(2))/len(neighbors)
        gradient=previous_gradient
    return gradient


def sample(graph: list, seeds: list[int], fanouts: list[int]) -> dict:
    if not seeds or len(set(seeds))!=len(seeds) or any(type(n) is not int or not 0<=n<len(graph) for n in seeds):
        raise ValueError("seeds must be distinct valid global node ids")
    if not fanouts or any(type(n) is not int or (n!=-1 and n<1) for n in fanouts):
        raise ValueError("fanouts must be positive integers or -1")
    ids=list(seeds)
    known=set(ids)
    expanded=set()
    frontier=list(seeds)
    edges={}
    hop_new_nodes=[]
    for fanout in fanouts:
        following=[]
        for node in frontier:
            if node in expanded:
                continue
            expanded.add(node)
            # Deterministic first-neighbor fixture; not a claim of uniform sampling.
            chosen=graph[node][:] if fanout==-1 else graph[node][:fanout]
            edges[node]=chosen
            for neighbor in chosen:
                if neighbor not in known:
                    known.add(neighbor)
                    ids.append(neighbor)
                    following.append(neighbor)
        hop_new_nodes.append(len(following))
        frontier=following
    local={node:index for index,node in enumerate(ids)}
    return {
        "n_id":ids,"batch_size":len(seeds),"hop_new_nodes":hop_new_nodes,
        "local_graph":[[local[neighbor] for neighbor in edges.get(node,[])] for node in ids],
        "expanded_global_nodes":sorted(expanded),
    }


def sampled_result(fanouts: list[int], wrong_mapping: bool=False, reverse_edges: bool=False, full_degree: bool=False) -> dict:
    graph=[row[:] for row in GRAPH]
    if reverse_edges:
        graph=[[] for _ in GRAPH]
        for destination,neighbors in enumerate(GRAPH):
            for source in neighbors:
                graph[source].append(destination)
    metadata=sample(graph,SEEDS,fanouts)
    ids=metadata["n_id"]
    features=[FEATURES[i][:] for i in (range(len(ids)) if wrong_mapping else ids)]
    divisors=[len(graph[node]) for node in ids] if full_degree else None
    history,_=forward(metadata["local_graph"],features,divisors=divisors)
    return {
        "metadata":metadata,
        "seed_output":history[-1][:len(SEEDS)],
        "seed_loss":seed_loss(history[-1],list(range(len(SEEDS))),LABELS),
    }


@dataclass(frozen=True)
class Budget:
    graph_nodes: int=1000000
    seeds: int=1024
    feature_dimension: int=128
    scalar_bytes: int=2
    fanouts: tuple[int,...]=(15,10)
    bandwidth_gbps: float=50.0
    transfer_setup_us: float=10.0


def estimate(b: Budget) -> dict:
    for name in ("graph_nodes","seeds","feature_dimension","scalar_bytes"):
        if type(getattr(b,name)) is not int or getattr(b,name)<1:
            raise ValueError(f"{name} must be a positive integer")
    if b.seeds>b.graph_nodes or b.scalar_bytes not in (2,4):
        raise ValueError("invalid seed count or scalar width")
    if not b.fanouts or any(type(n) is not int or n<1 for n in b.fanouts):
        raise ValueError("budget fanouts must be positive integers; full-neighbor sizes need graph degrees")
    if not math.isfinite(b.bandwidth_gbps) or b.bandwidth_gbps<=0:
        raise ValueError("bandwidth must be finite and positive")
    if not math.isfinite(b.transfer_setup_us) or b.transfer_setup_us<0:
        raise ValueError("setup must be finite and nonnegative")
    multiplicity=b.seeds
    frontier=b.seeds
    for fanout in b.fanouts:
        frontier*=fanout
        multiplicity+=frontier
    unique_bound=min(b.graph_nodes,multiplicity)
    subset_bytes=unique_bound*b.feature_dimension*b.scalar_bytes
    full_bytes=b.graph_nodes*b.feature_dimension*b.scalar_bytes
    return {
        "input":asdict(b),"tree_occurrence_bound":multiplicity,"unique_node_upper_bound":unique_bound,
        "actual_unique_nodes":None,"feature_subset_bound_bytes":subset_bytes,"full_graph_feature_bytes":full_bytes,
        "subset_transfer_modeled_ms":subset_bytes/(b.bandwidth_gbps*1e6)+b.transfer_setup_us/1000,
        "full_transfer_modeled_ms":full_bytes/(b.bandwidth_gbps*1e6)+b.transfer_setup_us/1000,
        "gpu_latency_ms":None,"model_accuracy":None,
        "scope":"uncached feature transfer bound only; no adjacency, weights, activations, sampling or GPU computation",
    }


def nonlinear_expectations() -> list:
    values=[-2.0,1.0,1.0]
    full_mean=sum(values)/len(values)
    rows=[]
    for count in (1,2,3):
        means=[sum(choice)/count for choice in itertools.combinations(values,count)]
        rows.append({
            "fanout":count,"equally_likely_subsets":len(means),"full_linear_mean":full_mean,
            "expected_sample_mean":sum(means)/len(means),
            "full_relu_mean":max(full_mean,0.0),
            "expected_relu_of_sample_mean":sum(max(value,0.0) for value in means)/len(means),
        })
    return rows


def simulate() -> dict:
    full,_=forward(GRAPH,FEATURES)
    reference=[full[-1][node] for node in SEEDS]
    cases=[]
    configurations=[
        ("full-neighbor-closure",[-1,-1],{},False),
        ("one-hop-for-two-layers",[-1],{},False),
        ("wrong-local-feature-index",[-1,-1],{"wrong_mapping":True},False),
        ("reverse-message-direction",[-1,-1],{"reverse_edges":True},False),
        ("finite-fanout-first-neighbor",[1,1],{},True),
        ("sample-mean-with-full-degree",[1,1],{"full_degree":True},False),
    ]
    for name,fanouts,options,intentional in configurations:
        result=sampled_result(fanouts,**options)
        compared_to="full-graph-seed-output"
        target=reference
        if options.get("full_degree"):
            target=sampled_result([1,1])["seed_output"]
            compared_to="same-sampled-edges-with-sampled-degree-mean"
        error=max(abs(a-b) for x,y in zip(result["seed_output"],target) for a,b in zip(x,y))
        cases.append({
            "id":name,**result,"maximum_absolute_error":error,"matches_reference":error<1e-12,
            "reference_scope":compared_to,"intentional_sampling_approximation":intentional,"confidence":None,
        })
    base=Budget()
    return {
        "schema_version":1,"round":97,"provenance":"synthetic","independently_reproduced":False,
        "physical_gpu_measurement":False,
        "source":"https://github.com/pyg-team/pytorch_geometric/blob/5570242624e163e541716a993918b7d534de6369/torch_geometric/loader/neighbor_loader.py",
        "source_scope":"hop counts, directional-subgraph dependency limits, seed-first order and n_id mappings; not backend reproduction",
        "fixture":{"incoming_neighbors":GRAPH,"features":FEATURES,"seeds":SEEDS,"labels":LABELS,
                   "layers":2,"tolerance":1e-12,"reference_seed_output":reference},
        "cases":cases,"uniform_subset_enumeration":nonlinear_expectations(),
        "budget":estimate(base),
        "scaling":[estimate(replace(base,fanouts=fanouts)) for fanouts in ((5,5),(10,10),(15,10),(25,15))],
        "quality":None,
        "notes":[
            "Original scalar two-layer mean-message model with ReLU, not PyG/GPU execution.",
            "Finite-fanout fixture selects first neighbors deterministically, not uniformly or representatively.",
            "Uniform subset enumeration is a separate exact three-value example, not a Monte Carlo accuracy benchmark.",
            "NeighborLoader directional/bidirectional/induced policies are different; this toy models directed incoming dependencies.",
            "Tree occurrence and capped unique-node upper bounds are not measured unique-node counts.",
        ],
    }


def outputs() -> dict[Path,str]:
    from .render import render_outputs
    report=simulate()
    return {Path(__file__).with_name("report.json"):json.dumps(report,indent=2)+"\n",**render_outputs(report)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    for path,content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8")!=content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8",newline="\n")
    print("Round 97 artifacts match." if args.check else "Wrote Round 97 artifacts.")


if __name__=="__main__":
    main()
