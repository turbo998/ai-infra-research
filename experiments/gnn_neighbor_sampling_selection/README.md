# GNN neighbor dependencies, seed identity and fanout bounds

Round 97. **Synthetic; independently_reproduced=false; no PyG/GPU execution or
model accuracy measurement.** The original two-layer scalar model uses incoming
mean messages, self features and ReLU. It is not a PyG backend reproduction.

```powershell
python -m experiments.gnn_neighbor_sampling_selection.run
python -m experiments.gnn_neighbor_sampling_selection.run --check
python -m unittest tests.test_gnn_neighbor_sampling_selection
```

Full-neighbor two-hop closure reproduces the full-graph seed outputs and
seed-only feature gradients. Every input-feature derivative also has an
independent central-difference check. Wrong controls cover insufficient hops,
local/global feature-index confusion, changed message direction and dividing
sampled means by full degrees.

Finite fanouts select first neighbors deterministically in the original
fixture. That is explicitly an approximation, not a uniform-sampling claim.
A separate exact enumeration of uniform subsets of `[-2,1,1]` shows that an
unbiased sample mean does not imply unbiased post-ReLU predictions.

The fanout calculator counts tree occurrences and caps a unique-node upper
bound by the full graph size; actual unique nodes stay null. Only uncached
feature-transfer bytes/bandwidth plus assumed setup are modeled. Adjacency,
sampling, model weights, activations and GPU compute are excluded.

The pinned NeighborLoader documentation supplies hop, subgraph-policy,
seed-first and n_id contracts. Its established source date is disclosed, not
presented as a new launch or popularity ranking. Static cards are script-free
under strict CSP; the separate local workbench permits inline calculation.
