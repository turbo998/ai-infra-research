# AI Cluster L3 Routing Control Plane, BGP Unnumbered & RoCEv2 Multi-Rail Underlay Selection (`ai_cluster_bgp_evpn_rocev2_rail_underlay_selection`)

Deterministic Python 3.12 analytical evaluator for **SA Guide**: comparing `Legacy L2 VLAN / MLAG + 180s BGP Hold-Timer` (Tier 1), `Manual /31 IPv4 eBGP Clos + Equal ECMP` (Tier 2), `RFC 5549 BGP Unnumbered + Rail-Independent ASN + 300ms BFD` (Tier 3), and `BGP Unnumbered + Hardware 15ms BFD + BGP Link-Bandwidth WCMP + EVPN VRF` (Tier 4).

## Reproduction

```bash
python -m experiments.ai_cluster_bgp_evpn_rocev2_rail_underlay_selection.run
python -m unittest tests/test_ai_cluster_bgp_evpn_rocev2_rail_underlay_selection.py -q
```

## Provenance

- `type`: `synthetic_analytical_model_calibrated_to_published_network_rfcs_and_sonic_designs`
- `independently_reproduced`: `false`
- `physical_gpu_cluster_measurement`: `not_measured`
