from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "ai_content_provenance_watermarking_c2pa_synthid_tree_ring_selection",
        "title": "Enterprise AI Content Provenance & Watermarking Selection: HSM-Signed C2PA 2.1 + Latent SynthID/Tree-Ring vs. EXIF & Post-Hoc LSB",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "compliance_standards": "EU AI Act Art. 50 & China GB 45438-2025 Mandatory AIGC Implicit/Explicit Labeling",
            "adversarial_channel": "Social Media Metadata Strip + JPEG Q=60 Compression + 30% Crop/Screenshot",
        },
        "tiers": [
            {
                "tier": "Tier 1: Unencrypted EXIF / XMP Header + Spatial LSB Pixel Watermark",
                "survival_after_social_media_and_jpeg_pct": 2.5,
                "cryptographic_non_repudiation": False,
                "added_generation_latency_ms": 1.2,
                "perceptual_quality_degradation_psnr_db": 0.8,
                "recommended_scenario": "Anti-pattern (stripped by 100% of social platforms and destroyed by JPEG Q=75)",
            },
            {
                "tier": "Tier 2: Standalone C2PA 2.1 Cryptographic JUMBF Manifest (Without Signal Watermark)",
                "survival_after_social_media_and_jpeg_pct": 18.0,
                "cryptographic_non_repudiation": True,
                "added_generation_latency_ms": 4.5,
                "perceptual_quality_degradation_psnr_db": 0.0,
                "recommended_scenario": "Internal DAM pipelines where container headers are never stripped by screenshots",
            },
            {
                "tier": "Tier 3: Post-Generation CNN Encoder Watermark Sidecar (RivaGAN / HiDDeN / AudioSeal)",
                "survival_after_social_media_and_jpeg_pct": 88.5,
                "cryptographic_non_repudiation": False,
                "added_generation_latency_ms": 45.0,
                "perceptual_quality_degradation_psnr_db": 1.4,
                "recommended_scenario": "Retrofitting black-box third-party models where latent noise/logits are inaccessible",
            },
            {
                "tier": "Tier 4 (Recommended): Dual-Layer HSM C2PA 2.1 + In-Generation Latent/Logit Watermark (Tree-Ring / Gaussian Shading / SynthID)",
                "survival_after_social_media_and_jpeg_pct": 97.8,
                "cryptographic_non_repudiation": True,
                "added_generation_latency_ms": 4.8,
                "perceptual_quality_degradation_psnr_db": 0.0,
                "recommended_scenario": "Full EU AI Act Art. 50 & GB 45438-2025 compliance (97.8% survival + PKI non-repudiation)",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
