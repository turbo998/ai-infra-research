# Enterprise AI Content Provenance & Watermarking Selection Experiment (`C2PA 2.1 + SynthID / Tree-Ring` vs. `EXIF / LSB`)

Reproducible analytical model comparing 4 AI content provenance and watermarking architectures under social media metadata stripping, `JPEG Q=60` re-compression, and `30%` screenshot cropping (`EU AI Act Art. 50` & `GB 45438-2025` compliance):
- **Tier 1 (`EXIF / XMP Header + Spatial LSB Pixel Watermark`)**: `2.5%` survival rate after social media upload/JPEG, zero cryptographic non-repudiation.
- **Tier 2 (`Standalone C2PA 2.1 Cryptographic JUMBF Manifest`)**: `100%` PKI non-repudiation when intact, **but `18.0%` survival in the wild after screenshot/social-media container stripping**.
- **Tier 3 (`Post-Generation CNN Watermark Sidecar RivaGAN / AudioSeal`)**: `88.5%` survival, `+45.0 ms` latency.
- **Tier 4 (`Dual-Layer HSM-Signed C2PA 2.1 + In-Generation Latent/Logit Watermark [Tree-Ring / Gaussian Shading / SynthID] + Soft-Binding Registry`, Recommended)**: **`97.8%` survival after JPEG/crop/screenshot**, full PKI cryptographic non-repudiation, **`0.0 dB` distortionless quality**, and **`+4.8 ms` HSM signing latency**.

## Run & Verify

```bash
python experiments/ai_content_provenance_watermarking_c2pa_synthid_tree_ring_selection/run.py
python experiments/ai_content_provenance_watermarking_c2pa_synthid_tree_ring_selection/run.py --check
```
