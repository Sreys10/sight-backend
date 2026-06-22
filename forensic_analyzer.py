import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
from datetime import datetime as dt
import subprocess
import json as jsonlib

class HybridForensicAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.score = 0
        self.reasons = []

    # ---- METADATA EXTRACTION ----
    def extract_metadata(self):
        """Try exiftool first, fall back to PIL EXIF"""
        metadata = {}

        # Try exiftool (richer metadata)
        try:
            result = subprocess.run(
                ["exiftool", "-json", self.image_path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                raw = jsonlib.loads(result.stdout)[0]
                # Filter out binary/thumbnail fields and convert values
                skip_keys = ["ThumbnailImage", "PreviewImage", "JFIFThumbnail",
                             "SourceFile", "Directory", "ExifToolVersion"]
                for k, v in raw.items():
                    if k in skip_keys:
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        metadata[k] = v
                    else:
                        metadata[k] = str(v)
                return metadata, "exiftool"
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            print(f"exiftool not available, falling back to PIL: {e}")

        # Fallback: PIL EXIF
        try:
            image = Image.open(self.image_path)
            exif_data = image._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    decoded = TAGS.get(tag, tag)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='replace')
                        except:
                            value = str(value)
                    elif not isinstance(value, (str, int, float, bool)):
                        value = str(value)
                    metadata[decoded] = value
        except Exception:
            pass

        return metadata, "PIL"

    # ---- METADATA ANALYSIS ----
    def analyze_metadata(self, metadata):
        flags = []

        # Check 1: Camera info
        if "Make" not in metadata and "Camera Make" not in metadata:
            flags.append({"text": "Camera information missing", "severity": "info", "points": 0})

        # Check 2: Editing software
        software = metadata.get("Software", metadata.get("Creator Tool", ""))
        if software:
            editors = ["Adobe", "Photoshop", "GIMP", "Lightroom", "Snapseed", "PicsArt", "Canva", "Figma"]
            if any(tool.lower() in str(software).lower() for tool in editors):
                self.score += 3
                flags.append({"text": f"Edited using: {software}", "severity": "high", "points": 3})
            else:
                flags.append({"text": f"Software: {software}", "severity": "info", "points": 0})

        # Check 3: Missing original date
        has_date = any(k in metadata for k in ["DateTimeOriginal", "Date/Time Original", "CreateDate", "Create Date"])
        if not has_date:
            flags.append({"text": "Missing original capture date", "severity": "info", "points": 0})

        # Check 4: Resolution anomaly
        x_res = metadata.get("XResolution", metadata.get("X Resolution", None))
        if x_res is not None and str(x_res).strip() in ["1", "1.0"]:
            flags.append({"text": "Resolution = 1 (export/web artifact)", "severity": "info", "points": 0})

        # Check 5: GPS data present
        gps_keys = [k for k in metadata if 'GPS' in str(k).upper()]
        if gps_keys:
            flags.append({"text": f"GPS location data found ({len(gps_keys)} fields)", "severity": "info", "points": 0})

        # Check 6: File size vs dimensions check
        file_size = metadata.get("FileSize", metadata.get("File Size", ""))
        if file_size and "KB" in str(file_size).upper():
            # Very small file might indicate heavy compression/editing
            try:
                size_val = float(str(file_size).split()[0])
                if size_val < 50:
                    flags.append({"text": f"Unusually small file size: {file_size}", "severity": "info", "points": 0})
            except:
                pass

        return flags

    # ---- ELA ANALYSIS ----
    def perform_ela(self, quality=90):
        ela_result = {
            "performed": False,
            "meanIntensity": 0,
            "elaImage": None,
            "interpretation": ""
        }

        try:
            original = Image.open(self.image_path).convert("RGB")

            # Save compressed version to temp file
            ela_temp = self.image_path + "_ela_temp.jpg"
            original.save(ela_temp, "JPEG", quality=quality)
            compressed = Image.open(ela_temp)

            # Compute difference
            ela_image = ImageChops.difference(original, compressed)

            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1

            # Compute mean intensity of unscaled differences
            ela_np_unscaled = np.array(ela_image)
            mean_intensity = float(np.mean(ela_np_unscaled))

            scale = 255.0 / max_diff
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

            # Score based on ELA (unscaled mean intensity)
            if mean_intensity > 12.0:
                self.score += 5
                interpretation = "High compression inconsistency — strong indicator of tampering"
                self.reasons.append(f"ELA: High inconsistency (mean={mean_intensity:.1f}) (+5)")
            elif mean_intensity > 6.0:
                self.score += 3
                interpretation = "Moderate compression inconsistency — possible editing detected"
                self.reasons.append(f"ELA: Moderate inconsistency (mean={mean_intensity:.1f}) (+3)")
            else:
                interpretation = "Uniform compression pattern — low suspicion of tampering"
                self.reasons.append(f"ELA: Uniform pattern (mean={mean_intensity:.1f})")

            # Save ELA image to temp and convert to base64
            ela_output_path = self.image_path + "_ela_output.jpg"
            ela_image.save(ela_output_path)

            with open(ela_output_path, "rb") as f:
                ela_base64 = base64.b64encode(f.read()).decode("utf-8")

            ela_result = {
                "performed": True,
                "meanIntensity": round(mean_intensity, 2),
                "elaImage": f"data:image/jpeg;base64,{ela_base64}",
                "interpretation": interpretation
            }

            # Clean up temp files
            for p in [ela_temp, ela_output_path]:
                if os.path.exists(p):
                    os.unlink(p)

        except Exception as e:
            print(f"ELA error: {e}")
            import traceback
            traceback.print_exc()
            ela_result["interpretation"] = f"ELA analysis failed: {str(e)}"

        return ela_result

    # ---- PRNU NOISE RESIDUAL ANALYSIS ----
    def perform_prnu_analysis(self, block_size=64):
        prnu_result = {
            "performed": False,
            "noiseMap": None,
            "blockVariances": [],
            "uniformityScore": 0,
            "suspiciousBlocks": 0,
            "totalBlocks": 0,
            "interpretation": ""
        }

        try:
            import pywt
            from scipy.ndimage import uniform_filter

            img = cv2.imread(self.image_path)
            if img is None:
                prnu_result["interpretation"] = "Could not load image for PRNU analysis"
                return prnu_result

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

            # --- Wavelet-based noise extraction (inspired by polimi-ispl/prnu-python) ---
            levels = 4
            sigma = 5.0
            noise_var = sigma ** 2

            # Process grayscale channel
            wlet = pywt.wavedec2(gray, 'db4', level=levels)

            # Wiener adaptive filter on detail coefficients
            for level_idx in range(len(wlet) - 1):
                detail_level = wlet[level_idx + 1]
                filtered = []
                for coeff in detail_level:
                    # Adaptive Wiener filter
                    energy = coeff ** 2
                    avg_energy = uniform_filter(energy, size=3, mode='constant')
                    threshold_val = np.maximum(avg_energy - noise_var, 0)
                    filtered_coeff = coeff * noise_var / (threshold_val + noise_var)
                    filtered.append(filtered_coeff)
                wlet[level_idx + 1] = tuple(filtered)

            # Zero out approximation coefficients
            wlet[0] = np.zeros_like(wlet[0])

            # Reconstruct noise residual
            noise_residual = pywt.waverec2(wlet, 'db4')
            noise_residual = noise_residual[:gray.shape[0], :gray.shape[1]]

            # --- Block-wise variance analysis ---
            h, w = noise_residual.shape
            block_h = h // block_size
            block_w = w // block_size

            if block_h < 2 or block_w < 2:
                # Image too small for block analysis
                block_size = min(h, w) // 4
                if block_size < 8:
                    prnu_result["interpretation"] = "Image too small for PRNU block analysis"
                    return prnu_result
                block_h = h // block_size
                block_w = w // block_size

            block_variances = np.zeros((block_h, block_w))
            for i in range(block_h):
                for j in range(block_w):
                    block = noise_residual[
                        i * block_size:(i + 1) * block_size,
                        j * block_size:(j + 1) * block_size
                    ]
                    block_variances[i, j] = np.var(block)

            # Statistical analysis of block variances
            mean_var = np.mean(block_variances)
            std_var = np.std(block_variances)
            total_blocks = block_h * block_w

            # Flag blocks with variance significantly different from mean
            # Add standard deviation floor (20% of mean variance) to avoid false positives in uniform images
            min_increment = 0.2 * mean_var if mean_var > 0 else 0.1
            threshold = mean_var + max(2.0 * std_var, min_increment)
            suspicious_mask = block_variances > threshold
            suspicious_count = int(np.sum(suspicious_mask))
            suspicious_ratio = suspicious_count / total_blocks if total_blocks > 0 else 0

            # Uniformity score (coefficient of variation)
            cv_score = float(std_var / mean_var) if mean_var > 0 else 0

            # Generate heatmap
            variance_norm = block_variances.copy()
            if variance_norm.max() > 0:
                variance_norm = (variance_norm / variance_norm.max() * 255).astype(np.uint8)
            else:
                variance_norm = variance_norm.astype(np.uint8)

            # Resize heatmap to original image size
            heatmap = cv2.resize(variance_norm, (w, h), interpolation=cv2.INTER_NEAREST)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Blend with original image
            original_bgr = cv2.imread(self.image_path)
            original_resized = cv2.resize(original_bgr, (w, h))
            blended = cv2.addWeighted(original_resized, 0.5, heatmap_color, 0.5, 0)

            # Encode to base64
            _, buffer = cv2.imencode('.jpg', blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
            noise_map_b64 = base64.b64encode(buffer).decode('utf-8')

            # Scoring
            if suspicious_ratio > 0.25:
                self.score += 5
                interpretation = "Highly non-uniform noise pattern — strong indicator of splicing or compositing"
                self.reasons.append(f"PRNU: {suspicious_count}/{total_blocks} blocks anomalous ({suspicious_ratio:.0%}) (+5)")
            elif suspicious_ratio > 0.10:
                self.score += 3
                interpretation = "Some non-uniform noise blocks detected — possible local editing"
                self.reasons.append(f"PRNU: {suspicious_count}/{total_blocks} blocks anomalous ({suspicious_ratio:.0%}) (+3)")
            elif cv_score > 1.0:
                self.score += 1
                interpretation = "Slightly irregular noise distribution — minor anomalies detected"
                self.reasons.append(f"PRNU: Noise CV={cv_score:.2f} (+1)")
            else:
                interpretation = "Uniform noise pattern — consistent with an untampered image"
                self.reasons.append(f"PRNU: Uniform pattern (CV={cv_score:.2f})")

            prnu_result = {
                "performed": True,
                "noiseMap": f"data:image/jpeg;base64,{noise_map_b64}",
                "uniformityScore": round(cv_score, 3),
                "suspiciousBlocks": suspicious_count,
                "totalBlocks": total_blocks,
                "suspiciousRatio": round(suspicious_ratio, 4),
                "meanVariance": round(float(mean_var), 4),
                "interpretation": interpretation
            }

        except ImportError:
            prnu_result["interpretation"] = "PRNU analysis requires PyWavelets (pywt). Install with: pip install PyWavelets"
        except Exception as e:
            print(f"PRNU error: {e}")
            import traceback
            traceback.print_exc()
            prnu_result["interpretation"] = f"PRNU analysis failed: {str(e)}"

        return prnu_result

    # ---- FULL ANALYSIS ----
    def run(self):
        metadata, source = self.extract_metadata()
        has_metadata = len(metadata) > 0

        # Metadata analysis
        if has_metadata:
            metadata_flags = self.analyze_metadata(metadata)
        else:
            metadata_flags = [{"text": "No metadata found — may have been stripped", "severity": "info", "points": 0}]

        # ELA analysis
        ela_result = self.perform_ela()

        # PRNU noise analysis
        prnu_result = self.perform_prnu_analysis()

        # Compute verdict
        if self.score <= 3:
            verdict = "Likely Original"
            risk = "LOW"
        elif self.score <= 7:
            verdict = "Possibly Edited"
            risk = "MEDIUM"
        elif self.score <= 12:
            verdict = "Highly Suspicious"
            risk = "HIGH"
        else:
            verdict = "Very High Tampering Probability"
            risk = "CRITICAL"

        return {
            "metadata": metadata,
            "metadataSource": source,
            "hasMetadata": has_metadata,
            "metadataFlags": metadata_flags,
            "ela": ela_result,
            "prnu": prnu_result,
            "score": self.score,
            "maxScore": 24,
            "risk": risk,
            "verdict": verdict,
            "reasons": self.reasons
        }
