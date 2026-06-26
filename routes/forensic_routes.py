import os
import tempfile
import base64
import logging
import requests
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status, Request, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

logger = logging.getLogger("forensic_routes")

router = APIRouter(tags=["Forensics & Tampering"])

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(x_api_key: Optional[str] = Security(api_key_header)):
    """Verifies the EviCheck API key header."""
    expected_key = os.getenv("EVI_CHECK_API_KEY", "evi-check-secure-key-2026")
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing API Key"
        )
    return x_api_key

# Request model for metadata ELA base64 option
class MetadataAnalyzeRequest(BaseModel):
    image: str  # Base64 string

@router.post("/detect")
async def detect_tampering_endpoint(
    image: UploadFile = File(..., description="Image to analyze for tampering"),
    api_key: str = Depends(verify_api_key)
):
    """Detect image tampering using the ELA / PRNU model."""
    try:
        from image_detector import detect_tampering
    except ImportError:
        raise HTTPException(status_code=503, detail="Tampering detection model is not available on this server.")

    # Save uploaded file temporarily
    ext = os.path.splitext(image.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        contents = await image.read()
        tmp_file.write(contents)
        tmp_path = tmp_file.name

    try:
        result = detect_tampering(tmp_path)
        return result
    except Exception as e:
        logger.error(f"Error in tampering detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


@router.post("/weapon/detect")
async def weapon_detect_endpoint(
    image: UploadFile = File(..., description="Image to scan for weapons"),
    api_key: str = Depends(verify_api_key)
):
    """Detect weapons in an uploaded image using Cloud Verification REST API (SightEngine)."""
    api_user = os.getenv('IMAGE_DETECTION_API_USER')
    api_secret = os.getenv('IMAGE_DETECTION_API_SECRET')

    if not api_user or not api_secret:
        raise HTTPException(
            status_code=500,
            detail="Verification API credentials (IMAGE_DETECTION_API_USER / IMAGE_DETECTION_API_SECRET) are not configured."
        )

    try:
        image_data = await image.read()
        
        # Prepare multipart payloads
        files = {'media': ('image.jpg', image_data, image.content_type or 'image/jpeg')}
        data = {
            'models': 'weapon',
            'api_user': api_user,
            'api_secret': api_secret
        }

        # Query verification check API
        response = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=data)

        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Verification API returned status code {response.status_code}: {response.text}"
            )

        res_data = response.json()
        if res_data.get('status') != 'success':
            error_msg = res_data.get('error', {}).get('message', 'Verification request failed')
            raise HTTPException(
                status_code=400,
                detail=f"Verification API Error: {error_msg}"
            )

        # Extract classification scores
        weapon_info = res_data.get('weapon', {})
        classes = weapon_info.get('classes', {})

        detections = []
        classes_found = set()
        anomalies = []

        class_mapping = {
            'firearm': 'Pistol',
            'knife': 'Knife',
            'firearm_gesture': 'Unknown',
            'firearm_toy': 'Unknown'
        }

        DETECTION_THRESHOLD = 0.2

        for se_class, score in classes.items():
            if score >= DETECTION_THRESHOLD:
                frontend_class = class_mapping.get(se_class, 'Unknown')
                conf_percent = round(score * 100, 2)

                detections.append({
                    'class': frontend_class,
                    'confidence': conf_percent,
                    'bbox': {'x': 50.0, 'y': 50.0, 'width': 100.0, 'height': 100.0}
                })

                if frontend_class != 'Unknown':
                    classes_found.add(frontend_class)
                else:
                    classes_found.add(se_class.replace('_', ' ').title())

                if score >= 0.8:
                    anomalies.append(f"{frontend_class} ({se_class}) detected with {conf_percent:.1f}% confidence — high certainty threat")
                else:
                    anomalies.append(f"{frontend_class} ({se_class}) detected with {conf_percent:.1f}% confidence")

        weapons_found = any(classes.get(w, 0.0) >= DETECTION_THRESHOLD for w in ['firearm', 'knife'])

        raw_result = {
            'model': 'Cloud Verification API (weapon)',
            'total_detections': len(detections),
            'classes': list(classes_found),
            'confidence_threshold': DETECTION_THRESHOLD,
            'raw_api_response': res_data
        }

        return {
            'success': True,
            'weaponsFound': weapons_found,
            'weaponsDetected': list(classes_found),
            'detections': detections,
            'anomalies': anomalies,
            'totalDetections': len(detections),
            'rawResult': raw_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in weapon detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metadata/analyze")
async def analyze_metadata_endpoint(
    image: Optional[UploadFile] = File(None, description="Image to analyze"),
    payload: Optional[MetadataAnalyzeRequest] = None,
    api_key: str = Depends(verify_api_key)
):
    """Hybrid forensic analysis: metadata extraction + ELA (Error Level Analysis) + PRNU noise check."""
    try:
        from app import HybridForensicAnalyzer
    except ImportError:
        raise HTTPException(status_code=503, detail="Hybrid Forensic Analyzer is not available on this server.")

    temp_path = None
    try:
        # Option A: File Upload
        if image is not None:
            ext = os.path.splitext(image.filename or "")[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                contents = await image.read()
                tmp_file.write(contents)
                temp_path = tmp_file.name
                
        # Option B: Base64 string in JSON payload
        elif payload is not None and payload.image:
            image_data = payload.image
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            image_bytes = base64.b64decode(image_data)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_bytes)
                temp_path = tmp_file.name
        else:
            raise HTTPException(status_code=400, detail="No image file or base64 payload provided.")

        analyzer = HybridForensicAnalyzer(temp_path)
        result = analyzer.run()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in forensic metadata analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
