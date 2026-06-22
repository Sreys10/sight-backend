"""
Image Detection Application
Detects deepfake, AI-generated content, image quality, and scammer detection
"""

import requests
import json
import os
import sys
from typing import Dict, Optional, Union
from pathlib import Path


class ImageDetector:
    """Main class for AI image detection"""
    
    def __init__(self, api_user: str, api_secret: str):
        """
        Initialize the detector with API credentials
        
        Args:
            api_user: API user ID
            api_secret: API secret key
        """
        self.api_user = api_user
        self.api_secret = api_secret
        self.base_url = "https://api.sightengine.com/1.0/check.json"
    
    def check_deepfake(self, image_source: Union[str, Path]) -> Dict:
        """
        Check if image is a deepfake
        
        Args:
            image_source: Image URL or file path
            
        Returns:
            Dictionary with deepfake detection results
        """
        return self._make_request(image_source, "deepfake")
    
    def check_ai_generated(self, image_source: Union[str, Path]) -> Dict:
        """
        Check if image is AI-generated
        
        Args:
            image_source: Image URL or file path
            
        Returns:
            Dictionary with AI-generated detection results
        """
        return self._make_request(image_source, "genai")
    
    def check_quality(self, image_source: Union[str, Path]) -> Dict:
        """
        Check image quality
        
        Args:
            image_source: Image URL or file path
            
        Returns:
            Dictionary with quality detection results
        """
        return self._make_request(image_source, "quality")
    
    def check_scammer(self, image_source: Union[str, Path]) -> Dict:
        """
        Check for scammer detection
        
        Args:
            image_source: Image URL or file path
            
        Returns:
            Dictionary with scammer detection results
        """
        return self._make_request(image_source, "scam")
    
    def _is_url(self, source: Union[str, Path]) -> bool:
        """Check if source is a URL"""
        source_str = str(source)
        return source_str.startswith(('http://', 'https://'))
    
    def _make_request(self, image_source: Union[str, Path], model: str) -> Dict:
        """
        Make API request for image detection
        
        Args:
            image_source: Image URL or file path
            model: Detection model to use (deepfake, genai, quality, scam)
            
        Returns:
            API response as dictionary
        """
        try:
            if self._is_url(image_source):
                # Use GET request for URL
                params = {
                    'models': model,
                    'api_user': self.api_user,
                    'api_secret': self.api_secret,
                    'url': str(image_source)
                }
                response = requests.get(self.base_url, params=params, timeout=30)
            else:
                # Use POST request for file upload
                with open(image_source, 'rb') as f:
                    files = {'media': f}
                    data = {
                        'models': model,
                        'api_user': self.api_user,
                        'api_secret': self.api_secret
                    }
                    response = requests.post(self.base_url, files=files, data=data, timeout=30)
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'error': str(e),
                'model': model
            }
    
    def analyze_image(self, image_source: Union[str, Path]) -> Dict:
        """
        Perform comprehensive image analysis using a single combined API call
        
        Args:
            image_source: Image URL or file path
            
        Returns:
            Dictionary with all detection results mapped for backward compatibility
        """
        print(f"Analyzing image: {image_source}", file=sys.stderr)
        
        # Make a single combined request to API for all models
        print("Running combined analysis...", file=sys.stderr)
        combined_response = self._make_request(image_source, "deepfake,genai,quality,scam,weapon")
        
        results = {
            'image_source': str(image_source),
            'status': 'success'
        }
        
        # Reconstruct individual check responses for perfect backward compatibility
        if combined_response.get('status') == 'success':
            # 1. Deepfake
            results['deepfake'] = {
                'status': 'success',
                'type': {
                    'deepfake': combined_response.get('type', {}).get('deepfake', 0.0)
                }
            }
            # 2. AI-Generated
            results['ai_generated'] = {
                'status': 'success',
                'type': {
                    'ai_generated': combined_response.get('type', {}).get('ai_generated', 0.0)
                }
            }
            # 3. Quality
            results['quality'] = {
                'status': 'success',
                'quality': {
                    'score': combined_response.get('quality', {}).get('score', 0.0)
                },
                'media': combined_response.get('media', {})
            }
            # 4. Scammer (scam model returns 'scam' object in root)
            results['scammer'] = {
                'status': 'success',
                'scam': {
                    'prob': combined_response.get('scam', {}).get('prob', 0.0)
                },
                'faces': combined_response.get('scam', {}).get('faces', [])
            }
            
            # 5. Weapon Detection
            weapon_info = combined_response.get('weapon', {})
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
                        'bbox': {
                            'x': 50.0,
                            'y': 50.0,
                            'width': 100.0,
                            'height': 100.0
                        }
                    })
                    
                    if frontend_class != 'Unknown':
                        classes_found.add(frontend_class)
                    else:
                        readable_name = se_class.replace('_', ' ').title()
                        classes_found.add(readable_name)
                    
                    if score >= 0.8:
                        anomalies.append(f"{frontend_class} ({se_class}) detected with {conf_percent:.1f}% confidence — high certainty threat")
                    else:
                        anomalies.append(f"{frontend_class} ({se_class}) detected with {conf_percent:.1f}% confidence")
            
            weapons_found = any(classes.get(w, 0.0) >= DETECTION_THRESHOLD for w in ['firearm', 'knife'])
            
            results['weapon_detection'] = {
                'weaponsFound': weapons_found,
                'weaponsDetected': list(classes_found),
                'detections': detections,
                'anomalies': anomalies,
                'totalDetections': len(detections),
                'rawResult': {
                    'model': 'Cloud Verification Engine (weapon)',
                    'total_detections': len(detections),
                    'classes': list(classes_found),
                    'confidence_threshold': DETECTION_THRESHOLD,
                    'raw_response': combined_response
                }
            }
        else:
            # Fallback to empty default/error responses
            err_msg = combined_response.get('error', 'API Request Failed')
            results['status'] = 'error'
            results['error'] = err_msg
            results['deepfake'] = {'status': 'error', 'error': err_msg}
            results['ai_generated'] = {'status': 'error', 'error': err_msg}
            results['quality'] = {'status': 'error', 'error': err_msg}
            results['scammer'] = {'status': 'error', 'error': err_msg}
            results['weapon_detection'] = {
                'weaponsFound': False,
                'weaponsDetected': [],
                'detections': [],
                'anomalies': [f"Error running weapon detection: {err_msg}"],
                'totalDetections': 0,
                'rawResult': None
            }
            
        print("Analysis complete", file=sys.stderr)
        return results

    
    def generate_report(self, results: Dict) -> str:
        """
        Generate a human-readable report from detection results
        
        Args:
            results: Dictionary containing all detection results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("\n" + "=" * 60)
        report.append("IMAGE DETECTION REPORT")
        report.append("=" * 60)
        report.append(f"\nImage Source: {results['image_source']}\n")
        
        # Deepfake results
        if 'deepfake' in results:
            deepfake_data = results['deepfake']
            if 'type' in deepfake_data and 'deepfake' in deepfake_data['type']:
                prob = deepfake_data['type']['deepfake']
                report.append(f"Deepfake Probability: {prob * 100:.2f}%")
                if prob > 0.5:
                    report.append("⚠️  WARNING: High deepfake probability detected!")
        
        # AI-generated results
        if 'ai_generated' in results:
            ai_data = results['ai_generated']
            if 'type' in ai_data and 'ai_generated' in ai_data['type']:
                prob = ai_data['type']['ai_generated']
                report.append(f"AI-Generated Probability: {prob * 100:.2f}%")
                if prob > 0.5:
                    report.append("⚠️  WARNING: AI-generated content detected!")
        
        # Quality results
        if 'quality' in results:
            quality_data = results['quality']
            if 'quality' in quality_data and 'score' in quality_data['quality']:
                score = quality_data['quality']['score']
                report.append(f"Image Quality Score: {score * 100:.2f}%")
        
        # Scammer results
        if 'scammer' in results:
            scammer_data = results['scammer']
            if 'scam' in scammer_data and 'prob' in scammer_data['scam']:
                prob = scammer_data['scam']['prob']
                report.append(f"Scammer Detection Probability: {prob * 100:.2f}%")
                if prob > 0.5:
                    report.append("⚠️  WARNING: Scammer indicators detected!")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)


def detect_tampering(image_path):
    # API credentials
    API_USER = os.getenv('IMAGE_DETECTION_API_USER', '1969601374')
    API_SECRET = os.getenv('IMAGE_DETECTION_API_SECRET', 'uk7Rwq4Bh8kURjU3WauN3J7nhtGgjSQz')

def detect_tampering(image_path):
    # API credentials
    API_USER = os.getenv('IMAGE_DETECTION_API_USER', '1969601374')
    API_SECRET = os.getenv('IMAGE_DETECTION_API_SECRET', 'uk7Rwq4Bh8kURjU3WauN3J7nhtGgjSQz')

    # Initialize detector
    detector = ImageDetector(API_USER, API_SECRET)
    return detector.analyze_image(image_path)


def main():
    """Main function to run the application"""
    # API credentials
    API_USER = os.getenv('IMAGE_DETECTION_API_USER', '1969601374')
    API_SECRET = os.getenv('IMAGE_DETECTION_API_SECRET', 'uk7Rwq4Bh8kURjU3WauN3J7nhtGgjSQz')

    # Initialize detector
    detector = ImageDetector(API_USER, API_SECRET)
    
    # Example usage
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Analyzing image: {image_path}")
        results = detector.analyze_image(image_path)
        print(detector.generate_report(results))
        print("\nFull JSON Results:")
        print(json.dumps(results, indent=2))
    else:
        print("Usage: python image_detector.py <image_path_or_url>")
        sys.exit(1)


if __name__ == '__main__':
    main()





