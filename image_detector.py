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
        Perform comprehensive image analysis
        
        Args:
            image_source: Image URL or file path
            
        Returns:
            Dictionary with all detection results
        """
        print(f"Analyzing image: {image_source}", file=sys.stderr)
        
        results = {
            'image_source': str(image_source),
            'status': 'success'
        }
        
        # Run all checks
        print("Checking for deepfake...", file=sys.stderr)
        results['deepfake'] = self.check_deepfake(image_source)
        
        print("Checking for AI-generated content...", file=sys.stderr)
        results['ai_generated'] = self.check_ai_generated(image_source)
        
        print("Checking image quality...", file=sys.stderr)
        results['quality'] = self.check_quality(image_source)
        
        print("Checking for scammer detection...", file=sys.stderr)
        results['scammer'] = self.check_scammer(image_source)
        
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

