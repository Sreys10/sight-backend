import os
from forensic_analyzer import HybridForensicAnalyzer

def main():
    img_path = "../ChatGPT Image Jun 17, 2026, 11_21_21 PM.png"
    abs_path = os.path.abspath(img_path)
    print(f"Testing analyzer on: {abs_path}")
    if not os.path.exists(abs_path):
        print("Test image not found at path.")
        return
        
    analyzer = HybridForensicAnalyzer(abs_path)
    res = analyzer.run()
    
    print("\n--- RESULTS ---")
    print(f"Score: {res['score']} / {res['maxScore']}")
    print(f"Risk: {res['risk']}")
    print(f"Verdict: {res['verdict']}")
    print("\nReasons:")
    for reason in res['reasons']:
        print(f" - {reason}")
    print("\nMetadata Flags:")
    for flag in res['metadataFlags']:
        print(f" - {flag['text']} (Points: {flag['points']}, Severity: {flag['severity']})")

if __name__ == "__main__":
    main()
