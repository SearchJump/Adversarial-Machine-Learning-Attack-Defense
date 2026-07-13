import os
import sys
import subprocess

def run_command(cmd, cwd=None):
    """Safely runs a python command and streams output in real-time."""
    print(f"\nExecuting: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        rc = process.poll()
        return rc == 0
    except Exception as e:
        print(f"Execution failed: {e}")
        return False

def run_verification_pipeline():
    print("==================================================")
    print("         AI RED TEAMING PORTFOLIO RUNNER          ")
    print("==================================================")
    
    # Check for core assets
    if not os.path.exists("data/labels.txt") or not os.path.exists("src/test_image.png"):
        print("Missing base assets. Running Setup_Assets.py first...")
        run_command([sys.executable, "src/Setup_Assets.py"])
        
    print("\nSelect an action:")
    print(" [1] Run all Attacks to generate artifacts")
    print(" [2] Run all Defenses to verify security controls")
    print(" [3] Run complete end-to-end Pipeline (Attacks + Defenses)")
    print(" [4] Verify generated portfolio artifacts inside 'my_assessment/'")
    print(" [q] Exit")
    
    choice = input("\nEnter choice: ").strip().lower()
    
    if choice == 'q':
        return

    # Execute scripts within 'src' to respect their internal paths
    cwd_src = "src"
    
    if choice in ['1', '3']:
        print("\n>>> Running Attacks to generate artifacts...")
        run_command([sys.executable, "Evasion.py"], cwd=cwd_src)
        run_command([sys.executable, "Poisoning.py"], cwd=cwd_src)
        run_command([sys.executable, "LLMs.py"], cwd=cwd_src)
        
        # Check if pre-trained weight file for inversion is present
        if os.path.exists("src/other_mnist.pt") or os.path.exists("other_mnist.pt"):
            run_command([sys.executable, "Inversion.py"], cwd=cwd_src)
        else:
            print("\nNote: Inversion.py skipped (missing external 'other_mnist.pt' weights)")
            
        run_command([sys.executable, "Assessment.py"], cwd=cwd_src)
        run_command([sys.executable, "Extraction.py"], cwd=cwd_src)

    if choice in ['2', '3']:
        print("\n>>> Evaluating Defenses against attacks...")
        run_command([sys.executable, "Evasion_Defense.py"], cwd=cwd_src)
        run_command([sys.executable, "Poisoning_Defense.py"], cwd=cwd_src)
        run_command([sys.executable, "LLMs_Defense.py"], cwd=cwd_src)

    if choice in ['1', '2', '3', '4']:
        print("\n" + "="*60)
        print("         PORTFOLIO ARTIFACT VERIFICATION STATUS")
        print("="*60)
        
        expected_artifacts = {
            "my_assessment/assessments_submission.json": "Anchor Explanation Keywords (Assessment.py)",
            "my_assessment/poisoning_dataset.npz": "Poisoned CIFAR-10 Dataset (Poisoning.py)",
            "my_assessment/llm_submission.txt": "Prompt Injection Payload (LLMs.py)",
            "my_assessment/extraction_counts.json": "Extraction Query Stats (Extraction.py)",
            "my_assessment/extraction_proxy_scores.npy": "Proxy Model Copycat Scores (Extraction.py)",
            "my_assessment/inversion_submission.npy": "Reconstructed Feature Tensor (Inversion.py)",
            "src/evasion_image.png": "Adversarial Image Output (Evasion.py)"
        }
        
        all_ok = True
        for path, description in expected_artifacts.items():
            exists = os.path.exists(path)
            status = "[FOUND]" if exists else "[MISSING]"
            print(f" {status:<10} {path:<45} : {description}")
            if not exists and "inversion" not in path:
                all_ok = False
                
        print("="*60)
        if all_ok:
            print("Success: All core portfolio artifacts are ready for commit!")
            print("Your directory structure is cleanly configured.")
        else:
            print("Notice: Some artifacts are missing. Run option [1] or [3] to generate.")
        print("="*60 + "\n")

if __name__ == "__main__":
    run_verification_pipeline()
