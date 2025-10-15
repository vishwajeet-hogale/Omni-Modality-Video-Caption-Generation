#!/usr/bin/env python3
"""
Script to run temporal window evaluation with LoRA-enabled Swin model
"""
import subprocess
import sys
import os
from datetime import datetime

def run_temporal_evaluation():
    """Run the temporal window evaluation script"""
    print("🚀 Starting temporal window evaluation with LoRA-enabled Swin model")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check if the evaluation script exists
    eval_script = "evaluate_temporal_windows.py"
    if not os.path.exists(eval_script):
        print(f"❌ Error: {eval_script} not found!")
        return False
    
    try:
        # Run the evaluation script
        result = subprocess.run([
            sys.executable, eval_script
        ], check=True, capture_output=True, text=True)
        
        print("✅ Temporal evaluation completed successfully!")
        print("\n📊 Output:")
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️ Warnings/Errors:")
            print(result.stderr)
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running temporal evaluation: {e}")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function"""
    print("🔧 Quick Testing Temporal Window Evaluation")
    print("="*60)
    print("Configuration:")
    print("- Batch size: 2 (ultra-small to prevent overheating)")
    print("- Epochs: 2 (quick testing only)")
    print("- Learning rate: 3e-5 (current)")
    print("- LoRA: Auto-detect from config (will add _lora or _wo_lora suffix)")
    print("- Accelerator: Auto-detect (MPS/CPU for Mac)")
    print("- Temporal windows: 5, 16 frames")
    print("- Caption contexts: 5, 15")
    print("- Total configurations: 4 (ultra-reduced for quick testing)")
    print("- ⚠️  WARNING: This is for quick testing only!")
    print("="*60)
    
    success = run_temporal_evaluation()
    
    if success:
        print("\n🎉 Evaluation completed successfully!")
        print("📁 Check the outputs/ directory for results")
    else:
        print("\n💥 Evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
