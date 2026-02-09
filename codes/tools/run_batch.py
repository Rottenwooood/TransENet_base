#!/usr/bin/env python3
"""
Batch Training Runner (Refactored)
Executes experiments from a static JSON manifest.
Design Principle: "Dumb Runner, Smart Generator"
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class ExperimentRunner:
    def __init__(self, manifest_file: str):
        self.manifest_path = Path(manifest_file)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
            
        self.experiments = self.load_manifest()
        self.results_file = Path("../experiment/batch_results_log.csv")

    def load_manifest(self) -> List[Dict[str, Any]]:
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Manifest must be a JSON list of experiment objects.")
        return data

    def run_all(self, dry_run: bool = False, target_ids: List[str] = None):
        print(f"📋 Loaded {len(self.experiments)} experiments from {self.manifest_path}")
        
        for exp in self.experiments:
            exp_id = str(exp.get("id", "unknown"))
            
            # Filter logic
            if target_ids and exp_id not in target_ids:
                continue
                
            self.run_experiment(exp, dry_run)

    def run_experiment(self, exp: Dict[str, Any], dry_run: bool):
        exp_id = exp.get("id", "unknown")
        args = exp.get("args", {})
        
        print(f"\n{'='*60}")
        print(f"🧪 Experiment ID: {exp_id}")
        
        cmd = self.build_command(args)
        
        if dry_run:
            print(f"🔍 [DRY RUN] Command:\n{cmd}")
            return

        print(f"🚀 Executing...")
        start_time = time.time()
        
        try:
            # Execute training script
            # We use shell=True to easily handle the command string, ensure security in production if needed
            result = subprocess.run(cmd, shell=True, text=True)
            
            duration = time.time() - start_time
            status = "SUCCESS" if result.returncode == 0 else "FAILED"
            
            print(f"🏁 Finished: {status} ({duration:.1f}s)")
            self.log_result(exp_id, status, duration)
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            self.log_result(exp_id, f"ERROR: {str(e)}", 0)

    def build_command(self, args: Dict[str, Any]) -> str:
        # Robustly determine training script path
        # train_enhanced.py is in the parent directory of this script (codes/)
        current_dir = Path(__file__).parent.resolve()
        script_path = current_dir.parent / "train_enhanced.py"
        
        if not script_path.exists():
            print(f"⚠️ Warning: Training script not found at {script_path}")
            # Fallback for some environments
            script_path = Path("codes/train_enhanced.py")

        cmd_parts = ["python3", str(script_path)]
        
        for key, value in args.items():
            if value is None:
                continue
                
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{key}")
            else:
                cmd_parts.append(f"--{key}")
                # Handle lists like [2,2,2] -> "2,2,2" for command line
                if isinstance(value, list):
                    cmd_parts.append(','.join(map(str, value)))
                else:
                    cmd_parts.append(str(value))
                    
        return " ".join(cmd_parts)

    def log_result(self, exp_id: str, status: str, duration: float):
        timestamp = datetime.now().isoformat()
        log_entry = f"{timestamp},{exp_id},{status},{duration:.2f}\n"
        
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)

def main():
    parser = argparse.ArgumentParser(description="SymUNet Batch Runner (Manifest Based)")
    parser.add_argument("--manifest", type=str, required=True, help="Path to JSON manifest file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--ids", type=str, help="Comma-separated list of Experiment IDs to run (e.g., exp_001,exp_002)")
    
    args = parser.parse_args()
    
    target_ids = args.ids.split(',') if args.ids else None
    
    runner = ExperimentRunner(args.manifest)
    runner.run_all(dry_run=args.dry_run, target_ids=target_ids)

if __name__ == "__main__":
    main()

