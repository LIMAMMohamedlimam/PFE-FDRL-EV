"""
time_logger.py
==============
Centralized utility to track and safely log execution times of all simulations
into a thread-safe JSON registry (execution_times.json).
"""

import json
import os
import fcntl
from datetime import datetime

def log_execution_time(run_id: str, run_info: dict, execution_time_s: float):
    """
    Safely append execution timing and run config data to results/execution_times.json.
    Uses an exclusive file lock to prevent corruption from parallel ComparisonPipeline runs.
    """
    json_path = 'results/execution_times.json'
    os.makedirs('results', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    entry = run_info.copy()
    entry["execution_timestamp"] = timestamp
    entry["execution_time_s"] = execution_time_s
    entry["execution_time_formatted"] = f"{execution_time_s/60:.2f} mins"
    
    with open(json_path, 'a+') as f:
        # Acquire an exclusive lock (blocks if another process is writing)
        fcntl.flock(f, fcntl.LOCK_EX)
        
        # Read existing data
        f.seek(0)
        try:
            data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            data = {}
            
        # Update data
        data[run_id] = entry
        
        # Write back
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=4)
        
        # Release lock
        fcntl.flock(f, fcntl.LOCK_UN)
        
    print(f"-> Execution time ({execution_time_s/60:.2f} mins) logged to {json_path}")
