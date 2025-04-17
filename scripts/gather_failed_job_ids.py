import os
import re

# === Configuration ===
LOG_DIR = os.path.expanduser("~/R-GIANT/logs")
OUTPUT_FILE = os.path.expanduser("~/R-GIANT/failed_jobs.txt")

# === Compile regex pattern ===
error_pattern = re.compile(r"\[ERROR\]\s+-\s+\[(\d{4})-(\d{4})\]")

# === Gather failed job IDs ===
failed_jobs = set()

for filename in os.listdir(LOG_DIR):
    if filename.endswith(".err"):
        filepath = os.path.join(LOG_DIR, filename)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = error_pattern.search(line)
                if match:
                    patient_id, session_id = match.groups()
                    failed_jobs.add(f"{patient_id}_{session_id}")

# === Save to output ===
with open(OUTPUT_FILE, "w") as out:
    for job_id in sorted(failed_jobs):
        out.write(job_id + "\n")

print(f"Found {len(failed_jobs)} failed jobs. Results saved to {OUTPUT_FILE}")