from flask import Flask, request, jsonify
import os
import subprocess
import re
from pathlib import Path
import requests

app = Flask(__name__)

def download_file(pc_url: str, box_id: str, dgx_id: str) -> Path:
    ml_root = Path(os.getcwd())
    # Ensure clean separation of inputs
    input_folder = ml_root / "data" / "input" 
    input_folder.mkdir(parents=True, exist_ok=True)
    
    filename = f"box-{box_id}-{dgx_id}.ply"
    full_path = input_folder / filename

    print(f"Downloading to {full_path}...")

    storage_api_key = os.getenv("STORAGE_API_KEY", "")
    headers = {"x-api-key": storage_api_key} if storage_api_key else {}

    try:
        with requests.get(pc_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(full_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return full_path
    except Exception as e:
        print(f"Download failed: {e}")
        raise e

def run_pipeline(box_id: str, dgx_id: str, input_ply_path: Path):
    # 1. Setup Environment Variables (Mirroring Bash)
    # Note: We use absolute paths for python to ensure we use the active conda env
    python_exec = "python" 
    
    # --- Bash Variable Alignment ---
    INPUT_PLY = str(input_ply_path)
    BOX_ID = box_id
    DGX_ID = dgx_id
    
    PROJECT_NAME = f"box-{BOX_ID}-{DGX_ID}-txt"
    # Path: data/PROJECT_NAME/PROJECT_NAME/PROJECT_NAME/Annotations/PROJECT_NAME.txt
    TXT_FILE_PATH = f"data/{PROJECT_NAME}/{PROJECT_NAME}/{PROJECT_NAME}/Annotations/{PROJECT_NAME}.txt"
    PREPROCESS_OUTPUT = f"data/custom_box/output-preprocess/box-{BOX_ID}-{DGX_ID}"
    EXP_NAME = "semseg_pt_v3m1_s3dis_2_custom_box_f"
    RESULT_NPY = f"exp/custom_box/{EXP_NAME}/result/{PROJECT_NAME}.npy"
    
    # --- Execution Steps (Mirroring Bash) ---
    commands = [
        # Step 1: Convert PLY to TXT
        f"{python_exec} ply_to_txt.py -n '{PROJECT_NAME}' -r '{INPUT_PLY}'",
        
        # Step 2: Add dummy color
        f"{python_exec} add_color.py -n 'box-{BOX_ID}-{DGX_ID}-color' -r '{TXT_FILE_PATH}'",
        
        # Step 3: Preprocess
        f"{python_exec} preprocess.py --dataset_root 'data/{PROJECT_NAME}' --output_root '{PREPROCESS_OUTPUT}'",
        
        # Step 4: Inference
        # Note: -s points to the preprocessed output
        f"sh scripts/pred.sh -g 1 -p {python_exec} -d custom_box -n '{EXP_NAME}' -w model_best -s 'output-preprocess/box-{BOX_ID}-{DGX_ID}/{PROJECT_NAME}'",
        
        # Step 5: Volume Estimation
        f"{python_exec} volume-est.py '{INPUT_PLY}' '{RESULT_NPY}'"
    ]

    cwd = os.getcwd()
    last_stdout = ""

    for i, cmd in enumerate(commands, 1):
        print(f"--- Executing Step {i} ---")
        print(f"Command: {cmd}")
        
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        
        if proc.returncode != 0:
            print(f"Error in step {i}: {proc.stderr}")
            return None, f"Step {i} failed: {proc.stderr}"
        
        last_stdout = proc.stdout
        # Print stdout for debugging logs in Docker
        print(proc.stdout)

    # Parse Final Output
    # Regex captures floating point numbers (e.g., 10 or 10.5)
    dimensions = re.search(r"Dimensions \(L×W×H\): (\d+(?:\.\d+)?) cm × (\d+(?:\.\d+)?) cm × (\d+(?:\.\d+)?) cm", last_stdout)
    volume = re.search(r"Volume: (\d+(?:\.\d+)?) cm³", last_stdout)

    if not (dimensions and volume):
        return None, "Pipeline finished but failed to parse dimensions/volume from output."

    return {
        "length": float(dimensions.group(1)),
        "width": float(dimensions.group(2)),
        "height": float(dimensions.group(3)),
        "volume": float(volume.group(1))
    }, None

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/infer")
def infer():
    try:
        body = request.get_json(force=True)
        box_id = str(body.get("box_id"))
        dgx_id = str(body.get("dgx_id"))
        pc_url = body.get("pcUrl")

        if not all([box_id, dgx_id, pc_url]):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        # 1. Download
        ply_path = download_file(pc_url, box_id, dgx_id)
        
        # 2. Run Pipeline
        result, err = run_pipeline(box_id, dgx_id, ply_path)
        
        if err:
            return jsonify({"status": "error", "message": err}), 500
            
        return jsonify(result)

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Development run
    app.run(host="0.0.0.0", port=5001, threaded=False)