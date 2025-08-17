import os
import shutil
import uuid
import json
from pathlib import Path
import asyncio
import logging
import re
from fastapi import FastAPI, Request, HTTPException, Response
from dotenv import load_dotenv
import uvicorn
from sentence_transformers import SentenceTransformer

# Import custom modules
from services.llm_planner import generate_script
from agent.executor import run_script_locally

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initial Setup ---
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    logger.error("FATAL ERROR: GOOGLE_API_KEY environment variable is not set.")
    exit(1)

# Initialize the FastAPI app
app = FastAPI(
    title="Data Analyst Agent API (Serverless)",
    description="An API that uses LLMs to source, prepare, analyze, and visualize data.",
    version="1.2.0"
)

# --- Model Caching on Startup ---
@app.on_event("startup")
async def startup_event():
    """
    On application startup, download and cache the sentence transformer model.
    This runs once when the serverless function starts (on a cold start).
    """
    model_name = 'all-MiniLM-L6-v2'
    # Vercel provides a writable /tmp directory for caching
    cache_dir = "/tmp/sentence_transformers_cache"
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir

    logger.info(f"Attempting to download and cache model: {model_name} to {cache_dir}")
    try:
        # This line triggers the download and saves the model to the specified cache directory
        SentenceTransformer(model_name)
        logger.info(f"Successfully cached model: {model_name}")
    except Exception as e:
        # Log the error but don't prevent the app from starting.
        # The generated script might still fail later if it needs the model.
        logger.error(f"Failed to download or cache model {model_name}. Error: {e}")

# --- API Endpoint ---
@app.post("/api/")
async def analyze_data(request: Request):
    """
    Receives a data analysis task, generates a Python script, executes it locally, and returns the result.
    """
    temp_dir = Path("/tmp") / str(uuid.uuid4())
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        async with asyncio.timeout(180):
            form_data = await request.form()

            if "questions.txt" not in form_data:
                raise HTTPException(status_code=400, detail="A 'questions.txt' file is required.")

            # Read the content of the uploaded file
            questions_file = form_data["questions.txt"]
            full_task_description = (await questions_file.read()).decode()
            
            uploaded_file_names = []
            for name, file in form_data.items():
                if not file.filename: continue
                
                # Save uploaded files to the temp directory
                file_path = temp_dir / Path(file.filename).name
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)

                # Keep track of data file names (exclude the questions file)
                if name != "questions.txt":
                    uploaded_file_names.append(file.filename)
            
            if not full_task_description:
                 raise HTTPException(status_code=400, detail="questions.txt is empty.")

            script_code = generate_script(full_task_description, uploaded_file_names)
            if not script_code:
                raise HTTPException(status_code=500, detail="Failed to generate analysis script.")

            task_payload = {"task": full_task_description}
            task_json_string = json.dumps(task_payload)

            script_env = {
                "USER_TASK_JSON": task_json_string,
                "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
            }

            output, error = run_script_locally(script_code, temp_dir, script_env)

            if error:
                raise HTTPException(status_code=500, detail=f"Script execution failed: {error}")

            try:
                json.loads(output)
            except (json.JSONDecodeError, TypeError):
                raise HTTPException(status_code=500, detail=f"Script produced invalid JSON output: {output}")

            return Response(content=output, media_type="application/json")

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out.")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

# --- Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"status": "Data Analyst Agent is running"}

# --- Uvicorn Runner for local testing ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
