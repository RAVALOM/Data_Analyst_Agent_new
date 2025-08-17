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

# Import the new local executor function
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
    version="1.1.0"
)

# --- API Endpoint ---
@app.post("/api/")
async def analyze_data(request: Request):
    """
    Receives a data analysis task, generates a Python script, executes it locally, and returns the result.
    """
    # Use /tmp for Vercel's writable directory, not a local 'temp' folder
    temp_dir = Path("/tmp") / str(uuid.uuid4())
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Set a timeout for the entire request
        async with asyncio.timeout(180):
            form_data = await request.form()

            if "questions.txt" not in form_data:
                raise HTTPException(
                    status_code=400,
                    detail="A 'questions.txt' file is required in the form data."
                )

            full_task_description = ""
            uploaded_file_names = []

            # Process uploaded files
            for name, file in form_data.items():
                if not file.filename:
                    continue

                file_path = temp_dir / Path(file.filename).name
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)

                if name == "questions.txt":
                    full_task_description = file_path.read_text()

                uploaded_file_names.append(file.filename)

            if not full_task_description:
                 raise HTTPException(status_code=400, detail="questions.txt is empty or was not provided.")

            logger.info(f"Uploaded files: {uploaded_file_names}")

            # Extract a JSON block if it exists to create a more focused task
            json_match = re.search(r'\{[\s\S]*\}', full_task_description)
            task_description = full_task_description

            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(0))
                    task_description = json.dumps(parsed_json, indent=2)
                    logger.info("Successfully parsed JSON block. Using it as the task.")
                except json.JSONDecodeError:
                    logger.warning("Found a JSON-like block, but it failed to parse. Using full description.")

            # Generate the script
            script_code = generate_script(task_description, uploaded_file_names)
            if not script_code:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate analysis script from LLM."
                )

            logger.debug(f"Generated script snippet: {script_code[:200]}...")

            # Prepare the environment variables for the script
            task_payload = {"task": task_description}
            task_json_string = json.dumps(task_payload)

            script_env = {
                "USER_TASK_JSON": task_json_string,
                "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
            }

            # Call the new local executor function instead of the Docker one
            output, error = run_script_locally(script_code, temp_dir, script_env)

            if error:
                raise HTTPException(status_code=500, detail=f"Script execution failed: {error}")

            # Validate that the script's output is valid JSON
            try:
                json.loads(output)
            except (json.JSONDecodeError, TypeError) as e:
                raise HTTPException(status_code=500, detail=f"Script produced invalid JSON output:\n{output}\nError: {str(e)}")

            return Response(content=output, media_type="application/json")

    except asyncio.TimeoutError:
        logger.error("Request timed out after 180 seconds")
        raise HTTPException(status_code=504, detail="Request timed out after 180 seconds")
    finally:
        # Clean up the temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

# --- Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"status": "Data Analyst Agent is running"}

# --- Uvicorn Runner for local testing ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
