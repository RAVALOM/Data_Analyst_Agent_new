# In agent/executor.py

import subprocess
import os
import uuid
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
# Set a timeout slightly less than the serverless function's maximum timeout
TIMEOUT_SECONDS = 170

def run_script_locally(script_code: str, temp_dir: Path, env_vars: dict) -> tuple[str | None, str | None]:
    """
    Executes a Python script locally using a subprocess.

    This method is less secure than sandboxing with Docker because it runs
    LLM-generated code directly on the execution environment. It's a trade-off
    for compatibility with serverless platforms.
    """
    script_filename = f"script_{uuid.uuid4()}.py"
    # Ensure the script is written to a writable directory, like the one provided
    script_path = temp_dir / script_filename
    script_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Write the generated Python code to a temporary file
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_code)
        logger.debug(f"Wrote script to {script_path}")
    except Exception as e:
        logger.error(f"Failed to write script to {script_path}: {e}")
        return None, f"Failed to write script file: {e}"

    # Prepare the environment for the subprocess
    # It inherits the parent environment and adds specific keys for the script
    execution_env = os.environ.copy()
    execution_env.update(env_vars)

    try:
        logger.info(f"Running script '{script_filename}' locally via subprocess...")
        # Execute the script using 'python'
        process = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,  # Capture stdout and stderr
            text=True,            # Decode output as text
            timeout=TIMEOUT_SECONDS,
            env=execution_env,
            cwd=script_path.parent # Set the working directory to where the script is
        )

        # Check the script's exit code
        if process.returncode == 0:
            logger.info("Script executed successfully.")
            return process.stdout.strip(), None # Return the standard output
        else:
            # If the script failed, return the standard error
            error_msg = f"Script execution failed with return code {process.returncode}:\n{process.stderr.strip()}"
            logger.error(error_msg)
            return None, error_msg

    except subprocess.TimeoutExpired:
        logger.error(f"Script execution timed out after {TIMEOUT_SECONDS} seconds")
        return None, f"Script execution timed out after {TIMEOUT_SECONDS} seconds"
    except Exception as e:
        logger.error(f"An unexpected error occurred during subprocess execution: {e}")
        return None, f"An unexpected error occurred: {e}"
