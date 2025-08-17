# pre_cache_models.py
import os
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The model to be downloaded. This is a good, general-purpose model.
MODEL_NAME = 'all-MiniLM-L6-v2'

def main():
    """
    Downloads and caches the specified sentence transformer model.
    For Vercel, this script should be run as part of the build command.
    It will store the model in a cache directory that can be accessed
    by the serverless function at runtime.
    """
    # On Vercel, you can use a specific cache directory.
    # Ensure this same path is set as an environment variable in your project settings.
    cache_dir = "/tmp/sentence_transformers_cache"
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
    
    logger.info(f"Attempting to download and cache model: {MODEL_NAME} to {cache_dir}")
    try:
        # This line triggers the download and saves the model to the specified cache directory
        SentenceTransformer(MODEL_NAME)
        logger.info(f"Successfully cached model: {MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to download model {MODEL_NAME}. Error: {e}")
        # Exit with a non-zero status code to fail the build if caching fails
        exit(1)

if __name__ == "__main__":
    main()
