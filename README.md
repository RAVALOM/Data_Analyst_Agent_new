# Data Analyst Agent API 🤖

This project implements a "Data Analyst Agent," an API that uses a Large Language Model (LLM) to automatically source, prepare, analyze, and visualize data based on natural language instructions.

## ✨ Features

* **LLM-Powered:** Uses Google's Gemini models to interpret tasks and generate Python code on the fly.
* **Flexible Data Handling:** Accepts tasks and data files (like `.csv`) via a simple API endpoint.
* **Versatile Analysis:** Capable of web scraping, complex data querying (with DuckDB), statistical analysis, and more.
* **Secure Execution:** All generated code is run in an isolated, secure Docker container to prevent security risks.
* **Dynamic Visualizations:** Generates plots and charts as base64-encoded data URIs directly in the JSON response.

---

## 🏛️ Architecture

The application follows a simple but robust workflow for each API request:

1.  **Receive Request:** A FastAPI server accepts a `POST` request containing a task description (`questions.txt`) and optional data files.
2.  **Generate Plan:** The task is sent to an LLM (Gemini), which generates a complete Python script to solve the task.
3.  **Execute Securely:** The generated script is run inside a temporary, isolated Docker container with all necessary data science libraries.
4.  **Capture Result:** The application captures the script's output, which is a single JSON string.
5.  **Send Response:** The captured JSON is sent back to the client as the API response.



---

## 🚀 Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

* Python 3.10+
* Docker Desktop (must be running)
* A Google API Key for the Gemini API

### 1. Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)[Your-Username]/data-analyst-agent.git
cd data-analyst-agent
```

### 2. Set Up Environment Variables

Create a file named `.env` in the project root and add your Google API key.

```env
# Get your API key from Google AI Studio: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
GOOGLE_API_KEY="PASTE_YOUR_GOOGLE_API_KEY_HERE"
```

### 3. Install Python Dependencies

It's recommended to use a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required libraries
pip install -r requirements.txt
```

### 4. Build the Docker Sandbox Image

This command builds the secure container environment defined in the `Dockerfile`.

```bash
docker build -t data-analyst-sandbox:latest .
```

---

## ▶️ Running the Application

Once setup is complete, you can start the API server.

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can see the auto-generated documentation at `http://127.0.0.1:8000/docs`.

---

## 🛠️ API Usage

To use the agent, send a `POST` request to the `/api/` endpoint with `multipart/form-data`.

You must include a file part named `questions.txt`. You can include other data files as needed.

### Example `curl` Request

This example assumes you have two files in your current directory:
* `questions.txt`: Contains the analysis instructions.
* `data.csv`: Contains the data to be analyzed.

```bash
curl -X POST "[http://127.0.0.1:8000/api/](http://127.0.0.1:8000/api/)" \
-F "questions.txt=@questions.txt" \
-F "data.csv=@data.csv"
```

The server will process the request and, if successful, return a `200 OK` response with a JSON body containing the answers.