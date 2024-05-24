# QnA API Application

This is a Python-based API application that provides question and answer services using the Flask framework. The application can vectorize various document types, process web pages, and manage API keys and user questions. It uses SQLite for database storage and Chroma for vector storage.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [License](#license)

## Features

- API key management
- Upload and process documents (TXT, PDF, DOCX, CSV)
- Process web pages
- Handle user questions and store answers
- Vectorize documents for similarity search

## Requirements

- Docker
- Python 3.11
- Flask
- Chroma
- OpenAI API

## Installation

### Using Docker

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Build the Docker image:

   ```bash
   docker build -t qna-api .
   ```

3. Run the Docker container:
   ```bash
   docker run -p 5001:5001 --env-file .env qna-api
   ```

### Without Docker

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment variables (refer to the [Environment Variables](#environment-variables) section).

4. Run the application:
   ```bash
   python app.py
   ```

## Usage

- The application will be available at `http://localhost:5001`.

## API Endpoints

### Public Endpoints

- **Test Connection**
  - `GET /test_connection`
  - Response: `{ "message": "Success" }`

### Protected Endpoints

These endpoints require an API key for access. Include the key in the `Authorization` header as `Bearer <API_KEY>`.

- **Set API Key**

  - `POST /set_api_key`
  - Request body: `{ "key_name": "OPENAI_API_KEY", "key_value": "<your-api-key>" }`
  - Response: `{ "message": "API key updated successfully." }`

- **Populate Database**

  - `POST /populate_db`
  - Request body: `{ "reset": true/false }`
  - Response: `{ "message": "Database population initiated successfully." }`

- **Submit Texts**

  - `POST /submit_text`
  - Request body: `[{"text_type": "<type>", "text": "<text>"}]`
  - Response: `{ "message": "Texts updated successfully." }`

- **Question and Answer**

  - `POST /qna`
  - Request body: `{ "question": "<your-question>", "question_session_id": "<session-id>" }`
  - Response: `{ "answer": "<answer>" }`

- **Upload File**

  - `POST /upload`
  - Request body: Form-data with key `file` and the file to upload.
  - Response: `{ "message": "File uploaded and processed successfully." }`

- **Process Webpage**

  - `POST /process_webpage`
  - Request body: `{ "url": "<webpage-url>" }`
  - Response: `{ "message": "Webpage content processed successfully." }`

- **Remove Vectorized Data**

  - `DELETE /remove_vectorized_data`
  - Request body: `{ "filename": "<filename>" }`
  - Response: `{ "message": "File and associated data deleted successfully." }`

- **Get All Chunk IDs**
  - `GET /get_all_chunk_ids`
  - Response: `{ "chunk_ids": [<list-of-chunk-ids>] }`

## Environment Variables

Create a `.env` file in the project root and add the following environment variables:

```env
SECRET_KEY=<your-secret-key>
OPENAI_API_KEY=<your-openai-api-key>
STATIC_API_TOKEN=<your-static-api-token>
```
