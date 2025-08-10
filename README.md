# RAG AI Chatbot

A starter template for a Retrieval-Augmented Generation (RAG) chatbot that leverages PDF files as its primary knowledge source. Designed for easy setup, customization, and future expansion.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)

## Installation
1. Clone the repo:
```
git clone https://github.com/aaroncanillas/rag-chatbot-assistant.git
cd rag-chatbot-assistant
```

2. Create .env file and input your own configuration
```
cp .env.sample .env
```

3. Create and activate a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate    # On Windows use `venv\Scripts\activate`
```

4. Install dependencies:
```
pip install -r requirements.txt
```

## Usage 

1. Create a folder named `data/` and place your PDF documents there.

2. Ingest documents using:
```
python ingest.py
```

3. Start the chatbot:

```
python main.py
```

4. You can modify settings in `config.py` file 👽

## Features
- 📄 Support PDF document ingestion
- 🧠 RAG-based conversational AI
- ⚙️ Configurable via config.py 
- 🛠️ Easy to extend for additional file types and features


## Future enhancements
- Additional file types support
- Web-based UI

