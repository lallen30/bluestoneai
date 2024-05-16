# rag-application

Install dependencies.

```python
pip install -r requirements.txt
```

Create the Chroma DB.

```python
python populate_database.py
```

Query the Chroma DB.

```python
python query_data.py "How does Alice meet the Mad Hatter?"
```