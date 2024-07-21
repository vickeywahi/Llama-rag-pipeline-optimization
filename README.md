# Llama-rag-pipeline-optimization 
 ## Project Description
 Iterative Optimization of LlamaIndex RAG Pipeline: A Step-by-Step Approach:

1. Baseline evaluation: Evaluation for Relevancy and Faithfulness metrics 
2. Changing the embedding model 
3. Incorporating a Reranker 
4. Employing Deep Memory 


This project explores the iterative optimization of a RAG (Retrieval Augmented Generation) pipeline built using LlamaIndex. We systematically evaluate and improve the pipeline's performance by experimenting with different components and configurations. 
## Steps: 
  1. **Baseline Evaluation:** We establish a performance baseline using a standard LlamaIndex RAG pipeline and evaluate it for relevancy and faithfulness. 
2. **Changing the Embedding Model:** We test different embedding models (e.g., OpenAI's `text-embedding-ada-002` and Cohere's `embed-english-v3.0`) to find the most effective one. 
3. **Incorporating a Reranker:** We implement various rerankers (e.g., cross-encoder, LLMRerank, CohereRerank) to refine the document selection process. 
4. **Employing Deep Memory:** We explore using ActiveLoop's Deep Memory feature to potentially improve retrieval accuracy. 

 ## Installation: 
1. **Create a virtual environment:** `python -m venv llama-rag-env3.10`
2. **Activate the environment:** `pyenv activate llama-rag-env3.10` 
3. **Install dependencies:** `pip install -r requirements.txt requirements-dev.txt` 
4. **Set up environment variables:** Create a `.env` file and add your OpenAI API key and 
ActiveLoop token (if using Deep Memory):
OPENAI_API_KEY=<your_openai_api_key> 
ACTIVELOOP_TOKEN=<your_activeloop_token>

## Usage: 
```bash 
# Run main.py with default values 
python main.py 
 # Run main.py with custom top_k and num_eval_queries 
python main.py --top_k 5 --num_eval_queries 10 

## Data Source

https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt

To replace this source, edit the main.py file and update the DATA_URL environment variable in the .env file.

##Evaluation Metrics:
**Hit Rate:** Measures how often the correct document is retrieved within the top-k results.
**Mean Reciprocal Rank (MRR):**  Indicates how close to the top of the retrieved documents the correct document typically ranks.
**Relevancy:**  Assesses whether the retrieved context and answer are relevant to the query.
**Faithfulness:**  Evaluates if the answer is faithful to the retrieved context (i.e., not a hallucination).

