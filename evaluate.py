#main branch evaluate.py to start building evaluate.py on the feature-b branch
import os
from llama_index.core.evaluation import generate_question_context_pairs, EmbeddingQAFinetuneDataset
from llama_index.core.evaluation import RetrieverEvaluator, RelevancyEvaluator, FaithfulnessEvaluator, BatchEvalRunner
from llama_index.llms.openai import OpenAI
from llama_index.core import ServiceContext
import json #Introducing caching

import asyncio

import pandas as pd

# Function to generate dataset (optional)
def generate_dataset(nodes, llm):
    """Function for converting LlamaIndex dataset to correct format for deep memory training"""
    qc_dataset = generate_question_context_pairs(
        nodes,
        llm=llm,
        num_questions_per_chunk=1
    )
    # We can save the dataset as a json file for later use.
    qc_dataset.save_json("qc_dataset.json")
    return qc_dataset

def get_dataset(nodes, llm):
    """Loads the dataset from a file, or generates a new one if it doesn't exist."""
    try:
        qc_dataset = EmbeddingQAFinetuneDataset.from_json("qc_dataset.json")
    except FileNotFoundError:
        print("Dataset not found. Generating...")
        qc_dataset = generate_dataset(nodes, llm)  # Save the generated dataset
    return qc_dataset


# First, we define a function to display the Retrieval evaluation results in table format.
def display_results_retriever(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"Retriever Name": [name], "Hit Rate": [hit_rate], "MRR": [mrr]}
    )

    return metric_df

# Define the run_evaluations function
async def run_evaluations(vector_index, nodes, llm, service_context, num_eval_queries):  # Added num_eval_queries
    # You can load the dataset from your local disk if you have already generated
    qc_dataset = get_dataset(nodes, llm)  # Get the dataset (load or generate)
    #Introducing Caching
    cache_file = "evaluation_cache.json"  

    def load_cache():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e: 
            print(f"Error loading cache file: {e}")
            # Optionally, print the contents of the cache file for debugging
            with open(cache_file, 'r') as f:
                print(f"Cache file contents:\n{f.read()}")
            return {}  
    
    def save_cache(cache):
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)

    # Load results cache (if exists)
    cache = load_cache()  

    # Retriever Evaluation with different top_k values
    for i in [2, 4, 6, 8, 10]:
        cache_key = f"retriever_top_{i}" ##Introducing caching
        if cache_key in cache:
            retriever_results.append(cache[cache_key])
            print(f"Loaded retriever top_{i} results from cache.")
        else:        
            while True:
                try:
                    retriever = vector_index.as_retriever(similarity_top_k=i)
                    retriever_evaluator = RetrieverEvaluator.from_metric_names(
                        ["mrr", "hit_rate"], retriever=retriever
                    )
                    eval_results = await retriever_evaluator.aevaluate_dataset(qc_dataset)
                    
                    # Store the first element of the list (assuming it's the relevant one)
                    cache[cache_key] = eval_results[0].metric_vals_dict  
                    save_cache(cache) 
                    print(display_results_retriever(f"Retriever top_{i}", eval_results))
                    break
                except RateLimitError as e:
                    retry_after = e.retry_after if e.retry_after else 1  # Ensure a minimum wait of 1 second
                    print(f"Rate limit exceeded. Retrying in {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
      
    # Evaluation for Relevancy and Faithfulness metrics
    for i in [2, 4, 6, 8, 10]:
        # Set Faithfulness and Relevancy evaluators
        query_engine = vector_index.as_query_engine(similarity_top_k=i)

        # While we use GPT3.5-Turbo to answer questions
        # we tried using GPT4 to evaluate the answers, its expensive, so try GPT3.5-Turbo
        llm_evaluator = OpenAI(temperature=0, model="gpt-3.5-turbo-1106")
        
        service_context_evaluator = ServiceContext.from_defaults(llm=llm_evaluator)
        
        faithfulness_evaluator = FaithfulnessEvaluator(service_context=service_context_evaluator)
        
        relevancy_evaluator = RelevancyEvaluator(service_context=service_context_evaluator)

        # Run evaluation
        queries = list(qc_dataset.queries.values())
        batch_eval_queries = queries[:num_eval_queries]  # Use num_eval_queries here

        runner = BatchEvalRunner(
        {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
        workers=8,
        )
        eval_results = await runner.aevaluate_queries(
            query_engine, queries=batch_eval_queries
        )
        faithfulness_score = sum(result.passing for result in eval_results['faithfulness']) / len(eval_results['faithfulness'])
        print(f"top_{i} faithfulness_score: {faithfulness_score}")

        relevancy_score = sum(result.passing for result in eval_results['relevancy']) / len(eval_results['relevancy'])  # Fixed calculation
        print(f"top_{i} relevancy_score: {relevancy_score}") 