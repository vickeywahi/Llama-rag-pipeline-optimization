
import os
from llama_index.core.evaluation import generate_question_context_pairs, EmbeddingQAFinetuneDataset
from llama_index.core.evaluation import RetrieverEvaluator, RelevancyEvaluator, FaithfulnessEvaluator, BatchEvalRunner
from llama_index.llms.openai import OpenAI
from llama_index.core import ServiceContext
import json

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

async def run_evaluations(vector_index, nodes, llm, service_context, num_eval_queries):
    qc_dataset = get_dataset(nodes, llm)
    cache_file = "evaluation_cache.json"

    def load_cache():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_cache(cache):
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
            
    # Load results cache (if exists)
    cache = load_cache()

    # Retriever Evaluation with different top_k values
    retriever_results = []
    for i in [2, 4, 6, 8, 10]:
        cache_key = f"retriever_top_{i}"
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
                    retriever_results.append(eval_results)
                    cache[cache_key] = eval_results # Store results in cache
                    save_cache(cache) # Save updated cache
                    print(display_results_retriever(f"Retriever top_{i}", eval_results))
                    break
                except RateLimitError as e:
                    retry_after = e.retry_after if e.retry_after else 1
                    print(f"Rate limit exceeded. Retrying in {retry_after} seconds...")
                    await asyncio.sleep(retry_after)

    # Combined Relevancy and Faithfulness Evaluation
    faithfulness_scores = {}
    relevancy_scores = {}
    queries = list(qc_dataset.queries.values())
    batch_eval_queries = queries[:num_eval_queries]
    
    # While we use GPT3.5-Turbo to answer questions
        # we tried using GPT4 to evaluate the answers, its expensive, so try GPT3.5-Turbo
        llm_evaluator = OpenAI(temperature=0, model="gpt-3.5-turbo-1106")
        service_context_evaluator = ServiceContext.from_defaults(llm=llm_evaluator)
        faithfulness_evaluator = FaithfulnessEvaluator(service_context=service_context_evaluator)
        relevancy_evaluator = RelevancyEvaluator(service_context=service_context_evaluator)

    for i in [2, 4, 6, 8, 10]:
        cache_key = f"eval_top_{i}"
        if cache_key in cache:
            faithfulness_scores[i] = cache[cache_key]['faithfulness_score']
            relevancy_scores[i] = cache[cache_key]['relevancy_score']
            print(f"Loaded evaluation top_{i} results from cache.")
        else:
            while True:  
                try:
                    query_engine = vector_index.as_query_engine(similarity_top_k=i)
                    runner = BatchEvalRunner(
                        {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
                        workers=8,
                    )
                    eval_results = await runner.aevaluate_queries(
                        query_engine, queries=batch_eval_queries
                    )
                    faithfulness_score = sum(result.passing for result in eval_results['faithfulness']) / len(eval_results['faithfulness'])
                    relevancy_score = sum(result.passing for result in eval_results['relevancy']) / len(eval_results['relevancy'])  
                    print(f"top_{i} faithfulness_score: {faithfulness_score}")
                    print(f"top_{i} relevancy_score: {relevancy_score}")
                    cache[cache_key] = {
                      'faithfulness_score': faithfulness_score,
                      'relevancy_score': relevancy_score
                      }
                    save_cache(cache) # Save updated cache
                    break
                except RateLimitError as e:
                    retry_after = e.retry_after if e.retry_after else 1  # Ensure a minimum wait of 1 second
                    print(f"Rate limit exceeded. Retrying in {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
