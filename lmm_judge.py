import os
import json
import time
import logging
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LMM Judge for Video Auto Arena")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Judge model name")
    parser.add_argument("--api_key", type=str, default=None, help="API key")
    parser.add_argument("--api_base", type=str, default=None, help="API base URL")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    return parser.parse_args()


SYSTEM_PROMPT = """You are a fair and objective judge. Your task is to evaluate the quality of responses 
provided by two AI assistants to a user's question about a video. You should consider the following criteria:
1. Accuracy: How accurate is the response in describing the video content?
2. Relevance: How relevant is the response to the user's question?
3. Detail: How detailed and comprehensive is the response?
4. Coherence: How well-organized and coherent is the response?

Please evaluate both responses and determine which one is better. Output your judgment in the following JSON format:
{"winner": "A" or "B" or "tie", "reason": "your explanation"}
"""


def create_judge_prompt(question, response_a, response_b):
    prompt = f"""User's Question: {question}

Assistant A's Response: {response_a}

Assistant B's Response: {response_b}

Please evaluate both responses based on accuracy, relevance, detail, and coherence. 
Which response is better? Output your judgment as JSON."""
    return prompt


def call_judge_api(client, model, system_prompt, user_prompt, temperature=0.0, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"All {max_retries} attempts failed")
                return {"winner": "error", "reason": str(e)}


def process_single_item(item, client, model, temperature, max_retries):
    question = item["question"]
    response_a = item["response_a"]
    response_b = item["response_b"]
    
    user_prompt = create_judge_prompt(question, response_a, response_b)
    result = call_judge_api(client, model, SYSTEM_PROMPT, user_prompt, temperature, max_retries)
    
    item["judgment"] = result
    return item


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    
    client = OpenAI(
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=args.api_base
    )
    
    # Load input data
    data = []
    with open(args.input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(data)} items from {args.input_file}")
    
    # Process items
    results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_single_item, item, client, args.model, 
                args.temperature, args.max_retries
            )
            for item in data
        ]
        
        for future in tqdm(futures, total=len(futures), desc="Judging"):
            results.append(future.result())
    
    # Save results
    with open(args.output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')
    
    # Print summary
    winners = [r["judgment"].get("winner", "error") for r in results]
    logger.info(f"Results: A={winners.count('A')}, B={winners.count('B')}, "
                f"Tie={winners.count('tie')}, Error={winners.count('error')}")


if __name__ == "__main__":
    main()
