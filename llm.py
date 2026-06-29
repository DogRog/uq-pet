import os
import math
import asyncio
import nest_asyncio
from collections import Counter
from openai import AsyncOpenAI
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()  # Load environment variables from .env file

# 1. Initialize the OpenRouter client using the OpenAI SDK
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

async def get_single_sample(prompt: str, model: str, temperature: float) -> str:
    """Fetches a single response from OpenRouter."""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=50 # Keep low for simple QA to save costs
    )
    return response.choices[0].message.content.strip()

async def get_multiple_samples(prompt: str, model: str, num_samples: int = 5, temperature: float = 0.7) -> list:
    """Fetches multiple samples concurrently for speed."""
    # Note: We use concurrent requests instead of the `n` parameter because 
    # the `n` parameter is not supported by all upstream providers on OpenRouter.
    tasks = [get_single_sample(prompt, model, temperature) for _ in range(num_samples)]
    return await asyncio.gather(*tasks)

def calculate_entropy(samples: list) -> float:
    """
    Calculates Shannon Entropy based on the frequency of exact matches.
    Formula: H = - sum(p * log2(p))
    """
    # Count the frequency of each unique answer
    counts = Counter(samples)
    total_samples = len(samples)
    
    entropy = 0.0
    for count in counts.values():
        probability = count / total_samples
        entropy -= probability * math.log2(probability)
        
    return entropy, counts

async def main():
    # Model to use via OpenRouter
    MODEL = "meta-llama/llama-3-8b-instruct" 
    
    # Example 1: High Confidence Expected
    prompt_easy = "Answer with a single word. What is the capital of France?"
    
    # Example 2: High Uncertainty Expected (Edge case / obscure fact)
    prompt_hard = "Answer with a single word. What is the favorite color of King Henry VIII?"

    print("Testing Easy Prompt...")
    samples_easy = await get_multiple_samples(prompt_easy, MODEL, num_samples=5, temperature=0.8)
    entropy_easy, counts_easy = calculate_entropy(samples_easy)
    
    print("Testing Hard Prompt...")
    samples_hard = await get_multiple_samples(prompt_hard, MODEL, num_samples=5, temperature=0.8)
    entropy_hard, counts_hard = calculate_entropy(samples_hard)

    print("\n--- RESULTS ---")
    print(f"Easy Prompt Answers: {dict(counts_easy)}")
    print(f"Easy Prompt Uncertainty (Entropy): {entropy_easy:.4f} bits\n")
    
    print(f"Hard Prompt Answers: {dict(counts_hard)}")
    print(f"Hard Prompt Uncertainty (Entropy): {entropy_hard:.4f} bits")

if __name__ == "__main__":
    asyncio.run(main())