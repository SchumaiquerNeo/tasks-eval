import os
import json
import re
import asyncio
import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
import yaml

load_dotenv()

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

MODELS = cfg["models"]
N_FEWSHOT = cfg["n_fewshot"]
MAX_SAMPLES = cfg["max_samples"]
NUM_CONCURRENT = cfg["num_concurrent"]
OUTPUT_DIR = cfg["output_dir"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

SUBJECTS = [s for s in get_dataset_config_names("cais/mmlu") if s not in ["all", "auxiliary_train", "validation", "dev"]]

# Extrair letra da resposta
def extract_choice(text: str):
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else None

# Construir prompt few-shot
def build_prompt(examples, question, choices):
    prompt = "Responda apenas com a letra (A, B, C ou D) correspondente.\n\n"
    for ex in examples:
        prompt += f"Pergunta: {ex['question']}\n"
        for i, choice in enumerate(ex['choices']):
            prompt += f"  {chr(65+i)}. {choice}\n"
        prompt += f"Resposta: {chr(65+ex['answer'])}\n\n"

    prompt += f"Pergunta: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"  {chr(65+i)}. {choice}\n"
    prompt += "Resposta:"
    return prompt

# Retry para chamadas da API
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=30))
async def call_api(model_name, prompt):
    resp = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip()

# Avaliar uma subcategoria
async def evaluate_subject(model_name, subject):
    dataset = load_dataset("cais/mmlu", subject)
    test_data = dataset["test"]

    if MAX_SAMPLES:
        test_data = test_data.select(range(min(MAX_SAMPLES, len(test_data))))

    correct, total = 0, 0
    sem = asyncio.Semaphore(NUM_CONCURRENT)

    async def process_example(i, example):
        nonlocal correct, total
        fewshot_examples = test_data.select(
            range(min(N_FEWSHOT, len(test_data)))
        ) if i >= N_FEWSHOT else test_data.select(range(i))

        prompt = build_prompt(fewshot_examples, example["question"], example["choices"])

        async with sem:
            try:
                answer_text = await call_api(model_name, prompt)
                pred = extract_choice(answer_text)
                if pred == chr(65 + example["answer"]):
                    correct += 1
                total += 1
            except Exception as e:
                print(f"[{model_name}] Erro em {subject}, exemplo {i}: {e}")

    await tqdm_asyncio.gather(
        *[process_example(i, ex) for i, ex in enumerate(test_data)],
        desc=f"{model_name} - {subject}"
    )

    accuracy = correct / total if total > 0 else 0.0
    return {"subject": subject, "accuracy": accuracy, "correct": correct, "total": total}

async def main():
    all_results = []
    for model in MODELS:
        print(f"\n=== Avaliando {model} em todas as categorias ===")
        for subject in SUBJECTS:
            res = await evaluate_subject(model, subject)
            res["model"] = model
            all_results.append(res)
            print(f"  {subject}: {res['accuracy']*100:.2f}% ({res['correct']}/{res['total']})")

    json_path = os.path.join(OUTPUT_DIR, "mmlu_full_results_async.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    df = pd.DataFrame(all_results)
    summary = df.groupby("model").agg(
        mean_accuracy=("accuracy", "mean"),
        total_correct=("correct", "sum"),
        total_questions=("total", "sum")
    )
    summary["overall_accuracy"] = summary["total_correct"] / summary["total_questions"]

    csv_path = os.path.join(OUTPUT_DIR, "mmlu_summary_async.csv")
    summary.to_csv(csv_path)

    print("\n=== Resultados Finais ===")
    print(summary)
    print(f"\nResultados salvos em:\n- {json_path}\n- {csv_path}")

if __name__ == "__main__":
    asyncio.run(main())
