# 📊 Benchmark Runner (MMLU & GSM8K)

Este projeto permite rodar benchmarks **MMLU** e **GSM8K** em modelos da OpenAI (ex.: `gpt-4o`, `gpt-4o-mini`, `gpt-4`) de forma simples e configurável.  
Ele integra:
- Configuração via **YAML** (`config.yaml`)
- **.env** para manter sua chave de API segura
- **Makefile** para automatizar instalação e execução
- Suporte a concorrência assíncrona (para acelerar execução no MMLU)

---

## 🚀 Setup

### 1. Clone e crie ambiente virtual
```bash
git clone <repo>
cd mmlu-run
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instale dependências

```bash
pip install -r requirements.txt
```

### 🔑 Configuração da API

Crie um arquivo .env na raiz do projeto:

```ini
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

## ⚙️ Configuração de benchmark

Ajuste config.yaml:

```yaml
models:
  - gpt-4o
  - gpt-4o-mini

# Escolha o benchmark: "mmlu" ou "gsm8k"
task: gsm8k

# Configurações gerais
n_fewshot: 5
max_samples: 200   # use None para dataset completo
num_concurrent: 5
output_dir: results

```

## ▶️ Execução

O Makefile já cuida de tudo:

```bash
make run
```
- Se task: mmlu → roda o script assíncrono run_mmlu_all_async.py
- Se task: gsm8k → roda lm_eval diretamente, um JSON por modelo

## 📂 Resultados

### GSM8K

Saem na pasta results/, um JSON por modelo:

```bash
results/gsm8k_gpt-4o.json
results/gsm8k_gpt-4o-mini.json
```

### MMLU

- Resultados detalhados por categoria:

```bash
results/mmlu_full_results_async.json
```

- Resumo geral por modelo (CSV):

```bash
results/mmlu_summary_async.csv
```