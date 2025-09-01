# Variáveis principais
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
REQ := requirements.txt
CONFIG := config.yaml
SCRIPT_MMLU := run_mmlu_all_async.py

# Ler configs do YAML via Python + pyyaml
TASK := $(shell $(PYTHON) -c "import yaml;print(yaml.safe_load(open('$(CONFIG)'))['task'])")
MODELS := $(shell $(PYTHON) -c "import yaml;print(' '.join(yaml.safe_load(open('$(CONFIG)'))['models']))")
N_FEWSHOT := $(shell $(PYTHON) -c "import yaml;print(yaml.safe_load(open('$(CONFIG)'))['n_fewshot'])")
OUTPUT_DIR := $(shell $(PYTHON) -c "import yaml;print(yaml.safe_load(open('$(CONFIG)'))['output_dir'])")

# Criar venv
$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	@echo "✅ Virtualenv criada em $(VENV)."

# Instalar dependências
install: $(VENV)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQ)

# Rodar benchmark (decide com base no config.yaml)
run: install
ifeq ($(TASK),mmlu)
	@echo " - Rodando MMLU com modelos: $(MODELS)"
	$(PYTHON) $(SCRIPT_MMLU)
endif
ifeq ($(TASK),gsm8k)
	@echo " - Rodando GSM8K com modelos: $(MODELS)"
	for model in $(MODELS); do \
		OPENAI_API_KEY=$$OPENAI_API_KEY lm_eval \
			--model openai-chat-completions \
			--model_args model=$$model \
			--tasks gsm8k \
			--num_fewshot $(N_FEWSHOT) \
			--batch_size 1 \
			--apply_chat_template \
			--output_path $(OUTPUT_DIR)/gsm8k_$${model}.json ; \
	done
endif
