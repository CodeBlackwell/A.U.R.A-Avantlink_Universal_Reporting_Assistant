from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TheBloke/WizardCoder-15B-1.0-GPTQ")
model = AutoModelForCausalLM.from_pretrained("TheBloke/WizardCoder-15B-1.0-GPTQ")
