"""
Model loading and generation: 4-bit quantized causal LM + MiniLM sentence encoder.
"""

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Gemma3ForConditionalGeneration


#MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# Alternates:
# MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
#MODEL_NAME = "google/gemma-3-4b-it"
#MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def load_sentence_encoder():
    print("Loading sentence encoder...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    encoder.eval()
    print(f"Encoder loaded. Embedding dim: {encoder.get_sentence_embedding_dimension()}")
    return encoder


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16,
        bnb_4bit_use_double_quant = True,
    )

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = torch.float16,
    )
    model.eval()
    print("Model loaded.")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model, tokenizer


def generate_response(
    model, tokenizer, question: str,
    temperature:        float = 0.9,
    max_new_tokens:     int   = 20,
    repetition_penalty: float = 1.15,
) -> str:
    """
    Generate one SHORT answer (name, number, phrase — not a full sentence).
    Uses apply_chat_template so the prompt format adapts to any chat model.
    """
    messages = [{
        "role": "user",
        "content": (
            "Answer the question with only a name, number, or short phrase. "
            "Do not write a full sentence. Do not explain.\n\n"
            "Examples:\n"
            "Q: Who wrote Romeo and Juliet? A: William Shakespeare\n"
            "Q: What is the capital of Japan? A: Tokyo\n"
            "Q: How many planets are in the solar system? A: 8\n\n"
            f"Q: {question} A:"
        ),
    }]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens     = max_new_tokens,
            do_sample          = True,
            temperature        = temperature,
            top_p              = 0.95,
            repetition_penalty = repetition_penalty,
            pad_token_id       = tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    response  = tokenizer.decode(generated, skip_special_tokens=True).strip()

    for stop in [".", "\n", "because", "since", "who", "which", "[", "]"]:
        if stop in response:
            response = response[:response.index(stop)].strip()

    return response


def generate_batch(model, tokenizer, question, n, temperature=0.9):
    return [
        generate_response(model, tokenizer, question, temperature)
        for _ in range(n)
    ]
