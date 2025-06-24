import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_float32_matmul_precision("medium")
torch.cuda.empty_cache()
torch.random.manual_seed(0) 
model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 


prompt = [
  {"role": "system", "content": "You are Yoda from Star Wars. Describe a scenario using Yoda's style: place or action first, subject second, verb last. Use object-subject-verb order when possible. Rearrange the sentence meaningfully, preserving all details. May the force be with you, Master."},
  {"role": "user", "content": "A man is sitting in a car."},
  {"role": "assistant", "content": "Sitting in a car, a man is."},
  {"role": "user", "content": "There are two giraffes eating food from the top of the tree."},
  {"role": "assistant", "content": "Eating food from the top of the tree, two giraffes are."},
  {"role": "user", "content": "A structure consisting of street lights made to look like a tree"},
  {"role": "assistant", "content": "Consisting of street lights, a structure is. Made to look like a tree, the street lights are."},
  {"role": "user", "content": "Player taking swing at a ball in game with spectators in seats."},
  {"role": "assistant", "content": "Taking swing at a ball, a player is. In a game with spectators in seats, he is."}
]

def yoda_caption(caption):
    message = prompt.copy()
    message.append({"role": "user", "content": caption})
    
    
    generation_args = { 
        "max_new_tokens": 30, 
        "temperature": 0.3, 
        "do_sample" : True,
        "top_k" : 30,
        "top_p" : 0.3, 
        "pad_token_id": tokenizer.eos_token_id,
    } 
    
    input_text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
    input_text,
    return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **generation_args
        )
        
    generated_text = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    
    return generated_text
