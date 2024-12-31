import tqdm
import json
import torch

from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEEPSEEK_RESPONSE = 'deepseek_response'
PHI_RESPONSE = 'phi-3-mini_response'
CONVERSATIONS = 'conversations'
SYSTEM = 'system'
VALUE = 'value'
FROM = 'from'
REWARD = 'reward'


def get_messages_no_system(element, include_last = False):
    messages = []

    system_prompt = element.get(SYSTEM)
    
    for message in element[CONVERSATIONS]:
        if message[FROM] == 'human':
            if system_prompt:
                messages.append({'role':'user', 'content':system_prompt + "\n\n---\n\n" + message[VALUE]})
                system_prompt = None
            else:
                messages.append({'role':'user', 'content':message[VALUE]})
        elif message[FROM] == 'gpt':
            messages.append({'role':'assistant', 'content':message[VALUE]})
    
    if messages[-1]['role'] == 'assistant' and not include_last:
        messages = messages[:-1]

    return messages



def get_reward_scores(
    conversations
):
    def get_batch_size(token_length):
        if token_length <= 128:
            return 128
        if token_length <= 256:
            return 64
        if token_length <= 512:
            return 32
        elif token_length <= 1024:
            return 16
        elif token_length <= 2048:
            return 8
        elif token_length <= 3072:
            return 6
        else:
            return 4
    
    #model_name = 'Skywork/Skywork-Reward-Gemma-2-27B-v0.2'
    model_name = '/home/hidelord/text-generation-webui-snapshot-2024-04-14/models/Skywork-Reward-Gemma-2-27B-v0.2'

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        num_labels=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    tokenized_conversations = [
        tokenizer.apply_chat_template(
            conv, tokenize=False
        ) for conv in conversations
    ]
    
    conversation_lengths = [len(tokenizer(conv)['input_ids']) for conv in tokenized_conversations]
    
    sorted_conversations_with_indices = sorted(
        enumerate(conversations), 
        key=lambda x: conversation_lengths[x[0]]
    )
    
    sorted_conversations = [conv for _, conv in sorted_conversations_with_indices]
    original_indices = [idx for idx, _ in sorted_conversations_with_indices]
    sorted_lengths = [conversation_lengths[idx] for idx, _ in sorted_conversations_with_indices]

    all_scores = [None] * len(conversations)
    current_idx = 0

    # Create progress bar
    pbar = tqdm.tqdm(total=len(sorted_conversations), desc="Processing conversations")
    
    with torch.no_grad():
        while current_idx < len(sorted_conversations):
            current_length = sorted_lengths[current_idx]
            batch_size = get_batch_size(current_length)
            success = False
            
            while not success and batch_size >= 1:
                try:
                    end_idx = min(current_idx + batch_size, len(sorted_conversations))
                    batch_conversations = sorted_conversations[current_idx:end_idx]
                    
                    batch_tokenized = tokenizer.apply_chat_template(
                        batch_conversations,
                        tokenize=True,
                        padding=True,
                        return_tensors="pt"
                    ).to("cuda:0")
                    
                    attention_mask = (batch_tokenized != tokenizer.pad_token_id).long()
                    
                    outputs = model(
                        input_ids=batch_tokenized,
                        attention_mask=attention_mask
                    )
                    
                    batch_scores = outputs.logits[:, 0].tolist()
                    
                    for i, score in enumerate(batch_scores):
                        original_idx = original_indices[current_idx + i]
                        all_scores[original_idx] = score
                    
                    del batch_tokenized
                    del attention_mask
                    del outputs
                    
                    success = True
                    
                except Exception as e:
                    print(f"Error processing batch starting at index {current_idx} with batch size {batch_size}: {str(e)}")
                    batch_size = max(batch_size // 2, 1)  # Reduce batch size, but keep minimum of 1
                    torch.cuda.empty_cache()  # Clear GPU memory
                    
                    if batch_size == 1:  # If we've reached batch size 1 and still failing
                        print(f"Failed to process item at index {current_idx} even with batch size 1")
                        original_idx = original_indices[current_idx]
                        all_scores[original_idx] = None
                        success = True  # Move on to next item
            
            # Update progress bar
            increment = end_idx - current_idx if success else 1
            pbar.update(increment)
            current_idx = end_idx if success else current_idx + 1

    pbar.close()
    del tokenizer
    del model
    torch.cuda.empty_cache()
    
    return all_scores

def download_data():
    dataset = load_dataset("OpenLeecher/lmsys_chat_1m_clean", )
    data = []

    for split in dataset.keys():
        for element in dataset[split]:
            data.append(element)

    return data

def upload_data(data):
    train_ds = Dataset.from_list(data)
    train_ds.push_to_hub('lmsys_chat_1m_clean')

def get_deepseek_convos(data):
    convos = []
    for element in data:
        convos.append([
            {'role':'user', 'content':element[CONVERSATIONS][0][VALUE]},
            {'role':'assistant', 'content':element[DEEPSEEK_RESPONSE][VALUE]}
        ])

    return convos

def get_phi_convos(data):
    convos = []
    for element in data:
        convos.append([
            {'role':'user', 'content':element[CONVERSATIONS][0][VALUE]},
            {'role':'assistant', 'content':element[PHI_RESPONSE][VALUE]}
        ])

    return convos

def main():
    data = download_data()[:100]
    reward_scores = get_reward_scores(get_deepseek_convos(data))

    for element, reward in zip(data, reward_scores):
        element['deepseek_response'][REWARD] = reward

    reward_scores = get_reward_scores(get_phi_convos(data))

    for element, reward in zip(data, reward_scores):
        element['phi-3-mini_response'][REWARD] = reward

    with open('test.json', 'w') as file:
        json.dump(data, file)


if __name__ == "__main__":
    main()