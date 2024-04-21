from transformers import CLIPTokenizer, CLIPTextModel, GPT2Tokenizer
import torch
import tokenizations

# Initialize the CLIPTokenizer and CLIPTextModel
clip_tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer")
clip_model = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="text_encoder")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

clip_embeddings = clip_model.get_input_embeddings().weight
gpt_vocab_size = gpt_tokenizer.vocab_size

def get_gpt2_logits(prompt):
    # Tokenize the prompt
    tokens = gpt_tokenizer.encode(prompt, add_special_tokens=False)

    # Create a tensor to hold the one-hot encodings
    # Shape: [sequence_length, vocab_size]
    one_hot_encodings = torch.zeros((len(tokens), gpt_vocab_size))

    # Fill the tensor with one-hot encodings
    for i, token_id in enumerate(tokens):
        one_hot_encodings[i, token_id] = 1

    return one_hot_encodings

def recover_text_from_one_hot(one_hot_encodings, tokenizer):
    # Get the token IDs from the one-hot encodings
    token_ids = one_hot_encodings.argmax(dim=-1).tolist()

    # Decode the token IDs to text
    text = tokenizer.decode(token_ids)
    return text

def convert_gpt2_to_clip_onehots(gpt2_onehots, transformation_matrix):
    # Assuming transformation_matrix is a sparse tensor
    # Perform sparse matrix multiplication
    return torch.sparse.mm(gpt2_onehots, transformation_matrix)



def create_sparse_transformation_matrix(tokens_gpt2, tokens_clip, a2b, gpt2_vocab_size, clip_vocab_size):
    # Prepare indices and values for the sparse matrix
    indices = []
    values = []
    
    for gpt2_idx, alignments in enumerate(a2b):
        gpt2_token_id = gpt_tokenizer.convert_tokens_to_ids(tokens_gpt2[gpt2_idx])
        for clip_idx in alignments:
            clip_token_id = clip_tokenizer.convert_tokens_to_ids(tokens_clip[clip_idx])
            indices.append([gpt2_token_id, clip_token_id])
            values.append(1)  # We set the value to 1 to denote alignment
    
    # Convert lists to tensors
    indices = torch.LongTensor(indices).t()  # Transpose to fit COO format
    values = torch.FloatTensor(values)
    
    # Create sparse tensor
    transformation_matrix = torch.sparse.FloatTensor(indices, values, torch.Size([gpt2_vocab_size, clip_vocab_size]))
    return transformation_matrix


def tokenize_and_align(prompt):
    # Tokenize using GPT-2
    tokens_gpt2 = gpt_tokenizer.tokenize(prompt)
    # Tokenize using Stable Diffusion's CLIPTokenizer
    tokens_clip = clip_tokenizer.tokenize(prompt)

    # Get token alignments
    a2b, b2a = tokenizations.get_alignments(tokens_gpt2, tokens_clip)
    
    return tokens_gpt2, tokens_clip, a2b, b2a

def one_hot_to_embeddings(one_hot_encodings, embeddings):
    """
    Convert one-hot encodings to embeddings by matrix multiplication.
    one_hot_encodings: [sequence_length, vocab_size]
    embeddings: [vocab_size, embedding_dim]
    Returns:
    Tensor of shape [sequence_length, embedding_dim]
    """
    return torch.matmul(one_hot_encodings, embeddings)

def map_prompt_to_clip(one_hot_gpt2):
    prompt = recover_text_from_one_hot(one_hot_gpt2, gpt_tokenizer)
    tokens_gpt2, tokens_clip, a2b, b2a = tokenize_and_align(prompt)
    gpt2_vocab_size = len(gpt_tokenizer.get_vocab())
    clip_vocab_size = len(clip_tokenizer.get_vocab())
    transformation_matrix = create_sparse_transformation_matrix(tokens_gpt2, tokens_clip, a2b, gpt2_vocab_size, clip_vocab_size)
    one_hot_gpt2 = get_gpt2_logits(prompt)
    one_hot_clip = convert_gpt2_to_clip_onehots(one_hot_gpt2, transformation_matrix)
    diffusion_embeddings = one_hot_to_embeddings(one_hot_clip, clip_embeddings)
    return diffusion_embeddings