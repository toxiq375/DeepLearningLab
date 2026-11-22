# the core program of the whole model, it defines the model of the project
import torch
import torch.nn as nn

class SingleHead(nn.Module):
    def __init__(self, input_channels, attn_channels):
        # the singlehead model takes inputs, it needs to know the input data's channels
        # also, model needs to know the projecting channels of the attention, so that we can projects the input data 
        super().__init__()
        self.attn_channels = attn_channels
        self.q_proj = nn.Linear(input_channels, attn_channels, bias=False) # constrcut the projection function, with no bias
        self.k_proj = nn.Linear(input_channels, attn_channels, bias=False)
        self.v_proj = nn.Linear(input_channels, attn_channels, bias=False)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        token_size = x.shape[1]
        # input x [B,T,C], B is batch, T is token_size, and C is input_channels_size(embeded size)
        scores = Q @ K.transpose(-2, -1) / (self.attn_channels ** 0.5)
        # masked_allowed = torch.tril(torch.ones(token_size, token_size, dtype=torch.bool, device=x.device))
        # # torch.tril creates a lower triangular matrix of 1, the remaining elements are 0
        # masked_blocked = ~masked_allowed
        # masked_scores = scores.masked_fill(masked_blocked, float("-inf"))
        # ABOVE is the old implement of the masked scores
        mask = torch.triu(
            torch.full_like(scores, float("-inf")), # create a matrix with the same shapes of scores, and its elements are all -inf
            diagonal=1
        ) # torch.triu(,diagonal=1) means keeps the upper triangular and makes the ramaining elements 0
        scores = scores + mask
        A = torch.softmax(scores, dim=-1)
        output = A @ V

        return output

class MultiHead(nn.Module):
    def __init__(self, input_channels, attn_channels, output_channels, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead(input_channels, attn_channels) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*attn_channels, output_channels)

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        # Dont use self.heads(), ModuleList is Not a Function ! So u cant call it
        outputs = torch.cat(outputs, dim=-1)
        # torch.cat() takes a list of SingleHead's output, its shape is [B,T,d_attn], and it plus the last dim, namely the d_attn 
        outputs = self.proj(outputs)
        # so the function proj takes the num_heads*d_attn

        return outputs

class TransformerBlock(nn.Module):
    def __init__(self, input_channels, attn_channels, output_channels, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_channels)
        self.attn = MultiHead(input_channels, attn_channels, output_channels, num_heads)
        self.ln2 = nn.LayerNorm(output_channels)
        self.mlp = nn.Sequential(
            nn.Linear(output_channels, 4*output_channels),
            nn.GELU(),
            nn.Linear(4*output_channels, output_channels)
        )

    def forward(self, x):
        u = x + self.attn(self.ln1(x))
        y = u + self.mlp(self.ln2(u))

        return y

class miniGPT(nn.Module):
    def __init__(
            self, 
            vocab_size,
            context_size,
            embed_dim,
            input_channels,
            attn_channels,
            output_channels,
            num_heads,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_size, embed_dim)
        # nn.Embedding dont takes input, it creates a embedding table, its shape is (size, embedding_dims)
        self.attn_block = TransformerBlock(input_channels, attn_channels, output_channels, num_heads)
        # Here we make the dim of the embed_dim and the input_channels of the attn. If Not, we need to add a new projection head to project the embed_dim into attention's input_channels
        self.layernorm = nn.LayerNorm(output_channels)
        self.linear_head = nn.Linear(output_channels, vocab_size)

    def forward(self, x):
        token_embeddings = self.token_embedding(x)
        pos_embeddings = self.pos_embedding(torch.arange(x.shape[1], device=x.device))
        # when inference, the input token_size may not equals context_size, so that we need to query the actual input length, namely x.shape[1], from the pos embedding table
        embeddings = token_embeddings + pos_embeddings
        outputs = self.linear_head(self.layernorm(self.attn_block(embeddings)))
        # remenber to make the enbeddings input, not x

        return outputs
        