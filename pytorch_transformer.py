from torch import nn


class Pytorch_Transformer(nn.Module):
    def __init__(self, n_vocab, d_model, embedding_dim, num_layers, n_head, dim_feedforward, dropout):
        super(Pytorch_Transformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim
        )
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout

        )

        self.fc = nn.Linear(self.d_model, n_vocab)

    def forward(self, src, tgt):
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)
        output = self.transformer(src_embed, tgt_embed)
        logits = self.fc(output)

        return logits