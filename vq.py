# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim
        )
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        self.input_shape = inputs.shape
        input_shape = self.input_shape

        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        indices = encoding_indices.reshape(*input_shape[:-1]).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self._num_embeddings,
            device=inputs.device,
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(
            input_shape
        )

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return loss, quantized, perplexity, encodings, indices

    @torch.no_grad()
    def test(self, encoding_indices):
        bs = encoding_indices.shape[0]
        encoding_indices = encoding_indices.view(-1, 1).long()
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self._num_embeddings,
            device=encoding_indices.device,
        )
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(
            bs, *self.input_shape[1:]
        )
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon=1e-5,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim
        )
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(
            torch.Tensor(num_embeddings, self._embedding_dim)
        )
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        self.shape = input_shape

        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        indices = encoding_indices.reshape(*input_shape[:-1])
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self._num_embeddings,
            device=inputs.device,
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(
            input_shape
        )

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )
            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        return (
            loss,
            quantized.permute(0, 3, 1, 2).contiguous(),
            perplexity,
            encodings,
            indices,
        )

    @torch.no_grad()
    def asd(self, indices):
        indices = indices.view(-1, 1)
        encodings = torch.zeros(
            indices.shape[0], self._num_embeddings, device=indices.device
        )
        encodings.scatter_(1, indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(
            self.shape
        )
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized


class VQVAE(nn.Module):
    def __init__(self, enc, dec, emb_num, emb_dim, ccost=0.25):
        super().__init__()
        self._enc = enc
        self._dec = dec
        self._vq_vae = VectorQuantizer(emb_num, emb_dim, ccost)

    def forward(self, x):
        z = self._enc(x)
        vq_loss, quant, _, _, ind = self._vq_vae(z)
        asd = z.clone().detach()
        x_r = self._dec(quant)
        return vq_loss, x_r, asd, ind

    @torch.no_grad()
    def test(self, x):
        o = self._vq_vae.test(x)
        o = self._dec(o)
        return o

    def loss(
        self,
        vq_loss,
        x_r,
        asd,
        ind,
        y,
        opt=None,
        loss_fn=nn.functional.mse_loss,
        top_k=None,
    ):
        loss = loss_fn(x_r, y)
        if top_k:
            loss = loss_fn(x_r, y, reduce=False)
            loss = loss.mean(axis=list(range(len(loss.shape)))[1:])
            loss, _ = torch.topk(loss, top_k)
            loss = loss.mean()
        else:
            loss = loss_fn(x_r, y)
        loss += vq_loss
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item()


class VAE(nn.Module):
    def __init__(self, enc, dec, emb_num, emb_dim, ccost=0.25):
        super().__init__()
        self._enc = enc
        self._dec = dec
        # self._vq_vae = VectorQuantizer(emb_num, emb_dim, ccost)

    def forward(self, x):
        z = self._enc(x)
        # vq_loss, quant, _, _, ind = self._vq_vae(z)
        asd = z.clone().detach()
        x_r = self._dec(z)
        return None, x_r, asd, None

    @torch.no_grad()
    def test(self, x):
        # o = self._vq_vae.test(x)
        o = self._dec(x)
        return o

    def loss(
        self,
        vq_loss,
        x_r,
        asd,
        ind,
        y,
        opt=None,
        loss_fn=nn.functional.mse_loss,
        top_k=None,
    ):
        loss = loss_fn(x_r, y)
        if top_k:
            loss = loss_fn(x_r, y, reduce=False)
            loss = loss.mean(axis=list(range(len(loss.shape)))[1:])
            loss, _ = torch.topk(loss, top_k)
            loss = loss.mean()
        else:
            loss = loss_fn(x_r, y)
        # loss += vq_loss
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item()
