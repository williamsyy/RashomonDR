import torch
from torch import nn
from torch.nn import functional as F

class SinLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class ANNLayer(nn.Module):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        bias: bool = True,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
        activation: str = "relu",
        residual_connection: bool = False,
    ):
        super().__init__()
        self.bias = bias
        self.linear = nn.Linear(
            in_dims, out_dims, bias=bias, device=device, dtype=dtype
        )
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "sin":
            self.activation = SinLayer()
        elif activation is not None:
            raise ValueError(f"Activation {activation} is not yet supported.")
        else:
            self.activation = None
        self.residual_connection = residual_connection and in_dims == out_dims

    def eye_init(self):
        nn.init.eye_(self.linear.weight)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        output = self.linear(x)
        if self.activation is not None:
            output = self.activation(output)
        if self.residual_connection:
            return output + x
        return output


class ANN(nn.Module):
    """A simple Feed-Forward Neural Network that serve as the backbone."""

    def __init__(
        self,
        layer_size,
        eye_init: bool = False,
        bias: bool = True,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
        activation: str = "relu",
        residual_connection: bool = False,
    ):
        super().__init__()
        layers = []
        num_layers = len(layer_size) - 1
        self.activation = activation
        for i in range(num_layers):
            layer = nn.Linear(layer_size[i], layer_size[i + 1])
            layer = ANNLayer(
                in_dims=layer_size[i],
                out_dims=layer_size[i + 1],
                bias=bias,
                device=device,
                dtype=dtype,
                activation=None if i == num_layers - 1 else self.activation,
                residual_connection=residual_connection,
            )
            if eye_init:
                layer.eye_init()
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self._output_per_layer = False

    def set_output_per_layer(self, value):
        self._output_per_layer = value

    def forward(self, x: torch.Tensor):
        if self._output_per_layer:
            outputs = []
            for layer in self.layers:
                x = layer(x)
                outputs.append(x.detach())
            return outputs
        for layer in self.layers:
            x = layer(x)
        return x

class PolynomialNN(nn.Module):
    def __init__(self, input_dim, output_dim, degree=2):
        super().__init__()
        self.degree = degree  # The maximum degree of polynomial terms
        # choose degree from input_dim + degree
        import math
        # self.linear = nn.Linear(math.comb(input_dim + degree, degree)-1, output_dim)
        self.linear = ANNLayer(
                in_dims=math.comb(input_dim + degree, degree)-1,
                out_dims=output_dim,
                bias=True,
                device=torch.device("cuda"),
                dtype=torch.float32,
                activation=None,
                residual_connection=False,
            )


    def forward(self, X):
        # X is expected to be of shape (N, D)
        N, D = X.shape
        poly_features = [X]
        from itertools import combinations_with_replacement
        for d in range(2, self.degree + 1):
            for comb in combinations_with_replacement(range(D), d):
                new_feature = torch.prod(X[:, comb], dim=1, keepdim=True)
                poly_features.append(new_feature)
        
        X = torch.cat(poly_features, dim=1)

        # apply the linear layer to be the output
        return self.linear(X)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=10, activation="relu"):
        super().__init__()
        self.encoder = ANN(
            layer_size=[input_dim, hidden_dim, output_dim],
            activation=activation,
        )
        self.decoder = ANN(
            layer_size=[output_dim, hidden_dim, input_dim],
            activation=activation,
        )
    
    def forward(self, x):
        embedding_x = self.encoder(x)
        x = self.decoder(x)
        return embedding_x, x

class ParamPaCMAP(nn.Module):
    def __init__(
        self,
        input_dims: int = 100,
        output_dims: int = 2,
        model_dict: dict = {},
        n_samples: int = None,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.model_dict = model_dict
        self._output_per_layer = False
        self.n_classes = model_dict.get("n_classes", 2)
        self.is_parametric = True
        self.input_dims = input_dims
        self.backbone = self.get_backbone(input_dims, output_dims, model_dict)
        if model_dict["backbone"] == "embedding":
            self.is_parametric = False

    def set_output_per_layer(self, value: bool):
        self._output_per_layer = value
        if value and not (
            isinstance(self.backbone, ANN) or isinstance(self.backbone, nn.Embedding)
        ):
            raise ValueError("CNN does not support output per layer yet.")
        if isinstance(self.backbone, ANN):
            self.backbone.set_output_per_layer(value)

    def get_backbone(
        self, input_dims=100, output_dims=2, model_dict: dict = {}
    ) -> nn.Module:
        backbone = model_dict["backbone"]

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Fully Connected Layers
        if backbone == "ANN":
            eye_init = model_dict.get("eye_init", False)
            dtype = model_dict.get("dtype", torch.float32)
            residual = model_dict.get("residual", False)
            bias = model_dict.get("bias", True)
            activation = model_dict.get("activation", "relu")
            layer_size = [input_dims] + model_dict["layer_size"] + [output_dims]
            model = ANN(
                layer_size=layer_size,
                eye_init=eye_init,
                bias=bias,
                device=device,
                dtype=dtype,
                activation=activation,
                residual_connection=residual,
            )
            return model
        elif backbone == "CNN":
            module_list = []
            for i in range(len(model_dict["conv_size"])):
                # InChannel, OutChannel, Size
                module_list.append(
                    nn.Conv2d(
                        model_dict["conv_size"][i][0],
                        model_dict["conv_size"][i][1],
                        model_dict["conv_size"][i][2],
                        padding="same",
                    )
                )
                module_list.append(nn.BatchNorm2d(model_dict["conv_size"][i][1]))
                module_list.append(nn.ReLU())
            module_list.append(nn.Flatten())  # Flatten the intermediate layer
            layer_size = model_dict["layer_size"] + [output_dims]
            for i in range(len(layer_size) - 1):
                module_list.append(nn.Linear(layer_size[i], layer_size[i + 1]))
                module_list.append(nn.ReLU())
            module_list = module_list[:-1]
            return nn.Sequential(*module_list)
        elif backbone == "embedding":
            assert self.n_samples is not None
            embedding = nn.Embedding(self.n_samples, output_dims)
            # set up the initialization as N(0,0.001)
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.0001)
            return embedding
        else:
            raise NotImplementedError("Unsupported model backbone style.")

    def forward(self, sample):
        embedding = self.backbone(sample)
        return embedding

import torch
import torch.nn.functional as F

def dcg_at_k(relevance, k):
    """
    Compute DCG at rank k.
    """
    order = torch.argsort(relevance, descending=True)  # Sort by relevance
    gain = 2 ** relevance[order] - 1  # Gain transformation
    discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float, device=relevance.device))
    return torch.sum(gain[:k] / discounts[:k])

def ndcg_loss(y_pred, x_values, k=None):
    """
    Computes the NDCG loss for ranking regression outputs.
    
    Args:
        y_pred (torch.Tensor): Model's predicted scores.
        x_values (torch.Tensor): True ranking variable (e.g., ground-truth x values).
        k (int, optional): Cut-off rank (default: full list).
        
    Returns:
        torch.Tensor: NDCG loss.
    """
    if k is None:
        k = len(y_pred)  # Use full ranking list if k is not specified

    # Compute DCG for predicted ranking
    dcg = dcg_at_k(y_pred, k)
    
    # Compute Ideal DCG (IDCG) from ground-truth x-values
    idcg = dcg_at_k(x_values, k)

    # Compute NDCG (normalize DCG by IDCG)
    ndcg = dcg / (idcg + 1e-6)  # Avoid division by zero

    # Convert to loss (1 - NDCG so that we minimize it)
    return 1 - ndcg

def ranking_loss(x_values, y_pred, margin=1.0):
    """
    Custom ranking loss to ensure that predicted values respect the ranking of x_values.

    Args:
        x_values (torch.Tensor): True ranking variable (1D tensor).
        y_pred (torch.Tensor): Predicted regression values (1D tensor).
        margin (float): Margin to enforce separation (optional).

    Returns:
        torch.Tensor: Computed ranking loss.
    """
    loss = torch.tensor(0.0, device=y_pred.device)
    count = 0  # To normalize the loss

    for i in range(len(x_values)):
        for j in range(len(x_values)):
            if x_values[i] > x_values[j]:  # Ensure proper ranking order
                loss += F.relu(margin + y_pred[j] - y_pred[i])  # Ranking violation
                count += 1

    return loss / (count + 1e-6)  # Normalize the loss

def ndcg_loss(y_pred, y_true, k=10):
    """
    Calculates the NDCG loss.

    Args:
        y_pred (torch.Tensor): Predicted scores, shape (batch_size, list_length).
        y_true (torch.Tensor): Ground truth relevance scores, shape (batch_size, list_length).
        k (int): Cutoff rank.

    Returns:
        torch.Tensor: NDCG loss, a scalar.
    """
    relevant_elements_true = torch.sum(torch.where(y_true > 0, 1, 0), dim=-1, keepdim=True)
    k = torch.minimum(relevant_elements_true, torch.tensor(k))
    _, indices = torch.topk(y_pred, k=k.max(), dim=-1)
    
    y_true_sorted = torch.gather(y_true, dim=-1, index=indices)
    
    discounts = 1 / torch.log2(torch.arange(k.max()) + 2.0)
    discounts = discounts.to(y_true.device)
    
    dcg = (y_true_sorted * discounts[:k.max()]).sum(dim=-1)
    
    ideal_y_true_sorted, _ = torch.topk(y_true, k=k.max(), dim=-1)
    ideal_dcg = (ideal_y_true_sorted * discounts[:k.max()]).sum(dim=-1)
    
    ndcg = dcg / ideal_dcg
    ndcg[torch.isnan(ndcg)] = 0.0
    return 1 - ndcg.mean()


class PaCMAPLoss(nn.Module):
    def __init__(
        self,
        weight,
        thresholds=[None, None, None],
        exponents=[2, 2, 2],
        consts=[10, 1, 10000],
        label_weight=1e-1,
        task_type="classification",
        n_classes=2,
    ) -> None:
        super().__init__()
        self.weight = weight
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.nnloss = NNLoss(
            weight[0],
            threshold=thresholds[0],
            exponent=exponents[0],
            const=consts[0],
            device=device,
        )
        self.fploss = FPLoss(
            weight[1],
            threshold=thresholds[1],
            exponent=exponents[1],
            const=consts[1],
            device=device,
        )
        self.mnloss = MNLoss(
            weight[2],
            threshold=thresholds[2],
            exponent=exponents[2],
            const=consts[2],
            device=device,
        )
        self.task_type = task_type
        self.label_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        self.label_weight = label_weight
        self.n_classes = n_classes
        self.device = device

    def forward(
        self, basis, nn_pairs, fp_pairs, mn_pairs, predicted_labels=None, labels=None, fp=None, model_input=None, epoch=None
    ):
        # Based on the labels, generate the outputs
        nn_loss = self.nnloss(basis, nn_pairs)
        fp_loss = self.fploss(basis, fp_pairs)
        mn_loss = self.mnloss(basis, mn_pairs)
        loss = nn_loss + fp_loss + mn_loss
        if labels is not None:
            if self.task_type == "concept":
                vector = basis[:,:,0] - fp_pairs[:,:,0]
                label_difference = labels.unsqueeze(1) - fp.reshape(labels.shape[0], -1)
                label_mask = ((labels.unsqueeze(1) != -1) & (fp.reshape(labels.shape[0], -1) != -1)).view(-1)
                # set up the float32
                vector = vector.float()
                label_difference = label_difference.float()
                # standardize the vector
                vector = (vector - torch.mean(vector, dim=1, keepdim=True)) / (torch.std(vector, dim=1, keepdim=True)+1e-10)
                # standardize the label difference
                label_difference = (label_difference - torch.mean(label_difference, dim=1, keepdim=True)) / (torch.std(label_difference, dim=1, keepdim=True)+1e-10)
                # # ignore if the labels or fp is -1
                loss += self.label_weight * torch.mean((vector.view(-1)[label_mask] - label_difference.view(-1)[label_mask])**2)
                return loss  
            elif self.task_type == "pca":
                '''
                Tempt one: directly use the ratio of the distance between the fp pairs and the basis as the regualarization term
                '''
                vector = basis - fp_pairs
                pca_vector = labels.unsqueeze(1) - fp.reshape(labels.shape[0], -1, 2)
                cosine_similarity = F.cosine_similarity(vector, pca_vector, dim=2)

                loss += self.label_weight * torch.mean((cosine_similarity - 1)**2)
        return loss

    def update_weight(self, weight, const=None) -> None:
        self.weight = weight
        self.nnloss.weight = weight[0]
        self.fploss.weight = weight[1]
        self.mnloss.weight = weight[2]
        if const is not None:
            self.nnloss.const.fill_(const[0])
            self.fploss.const.fill_(const[1])
            self.mnloss.const.fill_(const[2])


class NNLoss(nn.Module):
    """NN Loss of PaCMAP."""

    def __init__(
        self, weight, threshold=None, exponent=2, const=10, device=torch.device("cuda")
    ) -> None:
        super().__init__()
        self.weight = weight
        self.threshold = threshold
        self.exponent = exponent
        self.multiplier = 0
        self.const = torch.tensor(
            [
                const,
            ],
            dtype=torch.float32,
            device=device,
        )

    def forward(self, basis, pair_components):
        diff = pair_components - basis  # N, P, D
        norm = torch.linalg.norm(diff, dim=2)
        d2 = norm**self.exponent + 1  # dist squared
        loss = d2 / (self.const + d2)
        if self.threshold is not None:
            loss = torch.where(loss > self.threshold, loss, 1.0)
        loss = torch.sum(loss, dim=1)
        return self.weight * torch.mean(loss)


class FPLoss(nn.Module):
    """FP Loss of PaCMAP."""

    def __init__(
        self, weight, threshold=None, exponent=2, const=1, device=torch.device("cuda")
    ) -> None:
        super().__init__()
        self.weight = weight
        self.threshold = threshold
        self.exponent = exponent
        self.const = torch.tensor(
            [
                const,
            ],
            dtype=torch.float32,
            device=device,
        )

    def forward(self, basis, pair_components):
        diff = pair_components - basis  # N, P, D
        norm = torch.linalg.norm(diff, dim=2)
        d2 = norm**self.exponent + 1  # dist squared
        loss = self.const / (self.const + d2)
        if self.threshold is not None:
            loss = torch.where(loss < self.threshold, loss, 0.0)
        loss = torch.sum(loss, dim=1)
        return self.weight * torch.mean(loss)


class MNLoss(nn.Module):
    """MN Loss of PaCMAP."""

    def __init__(
        self,
        weight,
        threshold=None,
        exponent=2,
        const=10000,
        device=torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.weight = weight
        self.threshold = threshold
        self.exponent = exponent
        self.const = torch.tensor(
            [
                const,
            ],
            dtype=torch.float32,
            device=device,
        )

    def forward(self, basis, pair_components):
        diff = pair_components - basis  # N, P, D
        norm = torch.linalg.norm(diff, dim=2)
        d2 = norm**self.exponent + 1  # dist squared
        loss = d2 / (self.const + d2)
        if self.threshold is not None:
            loss = torch.where(loss > self.threshold, loss, 1.0)
        loss = torch.sum(loss, dim=1)
        return self.weight * torch.mean(loss)
