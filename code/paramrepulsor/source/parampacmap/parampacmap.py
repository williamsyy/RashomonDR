"""Implementing the ParamPaCMAP Algorithm as a sklearn estimator."""

import time
from typing import Optional, Callable

import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
from sklearn import preprocessing, decomposition
from sklearn.base import BaseEstimator

from parampacmap.models import module, dataset
from parampacmap.utils import data, utils
from parampacmap import training


def pacmap_weight_schedule(epoch):
    if epoch < 100:
        w_mn = 10 * (100 - epoch) + 0.03 * epoch
        w_nn = 2.0
        w_fp = 1.0
    elif epoch < 200:
        w_mn = 3.0
        w_nn = 3.0
        w_fp = 1.0
    else:
        w_mn = 0.0
        w_nn = 1.0
        w_fp = 1.0
    weight = np.array([w_nn, w_fp, w_mn])
    return weight


def pacmap_opmn_weight_schedule(epoch):
    if epoch < 100:
        w_nn = 2.0
        w_fp = 1.0
        w_mn = 10 * (100 - epoch) + 0.03 * epoch
    elif epoch < 200:
        w_nn = 3.0
        w_fp = 1.0
        w_mn = 0.0
    else:
        w_nn = 1.0
        w_fp = 1.0
        w_mn = -4.0
    weight = np.array([w_nn, w_fp, w_mn])
    return weight


def paramrep_weight_schedule(epoch):
    if epoch < 100:
        w_nn = 2.0
        w_fp = 1.0
        w_mn = 0.0
    elif epoch < 200:
        w_nn = 3.0
        w_fp = 1.0
        w_mn = 0.0
    else:
        w_nn = 1.0
        w_fp = 1.0
        w_mn = -4.0
    weight = np.array([w_nn, w_fp, w_mn])
    return weight


def paramrep_weight_schedule2(epoch):
    if epoch < 200:
        w_nn = 3.0
        w_fp = 1.0
        w_mn = 0.0
    else:
        w_nn = 1.0
        w_fp = 1.0
        w_mn = -4.0
    weight = np.array([w_nn, w_fp, w_mn])
    return weight


def paramrep_weight_schedule3(epoch):
    if epoch < 200:
        w_nn = 4.0
        w_fp = 8.0
        w_mn = 0.0
    else:
        w_nn = 1.0
        w_fp = 8.0
        w_mn = -12.0
    weight = np.array([w_nn, w_fp, w_mn])
    return weight


def pacmap_vis_schedule(epoch):
    if epoch < 100:
        w_nn = 4.0
        w_fp = 1.0
        w_mn = 0.0
    elif epoch < 200:
        w_nn = 4.0
        w_fp = 1.0
        w_mn = 0.0
    else:
        w_nn = 1.0
        w_fp = 1.0
        w_mn = -12.0
    weight = np.array([w_nn, w_fp, w_mn])
    return weight


def pacmap_opmn_const_schedule(epoch):
    if epoch < 200:
        w_nn = 10.0
        w_fp = 1.0
        w_mn = 10000.0
    else:
        w_nn = 10.0
        w_fp = 1.0
        w_mn = 1.0
    const = np.array([w_nn, w_fp, w_mn])
    return const


class ParamPaCMAP(BaseEstimator):
    """Parametric PaCMAP implemented with Pytorch."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        n_FP: int = 20,
        n_MN: int = 5,
        distance: str = "euclidean",
        optim_type: str = "Adam",
        lr: float = 1e-3,
        lr_schedule: Optional[bool] = None,
        apply_pca: bool = True,
        apply_scale: Optional[str] = None,
        model_dict: Optional[dict] = utils.DEFAULT_MODEL_DICT,
        intermediate_snapshots: Optional[list] = [],
        loss_weight: Optional[list] = [1, 1, 1],
        batch_size: int = 1024,
        data_reshape: Optional[list] = None,
        num_epochs: int = 450,
        verbose: bool = False,
        weight_schedule: Callable = pacmap_weight_schedule,
        num_workers: int = 1,
        dtype: torch.dtype = torch.float32,
        embedding_init: str = "pca",
        thresholds: list[Optional[float]] = [None, None, None],
        exponents: list[float] = [2, 2, 2],
        loss_coeffs: list[float] = [1.0, 1.0, 1.0],
        use_ns_loader: bool = False,
        use_ibns_loader: bool = False,
        consts: list[float] = [10, 1, 10000],
        const_schedule: Optional[Callable] = None,
        torch_compile: bool = False,
        label_weight: float = 1e-1,
        seed: Optional[int] = None,
        save_pairs: bool = False,
        task_type: str = "concept",
    ):
        super().__init__()
        self.n_components = n_components  # output_dims
        self.n_neighbors = n_neighbors
        self.n_FP = n_FP
        self.n_MN = n_MN
        self.distance = distance
        self.optim_type = optim_type
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.apply_pca = apply_pca
        self.apply_scale = apply_scale
        # Placeholder for the model. The model is initialized during fit.
        self.model = None
        self.model_dict = model_dict
        self.intermediate_snapshots = intermediate_snapshots
        self.loss_weight = loss_weight
        self.batch_size = batch_size
        self.data_reshape = data_reshape
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.weight_schedule = weight_schedule
        self.num_workers = num_workers
        self._dtype = dtype
        self._scaler = None
        self._projector = None
        self.time_profiles = None
        # EXPERIMENTAL
        self.thresholds = thresholds
        self.exponents = exponents
        self.loss_coeffs = np.array(loss_coeffs)
        self.use_ns_loader = use_ns_loader
        self.use_ibns_loader = use_ibns_loader
        assert not (
            use_ns_loader and use_ibns_loader
        ), "NS loader and IBNS loader cannot be used together"
        self.consts = consts
        self.const_schedule = const_schedule
        self.torch_compile = torch_compile

        # (Semi)-Supervised related settings
        self.label_weight = label_weight

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if self._dtype == torch.float32:
            torch.set_float32_matmul_precision("medium")
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if embedding_init not in ["pca", "random"]:
            raise ValueError(
                f"Embedding init mode '{embedding_init}' is not supported."
            )
        self.embedding_init = embedding_init

        # Pair saving
        self.save_pairs = save_pairs
        self._pairs_saved = False
        self.pair_neighbors = None
        self.pair_MN = None
        self.pair_FP = None
        self.task_type = task_type
        self._used_projector = False

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        profile_only: bool = False,
        per_layer: bool = False,
    ) -> None:
        fit_begin = time.perf_counter()
        input_dims = X.shape[1]

        # Data Preprocessing
        if input_dims > 100 and self.apply_pca:
            self._projector = decomposition.PCA(n_components=100)
            self._used_projector = True
            X = self._projector.fit_transform(X)
            input_dims = X.shape[1]
        if self.apply_scale == "standard":
            self._scaler = preprocessing.StandardScaler()
            X = self._scaler.fit_transform(X)
        elif self.apply_scale == "minmax":
            self._scaler = preprocessing.MinMaxScaler()
            X = self._scaler.fit_transform(X)

        self.model = (
            module.ParamPaCMAP(
                input_dims=input_dims,
                output_dims=self.n_components,
                model_dict=self.model_dict,
                n_samples=X.shape[0],
            )
            .to(self.device)
            .to(self._dtype)
        )
        self.loss = module.PaCMAPLoss(
            weight=self.loss_weight,
            thresholds=self.thresholds,
            exponents=self.exponents,
            consts=self.consts,
            label_weight=self.label_weight,
            task_type=self.task_type,
            n_classes = len(np.unique(y)) - 1 if -1 in y else len(np.unique(y))
        ).to(self._dtype)
        self.intermediate_outputs = []

        # Constructing dataloader
        if (self.save_pairs and self._pairs_saved) or self.pair_neighbors is not None:
            pair_neighbors, pair_MN, pair_FP = (
                self.pair_neighbors,
                self.pair_MN,
                self.pair_FP,
            )
        else:
            pair_neighbors, pair_MN, pair_FP, _ = data.generate_pair(
                X,
                n_neighbors=self.n_neighbors,
                n_MN=self.n_MN,
                n_FP=self.n_FP,
                distance=self.distance,
                verbose=False,
            )
            if self.save_pairs:
                self.pair_neighbors = pair_neighbors
                self.pair_MN = pair_MN
                self.pair_FP = pair_FP
                self._pairs_saved = True

        nn_pairs, fp_pairs, mn_pairs = training.convert_pairs(
            pair_neighbors, pair_FP, pair_MN, X.shape[0]
        )

        if self.use_ns_loader:
            if (self.task_type == "pca") or (self.task_type == "concept"):
                train_loader_ctor = dataset.FastRegressionNSDataloader
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")
        elif self.use_ibns_loader:
            train_loader_ctor = dataset.FastIBNSDataloader
        else:
            if (self.task_type == "pca") or (self.task_type == "concept"):
                train_loader_ctor = dataset.FastRegressionDataloader
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

        # For non-parametric version, we will use the indices as the input.
        if not self.model.is_parametric:
            self._init_embedding(X)
            X = np.arange(X.shape[0])
            data_dtype = torch.int32
        else:
            data_dtype = self._dtype

        train_loader = train_loader_ctor(
            data=X,
            labels=y,
            nn_pairs=nn_pairs,
            fp_pairs=fp_pairs,
            mn_pairs=mn_pairs,
            batch_size=self.batch_size,
            device=self.device,
            shuffle=True,
            reshape=self.data_reshape,
            dtype=data_dtype,
        )
        test_set = dataset.TensorDataset(
            data=X, reshape=self.data_reshape, dtype=data_dtype
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=2 * self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

        parameter_set = [{"params": self.model.backbone.parameters()}]
        if self.optim_type == "Adam":
            optimizer = optim.Adam(parameter_set, lr=self.lr)
        elif self.optim_type == "SGD":
            optimizer = optim.SGD(parameter_set, lr=self.lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optim_type}")

        if profile_only:
            epoch_begin = time.perf_counter()
            print(
                f"Time Profile: Before Epoch\n"
                f"Preparation:{(epoch_begin - fit_begin):03.3f}s\n"
            )
            self._tune_weight(epoch=0)
            self._profile_epoch(train_loader, optimizer)
            self._embedding = None
            return

        if self.use_ibns_loader:
            train_func = self._train_epoch_ib
        else:
            train_func = self._train_epoch
        if self.torch_compile:
            train_func = torch.compile(train_func)

        for epoch in range(self.num_epochs):
            if epoch in self.intermediate_snapshots:
                if per_layer:
                    result = self._inference_per_layer(test_loader)
                else:
                    result = self._inference(test_loader)
                self.intermediate_outputs.append(result)
            # Tune the weights
            self._tune_weight(epoch=epoch)

            # Perform training for one epoch
            train_func(train_loader, epoch, optimizer)

        if per_layer:
            self._embedding = self._inference_per_layer(test_loader)
        else:
            self._embedding = self._inference(test_loader)

    def _tune_weight(self, epoch: int):
        """Automatically tune the weight."""
        # Decide weight based on the functions
        weight = self.weight_schedule(epoch) * self.loss_coeffs
        if self.const_schedule is not None:
            const = self.const_schedule(epoch)
        else:
            const = None
        self.loss.update_weight(weight, const)

    def _train_epoch(
        self, train_loader, epoch, optimizer: optim.Optimizer
    ):
        """Perform a single epoch of training."""
        for batch in train_loader:
            optimizer.zero_grad()
            num_items, model_input, fp_label, model_label = batch
            model_output = self.model(model_input)
            basis = model_output[:num_items]
            nn_pairs = model_output[num_items : num_items * (self.n_neighbors + 1)]
            fp_pairs = model_output[
                num_items
                * (self.n_neighbors + 1) : num_items
                * (self.n_neighbors + self.n_FP + 1)
            ]
            mn_pairs = model_output[num_items * (self.n_neighbors + self.n_FP + 1) :]
            basis = torch.unsqueeze(basis, 1)
            nn_pairs = nn_pairs.view(num_items, self.n_neighbors, nn_pairs.shape[1])
            fp_pairs = fp_pairs.view(num_items, self.n_FP, fp_pairs.shape[1])
            mn_pairs = mn_pairs.view(num_items, self.n_MN, mn_pairs.shape[1])
            loss = self.loss(
                basis,
                nn_pairs,
                fp_pairs,
                mn_pairs,
                labels=model_label,
                fp=fp_label
            )
            loss.backward()
            optimizer.step()
        if ((epoch + 1) % 20 == 0 or epoch == 0) and self.verbose:
            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f},",
                flush=True,
            )

    def _train_epoch_ib(
        self, train_loader, epoch, optimizer: optim.Optimizer
    ):
        for batch in train_loader:
            optimizer.zero_grad()
            num_items, model_input, model_label, fp_indices = batch
            model_output = self.model(model_input)
            basis = model_output[:num_items]
            nn_pairs = model_output[num_items : num_items * (self.n_neighbors + 1)]
            mn_pairs = model_output[num_items * (self.n_neighbors + 1) :]
            fp_pairs = torch.index_select(nn_pairs, dim=0, index=fp_indices)
            basis = torch.unsqueeze(basis, 1)
            nn_pairs = nn_pairs.view(num_items, self.n_neighbors, nn_pairs.shape[1])
            fp_pairs = fp_pairs.view(num_items, self.n_FP, fp_pairs.shape[1])
            mn_pairs = mn_pairs.view(num_items, self.n_MN, mn_pairs.shape[1])
            loss = self.loss(
                basis,
                nn_pairs,
                fp_pairs,
                mn_pairs,
                labels=model_label,
            )
            loss.backward()
            optimizer.step()
        if ((epoch + 1) % 20 == 0 or epoch == 0) and self.verbose:
            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f},",
                flush=True,
            )

    def _profile_epoch(
        self, train_loader, optimizer: optim.Optimizer
    ):
        """Perform a single epoch of training with detailed profiling."""
        time_profiles = []
        batch_begin = time.perf_counter()
        for batch in train_loader:
            torch.cuda.synchronize()
            time_dataloader = time.perf_counter()
            optimizer.zero_grad()
            # The pairs are under the format (i, num_pairs, ...)
            num_items, model_input = batch
            # model_input = torch.concat((basis, nn_pairs, fp_pairs, mn_pairs), dim=0)
            torch.cuda.synchronize()
            time_reshape = time.perf_counter()
            # Use the model to perform forward
            model_output = self.model(model_input)
            basis = model_output[:num_items]
            nn_pairs = model_output[num_items : num_items * (self.n_neighbors + 1)]
            fp_pairs = model_output[
                num_items
                * (self.n_neighbors + 1) : num_items
                * (self.n_neighbors + self.n_FP + 1)
            ]
            mn_pairs = model_output[num_items * (self.n_neighbors + self.n_FP + 1) :]
            torch.cuda.synchronize()
            time_forward = time.perf_counter()
            basis = torch.unsqueeze(basis, 1)
            nn_pairs = nn_pairs.view(num_items, self.n_neighbors, nn_pairs.shape[1])
            fp_pairs = fp_pairs.view(num_items, self.n_FP, fp_pairs.shape[1])
            mn_pairs = mn_pairs.view(num_items, self.n_MN, mn_pairs.shape[1])
            loss = self.loss(basis, nn_pairs, fp_pairs, mn_pairs)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            time_backward = time.perf_counter()
            time_series = [
                time_dataloader - batch_begin,
                time_reshape - time_dataloader,
                time_forward - time_reshape,
                time_backward - time_forward,
            ]
            batch_begin = time_backward
            time_profiles.append(time_series)
        self.time_profiles = np.array(time_profiles)
        # Generate a profile report
        time_summary = np.sum(self.time_profiles, axis=0)
        summary_text = (
            f"Time Profile: Sum in Epoch\n"
            f"Dataloader: {time_summary[0]:03.3f}s\n"
            f"Reshape:    {time_summary[1]:03.3f}s\n"
            f"Forward:    {time_summary[2]:03.3f}s\n"
            f"Backward:   {time_summary[3]:03.3f}s\n"
        )
        print(summary_text)

    def _inference(self, test_loader):
        """Perform a pure inference for the model."""
        results = []
        with torch.inference_mode():
            for batch in test_loader:
                result = self.model(batch.to(self.device))
                if isinstance(result, tuple):
                    result = result[0]  # Remove predicted labels
                results.append(result.detach())
            results = torch.concatenate(results)
            results = results.float().cpu().numpy()
        return results

    def _inference_with_prediction(self, test_loader):
        results = []
        probs = []
        with torch.inference_mode():
            for batch in test_loader:
                result, prob = self.model(batch.to(self.device))
                results.append(result.detach())
                probs.append(prob.detach())
            results = torch.concatenate(results)
            results = results.float().cpu().numpy()
            probs = torch.concatenate(probs)
            probs = probs.float().cpu().numpy()
        return results, probs

    def _inference_per_layer(self, test_loader):
        """Perform a pure inference for the model."""
        self.model.set_output_per_layer(True)
        results = []
        with torch.inference_mode():
            for batch in test_loader:
                result = self.model(
                    batch.to(self.device)
                )  # A list of multiple embeddings
                if isinstance(result, tuple):
                    result = result[0]  # Remove predicted labels
                results.append(result)
        all_same_size = all(len(result) == len(results[0]) for result in results)
        assert all_same_size
        num_layers = len(results[0])
        layer_results = []
        for i in range(num_layers):
            sub_result = [result[i] for result in results]
            layer_result = torch.concatenate(sub_result).float().cpu().numpy()
            layer_results.append(layer_result)
        self.model.set_output_per_layer(False)
        return layer_results

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, per_layer: bool = False
    ):
        self.fit(X, y=y, per_layer=per_layer)
        if len(self.intermediate_outputs) == 0:
            return self._embedding
        return self._embedding, self.intermediate_outputs

    def _prepare_test_loader(self, X: np.ndarray) -> torch.utils.data.DataLoader:
        if self.model.is_parametric:
            if self._projector is not None:
                X = self._projector.transform(X)
            if self._scaler is not None:
                X = self._scaler.transform(X)
        else:
            X = np.arange(X.shape[0])
        data_dtype = self._dtype if self.model.is_parametric else torch.int32
        test_set = dataset.TensorDataset(
            data=X, reshape=self.data_reshape, dtype=data_dtype
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=2 * self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return test_loader

    def transform(self, X: np.ndarray, per_layer: bool = False) -> np.ndarray:
        test_loader = self._prepare_test_loader(X)
        if per_layer:
            return self._inference_per_layer(test_loader)
        return self._inference(test_loader)

    def transform_and_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Transform the data and also generate the probability predicted by classifier."""
        test_loader = self._prepare_test_loader(X)
        return self._inference_with_prediction(test_loader)
    


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the labels for the embedding data."""
        test_set = dataset.TensorDataset(
            data=X, reshape=self.data_reshape, dtype=torch.float32
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=2 * self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        results = []
        for batch in test_loader:
            result = self.model.projector(batch.to(self.device))
            results.append(result.detach())
        results = torch.concatenate(results)
        results = results.float().cpu().numpy()
        if self._used_projector:
            results = self._projector.inverse_transform(results)
        return results
        
        

    def _init_embedding(self, X: np.ndarray) -> None:
        with torch.inference_mode():
            state_dict = self.model.backbone.state_dict()
            if self.embedding_init == "pca":
                state_dict["weight"] = (
                    torch.tensor(
                        X[:, : self.n_components],
                        dtype=self._dtype,
                        device=self.device,
                    )
                    * 0.01
                )
            elif self.embedding_init == "random":
                state_dict["weight"] = state_dict["weight"] * 0.0001
            self.model.backbone.load_state_dict(state_dict)