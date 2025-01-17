from typing import List, Dict, Union, Optional

import logging
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch.nn import DataParallel
from torch.utils.data.sampler import SequentialSampler

from haystack.schema import Document
from haystack.document_stores import BaseDocumentStore
from haystack.nodes.retriever.base import BaseRetriever
from haystack.modeling.model.tokenization import Tokenizer
from haystack.modeling.model.language_model import LanguageModel
from haystack.modeling.model.biadaptive_hash_model import BiAdaptiveHashModel
from haystack.modeling.model.prediction_head import TextSimilarityHead, BinarySimilarityHead
from haystack.modeling.data_handler.processor import TextSimilarityProcessor
from haystack.modeling.data_handler.data_silo import DataSilo
from haystack.modeling.data_handler.dataloader import NamedDataLoader
from haystack.modeling.model.optimization import initialize_optimizer
from haystack.modeling.training.base import Trainer, EarlyStopping
from haystack.modeling.utils import initialize_device_settings
from copy import deepcopy
import math

logger = logging.getLogger(__name__)


class BinaryPassageRetriever(BaseRetriever):
    """
    Retriever that uses a bi-encoder (one transformer for query, one transformer for passage).
    See the original paper for more details:
    Karpukhin, Vladimir, et al. (2020): "Dense Passage Retrieval for Open-Domain Question Answering."
    (https://arxiv.org/abs/2004.04906).
    """

    def __init__(
        self,
        document_store: BaseDocumentStore,
        query_embedding_model: Union[Path, str] = "facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model: Union[Path, str] = "facebook/dpr-ctx_encoder-single-nq-base",
        model_version: Optional[str] = None,
        max_seq_len_query: int = 64,
        max_seq_len_passage: int = 256,
        top_k: int = 10,
        use_gpu: bool = True,
        batch_size: int = 16,
        embed_title: bool = True,
        use_fast_tokenizers: bool = True,
        infer_tokenizer_classes: bool = False,
        similarity_function: str = "dot_product",
        global_loss_buffer_size: int = 150000,
        progress_bar: bool = True,
        devices: Optional[List[Union[int, str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        hashnet_gamma: float=0.1,
        candidates: int=1000,
    ):
        """
        Init the Retriever incl. the two encoder models from a local or remote model checkpoint.
        The checkpoint format matches huggingface transformers' model format

        **Example:**

                ```python
                |    # remote model from FAIR
                |    DensePassageRetriever(document_store=your_doc_store,
                |                          query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                |                          passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base")
                |    # or from local path
                |    DensePassageRetriever(document_store=your_doc_store,
                |                          query_embedding_model="model_directory/question-encoder",
                |                          passage_embedding_model="model_directory/context-encoder")
                ```

        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param query_embedding_model: Local path or remote name of question encoder checkpoint. The format equals the
                                      one used by hugging-face transformers' modelhub models
                                      Currently available remote names: ``"facebook/dpr-question_encoder-single-nq-base"``
        :param passage_embedding_model: Local path or remote name of passage encoder checkpoint. The format equals the
                                        one used by hugging-face transformers' modelhub models
                                        Currently available remote names: ``"facebook/dpr-ctx_encoder-single-nq-base"``
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param max_seq_len_query: Longest length of each query sequence. Maximum number of tokens for the query text. Longer ones will be cut down."
        :param max_seq_len_passage: Longest length of each passage/context sequence. Maximum number of tokens for the passage text. Longer ones will be cut down."
        :param top_k: How many documents to return per query.
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param batch_size: Number of questions or passages to encode at once. In case of multiple gpus, this will be the total batch size.
        :param embed_title: Whether to concatenate title and passage to a text pair that is then used to create the embedding.
                            This is the approach used in the original paper and is likely to improve performance if your
                            titles contain meaningful information for retrieval (topic, entities etc.) .
                            The title is expected to be present in doc.meta["name"] and can be supplied in the documents
                            before writing them to the DocumentStore like this:
                            {"text": "my text", "meta": {"name": "my title"}}.
        :param use_fast_tokenizers: Whether to use fast Rust tokenizers
        :param infer_tokenizer_classes: Whether to infer tokenizer class from the model config / name.
                                        If `False`, the class always loads `DPRQuestionEncoderTokenizer` and `DPRContextEncoderTokenizer`.
        :param similarity_function: Which function to apply for calculating the similarity of query and passage embeddings during training.
                                    Options: `dot_product` (Default) or `cosine`
        :param global_loss_buffer_size: Buffer size for all_gather() in DDP.
                                        Increase if errors like "encoded data exceeds max_size ..." come up
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param devices: List of GPU devices to limit inference to certain GPUs and not use all available ones (e.g. ["cuda:0"]).
                        As multi-GPU training is currently not implemented for DPR, training will only use the first device provided in this list.
        :param use_auth_token:  API token used to download private models from Huggingface. If this parameter is set to `True`,
                                the local token will be used, which must be previously created via `transformer-cli login`.
                                Additional information can be found here https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        super().__init__()

        if devices is not None:
            self.devices = devices
        else:
            self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)

        if batch_size < len(self.devices):
            logger.warning("Batch size is less than the number of devices. All gpus will not be utilized.")

        self.document_store = document_store
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.top_k = top_k
        self.hashnet_gamma = hashnet_gamma
        self.candidates = candidates

        if document_store is None:
            logger.warning(
                "DensePassageRetriever initialized without a document store. "
                "This is fine if you are performing DPR training. "
                "Otherwise, please provide a document store in the constructor."
            )

        self.infer_tokenizer_classes = infer_tokenizer_classes
        tokenizers_default_classes = {"query": "DPRQuestionEncoderTokenizer", "passage": "DPRContextEncoderTokenizer"}
        if self.infer_tokenizer_classes:
            tokenizers_default_classes["query"] = None  # type: ignore
            tokenizers_default_classes["passage"] = None  # type: ignore

        # Init & Load Encoders
        self.query_tokenizer = Tokenizer.load(
            pretrained_model_name_or_path=query_embedding_model,
            revision=model_version,
            do_lower_case=True,
            use_fast=use_fast_tokenizers,
            tokenizer_class=tokenizers_default_classes["query"],
            use_auth_token=use_auth_token,
        )
        self.query_encoder = LanguageModel.load(
            pretrained_model_name_or_path=query_embedding_model,
            revision=model_version,
            language_model_class="DPRQuestionEncoder",
            use_auth_token=use_auth_token,
        )
        self.passage_tokenizer = Tokenizer.load(
            pretrained_model_name_or_path=passage_embedding_model,
            revision=model_version,
            do_lower_case=True,
            use_fast=use_fast_tokenizers,
            tokenizer_class=tokenizers_default_classes["passage"],
            use_auth_token=use_auth_token,
        )
        self.passage_encoder = LanguageModel.load(
            pretrained_model_name_or_path=passage_embedding_model,
            revision=model_version,
            language_model_class="DPRContextEncoder",
            use_auth_token=use_auth_token,
        )

        self.processor = TextSimilarityProcessor(
            query_tokenizer=self.query_tokenizer,
            passage_tokenizer=self.passage_tokenizer,
            max_seq_len_passage=max_seq_len_passage,
            max_seq_len_query=max_seq_len_query,
            label_list=["hard_negative", "positive"],
            metric="text_similarity_metric",
            embed_title=embed_title,
            num_hard_negatives=0,
            num_positives=1,
        )
        prediction_head = BinarySimilarityHead(
            similarity_function=similarity_function, global_loss_buffer_size=global_loss_buffer_size
        )
        self.model = BiAdaptiveHashModel(
            language_model1=self.query_encoder,
            language_model2=self.passage_encoder,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=0.1,
            lm1_output_types=["per_sequence"],
            lm2_output_types=["per_sequence"],
            device=str(self.devices[0]),
            hashnet_gamma=self.hashnet_gamma,
        )

        self.model.connect_heads_with_processor(self.processor.tasks, require_labels=False)

        if len(self.devices) > 1:
            self.model = DataParallel(self.model, device_ids=self.devices)


    def retrieve(
        self,
        query: str,
        filters: dict = None,
        top_k: Optional[int] = None,
        index: str = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]: # TODO - should change so it uses query binary and dense representation. Generation and re-ranking
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        if top_k is None:
            top_k = self.top_k
        if not self.document_store:
            logger.error("Cannot perform retrieve() since DensePassageRetriever initialized with document_store=None")
            return []
        if index is None:
            index = self.document_store.index
        # query_embeddings, query_emb_binary = self.embed_queries(texts=[query])
        query_embeddings, query_emb_binary = self.embed_one_query(query=[{"query": query}])

        if self.candidates>len(self.document_store.all_documents):
            self.candidates=len(self.document_store.all_documents)

        bin_query_embeddings = np.packbits(np.where(query_emb_binary > 0, 1, 0), axis=-1).reshape(1, -1)

        raw_index = self.document_store.faiss_indexes[index]
        _, ids_arr = raw_index.search(bin_query_embeddings, self.candidates)

        passage_embeddings = np.vstack(
            [np.unpackbits(raw_index.reconstruct(int(id_))) for id_ in ids_arr.reshape(-1)]
        )

        passage_embeddings = passage_embeddings.reshape(
            query_embeddings.shape[0], passage_embeddings.shape[0], query_embeddings.shape[1]
        )
        passage_embeddings = passage_embeddings.astype(np.float32)

        passage_embeddings = passage_embeddings * 2 - 1 # Turns 1, 0 into 1, -1

        scores_arr = np.einsum("ijk,ik->ij", passage_embeddings, query_embeddings)
        sorted_indices = np.argsort(-scores_arr, axis=1)[0][:top_k]
        documents = [self.document_store.all_documents[ids_arr[0][i]] for i in sorted_indices]
        # documents = self.document_store.query_by_embedding(
        #     query_emb=query_emb_binary[0], top_k=top_k, filters=filters, index=index, headers=headers,
        #     return_embedding=False
        # )
        return documents

    def embed_one_query(self, query):
        dataset, tensor_names, _, baskets = self.processor.dataset_from_dicts(
            query, indices=[i for i in range(len(query))], return_baskets=True
        )

        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )

        # self.model.eval() # Changed to eval() method to avoid doing it every query

        for batch in data_loader:
            batch = {key: batch[key].to(self.devices[0]) for key in batch}
            with torch.no_grad():
                binary_query, binary_passage, dense_query = self.model.forward(**batch)[0]
                return dense_query.cpu().numpy(), binary_query.cpu().numpy()
        return None

    def _get_predictions(self, dicts):
        """
        Feed a preprocessed dataset to the model and get the actual predictions (forward pass + formatting).

        :param dicts: list of dictionaries
        examples:[{'query': "where is florida?"}, {'query': "who wrote lord of the rings?"}, ...]
                [{'passages': [{
                    "title": 'Big Little Lies (TV series)',
                    "text": 'series garnered several accolades. It received..',
                    "label": 'positive',
                    "external_id": '18768923'},
                    {"title": 'Framlingham Castle',
                    "text": 'Castle on the Hill "Castle on the Hill" is a song by English..',
                    "label": 'positive',
                    "external_id": '19930582'}, ...]
        :return: dictionary of embeddings for "passages" and "query"
        """
        dataset, tensor_names, _, baskets = self.processor.dataset_from_dicts(
            dicts, indices=[i for i in range(len(dicts))], return_baskets=True
        )

        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )
        all_embeddings = {"query": [], "passages": [], "dense_query": []}
        self.model.eval()

        # When running evaluations etc., we don't want a progress bar for every single query
        if len(dataset) == 1:
            disable_tqdm = True
        else:
            disable_tqdm = not self.progress_bar

        with tqdm(
            total=len(data_loader) * self.batch_size,
            unit=" Docs",
            desc=f"Create embeddings",
            position=0,
            leave=False,
            disable=disable_tqdm,
        ) as progress_bar:
            for batch in data_loader:
                batch = {key: batch[key].to(self.devices[0]) for key in batch}

                # get logits
                with torch.no_grad():
                    binary_query, binary_passage, dense_query = self.model.forward(**batch)[0]
                    if binary_query is not None:
                        all_embeddings["query"].append(binary_query.cpu().numpy())
                    if binary_passage is not None:
                        all_embeddings["passages"].append(binary_passage.cpu().numpy())
                    if dense_query is not None:
                        all_embeddings["dense_query"].append(dense_query.cpu().numpy())
                progress_bar.update(self.batch_size)

        if all_embeddings["passages"]:
            all_embeddings["passages"] = np.concatenate(all_embeddings["passages"])
        if all_embeddings["query"]:
            all_embeddings["query"] = np.concatenate(all_embeddings["query"])
        if all_embeddings["dense_query"]:
            all_embeddings["dense_query"] = np.concatenate(all_embeddings["dense_query"])
        return all_embeddings

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for a list of queries using the query encoder

        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        queries = [{"query": q} for q in texts]
        result = self._get_predictions(queries)
        return result["dense_query"], result["query"]

    def embed_documents(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings for a list of documents using the passage encoder

        :param docs: List of Document objects used to represent documents / passages in a standardized way within Haystack.
        :return: Embeddings of documents / passages shape (batch_size, embedding_dim)
        """
        if self.processor.num_hard_negatives != 0:
            logger.warning(
                f"'num_hard_negatives' is set to {self.processor.num_hard_negatives}, but inference does "
                f"not require any hard negatives. Setting num_hard_negatives to 0."
            )
            self.processor.num_hard_negatives = 0

        passages = [
            {
                "passages": [
                    {
                        "title": d.meta["name"] if d.meta and "name" in d.meta else "",
                        "text": d.content,
                        "label": d.meta["label"] if d.meta and "label" in d.meta else "positive",
                        "external_id": d.id,
                    }
                ]
            }
            for d in docs
        ]
        embeddings = self._get_predictions(passages)["passages"]

        return embeddings

    def train(
        self,
        data_dir: str,
        train_filename: str,
        dev_filename: str = None,
        test_filename: str = None,
        max_samples: int = None,
        max_processes: int = 128,
        multiprocessing_strategy: Optional[str] = None,
        early_stopping: Optional[EarlyStopping] = None,
        logging_wandb: bool=False,
        checkpoint_every: Optional[int] = None,
        checkpoint_root_dir: Optional[Path] = None,
        from_epoch: int = 0,
        from_step: int = 0,
        checkpoint_on_sigterm: bool = False,
        checkpoints_to_keep: int = 5,
        dev_split: float = 0,
        batch_size: int = 2,
        embed_title: bool = True,
        num_hard_negatives: int = 1,
        num_positives: int = 1,
        n_epochs: int = 3,
        evaluate_every: int = 1000,
        n_gpu: int = 1,
        learning_rate: float = 1e-5,
        epsilon: float = 1e-08,
        weight_decay: float = 0.0,
        num_warmup_steps: int = 100,
        grad_acc_steps: int = 1,
        use_amp: str = None,
        optimizer_name: str = "AdamW",
        optimizer_correct_bias: bool = True,
        save_dir: str = "../saved_models/dpr",
        query_encoder_save_dir: str = "query_encoder",
        passage_encoder_save_dir: str = "passage_encoder",
    ):
        """
        train a DensePassageRetrieval model
        :param data_dir: Directory where training file, dev file and test file are present
        :param train_filename: training filename
        :param dev_filename: development set filename, file to be used by model in eval step of training
        :param test_filename: test set filename, file to be used by model in test step after training
        :param max_samples: maximum number of input samples to convert. Can be used for debugging a smaller dataset.
        :param max_processes: the maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
                              It can be set to 1 to disable the use of multiprocessing or make debugging easier.
        :param multiprocessing_strategy: Set the multiprocessing sharing strategy, this can be one of file_descriptor/file_system depending on your OS.
                                         If your system has low limits for the number of open file descriptors, and you can’t raise them,
                                         you should use the file_system strategy.
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :param batch_size: total number of samples in 1 batch of data
        :param embed_title: whether to concatenate passage title with each passage. The default setting in official DPR embeds passage title with the corresponding passage
        :param num_hard_negatives: number of hard negative passages(passages which are very similar(high score by BM25) to query but do not contain the answer
        :param num_positives: number of positive passages
        :param n_epochs: number of epochs to train the model on
        :param evaluate_every: number of training steps after evaluation is run
        :param n_gpu: number of gpus to train on
        :param learning_rate: learning rate of optimizer
        :param epsilon: epsilon parameter of optimizer
        :param weight_decay: weight decay parameter of optimizer
        :param grad_acc_steps: number of steps to accumulate gradient over before back-propagation is done
        :param use_amp: Whether to use automatic mixed precision (AMP) or not. The options are:
                    "O0" (FP32)
                    "O1" (Mixed Precision)
                    "O2" (Almost FP16)
                    "O3" (Pure FP16).
                    For more information, refer to: https://nvidia.github.io/apex/amp.html
        :param optimizer_name: what optimizer to use (default: AdamW)
        :param num_warmup_steps: number of warmup steps
        :param optimizer_correct_bias: Whether to correct bias in optimizer
        :param save_dir: directory where models are saved
        :param query_encoder_save_dir: directory inside save_dir where query_encoder model files are saved
        :param passage_encoder_save_dir: directory inside save_dir where passage_encoder model files are saved
        """
        self.processor.embed_title = embed_title
        self.processor.data_dir = Path(data_dir)
        self.processor.train_filename = train_filename
        self.processor.dev_filename = dev_filename
        self.processor.test_filename = test_filename
        self.processor.max_samples = max_samples
        self.processor.dev_split = dev_split
        self.processor.num_hard_negatives = num_hard_negatives
        self.processor.num_positives = num_positives

        if isinstance(self.model, DataParallel):
            self.model.module.connect_heads_with_processor(self.processor.tasks, require_labels=True)
        else:
            self.model.connect_heads_with_processor(self.processor.tasks, require_labels=True)

        data_silo = DataSilo(
            processor=self.processor,
            batch_size=batch_size,
            distributed=False,
            max_processes=max_processes,
            multiprocessing_strategy=multiprocessing_strategy,
        )

        # 5. Create an optimizer
        self.model, optimizer, lr_schedule = initialize_optimizer(
            model=self.model,
            learning_rate=learning_rate,
            optimizer_opts={
                "name": optimizer_name,
                "correct_bias": optimizer_correct_bias,
                "weight_decay": weight_decay,
                "eps": epsilon,
            },
            schedule_opts={"name": "LinearWarmup", "num_warmup_steps": num_warmup_steps},
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=n_epochs,
            grad_acc_steps=grad_acc_steps,
            device=self.devices[0],  # Only use first device while multi-gpu training is not implemented
            use_amp=use_amp,
        )

        self.model.prediction_heads[0].n_passages = num_positives+num_hard_negatives

        # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=self.devices[0],  # Only use first device while multi-gpu training is not implemented
            use_amp=use_amp,
            early_stopping=early_stopping,
            logging_wandb=logging_wandb,
            eval_report=False,
            checkpoint_every=checkpoint_every,
            checkpoint_root_dir=checkpoint_root_dir,
            from_epoch=from_epoch,
            from_step=from_step,
            checkpoint_on_sigterm=checkpoint_on_sigterm,
            checkpoints_to_keep=checkpoints_to_keep,
            query_encoder_save_dir=query_encoder_save_dir,
            passage_encoder_save_dir=passage_encoder_save_dir,
            retriever=self
        )

        # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
        trainer.train()

        self.model.save(Path(save_dir), lm1_name=query_encoder_save_dir, lm2_name=passage_encoder_save_dir)
        self.query_tokenizer.save_pretrained(f"{save_dir}/{query_encoder_save_dir}")
        self.passage_tokenizer.save_pretrained(f"{save_dir}/{passage_encoder_save_dir}")

        if len(self.devices) > 1 and not isinstance(self.model, DataParallel):
            self.model = DataParallel(self.model, device_ids=self.devices)

    def save(
        self,
        save_dir: Union[Path, str],
        query_encoder_dir: str = "query_encoder",
        passage_encoder_dir: str = "passage_encoder",
    ):
        """
        Save DensePassageRetriever to the specified directory.

        :param save_dir: Directory to save to.
        :param query_encoder_dir: Directory in save_dir that contains query encoder model.
        :param passage_encoder_dir: Directory in save_dir that contains passage encoder model.
        :return: None
        """
        save_dir = Path(save_dir)
        self.model.save(save_dir, lm1_name=query_encoder_dir, lm2_name=passage_encoder_dir)
        save_dir = str(save_dir)
        self.query_tokenizer.save_pretrained(save_dir + f"/{query_encoder_dir}")
        self.passage_tokenizer.save_pretrained(save_dir + f"/{passage_encoder_dir}")

    @classmethod
    def load(
        cls,
        load_dir: Union[Path, str],
        document_store: BaseDocumentStore,
        max_seq_len_query: int = 64,
        max_seq_len_passage: int = 256,
        use_gpu: bool = True,
        batch_size: int = 16,
        embed_title: bool = True,
        use_fast_tokenizers: bool = True,
        similarity_function: str = "dot_product",
        query_encoder_dir: str = "query_encoder",
        passage_encoder_dir: str = "passage_encoder",
        infer_tokenizer_classes: bool = False,
    ):
        """
        Load DensePassageRetriever from the specified directory.
        """
        load_dir = Path(load_dir)
        dpr = cls(
            document_store=document_store,
            query_embedding_model=Path(load_dir) / query_encoder_dir,
            passage_embedding_model=Path(load_dir) / passage_encoder_dir,
            max_seq_len_query=max_seq_len_query,
            max_seq_len_passage=max_seq_len_passage,
            use_gpu=use_gpu,
            batch_size=batch_size,
            embed_title=embed_title,
            use_fast_tokenizers=use_fast_tokenizers,
            similarity_function=similarity_function,
            infer_tokenizer_classes=infer_tokenizer_classes,
        )
        logger.info(f"DPR model loaded from {load_dir}")

        return dpr