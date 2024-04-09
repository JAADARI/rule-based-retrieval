"""Retrieval augmented generation logic."""

import logging
import os
import pathlib
import re
import uuid
from typing import Any, Literal, TypedDict, cast, Dict, List

from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk
from langchain_core.documents import Document
from openai import OpenAI
from openai import AzureOpenAI

from pinecone import (
    Index,
    NotFoundException,
    Pinecone,
    PodSpec,
    ServerlessSpec,
)
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from whyhow_rbr.embedding import generate_embeddings
from whyhow_rbr.exceptions import (
    IndexAlreadyExistsException,
    IndexNotFoundException,
    OpenAIException,
)
from whyhow_rbr.processing import clean_chunks, parse_and_split
from typing import Optional

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_SPEC = ServerlessSpec(cloud="aws", region="us-west-2")


# Custom classes
class PineconeMetadata(BaseModel):
    text: str
    page_number: int
    chunk_number: int
    filename: str
    uuid: str


class PineconeDocument(BaseModel, extra="forbid"):
    """The actual document to be stored in Pinecone.

    Attributes
    ----------
    metadata : PineconeMetadata
        The metadata of the document.

    values : list[float] | None
        The embedding of the document. The None is used when querying
        the index since the values are not needed. At upsert time, the
        values are required.

    id : str | None
        The human-readable identifier of the document. This is generated
        automatically when creating the PineconeDocument unless it is
        provided.

    """

    metadata: PineconeMetadata
    values: list[float] | None = None
    id: str | None = None

    @model_validator(mode="after")
    def generate_human_readable_id(self) -> "PineconeDocument":
        """Generate a human-readable identifier for the document."""
        if self.id is None:
            meta = self.metadata
            hr_id = f"{meta.filename}-{meta.page_number}-{meta.chunk_number}"
            self.id = hr_id

        return self


class PineconeMatch(BaseModel):
    id: str
    score: Optional[float]
    metadata: PineconeMetadata

    class Config:
        extra = "ignore"


class Rule(BaseModel):
    """Retrieval rule.

    The rule is used to filter the documents in the index.

    Attributes
    ----------
    filename : str | None
        The filename of the document.

    uuid : str | None
        The UUID of the document.

    page_numbers : list[int] | None
        The page numbers of the document.

    keywords : list[str] | None
        The keywords to trigger a rule.
    """

    filename: list[str] | None = None
    uuid: str | None = None
    page_numbers: list[int] | None = None
    keywords: list[str] | None = None

    @field_validator("page_numbers", mode="before")
    @classmethod
    def convert_empty_to_none(cls, v: list[int] | None) -> list[int] | None:
        """Convert empty list to None."""
        if v is not None and not v:
            return None
        return v

    def convert_empty_str_to_none(cls, s: list[str] | None) -> list[str] | None:
        """Convert empty string list to None."""
        if s is not None and not s:
            return None
        return s

    def to_filter(self) -> dict[str, dict[str, list[dict[str, Any]]]] | None:
        """Convert rule to Elasticsearch filter format."""
        if not any([self.filename, self.uuid, self.page_numbers]):
            return None

        conditions: list[dict[str, Any]] = []

        if self.filename is not None:
            conditions.append({"term": {"filename": self.filename}})
        if self.uuid is not None:
            conditions.append({"term": {"uuid": self.uuid}})
        if self.page_numbers is not None:
            conditions.append({"terms": {"page_number": self.page_numbers}})

        filter_ = {"bool": {"must": conditions}}
        return filter_


class Input(BaseModel):
    """Example input for the prompt.

    Attributes
    ----------
    question : str
        The question to ask.

    contexts : list[str]
        The contexts to use for answering the question.
    """

    question: str
    contexts: list[str]


class Output(BaseModel):
    """Example output for the prompt.

    Attributes
    ----------
    answer : str
        The answer to the question.

    contexts : list[int]
        The indices of the contexts that were used to answer the question.
    """

    answer: str
    contexts: list[int]


input_example_1 = Input(
    question="What is the capital of France?",
    contexts=[
        "The capital of France is Paris.",
        "The capital of France is not London.",
        "Paris is beautiful and it is also the capital of France.",
    ],
)
output_example_1 = Output(answer="Paris", contexts=[0, 2])

input_example_2 = Input(
    question="What are the impacts of climate change on global agriculture?",
    contexts=[
        "Climate change can lead to more extreme weather patterns, affecting crop yields.",
        "Rising sea levels due to climate change can inundate agricultural lands in coastal areas, reducing arable land.",
        "Changes in temperature and precipitation patterns can shift agricultural zones, impacting food security.",
    ],
)

output_example_2 = Output(
    answer="Variable impacts including altered weather patterns, reduced arable land, shifting agricultural zones, increased pests and diseases, with potential mitigation through technology and sustainable practices",
    contexts=[0, 1, 2],
)

input_example_3 = Input(
    question="How has the concept of privacy evolved with the advent of digital technology?",
    contexts=[
        "Digital technology has made it easier to collect, store, and analyze personal data, raising privacy concerns.",
        "Social media platforms and smartphones often track user activity and preferences, leading to debates over consent and data ownership.",
        "Encryption and secure communication technologies have evolved as means to protect privacy in the digital age.",
        "Legislation like the GDPR in the EU has been developed to address privacy concerns and regulate data handling by companies.",
        "The concept of privacy is increasingly being viewed through the lens of digital rights and cybersecurity.",
    ],
)

output_example_3 = Output(
    answer="Evolving with challenges due to data collection and analysis, changes in legislation, and advancements in encryption and security, amidst ongoing debates over consent and data ownership",
    contexts=[0, 1, 2, 3, 4],
)

# Custom types
Metric = Literal["cosine", "euclidean", "dotproduct"]


class QueryReturnType(TypedDict):
    """The return type of the query method.

    Attributes
    ----------
    answer : str
        The answer to the question.

    matches : list[dict[str, Any]]
        The retrieved documents from the index.

    used_contexts : list[int]
        The indices of the matches that were actually used to answer the question.
    """

    answer: str
    matches: list[dict[str, Any]]
    used_contexts: list[int]


PROMPT_START = f"""\
You are a helpful assistant. I will give you a question and provide multiple
context documents. You will need to answer the question based on the contexts
and also specify in which context(s) you found the answer.
If you don't find the answer in the context, you can use your own knowledge, however,
in that case, the contexts array should be empty.

Both the input and the output are JSON objects.

# EXAMPLE INPUT
# ```json
# {input_example_1.model_dump_json()}
# ```

# EXAMPLE OUTPUT
# ```json
# {output_example_1.model_dump_json()}

# EXAMPLE INPUT
# ```json
# {input_example_2.model_dump_json()}
# ```

# EXAMPLE OUTPUT
# ```json
# {output_example_2.model_dump_json()}

# EXAMPLE INPUT
# ```json
# {input_example_3.model_dump_json()}
# ```

# EXAMPLE OUTPUT
# ```json
# {output_example_3.model_dump_json()}

"""

from elasticsearch import Elasticsearch
import os
from openai import OpenAI

class Client:
    """Synchronous client for Elasticsearch."""

    def __init__(
            self,
            openai_api_key: str | None = None,
            api_version: str | None = None,
            azure_endpoint: str | None = None,
            elasticsearch_host: str = "localhost",
            elasticsearch_port: int = 9201,
    ):
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key is None:
                raise ValueError(
                    "No OPENAI_API_KEY provided must be provided."
                )
        self.openai_client = AzureOpenAI(
            api_key=openai_api_key, api_version=api_version,
            azure_endpoint=azure_endpoint)
        self.elasticsearch_client = Elasticsearch(
            [{'host': elasticsearch_host, 'port': int(elasticsearch_port), 'scheme': 'http'}])

    def get_index(self, name: str):
        """Get an existing index from Elasticsearch.

        Parameters
        ----------
        name : str
            The name of the index.

        Returns
        -------
        Index
            The index.

        Raises
        ------
        IndexNotFoundException
            If the index does not exist.

        """
        try:
            if self.elasticsearch_client.indices.exists(index=name):
                return name
            else:
                raise NotFoundError(f"Index {name} does not exist")
        except NotFoundError as e:
            raise IndexNotFoundException(f"Index {name} does not exist") from e

    def create_index(
            self,
            name: str,
            dimension: int = 1536,
            metric: str = "cosine",
            spec: ServerlessSpec | PodSpec | None = None,
    ):
        """Create a new index in Elasticsearch.

        If the index does not exist, it creates a new index with the specified parameters.

        Parameters
        ----------
        name : str
            The name of the index.

        dimension : int
            The dimension of the index.

        metric : str
            The metric of the index.

        spec : ServerlessSpec | PodSpec | None
            The spec of the index. If None, it uses the default spec.

        Raises
        ------
        IndexAlreadyExistsException
            If the index already exists.

        """
        if self.elasticsearch_client.indices.exists(index=name):
            raise IndexAlreadyExistsException(f"Index {name} already exists")

        if spec is None:
            spec = DEFAULT_SPEC
            logger.info(f"Using default spec {spec}")

        # Create index mapping and settings (if necessary)
        body = {
            'mappings': {
                'properties': {
                    'embedding': {
                        'type': 'dense_vector',
                        "dims": dimension,
                        "similarity": metric
                    }
                }
            }}

        self.elasticsearch_client.indices.create(index=name, body=body)

        return name

    def upload_documents(
            self,
            index: str,
            documents: list[str | pathlib.Path],
            namespace: str,
            embedding_model: str = "text-embedding-3-small",
            batch_size: int = 100,
    ) -> None:
        documents = list(set(documents))
        if not documents:
            logger.info("No documents to upload")
            return

        logger.info(f"Parsing {len(documents)} documents")
        all_chunks: list[Document] = []
        for document in documents:
            chunks_ = parse_and_split(document)
            chunks = clean_chunks(chunks_)
            all_chunks.extend(chunks)

        logger.info(f"Embedding {len(all_chunks)} chunks")
        embeddings = generate_embeddings(
            # openai_api_key=self.openai_client.api_key,
            chunks=[c.page_content for c in all_chunks],
            model=embedding_model,
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"
        )

        if len(embeddings) != len(all_chunks):
            raise ValueError(
                "Number of embeddings does not match number of chunks"
            )

        pinecone_documents = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            metadata = {
                "text": chunk.page_content,
                "page_number": chunk.metadata["page"],
                "chunk_number": chunk.metadata["chunk"],
                "filename": chunk.metadata["source"],
                "uuid": str(uuid.uuid4())
            }
            document = {
                "_source": {
                    "metadata": metadata,
                    "values": embedding,
                    "id": str(uuid.uuid4())
                }}
            pinecone_documents.append(document)
            self.elasticsearch_client.index(index=f"{index}_{namespace}", id=document["_source"]["id"],
                                            body=document["_source"])

    def clean_text(
            self,
            text: str
    ) -> str:
        """Return a lower case version of text with punctuation removed.

        Parameters
        ----------
        text : str
            The raw text to be cleaned.

        Returns
        -------
        str: The cleaned text string.
        """
        text_processed = re.sub('[^0-9a-zA-Z ]+', '', text.lower())
        text_processed_further = re.sub(' +', ' ', text_processed)
        return text_processed_further

    def query(
            self,
            question: str,
            index: str,
            namespace: str,
            rules: list[Rule] | None = None,
            top_k: int = 5,
            chat_model: str = "gpt-35-turbo-16k",
            chat_temperature: float = 0.0,
            chat_max_tokens: int = 1000,
            chat_seed: int = 2,
            embedding_model: str = "text-embedding-3-small",
            process_rules_separately: bool = False,
            keyword_trigger: bool = False
    ) -> QueryReturnType:
        """Query the index.

        Parameters
        ----------
        question : str
            The question to ask.

        index : str
            The name of the Elasticsearch index to query.

        namespace : str
            The namespace within the index to use.

        rules : list[Rule] | None
            The rules to use for filtering the documents.

        top_k : int
            The number of matches to return per rule.

        chat_model : str
            The OpenAI chat model to use.

        chat_temperature : float
            The temperature for the chat model.

        chat_max_tokens : int
            The maximum number of tokens for the chat model.

        chat_seed : int
            The seed for the chat model.

        embedding_model : str
            The OpenAI embedding model to use.

        process_rules_separately : bool, optional
            Whether to process each rule individually and combine the results at the end.
            When set to True, each rule will be run independently, ensuring that every rule
            returns results. When set to False (default), all rules will be run as one joined
            query, potentially allowing one rule to dominate the others.
            Default is False.

        keyword_trigger : bool, optional
            Whether to trigger rules based on keyword matches in the question.
            Default is False.

        Returns
        -------
        QueryReturnType
            Dictionary with keys "answer", "matches", and "used_contexts".
            The "answer" is the answer to the question.
            The "matches" are the "top_k" matches from the index.
            The "used_contexts" are the indices of the matches
            that were actually used to answer the question.

        Raises
        ------
        OpenAIException
            If there is an error with the OpenAI API. Some possible reasons
            include the chat model not finishing or the response not being
            valid JSON.
        """
        print(f'Raw rules: {rules}')

        if rules is None:
            rules = []

        if keyword_trigger:
            triggered_rules = []
            clean_question = self.clean_text(question).split(' ')

            for rule in rules:
                if rule.keywords:
                    clean_keywords = [self.clean_text(keyword) for keyword in rule.keywords]
                    if bool(set(clean_keywords) & set(clean_question)):
                        triggered_rules.append(rule)

            rules = triggered_rules

        rule_filters = [rule.to_filter() for rule in rules if rule is not None]
        print(rule_filters, "ruleeeee")
        question_embedding = generate_embeddings(
            openai_api_key=self.openai_client.api_key,
            chunks=[question],
            model=embedding_model,
        )[0]

        matches = []  # Initialize matches outside the loop to collect matches from all queries
        match_texts = []

        # Check if there are any rule filters, and if not, proceed with a default query

        for rule_filter in rule_filters:
            print(rule_filter,"process_sepa")
            queries = []

            # Loop over all filenames
            for filename_term in rule_filter['bool']['must'][0]['term']['filename']:
                # Loop over all page numbers
                for page_number_term in rule_filter['bool']['must'][1]['terms']['page_number']:
                    # Create a query for each combination of filename and page number
                    query = {
                        "size": top_k,
                        "query": {
                            "bool": {
                                "must": [
                                    {"match": {"metadata.page_number": page_number_term}},
                                    {"match": {"metadata.filename": filename_term}}
                                ]
                            }
                        }
                    }
                    # Append the query to the list of queries
                    print("///////////",query)
                    queries.append(query)

            hitss = []
            for query in queries:
                response = self.elasticsearch_client.search(index=f"{index}_{namespace}", body=query)
                print("*-*-****--*--*-**-",response)
                res = response["hits"]["hits"]
                hitss.append(res)
                for hits in hitss:
                    for hit in hits:
                        match = hit["_source"]
                        matches.append(PineconeMatch(
                            id=match["id"],
                            score=None,  # Score not used in Elasticsearch
                            metadata=PineconeMetadata(
                                text=match["metadata"]["text"],
                                page_number=match["metadata"]["page_number"],
                                chunk_number=match["metadata"]["chunk_number"],
                                filename=match["metadata"]["filename"],
                                uuid=match["metadata"]["uuid"]
                            )
                        ))
                        match_texts.append(match["metadata"]["text"])

        # Proceed to create prompt, send it to OpenAI, and handle the response
        prompt = self.create_prompt(question, match_texts)
        response = self.openai_client.chat.completions.create(
            model=chat_model,
            seed=chat_seed,
            temperature=chat_temperature,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=chat_max_tokens,
        )

        output = self.process_response(response)

        return_dict: QueryReturnType = {
            "answer": output.answer,
            "matches": [m.model_dump() for m in matches],
            "used_contexts": output.contexts,
        }

        return return_dict

    def create_prompt(self, question: str, match_texts: list[str]) -> str:
        """Create the prompt for the OpenAI chat completion.

        Parameters
        ----------
        question : str
            The question to ask.

        match_texts : list[str]
            The list of context strings to include in the prompt.

        Returns
        -------
        str
            The generated prompt.
        """
        input_actual = Input(question=question, contexts=match_texts)
        prompt_end = f"""
        ACTUAL INPUT
        ```json
        {input_actual.model_dump_json()}
        ```

        ACTUAL OUTPUT
        """
        return f"{PROMPT_START}\n{prompt_end}"

    def process_response(self, response: Any) -> Output:
        """Process the OpenAI chat completion response.

        Parameters
        ----------
        response : Any
            The OpenAI chat completion response.

        Returns
        -------
        Output
            The processed output.

        Raises
        ------
        OpenAIException
            If the chat model did not finish or the response is not valid JSON.
        """
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise OpenAIException(
                f"Chat did not finish. Reason: {choice.finish_reason}"
            )

        response_raw = cast(str, response.choices[0].message.content)

        if response_raw.startswith("```json"):
            start_i = response_raw.index("{")
            end_i = response_raw.rindex("}")
            response_raw = response_raw[start_i: end_i + 1]

        try:
            output = Output.model_validate_json(response_raw)
        except ValidationError as e:
            raise OpenAIException(
                f"OpenAI did not return a valid JSON: {response_raw}"
            ) from e

        return output
