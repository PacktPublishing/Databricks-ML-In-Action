# currently the langchain integration is broken
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts.chat import ChatPromptTemplate
from langchain.llms.base import LLM
from langchain.schema.messages import HumanMessage
import requests


class ServingEndpointLLM(LLM):
    endpoint_url: str
    token: str
    temperature: float = 0.1
    max_length: int = 256

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            #raise ValueError("stop kwargs are not permitted.")
            pass

        header = {"Context-Type": "text/json", "Authorization": f"Bearer {self.token}"}

        if type(prompt) is str:
            dataset = {'inputs': {'prompt': [prompt]},
                  'params': {**{'max_tokens': self.max_length}, **kwargs}}
        elif type(prompt) is ChatPromptTemplate:
            text_prompt = prompt.format()
            dataset = {'inputs': {'prompt': [text_prompt]},
                  'params': {**{'max_tokens': self.max_length}, **kwargs}} 
        #print(dataset)
        try:
            response = requests.post(headers=header, url=self.endpoint_url, json=dataset)

            try:
                return str(response.json()['predictions']['candidates'][0])
            
            except KeyError:
                print(response)
                return str(response.json())

        
        except TypeError:
          print(dataset)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"endpoint_url": self.endpoint_url}  

# Embedding wrapper
from typing import Any, Dict, List, Mapping, Optional, Tuple
import requests
from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_dict_or_env


class ModelServingEndpointEmbeddings(BaseModel, Embeddings):
    """Databricks Model Serving embedding service.

    To use, you should have the
    environment variable ``DB_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.
    """

    endpoint_url: str = None
    """Endpoint URL to use."""
    embed_instruction: str = "Represent the document for retrieval: "
    """Instruction used to embed documents."""
    query_instruction: str = (
        "Represent the question for retrieving supporting documents: "
    )
    """Instruction used to embed the query."""
    retry_sleep: float = 1.0
    """How long to try sleeping for if a rate limit is encountered"""

    db_api_token: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        db_api_token = get_from_dict_or_env(
            values, "db_api_token", "DB_API_TOKEN"
        )
        values["db_api_token"] = db_api_token
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"endpoint_url": self.endpoint_url}

    def _embed(
        self, input: List[Tuple[str, str]], is_retry: bool = False
    ) -> List[List[float]]:
        #payload = {"input_strings": input}
        payload = {
            "dataframe_split": {
                "data": [
                    [
                        input
                    ]
                ]
            }
        }

        # HTTP headers for authorization
        headers = {
            "Authorization": f"Bearer {self.db_api_token}",
            "Content-Type": "application/json",
        }

        # send request
        try:
            response = requests.post(url=self.endpoint_url, headers=headers, json=payload)
        
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        try:
            if response.status_code == 429:
                if not is_retry:
                    import time
                    time.sleep(self.retry_sleep)
                    return self._embed(input, is_retry=True)
                raise ValueError(
                    f"Error raised by inference API: rate limit exceeded.\nResponse: "
                    f"{response.text}"
                )
            
            parsed_response = response.json()

        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised by inference API: {e}.\nResponse: {response.text}"
            )
        return parsed_response

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a MosaicML deployed instructor embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            embeddings = [self._embed(x)['predictions'][0] for x in texts]
        except KeyError:
            print([self._embed(x) for x in texts])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a Databricks Model Serving embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self._embed(text)
        return embedding['predictions']