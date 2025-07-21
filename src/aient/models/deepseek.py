import os
import json
import requests

from .base import BaseLLM

class deepseek(BaseLLM):
    """
    DeepSeek API wrapper supporting both chat and reasoning models (deepseek-chat, deepseek-coder, deepseek-reasoner).
    Handles reasoning_content in streaming responses for deepseek-reasoner.
    """
    def __init__(
        self,
        api_key: str = None,
        engine: str = os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat",
        api_url: str = None,
        system_prompt: str = None,
        temperature: float = 0.5,
        top_p: float = 1,
        timeout: float = 20,
    ):
        # Set endpoint and prompt for reasoner
        if engine == "deepseek-reasoner":
            api_url = api_url or "https://api.deepseek.com/v1/reasoning/chat/completions"
            system_prompt = system_prompt or "You are DeepSeek Reasoner, an advanced step-by-step reasoning AI."
        else:
            api_url = api_url or "https://api.deepseek.com/v1/chat/completions"
            system_prompt = system_prompt or "You are DeepSeek, a helpful and versatile AI assistant."
        super().__init__(api_key, engine, api_url, system_prompt, timeout=timeout, temperature=temperature, top_p=top_p)
        self.api_url = api_url

    def add_to_conversation(
        self,
        message: str,
        role: str,
        convo_id: str = "default",
        pass_history: int = 9999,
        total_tokens: int = 0,
    ) -> None:
        """
        Add a message to the conversation
        """
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id)
        self.conversation[convo_id].append({"role": role, "content": message})

        history_len = len(self.conversation[convo_id])
        history = pass_history
        if pass_history < 2:
            history = 2
        while history_len > history:
            self.conversation[convo_id].pop(1)
            history_len = history_len - 1

        if total_tokens:
            self.tokens_usage[convo_id] += total_tokens

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        self.conversation[convo_id] = list()
        self.system_prompt = system_prompt or self.system_prompt

    def ask_stream(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        model_max_tokens: int = 2048,
        system_prompt: str = None,
        **kwargs,
    ):
        self.system_prompt = system_prompt or self.system_prompt
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id)
        self.add_to_conversation(prompt, role, convo_id=convo_id, pass_history=pass_history)

        url = self.api_url
        headers = {
            "Authorization": f"Bearer {kwargs.get('DEEPSEEK_API_KEY', self.api_key)}",
            "Content-Type": "application/json",
        }

        self.conversation[convo_id][0] = {"role": "system","content": self.system_prompt}
        json_post = {
            "messages": self.conversation[convo_id] if pass_history else [{
                "role": "user",
                "content": prompt
            }],
            "model": model or self.engine,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": model_max_tokens,
            "top_p": kwargs.get("top_p", self.top_p),
            "stop": None,
            "stream": True,
        }

        try:
            response = self.session.post(
                url,
                headers=headers,
                json=json_post,
                timeout=kwargs.get("timeout", self.timeout),
                stream=True,
            )
        except ConnectionError:
            print("Connection error, please check server status or network connection.")
            return
        except requests.exceptions.ReadTimeout:
            print("Request timed out, please check network connection or increase timeout.")
            return
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return

        if response.status_code != 200:
            print(response.text)
            raise BaseException(f"{response.status_code} {response.reason} {response.text}")
        response_role: str = "assistant"
        full_response: str = ""
        for line in response.iter_lines():
            if not line:
                continue
            # Remove "data: "
            if line.decode("utf-8")[:6] == "data: ":
                line = line.decode("utf-8")[6:]
            else:
                print(line.decode("utf-8"))
                full_response = json.loads(line.decode("utf-8"))["choices"][0]["message"]["content"]
                yield full_response
                break
            if line == "[DONE]":
                break
            resp: dict = json.loads(line)
            choices = resp.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta")
            if not delta:
                continue
            # Handle reasoning_content for deepseek-reasoner
            if "reasoning_content" in delta and delta["reasoning_content"]:
                yield {"reasoning_content": delta["reasoning_content"]}
            if "role" in delta:
                response_role = delta["role"]
            if "content" in delta and delta["content"]:
                content = delta["content"]
                full_response += content
                yield content
        self.add_to_conversation(full_response, response_role, convo_id=convo_id, pass_history=pass_history)

    async def ask_stream_async(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        model_max_tokens: int = 2048,
        system_prompt: str = None,
        **kwargs,
    ):
        self.system_prompt = system_prompt or self.system_prompt
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id)
        self.add_to_conversation(prompt, role, convo_id=convo_id, pass_history=pass_history)

        url = self.api_url
        headers = {
            "Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY', self.api_key) or kwargs.get('api_key')}",
            "Content-Type": "application/json",
        }

        self.conversation[convo_id][0] = {"role": "system","content": self.system_prompt}
        json_post = {
            "messages": self.conversation[convo_id] if pass_history else [{
                "role": "user",
                "content": prompt
            }],
            "model": model or self.engine,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": model_max_tokens,
            "top_p": kwargs.get("top_p", self.top_p),
            "stop": None,
            "stream": True,
        }

        response_role: str = "assistant"
        full_response: str = ""
        try:
            async with self.aclient.stream(
                "post",
                url,
                headers=headers,
                json=json_post,
                timeout=kwargs.get("timeout", self.timeout),
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    print(response.text)
                    raise BaseException(f"{response.status_code} {response.reason} {response.text}")
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    # Remove "data: "
                    if line[:6] == "data: ":
                        line = line.lstrip("data: ")
                    else:
                        full_response = json.loads(line)["choices"][0]["message"]["content"]
                        yield full_response
                        break
                    if line == "[DONE]":
                        break
                    resp: dict = json.loads(line)
                    choices = resp.get("choices")
                    if not choices:
                        continue
                    delta = choices[0].get("delta")
                    if not delta:
                        continue
                    # Handle reasoning_content for deepseek-reasoner
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        yield {"reasoning_content": delta["reasoning_content"]}
                    if "role" in delta:
                        response_role = delta["role"]
                    if "content" in delta and delta["content"]:
                        content = delta["content"]
                        full_response += content
                        yield content
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            return

        self.add_to_conversation(full_response, response_role, convo_id=convo_id, pass_history=pass_history) 