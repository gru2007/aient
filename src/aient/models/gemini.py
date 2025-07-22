import os
import re
import json
import requests

from .base import BaseLLM
from ..core.utils import BaseAPI

import copy
from ..plugins import PLUGINS, get_tools_result_async, function_call_list
from ..utils.scripts import safe_get

USE_OPENAI_COMPAT = os.environ.get("GEMINI_USE_OPENAI_COMPAT", "false").lower() == "true"


class gemini(BaseLLM):
    def __init__(
        self,
        api_key: str = None,
        engine: str = os.environ.get("GPT_ENGINE") or "gemini-1.5-pro-latest",
        api_url: str = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{stream}?key={api_key}",
        system_prompt: str = "You are Gemini, a large language model trained by Google. Respond conversationally",
        temperature: float = 0.5,
        top_p: float = 0.7,
        timeout: float = 20,
        use_plugins: bool = True,
        print_log: bool = False,
    ):
        url = api_url.format(model=engine, stream="streamGenerateContent", api_key=os.environ.get("GOOGLE_AI_API_KEY", api_key))
        super().__init__(api_key, engine, url, system_prompt=system_prompt, timeout=timeout, temperature=temperature, top_p=top_p, use_plugins=use_plugins, print_log=print_log)
        self.conversation: dict[str, list[dict]] = {
            "default": [],
        }

    def add_to_conversation(
        self,
        message: str,
        role: str,
        convo_id: str = "default",
        pass_history: int = 9999,
        total_tokens: int = 0,
        function_arguments: str = "",
    ) -> None:
        """
        Add a message to the conversation
        """

        if convo_id not in self.conversation:
            self.reset(convo_id=convo_id)
        # print("message", message)

        if function_arguments:
            self.conversation[convo_id].append(
                {
                    "role": "model",
                    "parts": [function_arguments]
                }
            )
            function_call_name = function_arguments["functionCall"]["name"]
            self.conversation[convo_id].append(
                {
                    "role": "function",
                    "parts": [{
                    "functionResponse": {
                        "name": function_call_name,
                        "response": {
                            "name": function_call_name,
                            "content": {
                                "result": message,
                            }
                        }
                    }
                    }]
                }
            )

        else:
            if isinstance(message, str):
                message = [{"text": message}]
            self.conversation[convo_id].append({"role": role, "parts": message})

        history_len = len(self.conversation[convo_id])
        history = pass_history
        if pass_history < 2:
            history = 2
        while history_len > history:
            mess_body = self.conversation[convo_id].pop(1)
            history_len = history_len - 1
            if mess_body.get("role") == "user":
                mess_body = self.conversation[convo_id].pop(1)
                history_len = history_len - 1
                if safe_get(mess_body, "parts", 0, "functionCall"):
                    self.conversation[convo_id].pop(1)
                    history_len = history_len - 1

        if total_tokens:
            self.tokens_usage[convo_id] += total_tokens

    def reset(self, convo_id: str = "default", system_prompt: str = "You are Gemini, a large language model trained by Google. Respond conversationally") -> None:
        """
        Reset the conversation
        """
        self.system_prompt = system_prompt or self.system_prompt
        self.conversation[convo_id] = list()

    def get_openai_history(self, convo_id, prompt, pass_history, system_prompt=None, max_history=20):
        """
        Формирует историю для OpenAI-совместимого режима:
        - system prompt всегда первый
        - далее user/assistant
        - все сообщения в формате {"role": ..., "content": ...}
        - если истории слишком много — обрезать до последних max_history сообщений
        """
        messages = []
        system_prompt = system_prompt or self.system_prompt
        history = self.conversation.get(convo_id, []) if pass_history else []
        # Преобразуем историю в OpenAI-формат
        for msg in history:
            if msg.get("role") and msg.get("parts"):
                content = msg["parts"][0]["text"] if isinstance(msg["parts"][0], dict) and "text" in msg["parts"][0] else str(msg["parts"][0])
                messages.append({"role": msg["role"], "content": content})
        # Убедимся, что первый — system
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages
        # Если только system — добавим текущий prompt
        if len(messages) == 1:
            messages.append({"role": "user", "content": prompt})
        # Обрезаем историю (оставляем system + последние max_history сообщений)
        if len(messages) > max_history + 1:
            messages = [messages[0]] + messages[-max_history:]
        return messages

    def ask_stream(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        model_max_tokens: int = 4096,
        system_prompt: str = None,
        **kwargs,
    ):
        self.system_prompt = system_prompt or self.system_prompt
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id, system_prompt=self.system_prompt)
        self.add_to_conversation(prompt, role, convo_id=convo_id, pass_history=pass_history)
        # print(self.conversation[convo_id])

        headers = {
            "Content-Type": "application/json",
        }

        if USE_OPENAI_COMPAT:
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            url = base_url + "chat/completions"
            api_key = os.environ.get("GOOGLE_AI_API_KEY", self.api_key)
            headers["Authorization"] = f"Bearer {api_key}"
            messages = self.get_openai_history(convo_id, prompt, pass_history, system_prompt)
            json_post = {
                "model": model or self.engine,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": model_max_tokens,
            }
        else:
            json_post = {
                "contents": self.conversation[convo_id] if pass_history else [{
                    "role": "user",
                    "content": prompt
                }],
                "systemInstruction": {"parts": [{"text": self.system_prompt}]},
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ],
            }
            url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{stream}?key={api_key}".format(model=model or self.engine, stream="streamGenerateContent", api_key=os.environ.get("GOOGLE_AI_API_KEY", self.api_key) or kwargs.get("api_key"))
            self.api_url = BaseAPI(url)
            url = self.api_url.source_api_url

        if self.print_log:
            print("url", url)
            replaced_text = json.loads(re.sub(r';base64,([A-Za-z0-9+/=]+)', ';base64,***', json.dumps(json_post)))
            print(json.dumps(replaced_text, indent=4, ensure_ascii=False))

        try:
            response = self.session.post(
                url,
                headers=headers,
                json=json_post,
                timeout=kwargs.get("timeout", self.timeout),
                stream=True,
            )
        except ConnectionError:
            print("连接错误，请检查服务器状态或网络连接。")
            return
        except requests.exceptions.ReadTimeout:
            print("请求超时，请检查网络连接或增加超时时间。{e}")
            return
        except Exception as e:
            print(f"发生了未预料的错误: {e}")
            return

        if response.status_code != 200:
            print(response.text)
            raise BaseException(f"{response.status_code} {response.reason} {response.text}")
        response_role: str = "model"
        full_response: str = ""
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line and '"text": "' in line:
                    content = line.split('"text": "')[1][:-1]
                    content = "\n".join(content.split("\\n"))
                    content = content.encode('utf-8').decode('unicode-escape')
                    full_response += content
                    yield content
        except requests.exceptions.ChunkedEncodingError as e:
            print("Chunked Encoding Error occurred:", e)
        except Exception as e:
            print("An error occurred:", e)

        self.add_to_conversation([{"text": full_response}], response_role, convo_id=convo_id, pass_history=pass_history)

    async def ask_stream_async(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        system_prompt: str = None,
        language: str = "English",
        function_arguments: str = "",
        total_tokens: int = 0,
        **kwargs,
    ):
        self.system_prompt = system_prompt or self.system_prompt
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id, system_prompt=self.system_prompt)
        self.add_to_conversation(prompt, role, convo_id=convo_id, total_tokens=total_tokens, function_arguments=function_arguments, pass_history=pass_history)
        # print(self.conversation[convo_id])

        headers = {
            "Content-Type": "application/json",
        }

        if USE_OPENAI_COMPAT:
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            url = base_url + "chat/completions"
            api_key = os.environ.get("GOOGLE_AI_API_KEY", self.api_key)
            headers["Authorization"] = f"Bearer {api_key}"
            messages = self.get_openai_history(convo_id, prompt, pass_history, system_prompt)
            json_post = {
                "model": model or self.engine,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": 4096,
            }
        else:
            json_post = {
                "contents": self.conversation[convo_id] if pass_history else [{
                    "role": "user",
                    "content": prompt
                }],
                "systemInstruction": {"parts": [{"text": self.system_prompt}]},
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ],
            }
            url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{stream}?key={api_key}".format(model=model or self.engine, stream="streamGenerateContent", api_key=os.environ.get("GOOGLE_AI_API_KEY", self.api_key) or kwargs.get("api_key"))
            self.api_url = BaseAPI(url)
            url = self.api_url.source_api_url

        if self.print_log:
            print("url", url)
            replaced_text = json.loads(re.sub(r';base64,([A-Za-z0-9+/=]+)', ';base64,***', json.dumps(json_post)))
            print(json.dumps(replaced_text, indent=4, ensure_ascii=False))

        response_role: str = "model"
        full_response: str = ""
        function_full_response: str = "{"
        need_function_call = False
        revicing_function_call = False
        total_tokens = 0
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=json_post,
                    timeout=kwargs.get("timeout", self.timeout),
                ) as response:
                    if response.status != 200:
                        error_content = await response.text()
                        raise BaseException(f"{response.status}: {error_content}")
                    async for line in response.content:
                        line = line.decode("utf-8")
                        if line and '"text": "' in line:
                            content = line.split('"text": "')[1][:-1]
                            content = "\n".join(content.split("\\n"))
                            full_response += content
                            yield content
        except Exception as e:
            print(f"发生了未预料的错误: {e}")
            return

        self.add_to_conversation([{"text": full_response}], response_role, convo_id=convo_id, total_tokens=total_tokens, pass_history=pass_history)