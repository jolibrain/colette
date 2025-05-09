from openai import OpenAI


class VllmClient:
    def __init__(
        self,
        url,
        api_key,
        model,
        max_model_len=4096,
        min_pixels=28 * 28,
        max_pixels=1280 * 28 * 28,
    ):
        self.model = model
        self.max_model_len = max_model_len
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.client = OpenAI(base_url=url, api_key=api_key)

    def chat(self, messages, max_new_tokens):
        print("!!!!!!!!!!!!!!!!!!!!!! sending request", messages)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body={
                "mm_processor_kwargs": {
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                    "max_model_len": self.max_model_len,
                },
            },
            max_completion_tokens=max_new_tokens,
        )
        return completion.choices[0].message.content

    def to(self, *args):
        pass
