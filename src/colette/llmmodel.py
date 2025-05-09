## LLM wrapper model descriptor and information holder

from .apidata import LLMModelObj


class LLMModelBadParamException(Exception):
    pass


class LLMModelInternalException(Exception):
    pass


class LLMModel:
    def init(self, ad: LLMModelObj):
        self.llm_obj = ad
        if ad and hasattr(ad, "inference") and ad.inference and hasattr(ad.inference, "lib"):
            self.llm_lib = ad.inference.lib
            self.llm_source = ad.source
        else:
            self.llm_lib = None
            self.llm_source = None

    def delete_model(self):
        pass
