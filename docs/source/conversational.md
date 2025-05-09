# Conversational mode

## Preamble

By default, Colette does not use conversational mode. It works in pure LLM mode.
This means the `template_prompt` is passed "as is" to the LLM model.

In non-conversational mode, the `template_prompt` should contain the question from the user.

## Enable conversational mode

During service creation, to enable conversational mode, set the `parameters.llm.conversational` boolean.
This will activate chat templates, where each prompt is associated with a role (system, user or assistant).
The formated input is then passed to the LLM.

In conversational mode, the `template_prompt` is the system prompt and should not include the question from the user. It should contain the context if RAG is enabled.

## Persistance

Optionally, you can also activate persistance by passing a `parameters.input.session_id` string during the `/predict` call. This should be a unique identifier of the session for the user.

This is the default behavior for the builtin frontend app.
Previous questions from the user and answers from the assistant will be included in the input passed to the LLM.

## RAG

If RAG is enabled, the document search will be done using the last question by default.

Using the `parameters.input.summarize` string, it is possible to specify a prompt that will be used to generate a new question based on the conversation history.

For example: _"Using only the information below and NOT your own knowledge, build a question that will replace the last question by giving more context to a RAG for correct document retrieval. Only output the new question."_

In this case, the LLM is called twice: first to generate the question used for the RAG, then to generate the answer based on retrieved context and conversation history.

## With the "vllm" lib

This [requires](https://python.langchain.com/v0.2/docs/integrations/chat/vllm/) to launch vLLM in server mode.
```
vllm serve Qwen/Qwen2-0.5B-Instruct-AWQ
```

## With the "huggingface" lib

On 2024-09-04, this requires the upstream version of langchain-huggingface:
```
pip install -e "git+https://github.com/langchain-ai/langchain#egg=langchain_huggingface&subdirectory=libs/partners/huggingface"
```
Because of [this commit](https://github.com/langchain-ai/langchain/commit/4796b7eb15b4c3a352c950f56a30659f0379f6e2) that is not released.
