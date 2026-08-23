import os
from typing import Generator, List, Dict, Any, Optional

# Safeguard: If SSL_CERT_FILE points to a missing file, remove it from environ
# so httpx falls back to default certificates instead of crashing with Errno 2.
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

# System Prompt enforcing strict grounded answering
STRICT_SYSTEM_PROMPT = """You are a highly accurate, strict Document Q&A Agent.
Your sole purpose is to answer the user's question using ONLY the information provided in the DOCUMENT CONTEXT below.

CRITICAL RULES:
1. Rely ONLY on the clear facts directly mentioned in the DOCUMENT CONTEXT.
2. Do NOT extrapolate, speculate, or make assumptions that are not explicitly stated in the context.
3. If the answer cannot be found or deduced with certainty from the provided context, you MUST state clearly:
   "I cannot find the answer in the uploaded documents."
4. Whenever you provide information from the documents, cite the document name and section/page whenever available.
5. NEVER answer from external general knowledge if the information is missing from the uploaded files.
"""

PROVIDERS_CONFIG = {
    "Google Gemini": {
        "models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ],
        "default_model": "gemini-2.5-flash",
        "env_var": "GEMINI_API_KEY",
        "doc_url": "https://aistudio.google.com/app/apikey"
    },
    "OpenAI ChatGPT": {
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o3-mini"
        ],
        "default_model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
        "doc_url": "https://platform.openai.com/api-keys"
    },
    "Anthropic Claude": {
        "models": [
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229"
        ],
        "default_model": "claude-3-7-sonnet-20250219",
        "env_var": "ANTHROPIC_API_KEY",
        "doc_url": "https://console.anthropic.com/settings/keys"
    }
}


def build_user_prompt_with_context(query: str, context: str) -> str:
    """Combines retrieved context with the user query."""
    if not context or not context.strip():
        return f"User Question: {query}\n\n[Note: No documents have been uploaded or found.]"
        
    return f"""--- START OF DOCUMENT CONTEXT ---
{context}
--- END OF DOCUMENT CONTEXT ---

Based ONLY on the document context above, please answer this question:
User Question: {query}
"""


def stream_gemini_response(
    api_key: str,
    model_name: str,
    query: str,
    context: str,
    chat_history: List[Dict[str, str]],
    temperature: float = 0.0
) -> Generator[str, None, None]:
    """Streams response from Google Gemini."""
    try:
        # Prefer new google-genai SDK if available
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=api_key)
        prompt_content = build_user_prompt_with_context(query, context)
        
        # Build contents including system instruction
        config = types.GenerateContentConfig(
            system_instruction=STRICT_SYSTEM_PROMPT,
            temperature=temperature
        )
        
        # Format recent history for context
        contents = []
        for msg in chat_history[-6:]:
            if msg["role"] == "user":
                contents.append(msg["content"])
            elif msg["role"] == "assistant":
                contents.append(msg["content"])
        contents.append(prompt_content)
        
        response = client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=config
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e_new:
        # Fallback to google.generativeai
        try:
            import google.generativeai as gai
            gai.configure(api_key=api_key)
            
            gen_model = gai.GenerativeModel(
                model_name=model_name,
                system_instruction=STRICT_SYSTEM_PROMPT,
                generation_config={"temperature": temperature}
            )
            prompt_content = build_user_prompt_with_context(query, context)
            response = gen_model.generate_content(prompt_content, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"⚠️ **Gemini API Error:** {str(e)}"


def stream_openai_response(
    api_key: str,
    model_name: str,
    query: str,
    context: str,
    chat_history: List[Dict[str, str]],
    temperature: float = 0.0
) -> Generator[str, None, None]:
    """Streams response from OpenAI ChatGPT."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        messages = [{"role": "system", "content": STRICT_SYSTEM_PROMPT}]
        
        # Add recent conversation turns
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        # Add current user prompt with isolated context
        messages.append({
            "role": "user",
            "content": build_user_prompt_with_context(query, context)
        })
        
        kwargs = {
            "model": model_name,
            "messages": messages,
            "stream": True
        }
        # o3-mini or reasoning models might use max_completion_tokens or not support temperature=0
        if not model_name.startswith("o"):
            kwargs["temperature"] = temperature
            
        stream = client.chat.completions.create(**kwargs)
        
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
            if delta:
                yield delta
    except Exception as e:
        import traceback
        yield f"⚠️ **OpenAI API Error:** {str(e)}\n\n```python\n{traceback.format_exc()}\n```"


def stream_claude_response(
    api_key: str,
    model_name: str,
    query: str,
    context: str,
    chat_history: List[Dict[str, str]],
    temperature: float = 0.0
) -> Generator[str, None, None]:
    """Streams response from Anthropic Claude."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        messages = []
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        messages.append({
            "role": "user",
            "content": build_user_prompt_with_context(query, context)
        })
        
        with client.messages.stream(
            model=model_name,
            max_tokens=4096,
            system=STRICT_SYSTEM_PROMPT,
            temperature=temperature,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"⚠️ **Claude API Error:** {str(e)}"


def stream_llm_answer(
    provider: str,
    api_key: str,
    model_name: str,
    query: str,
    context: str,
    chat_history: List[Dict[str, str]],
    temperature: float = 0.0
) -> Generator[str, None, None]:
    """Unified dispatcher to stream answers from any chosen LLM provider."""
    if not api_key or not api_key.strip():
        yield f"⚠️ **Missing API Key:** Please enter your **{provider}** API Key in the sidebar."
        return

    if provider == "Google Gemini":
        yield from stream_gemini_response(api_key, model_name, query, context, chat_history, temperature)
    elif provider == "OpenAI ChatGPT":
        yield from stream_openai_response(api_key, model_name, query, context, chat_history, temperature)
    elif provider == "Anthropic Claude":
        yield from stream_claude_response(api_key, model_name, query, context, chat_history, temperature)
    else:
        yield f"⚠️ Unsupported provider: {provider}"
