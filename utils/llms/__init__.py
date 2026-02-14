OPEN_AI_PRICING = {  # Pricing per 1M tokens
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-5": {"input": 1.25, "output": 10.0},
    "gpt-5.1": {"input": 1.25, "output": 10.0},
    "gpt-5.2": {"input": 1.75, "output": 14.0},
    "gpt-5-mini": {"input": 0.25, "output": 2.0},
    "gpt-5-nano": {"input": 0.05, "output": 0.40}
}

OPEN_AI_TOOL_PRICING = {  # Pricing per call
    "web_search": 0.01,
    "web_search_preview": 0.01,
    "file_search": 0.0025,
}
