
# whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
def get_completion(
    client,
    message: str,
    model: str = "gpt-3.5-turbo-0125",
    max_tokens=1000,
    temperature=0,
    stop=None,
    seed=None,
    logprobs=True,
    top_logprobs=1,
) -> str:
    params = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    completion = client.chat.completions.create(**params)
    return completion