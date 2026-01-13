import os

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../../prompts")


def get_prompt(name):
    with open(os.path.join(PROMPT_DIR, name + ".txt")) as f:
        return "".join([line for line in f])


# def get_completion(prompt, model, max_tokens=1000, temperature=0):
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         max_tokens=max_tokens,
#         temperature=temperature,  # this is the degree of randomness of the model's output
#     )
#     return response.choices[0].message["content"]


def get_completion(prompt, model, client):
    messages = [{"role": "user", "content": prompt}]
    ct = 0
    while ct < 2:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0,  # this is the degree of randomness of the model's output
            )
            return response.choices[0].message.content
            break
        except Exception as e:
            print(e)
            ct += 1

    return response.choices[0].message.content


def get_scoring_completion(prompt, model, client):
    ct = 0
    all_responses = None
    while ct < 2:
        try:
            _response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                temperature=2,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n=5
            )
            all_responses = [_response.choices[i].message.content for i in
                             range(len(_response.choices))]
            break
        except Exception as e:
            print(e)

    return all_responses