from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="",
)


def get_llm_response(question, docs):
    completion = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[
            {
                "role": "user",
                "content": f"""
                    Given the following context, 
                    if relevant use it to answer the user qustion else use your own knowledge to answer.
                    
                    <context>
                        {docs}
                    <context>

                    <user_query>
                        {question}
                    <user_query>
                 """,
            }
        ],
        temperature=0.2,
        top_p=0.7,
        stream=True,
    )
    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content

    return response
