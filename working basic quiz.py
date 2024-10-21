from fastapi import FastAPI, HTTPException
import openai

app = FastAPI()

# Set your OpenAI API key
#openai.api_key = "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"

@app.get("/sapa/")
async def sapa():
    try:
        # Define the prompt and call the chat model correctly
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a quiz generator."},
                {"role": "user", "content": "Generate 1 multiple-choice question in the Science category with Medium difficulty. Provide four options (A, B, C, D) and specify the correct answer."}
            ],
            max_tokens=150,
            temperature=0.7
        )

        # Get the generated question from the response
        quiz = response['choices'][0]['message']['content']

        # Return the quiz
        return {"quiz": quiz}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))