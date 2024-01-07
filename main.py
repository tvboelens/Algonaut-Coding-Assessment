import ast
import utils
import os
import pandas as pd



if __name__ == "__main__":
    if not os.path.exists('data/embeddings.csv'):
        papers_list = utils.fetch_papers()
        df = utils.create_embedding(papers_list)       
    else:
        df = pd.read_csv('data/embeddings.csv')
        df['embedding'] = df['embedding'].apply(ast.literal_eval)


    """
    Main interaction loop for the chatbot.
    """
    print("Welcome to Chatbot! Type 'quit' to exit.")

    user_input = ""
    while user_input.lower() != "quit":
        user_input = input("You: ")

        if user_input.lower() != "quit":
            response = utils.chat_with_openai(user_input,df)  # Pass user_input as an argument
            print(f"Chatbot: {response}")