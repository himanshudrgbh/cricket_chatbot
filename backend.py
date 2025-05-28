from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_chat(username, question, answer):
    response = supabase.table("chat_history").insert({
        "username": username,
        "question": question,
        "answer": answer
    }).execute()
    return response
