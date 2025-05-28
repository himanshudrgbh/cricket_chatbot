import os
import requests
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import re # Import regex for potential fallback or simpler parsing
import json # Import json for parsing LLM structured response

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys and system prompt
CRICKET_API_KEY = os.getenv('CRICKET_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT') 


# This class interacts with the cricketdata.org API
class CricketData:
    def __init__(self, API_KEY):
        self.API_KEY = API_KEY

    def get_url(self, url_type, id=None):
        """Constructs the API URL based on type and optional ID."""
        base_url = 'https://api.cricapi.com/v1/'
        if url_type == 'currentMatches':
            url = f'{base_url}{url_type}?apikey={self.API_KEY}&offset=0'
        else:
            url = f'{base_url}{url_type}?apikey={self.API_KEY}&offset=0&id={id}'
        return url

    def get_params(self, id=None):
        """Constructs request parameters."""
        params = {
            "apikey": self.API_KEY,
            "offset": 0
        }
        if id:
            params["id"] = id
        return params

    def _make_api_request(self, url_type, id=None):
        """Helper to make API requests and handle common errors."""
        url = self.get_url(url_type, id)
        params = self.get_params(id)
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            if data.get('status') != 'success':
                st.error(f"Cricket API call for {url_type} failed: {data.get('reason', 'Unknown error')}")
                return None
            return data.get('data', {})
        except requests.exceptions.RequestException as e:
            st.error(f"Network or API error for {url_type}: {e}")
            return None

    def get_match_info(self, match_id):
        """Fetches detailed information for a specific match."""
        return self._make_api_request('match_info', match_id)

    def get_match_squad(self, match_id):
        """Fetches squad details for a specific match."""
        return self._make_api_request('match_squad', match_id)

    def get_current_matches(self):
        """Fetches a list of all current matches."""
        # Note: currentMatches returns a list, not a dict like match_info/squad
        return self._make_api_request('currentMatches')


# This class handles interactions with the Gemini API
class LLM:
    def __init__(self, API_KEY, system_prompt):
        self.API_KEY = API_KEY
        genai.configure(api_key=self.API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        # Initialize a separate model for structured intent recognition
        self.intent_model = genai.GenerativeModel("gemini-2.0-flash")


        # Initialize chat history with system instructions
        system_instructions = {
            "role": "user",
            "parts": [system_prompt]
        }
        self.chat = self.model.start_chat(history=[system_instructions])

    def get_response(self, prompt, stream=True):
        """Sends a prompt to the Gemini model and returns the response."""
        response = self.chat.send_message(prompt, stream=stream)
        return response

    def get_match_intent(self, user_query, current_matches_summary):
        """
        Uses LLM to extract match number or name from user query.
        Returns a dictionary like {'match_number': int} or {'match_name': str} or None.
        """
        prompt = f"""
        Given the user query and the list of current matches, identify if the user is asking about a specific match by its number or name.

        Current Matches:
        {current_matches_summary}

        User Query: "{user_query}"

        If the user refers to a match by its number (e.g., "match 1", "first match"), extract the number.
        If the user refers to a match by its name (e.g., "Isle of Man Women vs Spain Women"), extract the full name.
        If no specific match is clearly identified, return an empty JSON object.

        Return your answer in JSON format. Example:
        {{ "match_number": 1 }}
        OR
        {{ "match_name": "Isle of Man Women vs Spain Women" }}
        OR
        {{}}
        """
        
        try:
            response = self.intent_model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": {
                        "type": "OBJECT",
                        "properties": {
                            "match_number": { "type": "INTEGER", "description": "The 1-based index of the match if referred by number" },
                            "match_name": { "type": "STRING", "description": "The full name of the match if referred by name" }
                        },
                        "additionalProperties": False
                    }
                }
            )
            # The response.text will be a JSON string
            parsed_json = json.loads(response.text)
            return parsed_json
        except Exception as e:
            st.warning(f"Could not parse match intent from LLM: {e}. Falling back to regex.")
            # Fallback to regex for simple number extraction if LLM fails
            match = re.search(r'match (\d+)', user_query.lower())
            if match:
                try:
                    return {"match_number": int(match.group(1))}
                except ValueError:
                    pass
            return {}


# --- Main Streamlit Application Logic ---

# Initialize API clients
cricket_data = CricketData(CRICKET_API_KEY)
llm_model = LLM(GEMINI_API_KEY, SYSTEM_PROMPT)

# Set up Streamlit page configuration
st.set_page_config(page_title="Fantasy Chat Assistant", layout="centered")
st.title("Fantasy Cricket Assistant")

# Initialize session state variables if they don't exist
if "chat" not in st.session_state:
    st.session_state['chat'] = llm_model.chat
if "messages" not in st.session_state:
    st.session_state['messages'] = []
# New session state variable to store current match data
if "current_match_data" not in st.session_state:
    st.session_state['current_match_data'] = None
# New session state variable to store simplified match list for LLM intent parsing
if "match_list_for_llm" not in st.session_state:
    st.session_state['match_list_for_llm'] = []

# --- Display Previous Messages ---
# Iterate through the history and display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

# --- Fetch Latest Matches Button ---
if st.button("Fetch Latest Matches"):
    with st.spinner("Fetching current matches..."):
        matches = cricket_data.get_current_matches()
        if matches:
            st.session_state.current_match_data = matches
            match_summary = []
            # Store the full match details in a more accessible way for intent parsing
            st.session_state.match_list_for_llm = [] 
            for i, m in enumerate(matches): # Iterate through all fetched matches
                # Only display top 5 for brevity in summary, but store all for lookup
                if i < 5:
                    match_summary.append(f"{i+1}. **{m.get('name', 'N/A')}** - {m.get('status', 'N/A')}")
                st.session_state.match_list_for_llm.append({
                    "index": i + 1,
                    "name": m.get('name', 'N/A'),
                    "id": m.get('id', 'N/A')
                })
            
            formatted_matches = "\n".join(match_summary)
            
            st.session_state.messages.append({
                "role": "assistant",
                "text": f"Here are some of the current matches:\n{formatted_matches}\n\nI've stored this information. Now you can ask me for fantasy recommendations based on these matches, or ask about a specific match by its number (e.g., 'tell me about match 1') or by its name."
            })
            # Re-run to display the new message immediately
            st.rerun()
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "text": "Could not fetch current matches. Please try again later."
            })
            st.rerun()


# --- User Input ---
user_input = st.chat_input("Let's Build your team")

if user_input:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Append user message to session state
    st.session_state.messages.append({"role": "user", "text": user_input})

    # --- Gemini Response ---
    with st.chat_message("assistant"):
        # Create a placeholder for the response to show a loading indicator
        response_area = st.empty()
        response_area.markdown("Thinking...") # Show a loading message

        try:
            full_prompt_to_gemini = user_input
            detailed_match_info_for_llm = ""
            
            # --- Step 1: Try to extract match intent using LLM ---
            selected_match_id = None
            selected_match_name = None # Initialize selected_match_name
            if st.session_state.current_match_data and st.session_state.match_list_for_llm:
                # Prepare a summary of current matches for the intent LLM
                current_matches_summary_text = "\n".join([
                    f"{m['index']}. {m['name']}" for m in st.session_state.match_list_for_llm
                ])
                
                match_intent = llm_model.get_match_intent(user_input, current_matches_summary_text)

                if match_intent:
                    if "match_number" in match_intent:
                        match_index = match_intent["match_number"] - 1 # Convert to 0-based index
                        if 0 <= match_index < len(st.session_state.current_match_data):
                            selected_match_id = st.session_state.current_match_data[match_index].get('id')
                            selected_match_name = st.session_state.current_match_data[match_index].get('name')
                            st.write(f"DEBUG: Identified match by number: {selected_match_name} (ID: {selected_match_id})")
                        else:
                            st.warning(f"Match number {match_intent['match_number']} is out of range.")
                    elif "match_name" in match_intent:
                        # Find match by name
                        for m in st.session_state.current_match_data:
                            if m.get('name', '').lower() == match_intent["match_name"].lower():
                                selected_match_id = m.get('id')
                                selected_match_name = m.get('name')
                                st.write(f"DEBUG: Identified match by name: {selected_match_name} (ID: {selected_match_id})")
                                break
                        if not selected_match_id:
                            st.warning(f"Could not find match with name: {match_intent['match_name']}.")
                else:
                    st.write("DEBUG: No specific match intent detected by LLM.")

            # --- Step 2: If a match ID was identified, fetch its details ---
            if selected_match_id:
                with st.spinner(f"Fetching details for {selected_match_name or 'the selected match'}..."):
                    match_info = cricket_data.get_match_info(selected_match_id)
                    match_squad = cricket_data.get_match_squad(selected_match_id)

                if match_info:
                    detailed_match_info_for_llm += f"Detailed Info for Match ID {selected_match_id} ({selected_match_name or 'N/A'}):\n"
                    for key, value in match_info.items():
                        if isinstance(value, (str, int, float, bool)):
                            detailed_match_info_for_llm += f"- {key}: {value}\n"
                    detailed_match_info_for_llm += "\n"

                if match_squad:
                    detailed_match_info_for_llm += "Squads:\n"
                    for team_squad in match_squad:
                        team_name = team_squad.get('name', 'Unknown Team')
                        players = [player.get('name', 'N/A') for player in team_squad.get('players', [])]
                        detailed_match_info_for_llm += f"  - {team_name}: {', '.join(players)}\n"
                    detailed_match_info_for_llm += "\n"
            elif st.session_state.current_match_data and (match_intent and ("match_number" in match_intent or "match_name" in match_intent)):
                 # If LLM identified a match but it wasn't found in our data or ID was missing
                 detailed_match_info_for_llm = "I understood you were asking about a specific match, but I couldn't find its details or ID. Please ensure the match number/name is correct and try again."


            # --- Step 3: Combine all available context for the main LLM ---
            context_for_llm = ""
            if st.session_state.current_match_data:
                # Always provide the general list of current matches if available
                general_match_data_for_llm = "\n".join([
                    f"Match: {m.get('name', 'N/A')}, Status: {m.get('status', 'N/A')}, ID: {m.get('id', 'N/A')}"
                    for m in st.session_state.current_match_data
                ])
                context_for_llm += f"General Current Match Data:\n{general_match_data_for_llm}\n\n"
            
            if detailed_match_info_for_llm:
                context_for_llm += detailed_match_info_for_llm + "\n"

            if context_for_llm:
                full_prompt_to_gemini = f"{context_for_llm}User Query: {user_input}"
            else:
                full_prompt_to_gemini = user_input # If no context, just send user query

            st.write(f"DEBUG: Sending to LLM (first 200 chars): {full_prompt_to_gemini[:200]}...") # Debugging line

            # Send message to Gemini and get response
            response = st.session_state.chat.send_message(full_prompt_to_gemini, stream=True)
            
            # Iterate through the streamed response chunks
            full_bot_reply = ""
            for chunk in response:
                if chunk.text:
                    full_bot_reply += chunk.text
                    response_area.markdown(full_bot_reply + "▌") # Add a blinking cursor effect
            
            # Remove blinking cursor after full response
            response_area.markdown(full_bot_reply)
            bot_reply = full_bot_reply

        except Exception as e:
            bot_reply = f"Error communicating with Gemini or fetching match details: {e}"
            response_area.markdown(bot_reply)
            st.error(f"An error occurred: {e}") # Display error in Streamlit UI for debugging

    # Save assistant message to session state
    st.session_state.messages.append({"role": "assistant", "text": bot_reply})
    st.rerun() # Rerun to clear the input box and update chat history
