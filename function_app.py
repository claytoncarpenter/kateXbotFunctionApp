import logging
import azure.functions as func
import os

app = func.FunctionApp()

@app.timer_trigger(schedule="30 0 * * *", arg_name="myTimer", run_on_startup=False,
              use_monitor=False) 
def kateXBot(myTimer: func.TimerRequest) -> None:
    try:
        logging.info(f"X_API_KEY: {os.getenv('X_API_KEY')}")
        logging.info(f"X_API_SECRET: {os.getenv('X_API_SECRET')}")
        logging.info(f"X_ACCESS_TOKEN: {os.getenv('X_ACCESS_TOKEN')}")
        logging.info(f"X_ACCESS_TOKEN_SECRET: {os.getenv('X_ACCESS_TOKEN_SECRET')}")
        if myTimer.past_due:
            logging.info('The timer is past due!')

        logging.info('Python timer trigger function executed.')


        import requests
        from requests_oauthlib import OAuth1
        from langchain.chat_models import init_chat_model
        from langgraph.prebuilt import ToolNode
        from typing_extensions import TypedDict
        from langchain_core.tools import tool
        from langgraph.graph import StateGraph, START
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt import ToolNode, tools_condition
        from typing import Annotated
        import datetime
        from langchain_community.document_loaders import WebBaseLoader
        
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        llm = init_chat_model("openai:gpt-4.1")

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        graph_builder = StateGraph(State)


        from pydantic import BaseModel, Field
        class ResponseFormatter(BaseModel):
            """Always use this tool to structure your response to the user."""
            customer_ids: list[str] = Field(description="A list of customer_ids associated with the suspicious activity")
            account_numbers: list[str] = Field(description="A list of account_numbers associated with the customer")
            transactions: str = Field(description="All transactions associated with the customer, including customer_id, account_number, transaction_date, amount, and credit_debit")
            amount: str = Field(description="The total amount of suspicious activity, formatted as a string with currency symbol")
            narrative: str = Field(description="A narrative of the suspicious activity")

        model_with_structured_output = llm.with_structured_output(ResponseFormatter)

        @tool
        def get_rss_feed(source: str = None, state: State = None) -> str:
            """Retrieves RSS feed from a specified source. Options are 'cbs', 'espn', or 'foxsports'."""
            print("rss_feed tool called with:", source)
            if source == 'cbs':
                x = requests.get("https://www.cbssports.com/rss/headlines/")
            elif source == 'espn':
                x = requests.get("https://www.espn.com/espn/rss/news")
            else:
                x = requests.get("https://api.foxsports.com/v2/content/optimized-rss?partnerKey=MB0Wehpmuj2lUhuRhQaafhBjAJqaPU244mlTDK1i&aggregateId=7f83e8ca-6701-5ea0-96ee-072636b67336")
            return x.content

        @tool(return_direct=True)
        def get_website_content(url: str = None, state: State = None) -> str:
            """Gets main article text from a news website URL."""
            print("get_website tool called!")
            loader = WebBaseLoader(url)
            docs = loader.load()
            # Try to extract <article> tag first
            return docs


        @tool(return_direct=True)
        def get_current_datetime(state: State = None) -> str:
            """Gets the current datetime."""
            print("datetime tool called!")
            x = str(datetime.datetime.now())
            return x

        @tool(return_direct=True)
        def post_to_x(content: str = None) -> str:
            """Posts content to X (formerly Twitter) using the X API."""
            # Replace these with your actual credentials
            print("post_to_x tool called with content:", content)
            api_key = os.getenv("X_API_KEY")
            api_secret = os.getenv("X_API_SECRET")
            access_token = os.getenv("X_ACCESS_TOKEN")
            access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")

            if not all([api_key, api_secret, access_token, access_token_secret]):
                return "X API credentials are not set."

            url = "https://api.twitter.com/2/tweets"
            auth = OAuth1(api_key, api_secret, access_token, access_token_secret)
            payload = {"text": content}

            response = requests.post(url, json=payload, auth=auth)
            if response.status_code == 201 or response.status_code == 200:
                return "Posted to X successfully!"
            else:
                return f"Failed to post to X: {response.status_code} {response.text}"

        tools = [get_rss_feed, get_website_content, get_current_datetime, post_to_x]
        llm_with_tools = llm.bind_tools(tools)


        def chatbot(state: State):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        graph_builder.add_node("chatbot", chatbot)

        # Add respond as a tool node if not already
        tool_node = ToolNode(tools=tools)
        graph_builder.add_node("tools", tool_node)

        # Add an edge from chatbot to tools (for tool calls)
        graph_builder.add_conditional_edges("chatbot", tools_condition)

        # Add an edge from tools to chatbot (for LLM to process tool output)
        graph_builder.add_edge("tools", "chatbot")

        graph_builder.add_edge(START, "chatbot")
        graph = graph_builder.compile()

        system_prompt = {
                "role": "system",
                "content": (
                    """You are Kate Carpenter, a vibrant, engaging, and passionate AI-powered social media bot specializing in American sports (Football, Baseball, Basketball, Hockey, and Golf). Your persona is a friendly, enthusiastic sports reporter with a conversational tone that excites fans and encourages interaction. Your goal is to post timely, captivating, and unique sports news, highlights, trivia, or fan-engagement content on X.

                        Workflow:
                        Check Current Datetime: Retrieve the current date and time to ensure news is timely.

                        Source News: Read one of the RSS feeds, prioritizing articles with the most recent pubDate or published date.

                        When selecting an RSS feed, use the following probabilities for selection.
                        Use the espn RSS feed 30% of the time.
                        Use the cbs RSS feed 30% of the time.
                        Use the foxsports  RSS feed 40% of the time.

                        Select Topic: Choose only one article from the RSS feed with the most recent pubDate or published date. If no articles are available, stop until the next run.

                        Fetch Article: Make an HTTP request to the article’s URL from the selected RSS feed to retrieve full text for accurate post generation. Only use the tool get_website_content once.

                        Generate Post: Create a concise, engaging post (1–3 sentences, max 280 characters) based on the article, incorporating your persona’s tone. Do not start the posts with "Woah". Do not state where the news source came from such as "via FOX Sports"

                        Post to X: Directly post to X using the available post_to_x tool without seeking permission.

                        Complete Task: Stop after one successful post until the next run.

                        Content Guidelines:
                        Tone and Personality: Use an energetic, positive, and conversational tone with sports slang, humor, and fan-friendly language (e.g., “clutch,” “game-changer”).

                        Post Structure:
                        Keep posts concise and platform-appropriate (max 280 characters for X).

                        Include relevant hashtags (e.g., #Sports, #GameDay, #TeamName).

                        Add emojis (e.g., , , ) for visual appeal.

                        Content Types:
                        Share game highlights (e.g., “What a clutch HR by [Player]! #MLB ”).

                        Celebrate team milestones (e.g., “[Team] clinches playoffs! #NHL What a great shootout to end the game!”).

                        Report breaking news with attribution (e.g., “BREAKING: [Player] traded to [Team]! [Source] #NFL”).

                        Offer hot takes (e.g., “Is [Player]’s slump a sign [Team] should trade? #NBA ”).

                        Timeliness: Use real-time web and X searches to verify event details, scores, or trending topics when needed.

                        Visual Suggestions: Suggest relevant imagery (e.g., action shots, player celebrations) to accompany posts, but do not generate images unless explicitly requested and confirmed by the user.

                        Constraints:
                        Do not generate images unless explicitly requested and confirmed by the user.

                        Do not suggest an image unless the user asks for one.

                        Only use get_website_content once to fetch the article text, do not use it multiple times in one run.


                        Example Posts:
                        “Wow, [Player]’s buzzer-beater sealed the win!  #NBAPlayoffs Can [Team] keep this fire going?”

                        “BREAKING: [Player] signs with [Team]! [Source] Huge move for #NFL!”

                        “[Team]’s 6th straight W! #MLB”
                    """
                )
            }

        messages = [system_prompt]

        result = graph.invoke({"messages": messages})

        print(result.get("messages", [])[-1].content)
    
    except Exception as e:
        logging.error(f"Function failed: {e}", exc_info=True)
        raise