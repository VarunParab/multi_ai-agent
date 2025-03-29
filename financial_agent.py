from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
load_dotenv()
# phi.api=os.getenv("PHI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
## web search agent
web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=["Alway include sources"],
    show_tools_calls=True,
    markdown=True,

)

## Financial agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-70b-8192"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,

)

multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    model=Groq(id="llama3-70b-8192"),
    instructions=["Always include sources","Use table to display the data"],
    show_tool_calls=True,
    markdown=True,
)

try:
    multi_ai_agent.print_response("Provide an overview of NVIDIA and its stock recommendations.", stream=True)
except Exception as e:
    print(f"Error: {e}")
    print(f"Details: {getattr(e, 'failed_generation', 'No additional details available')}")
