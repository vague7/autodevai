from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from functools import partial
from typing import Optional
import os
import logging

load_dotenv()

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode, InjectedState

import requests
import json
from typing import Annotated

logger = logging.getLogger(__name__)

# Warning: Brittle - None type will break this when it's injected into the state for the tool call, and it silently fails. So if it doesn't map state types properly from the frontend, it will break. (must be exactly what's defined here).
class LlamaPressState(MessagesState): 
    api_token: str
    agent_prompt: str
    page_id: str
    current_page_html: str
    selected_element: Optional[str]
    javascript_console_errors: Optional[str]

# Tools
@tool
def write_html_page(full_html_document: str, message_to_user: str, internal_thoughts: str, state: Annotated[dict, InjectedState]) -> str:
   """
   Write an HTML page to the filesystem.
   full_html_document is the full HTML document to write to the filesystem, including CSS and JavaScript.
   message_to_user is a string to tell the user what you're doing.
   internal_thoughts are your thoughts about the command.
   """
   # Debug logging
   logger.info(f"API TOKEN: {state.get('api_token')}")
   logger.info(f"Page ID: {state.get('page_id')}")
   logger.info(f"State keys: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}")
   
   # Configuration
   LLAMAPRESS_API_URL = os.getenv("LLAMAPRESS_API_URL")

   # Get page_id from state, with fallback
   page_id = state.get('page_id')
   if not page_id:
       return "Error: page_id is required but not provided in state"
   
   API_ENDPOINT = f"{LLAMAPRESS_API_URL}/pages/{page_id}.json"
   try:
       # Get API token from state
       api_token = state.get('api_token')
       if not api_token:
           return "Error: api_token is required but not provided in state"
       
       print(f"API TOKEN: LlamaBot {api_token}")

       # Make HTTP request to Rails API
       response = requests.put(
           API_ENDPOINT,
           json={'content': full_html_document},
           headers={'Content-Type': 'application/json', 'Authorization': f'LlamaBot {api_token}'},
           timeout=30  # 30 second timeout
       )

       # Parse the response
       if response.status_code == 200:
           data = response.json()
           return json.dumps(data, ensure_ascii=False, indent=2)
       else:
           return f"HTTP Error {response.status_code}: {response.text}"

   except requests.exceptions.ConnectionError:
       return "Error: Could not connect to Rails server. Make sure your Rails app is running."

   except requests.exceptions.Timeout:
       return "Error: Request timed out. The Rails request may be taking too long to execute."

   except requests.exceptions.RequestException as e:
       return f"Request Error: {str(e)}"

   except json.JSONDecodeError:
       return f"Error: Invalid JSON response from server. Raw response: {response.text}"

   except Exception as e:
       return f"Unexpected Error: {str(e)}"

   print("Write to filesystem!")
   return "HTML page written to filesystem!"

# Global tools list
tools = [write_html_page]

# Node
def llamapress(state: LlamaPressState):
   additional_instructions = state.get("agent_prompt")
   # System message
   sys_msg = SystemMessage(content=f"""You are LlamaPress, a helpful AI assistant.
                        In normal chat conversations, feel free to implement markdown formatting to make your responses more readable, if it's appropriate.
                        Here are additional instructions provided by the user: <ADDITIONAL_STATE_AND_CONTEXT> {state} </ADDITIONAL_STATE_AND_CONTEXT> 
                        <USER_INSTRUCTIONS> {additional_instructions} </USER_INSTRUCTIONS>""")

   llm = ChatOpenAI(model="o4-mini")
   llm_with_tools = llm.bind_tools(tools)
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

def build_workflow(checkpointer=None):
    # Graph
    builder = StateGraph(LlamaPressState)

    # Define nodes: these do the work
    builder.add_node("llamapress", llamapress)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "llamapress")
    builder.add_conditional_edges(
        "llamapress",
        # If the latest message (result) from llamapress is a tool call -> tools_condition routes to tools
        # If the latest message (result) from llamapress is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "llamapress")

    react_graph = builder.compile(checkpointer=checkpointer)

    return react_graph   

# Tools
# @tool
# def run_rails_console_command(rails_console_command: str, message_to_user: str, internal_thoughts: str, state: Annotated[LlamaBotState, InjectedState]) -> str:

# eventually make this a class 
# make it reusable/configurable to many workflows
# async def build_workflow_saas(context: WebSocketRequestContext):
#     workflow = StateGraph(LlamaPressWorkflowState)
    
#     #router setup
#     workflow.set_entry_point("router")
#     workflow.add_node("router", partial(route_request, context=context))
    
#     #router condition
#     workflow.add_conditional_edges(
#         "router",
#          lambda x: x["next"], 
#          {
#              "respond_naturally": "respond_naturally",
#              "coding_agent": "coding_agent_saas",
#              "end": "end"
#          }
#     )

#     workflow.add_node("end", partial(end_workflow, context=context))
#     workflow.add_node("coding_agent_saas", partial(coding_agent_saas, context=context))
#     workflow.add_node("respond_naturally", partial(respond_naturally, context=context))
#     workflow.add_node("remove_element", partial(remove_element, context=context))
#     workflow.add_node("improve_design", partial(improve_design, context=context))
#     workflow.add_node("clone_webpage", partial(clone, context=context))

#     # Add end as a valid endpoint
#     workflow.set_finish_point("end")

#     workflow.add_conditional_edges(
#         "coding_agent_saas",
#          lambda x: x["next"], 
#          {
#             "remove_element" : "remove_element",
#             "improve_design" : "improve_design",
#             "clone_router_agent" : "clone_router_agent",
#          }
#     )

#     workflow.add_node("clone_router_agent", partial(clone_router_agent, context=context))

#     workflow.add_conditional_edges(
#         "clone_router_agent",
#          lambda x: x["next"], 
#          {
#             "deep_vision_agent" : "deep_vision_agent",
#             "clone_webpage" : "clone_webpage",
#             "brand_clone_agent" : "brand_clone_agent"
#          }
#     )

#     workflow.add_node("deep_vision_agent", partial(deep_vision_agent, context=context))
#     workflow.add_edge("deep_vision_agent", "deep_clone_agent")

#     workflow.add_node("deep_clone_agent", partial(deep_clone_agent, context=context))
#     workflow.add_edge("deep_clone_agent", "end")

#     workflow.add_node("brand_clone_agent", partial(brand_clone_agent, context=context))
#     workflow.add_edge("brand_clone_agent", "end")

#     workflow.add_edge("respond_naturally", "end")
#     workflow.add_edge("remove_element", "end")
#     workflow.add_edge("clone_webpage", "end") #end after clone

#     workflow.add_conditional_edges(
#         "improve_design",
#          lambda x: x["next"], 
#          {
#             "improve_selected_element" : "improve_selected_element",
#             "design_requirements_agent" : "design_requirements_agent",
#             "debug_agent" : "debug_agent"
#          }
#     )

#     workflow.add_node("improve_selected_element", partial(improve_selected_element, context=context))
#     workflow.add_edge("improve_selected_element", "end")
    
#     workflow.add_node("design_requirements_agent", partial(design_requirements_agent, context=context))
#     workflow.add_edge("design_requirements_agent", "make_wholesale_change")

#     workflow.add_node("make_wholesale_change", partial(make_wholesale_change, context=context))
#     workflow.add_edge("make_wholesale_change", "end")

#     workflow.add_node("debug_agent", partial(debug_agent, context=context))
#     workflow.add_edge("debug_agent", "multiple_fragments")

#     workflow.add_node("multiple_fragments", partial(multiple_fragments, context=context))
#     workflow.add_edge("multiple_fragments", "end")

#     # We have to do this workaround, because otherwise Langgraph Studio won't compile and it will complain that 
#     # a checkpointer is already defined when running langgraph dev. (They set up their own checkpointer when running langgraph dev)
#     # I think this is because LangChain wants to incentivize people to use their own checkpointer
#     # that's set up when hosting their own LangGraph server.
#     # But we're not using their hosted LangGraph server, and we want to debug using langgraph dev, so we have do this workaround.
#     checkpointer = None
#     if type(context) != dict: # context will be a dict when we're running langgraph dev, but if we're running fastapi, context will be a WebSocketRequestContext object
#         checkpointer = context.langgraph_checkpointer
    
#     # It's confusing why context isn't passed as a parameter to the workflow.compile function.
#     # And it's because we're already passing it into each node using the partial function.
#     app = workflow.compile(checkpointer=checkpointer, debug=True)
#     return app