from asyncio import Lock, CancelledError

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState

from websocket.web_socket_request_context import WebSocketRequestContext
from typing import Dict, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.load import dumpd
from psycopg_pool import AsyncConnectionPool

from dotenv import load_dotenv
import json
import importlib
import os
import logging

logger = logging.getLogger(__name__)

load_dotenv()

from typing import Any, Dict, TypedDict

class RequestHandler:
    def __init__(self, app: FastAPI):
        self.locks: Dict[int, Lock] = {}
        self.app = app
    
    def _get_lock(self, websocket: WebSocket) -> Lock:
        """Get or create a lock for a specific websocket connection"""
        ws_id = id(websocket)
        if ws_id not in self.locks:
            self.locks[ws_id] = Lock()
        return self.locks[ws_id]

    def _is_websocket_open(self, websocket: WebSocket) -> bool:
        """Check if the WebSocket connection is still open"""
        return websocket.client_state == WebSocketState.CONNECTED

    async def handle_request(self, message: dict, websocket: WebSocket):
        """Handle incoming WebSocket requests with proper locking and cancellation"""
        ws_id = id(websocket)
        lock = self._get_lock(websocket)
        
        async with lock:
            try:
                app, state = self.get_langgraph_app_and_state(message)
                config = {
                    "configurable": {
                        "thread_id": f"{message.get('thread_id')}"
                    }
                } 

                async for chunk in app.astream(state, config=config, stream_mode=["updates", "messages"]):
                    is_this_chunk_an_llm_message = isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == 'messages'
                    is_this_chunk_an_update_stream_type = isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == 'updates'
                    if is_this_chunk_an_llm_message:
                        message_chunk_from_llm = chunk[1][0] #AIMessageChunk object -> https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html
                    
                    
                    elif is_this_chunk_an_update_stream_type: # This means that LangGraph has given us a state update. This will often include a new message from the AI.
                        state_object = chunk[1]
                        logger.info(f"🧠🧠🧠 LangGraph Output (State Update): {state_object}")
                    
                        # Handle dynamic agent key - look for messages in any nested dict
                        messages = None
                        for agent_key, agent_data in state_object.items():
                            did_agent_have_a_message_for_us = isinstance(agent_data, dict) and 'messages' in agent_data
                            if did_agent_have_a_message_for_us:
                                messages = agent_data['messages'] #Question: is this ALL messages coming through, or just the latest AI message?

                                # Safe check for tool calls with better error handling
                                did_agent_evoke_a_tool = False
                                tool_calls = []
                                
                                if messages and len(messages) > 0:
                                    message = messages[0]
                                    if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
                                        tool_calls_data = message.additional_kwargs.get('tool_calls')
                                        if tool_calls_data:
                                            did_agent_evoke_a_tool = True
                                            tool_calls = tool_calls_data
                                            
                                            # Log tool call details
                                            if len(tool_calls) > 0:
                                                tool_call_object = tool_calls[0]
                                                tool_call_name = tool_call_object.get("name")
                                                tool_call_args = tool_call_object.get("args")
                                                logger.info(f"🔨🔨🔨 Tool Call Name: {tool_call_name}")
                                                logger.info(f"🔨🔨🔨 Tool Call Args: {tool_call_args}")

                                    # AIMessage is not serializable to JSON, so we need to convert it to a string.
                                    messages_as_string = [msg.content if hasattr(msg, 'content') else str(msg) for msg in messages]

                                    #NOTE: I found we're able to serialize AIMessage into dict using dumpd.
                                    try:
                                        base_message_as_dict = dumpd(message)["kwargs"]
                                    except Exception as e:
                                        logger.warning(f"Failed to serialize message: {e}")
                                        base_message_as_dict = {"content": str(message), "type": "ai"}

                                    # Only send if WebSocket is still open
                                    if self._is_websocket_open(websocket):
                                        await websocket.send_json({
                                            "type": message.type if hasattr(message, 'type') else "ai",
                                            "content": messages_as_string[0] if messages_as_string else "",
                                            "tool_calls": tool_calls,
                                            "base_message": base_message_as_dict
                                        })
                                break
                        
                        logger.info(f"LangGraph Output (State Update): {chunk}")

                        # chunk will look like this:
                        # {'llamabot': {'messages': [AIMessage(content='Hello! I hear you loud and clear. I'm LlamaBot, your full-stack Rails developer assistant. How can I help you today?', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'o4-mini-2025-04-16', 'service_tier': 'default'}, id='run--ce385bc4-fecb-4127-81d2-1da5814874f8')]}}

                    else:
                        logger.info(f"Workflow output: {chunk}")

            except CancelledError as e:
                logger.info("handle_request was cancelled")
                # Only send error message if WebSocket is still open
                if self._is_websocket_open(websocket):
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Cancelled!"
                    })
                raise e
            except Exception as e:
                logger.error(f"Error handling request: {str(e)}", exc_info=True)
                # Only send error message if WebSocket is still open
                if self._is_websocket_open(websocket):
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Error processing request: {str(e)}"
                    })
                raise e

    async def get_chat_history(self, thread_id: str):
        # For chat history, we don't need a specific agent, just get any workflow to access the checkpointer
        # This is a bit of a hack - we should refactor this to not need the workflow for just getting history
        try:
            app, _ = self.get_langgraph_app_and_state({"agent_name": "llamabot", "message": "", "api_token": "", "agent_prompt": ""})
            config = {"configurable": {"thread_id": thread_id}}
            state_history = await app.aget_state(config=config)
            return state_history[0] #gets the actual state.
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return None

    def get_or_create_checkpointer(self):
        """Get persistent checkpointer, creating once if needed"""

        if self.app.state.async_checkpointer is not None:
            return self.app.state.async_checkpointer
        
        db_uri = os.getenv("DB_URI")
        self.app.state.async_checkpointer = MemorySaver() # save in RAM if postgres is not available
        if db_uri:
            try:
                # Create connection pool and PostgresSaver directly
                pool = AsyncConnectionPool(db_uri)
                self.app.state.async_checkpointer = AsyncPostgresSaver(pool)
                self.app.state.async_checkpointer.setup()  # Make this async
                logger.info("✅✅✅ Using PostgreSQL persistence!")
            except Exception as e:
                logger.warning(f"Failed to connect to PostgreSQL: {e}. Using MemorySaver.")
        else:
            logger.info("❌❌❌ No DB_URI found. Using MemorySaver for session-based persistence.")
        
        return self.app.state.async_checkpointer

    def cleanup_connection(self, websocket: WebSocket):
        """Clean up resources when a connection is closed"""
        ws_id = id(websocket)
        if ws_id in self.locks:
            del self.locks[ws_id]

    def get_workflow_from_langgraph_json(self, message: dict):
        # Try different paths for langgraph.json
        possible_paths = ["../langgraph.json", "../../langgraph.json", "langgraph.json"]
        langgraph_json = None
        
        for path in possible_paths:
            try:
                with open(path, 'r') as f:
                    langgraph_json = json.load(f)
                    break
            except FileNotFoundError:
                continue
        
        if langgraph_json is None:
            raise FileNotFoundError("Could not find the agent from the langgraph.json file")
        
        langgraph_workflow = langgraph_json.get("graphs").get(message.get("agent_name"))
        return langgraph_workflow
    
    def get_langgraph_app_and_state(self, message: dict):
        app = None
        state = message
        if message.get("agent_name") is not None:
            langgraph_workflow = self.get_workflow_from_langgraph_json(message)
            if langgraph_workflow is not None:
                app = self.get_app_from_workflow_string(langgraph_workflow)
                
                # Create messages from the message content
                messages = [HumanMessage(content=message.get("message"))]
                
                # Start with the transformed messages field
                state = {"messages": messages}
                
                # Pass through ALL fields except the ones used for system routingError processing request: cannot access local variable 'state' where it is not associated with a value
                system_routing_fields = {
                    "message",      # We transformed this into messages
                    "agent_name",   # Used for workflow routing only
                    "thread_id"     # Used for LangGraph config only
                }
                
                # Pass everything else through naturally
                for key, value in message.items():
                    if key not in system_routing_fields:
                        state[key] = value

                logger.info(f"Created state with keys: {list(state.keys())}")
                
            else:
                raise ValueError(f"Unknown workflow: {message.get('agent_name')}")
        
        return app, state
    
    def get_app_from_workflow_string(self, workflow_string: str):
        # Split the path into module path and function name
        module_path, function_name = workflow_string.split(':')
        # Remove './' if present and convert path to module format
        if module_path.startswith('./'):
            module_path = module_path[2:]
        module_path = module_path.replace('/', '.').replace('.py', '')

        # Dynamically import the module and get the function
        module = importlib.import_module(module_path)
        workflow_builder = getattr(module, function_name)

        # Build the workflow using the imported function
        return workflow_builder(checkpointer=self.get_or_create_checkpointer())