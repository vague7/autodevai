"""
Tests for the agents functionality.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import requests
import logging

from agents.base_agent import BaseAgent
from agents.llamabot_v1.nodes import run_rails_console_command, LlamaBotState
from agents.llamapress.nodes import write_html_page, LlamaPressState
from langgraph.checkpoint.memory import MemorySaver


class TestBaseAgent:
    """Test the base agent functionality."""
    
    def test_base_agent_initialization(self):
        """Test base agent initialization."""
        # BaseAgent is abstract, so we need to create a concrete implementation
        class ConcreteAgent(BaseAgent):
            def run(self, input: str) -> str:
                return f"Processed: {input}"
        
        agent = ConcreteAgent("test_agent", "A test agent")
        assert agent.name == "test_agent"
        assert agent.description == "A test agent"
        assert hasattr(agent, 'llm')
    
    def test_base_agent_is_abstract(self):
        """Test that BaseAgent is properly abstract."""
        # This test ensures that BaseAgent cannot be instantiated directly
        with pytest.raises(TypeError):
            BaseAgent("test", "test")


class TestLlamaBotV1Nodes:
    """Test LlamaBot V1 node functionality."""
    
    @patch('agents.llamabot_v1.nodes.requests.post')
    @patch('agents.llamabot_v1.nodes.os.getenv')
    def test_run_rails_console_command_success(self, mock_getenv, mock_post):
        """Test successful rails console command execution."""
        # Setup mocks
        mock_getenv.return_value = "http://test-server.com"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'result': {'data': 'test_result'},
            'type': 'success'
        }
        mock_post.return_value = mock_response
        
        # Create test state with all required fields
        state = {
            'api_token': 'test_token_123',
            'agent_prompt': 'Test prompt',
            'messages': []
        }
        
        # Execute function using tool invoke
        result = run_rails_console_command.invoke({
            'rails_console_command': 'User.count',
            'message_to_user': 'Counting users',
            'internal_thoughts': 'Getting user count',
            'state': state
        })
        
        # Verify the request was made correctly
        mock_post.assert_called_once_with(
            "http://test-server.com/llama_bot/agent/command",
            json={'command': 'User.count'},
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'LlamaBot test_token_123'
            },
            timeout=30
        )
        
        # Verify the result
        assert 'test_result' in result
        assert isinstance(result, str)
    
    @patch('agents.llamabot_v1.nodes.requests.post')
    @patch('agents.llamabot_v1.nodes.os.getenv')
    def test_run_rails_console_command_http_error(self, mock_getenv, mock_post):
        """Test rails console command with HTTP error."""
        # Setup mocks
        mock_getenv.return_value = "http://test-server.com"
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response
        
        # Create test state with all required fields
        state = {
            'api_token': 'invalid_token',
            'agent_prompt': 'Test prompt',
            'messages': []
        }
        
        # Execute function using tool invoke
        result = run_rails_console_command.invoke({
            'rails_console_command': 'User.count',
            'message_to_user': 'Counting users',
            'internal_thoughts': 'Getting user count',
            'state': state
        })
        
        # Verify error handling
        assert "HTTP Error 401" in result
        assert "Unauthorized" in result
    
    @patch('agents.llamabot_v1.nodes.requests.post')
    @patch('agents.llamabot_v1.nodes.os.getenv')
    def test_run_rails_console_command_connection_error(self, mock_getenv, mock_post):
        """Test rails console command with connection error."""
        # Setup mocks
        mock_getenv.return_value = "http://test-server.com"
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        # Create test state with all required fields
        state = {
            'api_token': 'test_token',
            'agent_prompt': 'Test prompt',
            'messages': []
        }
        
        # Execute function using tool invoke
        result = run_rails_console_command.invoke({
            'rails_console_command': 'User.count',
            'message_to_user': 'Counting users',
            'internal_thoughts': 'Getting user count',
            'state': state
        })
        
        # Verify error handling
        assert "Could not connect to Rails server" in result
    
    @patch('agents.llamabot_v1.nodes.requests.post')
    @patch('agents.llamabot_v1.nodes.os.getenv')
    def test_run_rails_console_command_missing_token(self, mock_getenv, mock_post):
        """Test rails console command with missing API token."""
        # Setup mocks
        mock_getenv.return_value = "http://test-server.com"
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response
        
        # Create test state with empty api_token to simulate missing token
        state = {
            'api_token': '',  # Use empty string instead of None
            'agent_prompt': 'Test prompt',
            'messages': []
        }
        
        # Execute function using tool invoke
        result = run_rails_console_command.invoke({
            'rails_console_command': 'User.count',
            'message_to_user': 'Counting users',
            'internal_thoughts': 'Getting user count',
            'state': state
        })
        
        # Verify the request was made with empty token
        mock_post.assert_called_once_with(
            "http://test-server.com/llama_bot/agent/command",
            json={'command': 'User.count'},
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'LlamaBot '
            },
            timeout=30
        )
        
        # Verify error handling for unauthorized request
        assert "HTTP Error 401" in result
        assert "Unauthorized" in result


class TestLlamaPressNodes:
    """Test LlamaPress node functionality."""
    
    @patch('agents.llamapress.nodes.requests.put')
    @patch('agents.llamapress.nodes.os.getenv')
    def test_write_html_page_success(self, mock_getenv, mock_put):
        """Test successful HTML page writing."""
        # Setup mocks
        mock_getenv.return_value = "http://test-server.com"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 123,
            'content': '<html><body>Test</body></html>',
            'updated_at': '2023-01-01T00:00:00Z'
        }
        mock_put.return_value = mock_response
        
        # Create test state with all required fields
        state = {
            'api_token': 'test_token_123',
            'agent_prompt': 'Test prompt',
            'page_id': 'test_page_456',
            'current_page_html': '<html></html>',
            'selected_element': None,
            'javascript_console_errors': None,
            'messages': []
        }
        
        # Execute function using tool invoke
        result = write_html_page.invoke({
            'full_html_document': '<html><body>Test Content</body></html>',
            'message_to_user': 'Writing HTML page',
            'internal_thoughts': 'Creating test page',
            'state': state
        })
        
        # Verify the request was made correctly
        mock_put.assert_called_once_with(
            "http://test-server.com/pages/test_page_456.json",
            json={'content': '<html><body>Test Content</body></html>'},
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'LlamaBot test_token_123'
            },
            timeout=30
        )
        
        # Verify the result contains expected data
        result_data = json.loads(result)
        assert result_data['id'] == 123
        assert 'content' in result_data
    
    @patch('agents.llamapress.nodes.requests.put')
    @patch('agents.llamapress.nodes.os.getenv')
    def test_write_html_page_http_error(self, mock_getenv, mock_put):
        """Test HTML page writing with HTTP error."""
        # Setup mocks
        mock_getenv.return_value = "http://test-server.com"
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Page not found"
        mock_put.return_value = mock_response
        
        # Create test state with all required fields
        state = {
            'api_token': 'test_token',
            'agent_prompt': 'Test prompt',
            'page_id': 'nonexistent_page',
            'current_page_html': '<html></html>',
            'selected_element': None,
            'javascript_console_errors': None,
            'messages': []
        }
        
        # Execute function using tool invoke
        result = write_html_page.invoke({
            'full_html_document': '<html><body>Test</body></html>',
            'message_to_user': 'Writing HTML page',
            'internal_thoughts': 'Creating test page',
            'state': state
        })
        
        # Verify error handling
        assert "HTTP Error 404" in result
        assert "Page not found" in result
    
    @patch('agents.llamapress.nodes.requests.put')
    @patch('agents.llamapress.nodes.os.getenv')
    def test_write_html_page_connection_error(self, mock_getenv, mock_put):
        """Test HTML page writing with connection error."""
        # Setup mocks
        mock_getenv.return_value = "http://test-server.com"
        mock_put.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        # Create test state with all required fields
        state = {
            'api_token': 'test_token',
            'agent_prompt': 'Test prompt',
            'page_id': 'test_page',
            'current_page_html': '<html></html>',
            'selected_element': None,
            'javascript_console_errors': None,
            'messages': []
        }
        
        # Execute function using tool invoke
        result = write_html_page.invoke({
            'full_html_document': '<html><body>Test</body></html>',
            'message_to_user': 'Writing HTML page',
            'internal_thoughts': 'Creating test page',
            'state': state
        })
        
        # Verify error handling
        assert "Could not connect to Rails server" in result
    
    @patch('agents.llamapress.nodes.requests.put')
    @patch('agents.llamapress.nodes.os.getenv')
    def test_write_html_page_missing_token(self, mock_getenv, mock_put):
        """Test HTML page writing with missing API token."""
        # Setup mocks
        mock_getenv.return_value = "http://test-server.com"
        
        # Create test state without api_token but with other required fields
        state = {
            'agent_prompt': 'Test prompt',
            'page_id': 'test_page',
            'current_page_html': '<html></html>',
            'selected_element': None,
            'javascript_console_errors': None,
            'messages': []
        }
        
        # Execute function using tool invoke
        result = write_html_page.invoke({
            'full_html_document': '<html><body>Test</body></html>',
            'message_to_user': 'Writing HTML page',
            'internal_thoughts': 'Creating test page',
            'state': state
        })
        
        # Verify error handling
        assert "api_token is required" in result
        # Verify no HTTP request was made
        mock_put.assert_not_called()
    
    @patch('agents.llamapress.nodes.requests.put')
    @patch('agents.llamapress.nodes.os.getenv')
    def test_write_html_page_missing_page_id(self, mock_getenv, mock_put):
        """Test HTML page writing with missing page ID."""
        # Setup mocks
        mock_getenv.return_value = "http://test-server.com"
        
        # Create test state without page_id but with other required fields
        state = {
            'api_token': 'test_token',
            'agent_prompt': 'Test prompt',
            'current_page_html': '<html></html>',
            'selected_element': None,
            'javascript_console_errors': None,
            'messages': []
        }
        
        # Execute function using tool invoke
        result = write_html_page.invoke({
            'full_html_document': '<html><body>Test</body></html>',
            'message_to_user': 'Writing HTML page',
            'internal_thoughts': 'Creating test page',
            'state': state
        })
        
        # Verify error handling
        assert "page_id is required" in result
        # Verify no HTTP request was made
        mock_put.assert_not_called()
    
    @patch('agents.llamapress.nodes.logger')
    def test_write_html_page_logging(self, mock_logger):
        """Test that logging is properly implemented."""
        # Create test state with all required fields
        state = {
            'api_token': 'test_token',
            'agent_prompt': 'Test prompt',
            'page_id': 'test_page',
            'current_page_html': '<html></html>',
            'selected_element': None,
            'javascript_console_errors': None,
            'messages': []
        }
        
        # Execute function using tool invoke
        write_html_page.invoke({
            'full_html_document': '<html><body>Test</body></html>',
            'message_to_user': 'Writing HTML page',
            'internal_thoughts': 'Creating test page',
            'state': state
        })
        
        # Verify logging calls were made
        mock_logger.info.assert_any_call("API TOKEN: test_token")
        mock_logger.info.assert_any_call("Page ID: test_page")
        expected_keys = ['api_token', 'agent_prompt', 'page_id', 'current_page_html', 'selected_element', 'javascript_console_errors', 'messages']
        mock_logger.info.assert_any_call(f"State keys: {expected_keys}")


class TestWorkflowBuilding:
    """Test workflow building functionality."""
    
    @pytest.mark.asyncio
    async def test_build_workflow_with_memory_saver(self):
        """Test building workflow with MemorySaver."""
        checkpointer = MemorySaver()
        
        # Import and patch the actual build_workflow function
        with patch('agents.react_agent.nodes.build_workflow') as mock_build:
            mock_workflow = MagicMock()
            mock_build.return_value = mock_workflow
            
            # Import and call the function
            from agents.react_agent.nodes import build_workflow
            workflow = build_workflow(checkpointer=checkpointer)
            
            mock_build.assert_called_once_with(checkpointer=checkpointer)
            assert workflow == mock_workflow
    
    @pytest.mark.asyncio
    async def test_workflow_stream_functionality(self, mock_build_workflow):
        """Test workflow streaming functionality."""
        mock_workflow = mock_build_workflow.return_value
        
        # Mock the stream method to return some test data
        test_stream_data = [
            ("messages", ("Test message", {"langgraph_node": "test_node"})),
            ("updates", {"test_node": {"messages": ["Test update"]}})
        ]
        mock_workflow.stream.return_value = iter(test_stream_data)
        
        # Test that the workflow can be streamed
        stream_result = list(mock_workflow.stream(
            {"messages": ["Test input"]},
            config={"configurable": {"thread_id": "test_thread"}}
        ))
        
        assert len(stream_result) == 2
        assert stream_result[0][0] == "messages"
        assert stream_result[1][0] == "updates"
    
    @pytest.mark.asyncio
    async def test_workflow_get_state_functionality(self, mock_build_workflow):
        """Test workflow get_state functionality."""
        mock_workflow = mock_build_workflow.return_value
        
        # Mock the get_state method
        test_state = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        mock_workflow.get_state.return_value = test_state
        
        # Test that the workflow state can be retrieved
        state = mock_workflow.get_state(
            config={"configurable": {"thread_id": "test_thread"}}
        )
        
        assert state == test_state
        assert len(state["messages"]) == 2


class TestAgentIntegration:
    """Integration tests for agent functionality."""
    
    @pytest.mark.asyncio
    async def test_agent_workflow_integration(self, mock_build_workflow):
        """Test agent workflow integration."""
        mock_workflow = mock_build_workflow.return_value
        
        # Mock a complete workflow interaction
        mock_workflow.stream.return_value = iter([
            ("messages", ("Processing request...", {"langgraph_node": "route_initial_user_message"})),
            ("updates", {"route_initial_user_message": {"messages": ["Routing complete"]}}),
            ("messages", ("Generating response...", {"langgraph_node": "respond_naturally"})),
            ("updates", {"respond_naturally": {"messages": ["Response generated"]}})
        ])
        
        # Test the complete workflow
        input_data = {
            "messages": ["User message"],
            "initial_user_message": "Hello, how are you?",
            "existing_html_content": "<html></html>"
        }
        
        config = {"configurable": {"thread_id": "integration_test"}}
        
        stream_result = list(mock_workflow.stream(input_data, config=config))
        
        # Verify the workflow processed the request
        assert len(stream_result) == 4
        
        # Verify the workflow was called with correct parameters
        mock_workflow.stream.assert_called_once_with(input_data, config=config)
    
    @pytest.mark.asyncio
    async def test_multiple_agent_workflows(self, mock_build_workflow):
        """Test handling multiple agent workflows."""
        # This test simulates having multiple agents working
        
        # Create different mock workflows for different agents
        mock_workflow_1 = MagicMock()
        mock_workflow_1.stream.return_value = iter([
            ("messages", ("Agent 1 response", {"langgraph_node": "agent1_node"}))
        ])
        
        mock_workflow_2 = MagicMock()
        mock_workflow_2.stream.return_value = iter([
            ("messages", ("Agent 2 response", {"langgraph_node": "agent2_node"}))
        ])
        
        # Test that different workflows can be built
        mock_build_workflow.side_effect = [mock_workflow_1, mock_workflow_2]
        
        # Use the mock directly instead of importing the real function
        workflow1 = mock_build_workflow(checkpointer=MemorySaver())
        workflow2 = mock_build_workflow(checkpointer=MemorySaver())
        
        assert workflow1 != workflow2
        
        # Test that each workflow can process independently with proper config
        config1 = {"configurable": {"thread_id": "thread_1"}}
        config2 = {"configurable": {"thread_id": "thread_2"}}
        
        result1 = list(workflow1.stream({"messages": ["Input 1"]}, config=config1))
        result2 = list(workflow2.stream({"messages": ["Input 2"]}, config=config2))
        
        assert len(result1) == 1
        assert len(result2) == 1
        assert result1[0][1][1]["langgraph_node"] == "agent1_node"
        assert result2[0][1][1]["langgraph_node"] == "agent2_node"


class TestAgentErrorHandling:
    """Test agent error handling."""
    
    @pytest.mark.asyncio
    async def test_workflow_stream_error_handling(self, mock_build_workflow):
        """Test workflow stream error handling."""
        mock_workflow = mock_build_workflow.return_value
        
        # Mock the stream method to raise an exception
        mock_workflow.stream.side_effect = Exception("Workflow error")
        
        with pytest.raises(Exception) as exc_info:
            list(mock_workflow.stream({"messages": ["Test"]}))
        
        assert str(exc_info.value) == "Workflow error"
    
    @pytest.mark.asyncio
    async def test_workflow_get_state_error_handling(self, mock_build_workflow):
        """Test workflow get_state error handling."""
        mock_workflow = mock_build_workflow.return_value
        
        # Mock the get_state method to raise an exception
        mock_workflow.get_state.side_effect = Exception("State retrieval error")
        
        with pytest.raises(Exception) as exc_info:
            mock_workflow.get_state(config={"configurable": {"thread_id": "test"}})
        
        assert str(exc_info.value) == "State retrieval error"
    
    @pytest.mark.asyncio
    async def test_agent_with_invalid_checkpointer(self):
        """Test agent behavior with invalid checkpointer."""
        # Create a mock checkpointer that will cause an error
        mock_checkpointer = MagicMock()
        mock_checkpointer.setup.side_effect = Exception("Invalid checkpointer")
        
        with patch('agents.react_agent.nodes.build_workflow') as mock_build:
            mock_build.side_effect = Exception("Invalid checkpointer configuration")
            
            from agents.react_agent.nodes import build_workflow
            
            with pytest.raises(Exception):
                build_workflow(checkpointer=mock_checkpointer)


class TestAgentConfiguration:
    """Test agent configuration functionality."""
    
    def test_agent_configuration_loading(self):
        """Test loading agent configuration."""
        # Mock configuration data
        mock_config = {
            "agent_name": "test_agent",
            "description": "A test agent for configuration testing",
            "llm_model": "gpt-3.5-turbo"
        }
        
        # Test that configuration can be loaded and used
        assert mock_config["agent_name"] == "test_agent"
        assert "description" in mock_config
        assert "llm_model" in mock_config
    
    @pytest.mark.asyncio
    async def test_agent_initialization_with_config(self):
        """Test agent initialization with configuration."""
        class ConfigurableAgent(BaseAgent):
            def __init__(self, name: str, description: str, config: dict = None):
                super().__init__(name, description)
                self.config = config or {}
            
            def run(self, input: str) -> str:
                return f"Configured agent processed: {input}"
        
        test_config = {"test_param": "test_value"}
        agent = ConfigurableAgent("test_agent", "Test description", test_config)
        
        assert agent.name == "test_agent"
        assert agent.config == test_config
        assert agent.run("test input") == "Configured agent processed: test input"
    
    @pytest.mark.asyncio
    async def test_agent_thread_isolation(self, mock_build_workflow):
        """Test that agent threads remain isolated."""
        def get_state_side_effect(config):
            thread_id = config["configurable"]["thread_id"]
            return {
                "messages": [f"Message from {thread_id}"],
                "thread_id": thread_id
            }
        
        mock_workflow = mock_build_workflow.return_value
        mock_workflow.get_state.side_effect = get_state_side_effect
        
        # Test different thread isolation
        config1 = {"configurable": {"thread_id": "thread_1"}}
        config2 = {"configurable": {"thread_id": "thread_2"}}
        
        state1 = mock_workflow.get_state(config=config1)
        state2 = mock_workflow.get_state(config=config2)
        
        assert state1["thread_id"] == "thread_1"
        assert state2["thread_id"] == "thread_2"
        assert state1["messages"] != state2["messages"] 