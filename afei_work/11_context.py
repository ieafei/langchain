from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from pydantic import BaseModel, Field
from typing import Callable

@dataclass
class Context:
    user_role: str
    environment: str

class AdminResponse(BaseModel):
    """Response with technical details for admins."""
    answer: str = Field(description="Answer")
    debug_info: dict = Field(description="Debug information")
    system_status: str = Field(description="System status")

class UserResponse(BaseModel):
    """Simple response for regular users."""
    answer: str = Field(description="Answer")

@wrap_model_call
def context_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select output format based on Runtime Context."""
    # Read from Runtime Context: user role and environment
    user_role = request.runtime.context.user_role
    environment = request.runtime.context.environment

    if user_role == "admin" and environment == "production":
        # Admins in production get detailed output
        request = request.override(response_format=AdminResponse)
    else:
        # Regular users get simple output
        request = request.override(response_format=UserResponse)

    return handler(request)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[context_based_output],
    context_schema=Context
)