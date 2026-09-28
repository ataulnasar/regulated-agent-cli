"""How a task is presented to the model. Shared by the agent and the fake provider."""

import re

TASK_PATTERN = re.compile(
    r"^Customer: (?P<customer_id>CUST-\d{3})\n"
    r"<customer_request>\n(?P<request>.*)\n</customer_request>",
    re.DOTALL,
)


def format_task(customer_id: str, request: str) -> str:
    """The first user message. The request is fenced so it reads as data, not instructions."""
    return f"Customer: {customer_id}\n<customer_request>\n{request}\n</customer_request>"


def parse_task(message: str) -> tuple[str, str]:
    match = TASK_PATTERN.match(message)
    if not match:
        raise ValueError("message is not a task produced by format_task")
    return match["customer_id"], match["request"]
