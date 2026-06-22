from dotenv import load_dotenv
import os
import sys
 
load_dotenv()

_REQUIRED_ENV = [
    "ANTHROPIC_API_KEY",
    "TAVILY_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT"]
 
def _check_env() -> None:
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        print(f"[error] missing environment variables: {', '.join(missing)}")
        print("        add them to a .env file in the project root and retry.")
        sys.exit(1)
 