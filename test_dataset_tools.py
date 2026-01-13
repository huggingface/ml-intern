"""
Test script for unified dataset inspection tool
"""

import asyncio
import sys
from typing import TypedDict
from unittest.mock import MagicMock


# Mock the types module before importing dataset_tools
class ToolResult(TypedDict, total=False):
    formatted: str
    totalResults: int
    resultsShared: int
    isError: bool


mock_types = MagicMock()
mock_types.ToolResult = ToolResult
sys.modules["agent.tools.types"] = mock_types

# Now import directly from the file
sys.path.insert(0, "/Users/akseljoonas/Documents/hf-agent/agent/tools")
from dataset_tools import hf_inspect_dataset_handler, inspect_dataset


async def test_inspect_dataset():
    """Test the unified inspect_dataset function"""
    print("=" * 70)
    print("Testing inspect_dataset()")
    print("=" * 70)

    # Test with akseljoonas/hf-agent-sessions as specified
    print("\n→ inspect_dataset('akseljoonas/hf-agent-sessions'):")
    result = await inspect_dataset("akseljoonas/hf-agent-sessions")
    print(f"   isError: {result['isError']}")
    print(f"   Output:\n{result['formatted']}")

    print("\n" + "=" * 70)

    # # Test with stanfordnlp/imdb
    # print("\n→ inspect_dataset('stanfordnlp/imdb'):")
    # result = await inspect_dataset("stanfordnlp/imdb")
    # print(f"   isError: {result['isError']}")
    # print(f"   Output:\n{result['formatted']}")

    # print("\n" + "=" * 70)

    # # Test with multi-config dataset
    # print("\n→ inspect_dataset('nyu-mll/glue', config='mrpc'):")
    # result = await inspect_dataset("nyu-mll/glue", config="mrpc")
    # print(f"   isError: {result['isError']}")
    # print(f"   Output:\n{result['formatted']}")


async def test_handler():
    """Test the handler (what the agent calls)"""
    print("\n" + "=" * 70)
    print("Testing hf_inspect_dataset_handler()")
    print("=" * 70)

    result, success = await hf_inspect_dataset_handler(
        {
            "dataset": "stanfordnlp/imdb",
            "sample_rows": 2,
        }
    )
    print("\n→ Handler result:")
    print(f"   success: {success}")
    print(f"   output:\n{result}")


if __name__ == "__main__":
    print("\nUnified Dataset Inspection Tool Test\n")
    asyncio.run(test_inspect_dataset())
    # asyncio.run(test_handler())
    print("\n" + "=" * 70)
    print("Done!")
