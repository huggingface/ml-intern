import asyncio
import json
from typing import Any, Dict

import litellm
from models import Correctness, JudgementResult

# from: https://github.com/centerforaisafety/hle/blob/7b6be5aad6f9b43af3857de7867f3b52f6e4acb3/hle_eval/run_judge_results.py#L16-L33
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()

CHOICE_STRINGS = ["yes", "no"]


async def evaluate_single_response(
    question: str,
    response: str,
    correct_answer: str,
    model: str = "gpt-4o-mini",
    semaphore: asyncio.Semaphore = None,
) -> Dict[str, Any]:
    """
    Evaluate a single response against the ground truth using LLM as judge.

    Args:
        question: The question being answered
        response: The response to evaluate
        correct_answer: The ground truth answer
        model: The LLM model to use for judging
        semaphore: Semaphore for rate limiting

    Returns:
        Dictionary containing the judgement result and metadata
    """
    if semaphore:
        async with semaphore:
            return await _evaluate_single_response_impl(
                question, response, correct_answer, model
            )
    else:
        return await _evaluate_single_response_impl(
            question, response, correct_answer, model
        )


async def _evaluate_single_response_impl(
    question: str, response: str, correct_answer: str, model: str
) -> Dict[str, Any]:
    """Internal implementation of single response evaluation"""

    prompt = GRADER_TEMPLATE.format(
        question=question, response=response, correct_answer=correct_answer
    )

    # Use litellm with structured output
    response = await litellm.acompletion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert judge evaluating answers for accuracy and equivalence.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=JudgementResult,
        temperature=0.0,
    )

    # Parse structured output
    result: JudgementResult = JudgementResult.model_validate_json(
        response.choices[0].message.content
    )
    return result


async def evaluate_dataset(
    input_file: str,
    eval_file: str,
    output_file: str = "evaluation_results.jsonl",
    model: str = "gpt-4o-mini",
    max_concurrent: int = 30,
    limit: int = None,
) -> None:
    """
    Evaluate all QA pairs in the input file using LLM as judge.

    Args:
        input_file: Path to input JSONL file with QA pairs
        output_file: Path to output JSONL file for results
        model: The LLM model to use for judging
        max_concurrent: Maximum number of concurrent API calls
        limit: Optional limit on number of examples to evaluate
    """
    to_evaluate = [json.loads(line) for line in open(input_file, "r")]
    if limit:
        to_evaluate = to_evaluate[:limit]

    print(f"Loaded {len(to_evaluate)} QA pairs to evaluate")

    # Load dataset
    print(f"Loading ground truth from {eval_file}...")
    with open(eval_file, "r") as f:
        ground_truths = [json.loads(line) for line in f]

    print(f"Loaded {len(ground_truths)} ground truths")

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create evaluation tasks
    tasks = []
    for qa_pair, ground_truth in zip(to_evaluate, ground_truths):
        question = ground_truth.get("question", "")
        ground_truth = ground_truth.get("solution", "")
        response = qa_pair.get("solution", "")

        task = evaluate_single_response(
            response=response,
            question=question,
            correct_answer=ground_truth,
            model=model,
            semaphore=semaphore,
        )
        tasks.append(task)

    # Run evaluations in parallel
    print(f"Running evaluations with {max_concurrent} parallel workers...")
    results = await asyncio.gather(*tasks)

    # Combine results with original data
    output_data = []
    correct_count = 0
    error_count = 0

    for qa_pair, result in zip(to_evaluate, results):
        print(result.model_dump_json())
        # output_entry = {**qa_pair, "evaluation": result}
        # output_data.append(output_entry)

        if result.correct == Correctness.yes:
            correct_count += 1
        else:
            error_count += 1

    # # Write results
    # print(f"Writing results to {output_file}...")
    # with open(output_file, "w") as f:
    #     for entry in output_data:
    #         f.write(entry.model_dump_json() + "\n")

    # Print summary
    total = len(to_evaluate)
    success_rate = (total - error_count) / total * 100 if total > 0 else 0
    accuracy = correct_count / total * 100 if total > 0 else 0

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total examples: {total}")
    print(f"Successful evaluations: {total - error_count}")
    print(f"Errors: {error_count}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 50)


#


async def main():
    """Main entry point for the evaluation script"""
    await evaluate_dataset(
        input_file="qa_pairs.jsonl",
        eval_file="qa_pairs.jsonl",
        output_file="evaluation_results.jsonl",
        model="gpt-4o-mini",
        max_concurrent=30,
        limit=100,  # Set to None to evaluate all, or a number to limit
    )


if __name__ == "__main__":
    asyncio.run(main())
