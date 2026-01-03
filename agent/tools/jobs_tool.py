"""
Hugging Face Jobs Tool - Using huggingface-hub library

Refactored to use official huggingface-hub library instead of custom HTTP client
"""

import asyncio
import base64
import os
from typing import Any, Dict, Literal, Optional

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, get_token_to_send

from agent.tools.types import ToolResult
from agent.tools.utilities import (
    format_job_details,
    format_jobs_table,
    format_scheduled_job_details,
    format_scheduled_jobs_table,
)

# Hardware flavors
CPU_FLAVORS = ["cpu-basic", "cpu-upgrade", "cpu-performance", "cpu-xl"]
GPU_FLAVORS = [
    "sprx8",
    "zero-a10g",
    "t4-small",
    "t4-medium",
    "l4x1",
    "l4x4",
    "l40sx1",
    "l40sx4",
    "l40sx8",
    "a10g-small",
    "a10g-large",
    "a10g-largex2",
    "a10g-largex4",
    "a100-large",
    "h100",
    "h100x8",
]
SPECIALIZED_FLAVORS = ["inf2x6"]
ALL_FLAVORS = CPU_FLAVORS + GPU_FLAVORS + SPECIALIZED_FLAVORS

# Operation names
OperationType = Literal[
    "run",
    "ps",
    "logs",
    "inspect",
    "cancel",
    "scheduled run",
    "scheduled ps",
    "scheduled inspect",
    "scheduled delete",
    "scheduled suspend",
    "scheduled resume",
]

# Constants
UV_DEFAULT_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"


def _add_environment_variables(params: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Automatically adds selected environment variables to the parameters passed by LLM.

    Args:
        params: Dictionary that may contain "HF_TOKEN" and other environment variables as keys

    Returns:
        Dictionary with environment variables added
    """

    result = {"HF_TOKEN": get_token_to_send(os.environ.get("HF_TOKEN", ""))}

    if params:
        result.update(params)

    return result


def _build_uv_command(
    script: str,
    with_deps: list[str] | None = None,
    python: str | None = None,
    script_args: list[str] | None = None,
) -> list[str]:
    """Build UV run command"""
    parts = ["uv", "run"]

    if with_deps:
        for dep in with_deps:
            parts.extend(["--with", dep])

    if python:
        parts.extend(["-p", python])

    parts.append(script)

    if script_args:
        parts.extend(script_args)

    # add defaults
    # parts.extend(["--push_to_hub"])
    return parts


def _wrap_inline_script(
    script: str,
    with_deps: list[str] | None = None,
    python: str | None = None,
    script_args: list[str] | None = None,
) -> str:
    """Wrap inline script with base64 encoding to avoid file creation"""
    encoded = base64.b64encode(script.encode("utf-8")).decode("utf-8")
    # Build the uv command with stdin (-)
    uv_command = _build_uv_command("-", with_deps, python, script_args)
    # Join command parts with proper spacing
    uv_command_str = " ".join(uv_command)
    return f'echo "{encoded}" | base64 -d | {uv_command_str}'


def _ensure_hf_transfer_dependency(deps: list[str] | None) -> list[str]:
    """Ensure hf-transfer is included in the dependencies list"""

    if isinstance(deps, list):
        deps_copy = deps.copy()  # Don't modify the original
        if "hf-transfer" not in deps_copy:
            deps_copy.append("hf-transfer")
        return deps_copy

    return ["hf-transfer"]


def _resolve_uv_command(
    script: str,
    with_deps: list[str] | None = None,
    python: str | None = None,
    script_args: list[str] | None = None,
) -> list[str]:
    """Resolve UV command based on script source (URL, inline, or file path)"""
    # If URL, use directly
    if script.startswith("http://") or script.startswith("https://"):
        return _build_uv_command(script, with_deps, python, script_args)

    # If contains newline, treat as inline script
    if "\n" in script:
        wrapped = _wrap_inline_script(script, with_deps, python, script_args)
        return ["/bin/sh", "-lc", wrapped]

    # Otherwise, treat as file path
    return _build_uv_command(script, with_deps, python, script_args)


async def _async_call(func, *args, **kwargs):
    """Wrap synchronous HfApi calls for async context"""
    return await asyncio.to_thread(func, *args, **kwargs)


def _job_info_to_dict(job_info) -> Dict[str, Any]:
    """Convert JobInfo object to dictionary for formatting functions"""
    return {
        "id": job_info.id,
        "status": {"stage": job_info.status.stage, "message": job_info.status.message},
        "command": job_info.command,
        "createdAt": job_info.created_at.isoformat(),
        "dockerImage": job_info.docker_image,
        "spaceId": job_info.space_id,
        "hardware_flavor": job_info.flavor,
        "owner": {"name": job_info.owner.name},
    }


def _scheduled_job_info_to_dict(scheduled_job_info) -> Dict[str, Any]:
    """Convert ScheduledJobInfo object to dictionary for formatting functions"""
    job_spec = scheduled_job_info.job_spec

    # Extract last run and next run from status
    last_run = None
    next_run = None
    if scheduled_job_info.status:
        if scheduled_job_info.status.last_job:
            last_run = scheduled_job_info.status.last_job.created_at
            if last_run:
                last_run = (
                    last_run.isoformat()
                    if hasattr(last_run, "isoformat")
                    else str(last_run)
                )
        if scheduled_job_info.status.next_job_run_at:
            next_run = scheduled_job_info.status.next_job_run_at
            next_run = (
                next_run.isoformat()
                if hasattr(next_run, "isoformat")
                else str(next_run)
            )

    return {
        "id": scheduled_job_info.id,
        "schedule": scheduled_job_info.schedule,
        "suspend": scheduled_job_info.suspend,
        "lastRun": last_run,
        "nextRun": next_run,
        "jobSpec": {
            "dockerImage": job_spec.docker_image,
            "spaceId": job_spec.space_id,
            "command": job_spec.command or [],
            "hardware_flavor": job_spec.flavor or "cpu-basic",
        },
    }


class HfJobsTool:
    """Tool for managing Hugging Face compute jobs using huggingface-hub library"""

    def __init__(self, hf_token: Optional[str] = None, namespace: Optional[str] = None):
        self.api = HfApi(token=hf_token)
        self.namespace = namespace

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the specified operation"""
        operation = params.get("operation")

        args = params

        # If no operation provided, return error
        if not operation:
            return {
                "formatted": "Error: 'operation' parameter is required. See tool description for available operations and usage examples.",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        # Normalize operation name
        operation = operation.lower()

        try:
            # Route to appropriate handler
            if operation == "run":
                return await self._run_job(args)
            elif operation == "ps":
                return await self._list_jobs(args)
            elif operation == "logs":
                return await self._get_logs(args)
            elif operation == "inspect":
                return await self._inspect_job(args)
            elif operation == "cancel":
                return await self._cancel_job(args)
            elif operation == "scheduled run":
                return await self._scheduled_run(args)
            elif operation == "scheduled ps":
                return await self._list_scheduled_jobs(args)
            elif operation == "scheduled inspect":
                return await self._inspect_scheduled_job(args)
            elif operation == "scheduled delete":
                return await self._delete_scheduled_job(args)
            elif operation == "scheduled suspend":
                return await self._suspend_scheduled_job(args)
            elif operation == "scheduled resume":
                return await self._resume_scheduled_job(args)
            else:
                return {
                    "formatted": f'Unknown operation: "{operation}"\n\n'
                    "Available operations:\n"
                    "- run, ps, logs, inspect, cancel\n"
                    "- scheduled run, scheduled ps, scheduled inspect, "
                    "scheduled delete, scheduled suspend, scheduled resume\n\n"
                    "Call this tool with no operation for full usage instructions.",
                    "totalResults": 0,
                    "resultsShared": 0,
                    "isError": True,
                }

        except HfHubHTTPError as e:
            return {
                "formatted": f"API Error: {str(e)}",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }
        except Exception as e:
            return {
                "formatted": f"Error executing {operation}: {str(e)}",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

    async def _wait_for_job_completion(
        self, job_id: str, namespace: Optional[str] = None
    ) -> tuple[str, list[str]]:
        """
        Stream job logs until completion, printing them in real-time.

        Returns:
            tuple: (final_status, all_logs)
        """
        all_logs = []

        # Fetch logs - generator streams logs as they arrive and ends when job completes
        logs_gen = self.api.fetch_job_logs(job_id=job_id, namespace=namespace)

        # Stream logs in real-time
        for log_line in logs_gen:
            print("\t" + log_line)
            all_logs.append(log_line)

        # After logs complete, fetch final job status
        job_info = await _async_call(
            self.api.inspect_job, job_id=job_id, namespace=namespace
        )
        final_status = job_info.status.stage

        return final_status, all_logs

    async def _run_job(self, args: Dict[str, Any]) -> ToolResult:
        """Run a job using HfApi.run_job() - smart detection of Python vs Docker mode"""
        try:
            script = args.get("script")
            command = args.get("command")

            # Validate mutually exclusive parameters
            if script and command:
                raise ValueError(
                    "'script' and 'command' are mutually exclusive. Provide one or the other, not both."
                )

            if not script and not command:
                raise ValueError(
                    "Either 'script' (for Python) or 'command' (for Docker) must be provided."
                )

            # Python mode: script provided
            if script:
                # Get dependencies and ensure hf-transfer is included
                deps = _ensure_hf_transfer_dependency(args.get("dependencies"))

                # Resolve the command based on script type (URL, inline, or file)
                command = _resolve_uv_command(
                    script=script,
                    with_deps=deps,
                    python=args.get("python"),
                    script_args=args.get("script_args"),
                )

                # Use UV image unless overridden
                image = args.get("image", UV_DEFAULT_IMAGE)
                job_type = "Python"

            # Docker mode: command provided
            else:
                image = args.get("image", "python:3.12")
                job_type = "Docker"

            # Run the job
            job = await _async_call(
                self.api.run_job,
                image=image,
                command=command,
                env=_add_environment_variables(args.get("env")),
                secrets=_add_environment_variables(args.get("secrets")),
                flavor=args.get("hardware_flavor", "cpu-basic"),
                timeout=args.get("timeout", "30m"),
                namespace=self.namespace,
            )

            # Wait for completion and stream logs
            print(f"{job_type} job started: {job.url}")
            print("Streaming logs...\n---\n")

            final_status, all_logs = await self._wait_for_job_completion(
                job_id=job.id,
                namespace=self.namespace,
            )

            # Format all logs for the agent
            log_text = "\n".join(all_logs) if all_logs else "(no logs)"

            response = f"""{job_type} job completed!

**Job ID:** {job.id}
**Final Status:** {final_status}
**View at:** {job.url}

**Logs:**
```
{log_text}
```"""
            return {"formatted": response, "totalResults": 1, "resultsShared": 1}

        except Exception as e:
            raise Exception(f"Failed to run job: {str(e)}")

    async def _list_jobs(self, args: Dict[str, Any]) -> ToolResult:
        """List jobs using HfApi.list_jobs()"""
        jobs_list = await _async_call(self.api.list_jobs, namespace=self.namespace)

        # Filter jobs
        if not args.get("all", False):
            jobs_list = [j for j in jobs_list if j.status.stage == "RUNNING"]

        if args.get("status"):
            status_filter = args["status"].upper()
            jobs_list = [j for j in jobs_list if status_filter in j.status.stage]

        # Convert JobInfo objects to dicts for formatting
        jobs_dicts = [_job_info_to_dict(j) for j in jobs_list]

        table = format_jobs_table(jobs_dicts)

        if len(jobs_list) == 0:
            if args.get("all", False):
                return {
                    "formatted": "No jobs found.",
                    "totalResults": 0,
                    "resultsShared": 0,
                }
            return {
                "formatted": 'No running jobs found. Use `{"operation": "ps", "all": true}` to show all jobs.',
                "totalResults": 0,
                "resultsShared": 0,
            }

        response = f"**Jobs ({len(jobs_list)} total):**\n\n{table}"
        return {
            "formatted": response,
            "totalResults": len(jobs_list),
            "resultsShared": len(jobs_list),
        }

    async def _get_logs(self, args: Dict[str, Any]) -> ToolResult:
        """Fetch logs using HfApi.fetch_job_logs()"""
        job_id = args.get("job_id")
        if not job_id:
            return {
                "formatted": "job_id is required",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }

        try:
            # Fetch logs (returns generator, convert to list)
            logs_gen = self.api.fetch_job_logs(job_id=job_id, namespace=self.namespace)
            logs = await _async_call(list, logs_gen)

            if not logs:
                return {
                    "formatted": f"No logs available for job {job_id}",
                    "totalResults": 0,
                    "resultsShared": 0,
                }

            log_text = "\n".join(logs)
            return {
                "formatted": f"**Logs for {job_id}:**\n\n```\n{log_text}\n```",
                "totalResults": 1,
                "resultsShared": 1,
            }

        except Exception as e:
            return {
                "formatted": f"Failed to fetch logs: {str(e)}",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }

    async def _inspect_job(self, args: Dict[str, Any]) -> ToolResult:
        """Inspect job using HfApi.inspect_job()"""
        job_id = args.get("job_id")
        if not job_id:
            return {
                "formatted": "job_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        job_ids = job_id if isinstance(job_id, list) else [job_id]

        jobs = []
        for jid in job_ids:
            try:
                job = await _async_call(
                    self.api.inspect_job,
                    job_id=jid,
                    namespace=self.namespace,
                )
                jobs.append(_job_info_to_dict(job))
            except Exception as e:
                raise Exception(f"Failed to inspect job {jid}: {str(e)}")

        formatted_details = format_job_details(jobs)
        response = f"**Job Details** ({len(jobs)} job{'s' if len(jobs) > 1 else ''}):\n\n{formatted_details}"

        return {
            "formatted": response,
            "totalResults": len(jobs),
            "resultsShared": len(jobs),
        }

    async def _cancel_job(self, args: Dict[str, Any]) -> ToolResult:
        """Cancel job using HfApi.cancel_job()"""
        job_id = args.get("job_id")
        if not job_id:
            return {
                "formatted": "job_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        await _async_call(
            self.api.cancel_job,
            job_id=job_id,
            namespace=self.namespace,
        )

        response = f"""✓ Job {job_id} has been cancelled.

To verify, call this tool with `{{"operation": "inspect", "job_id": "{job_id}"}}`"""

        return {"formatted": response, "totalResults": 1, "resultsShared": 1}

    async def _scheduled_run(self, args: Dict[str, Any]) -> ToolResult:
        """Create scheduled job using HfApi.create_scheduled_job() - smart detection of Python vs Docker mode"""
        try:
            script = args.get("script")
            command = args.get("command")
            schedule = args.get("schedule")

            if not schedule:
                raise ValueError("schedule is required for scheduled jobs")

            # Validate mutually exclusive parameters
            if script and command:
                raise ValueError(
                    "'script' and 'command' are mutually exclusive. Provide one or the other, not both."
                )

            if not script and not command:
                raise ValueError(
                    "Either 'script' (for Python) or 'command' (for Docker) must be provided."
                )

            # Python mode: script provided
            if script:
                # Get dependencies and ensure hf-transfer is included
                deps = _ensure_hf_transfer_dependency(args.get("dependencies"))

                # Resolve the command based on script type
                command = _resolve_uv_command(
                    script=script,
                    with_deps=deps,
                    python=args.get("python"),
                    script_args=args.get("script_args"),
                )

                # Use UV image unless overridden
                image = args.get("image", UV_DEFAULT_IMAGE)
                job_type = "Python"

            # Docker mode: command provided
            else:
                image = args.get("image", "python:3.12")
                job_type = "Docker"

            # Create scheduled job
            scheduled_job = await _async_call(
                self.api.create_scheduled_job,
                image=image,
                command=command,
                schedule=schedule,
                env=_add_environment_variables(args.get("env")),
                secrets=_add_environment_variables(args.get("secrets")),
                flavor=args.get("hardware_flavor", "cpu-basic"),
                timeout=args.get("timeout", "30m"),
                namespace=self.namespace,
            )

            scheduled_dict = _scheduled_job_info_to_dict(scheduled_job)

            response = f"""✓ Scheduled {job_type} job created successfully!

**Scheduled Job ID:** {scheduled_dict["id"]}
**Schedule:** {scheduled_dict["schedule"]}
**Suspended:** {"Yes" if scheduled_dict.get("suspend") else "No"}
**Next Run:** {scheduled_dict.get("nextRun", "N/A")}

To inspect, call this tool with `{{"operation": "scheduled inspect", "scheduled_job_id": "{scheduled_dict["id"]}"}}`
To list all, call this tool with `{{"operation": "scheduled ps"}}`"""

            return {"formatted": response, "totalResults": 1, "resultsShared": 1}

        except Exception as e:
            raise Exception(f"Failed to create scheduled job: {str(e)}")

    async def _list_scheduled_jobs(self, args: Dict[str, Any]) -> ToolResult:
        """List scheduled jobs using HfApi.list_scheduled_jobs()"""
        scheduled_jobs_list = await _async_call(
            self.api.list_scheduled_jobs,
            namespace=self.namespace,
        )

        # Filter jobs - default: hide suspended jobs unless --all is specified
        if not args.get("all", False):
            scheduled_jobs_list = [j for j in scheduled_jobs_list if not j.suspend]

        # Convert to dicts for formatting
        scheduled_dicts = [_scheduled_job_info_to_dict(j) for j in scheduled_jobs_list]

        table = format_scheduled_jobs_table(scheduled_dicts)

        if len(scheduled_jobs_list) == 0:
            if args.get("all", False):
                return {
                    "formatted": "No scheduled jobs found.",
                    "totalResults": 0,
                    "resultsShared": 0,
                }
            return {
                "formatted": 'No active scheduled jobs found. Use `{"operation": "scheduled ps", "all": true}` to show suspended jobs.',
                "totalResults": 0,
                "resultsShared": 0,
            }

        response = f"**Scheduled Jobs ({len(scheduled_jobs_list)} total):**\n\n{table}"
        return {
            "formatted": response,
            "totalResults": len(scheduled_jobs_list),
            "resultsShared": len(scheduled_jobs_list),
        }

    async def _inspect_scheduled_job(self, args: Dict[str, Any]) -> ToolResult:
        """Inspect scheduled job using HfApi.inspect_scheduled_job()"""
        scheduled_job_id = args.get("scheduled_job_id")
        if not scheduled_job_id:
            return {
                "formatted": "scheduled_job_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        scheduled_job = await _async_call(
            self.api.inspect_scheduled_job,
            scheduled_job_id=scheduled_job_id,
            namespace=self.namespace,
        )

        scheduled_dict = _scheduled_job_info_to_dict(scheduled_job)
        formatted_details = format_scheduled_job_details(scheduled_dict)

        return {
            "formatted": f"**Scheduled Job Details:**\n\n{formatted_details}",
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _delete_scheduled_job(self, args: Dict[str, Any]) -> ToolResult:
        """Delete scheduled job using HfApi.delete_scheduled_job()"""
        scheduled_job_id = args.get("scheduled_job_id")
        if not scheduled_job_id:
            return {
                "formatted": "scheduled_job_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        await _async_call(
            self.api.delete_scheduled_job,
            scheduled_job_id=scheduled_job_id,
            namespace=self.namespace,
        )

        return {
            "formatted": f"✓ Scheduled job {scheduled_job_id} has been deleted.",
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _suspend_scheduled_job(self, args: Dict[str, Any]) -> ToolResult:
        """Suspend scheduled job using HfApi.suspend_scheduled_job()"""
        scheduled_job_id = args.get("scheduled_job_id")
        if not scheduled_job_id:
            return {
                "formatted": "scheduled_job_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        await _async_call(
            self.api.suspend_scheduled_job,
            scheduled_job_id=scheduled_job_id,
            namespace=self.namespace,
        )

        response = f"""✓ Scheduled job {scheduled_job_id} has been suspended.

To resume, call this tool with `{{"operation": "scheduled resume", "scheduled_job_id": "{scheduled_job_id}"}}`"""

        return {"formatted": response, "totalResults": 1, "resultsShared": 1}

    async def _resume_scheduled_job(self, args: Dict[str, Any]) -> ToolResult:
        """Resume scheduled job using HfApi.resume_scheduled_job()"""
        scheduled_job_id = args.get("scheduled_job_id")
        if not scheduled_job_id:
            return {
                "formatted": "scheduled_job_id is required",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        await _async_call(
            self.api.resume_scheduled_job,
            scheduled_job_id=scheduled_job_id,
            namespace=self.namespace,
        )

        response = f"""✓ Scheduled job {scheduled_job_id} has been resumed.

To inspect, call this tool with `{{"operation": "scheduled inspect", "scheduled_job_id": "{scheduled_job_id}"}}`"""

        return {"formatted": response, "totalResults": 1, "resultsShared": 1}


# Tool specification for agent registration
HF_JOBS_TOOL_SPEC = {
    "name": "hf_jobs",
    "description": (
        "Manage Hugging Face CPU/GPU compute jobs. Run Python scripts with UV or custom Docker commands. "
        "List, schedule and monitor jobs/logs.\n\n"
        "## Operations\n"
        "**Jobs:** run, ps, logs, inspect, cancel\n"
        "**Scheduled Jobs:** scheduled run, scheduled ps, scheduled inspect, scheduled delete, scheduled suspend, scheduled resume\n\n"
        "## 'run' Operation Behavior\n"
        "The 'run' operation automatically detects what you want to do:\n"
        "- If 'script' provided → runs a Python script and auto installs dependencies in the env\n"
        "- If 'command' provided → runs a custom Docker command (full control)\n"
        "- 'script' and 'command' are MUTUALLY EXCLUSIVE - provide one or the other, not both\n\n"
        "## Available Hardware Flavors\n"
        "**CPU:** cpu-basic, cpu-upgrade, cpu-performance, cpu-xl\n"
        "**GPU:** t4-small, t4-medium, l4x1, l4x4, a10g-small, a10g-large, a10g-largex2, a10g-largex4, a100-large, h100, h100x8\n"
        "**Specialized:** inf2x6\n\n"
        "## Usage Examples\n"
        "**Run Python script with dependencies:**\n"
        "{'operation': 'run', 'script': 'import torch\\nprint(torch.cuda.is_available())', 'dependencies': ['torch', 'transformers'], 'hardware_flavor': 'a10g-small'}\n\n"
        "**Run Python with secrets:**\n"
        "{'operation': 'run', 'script': 'from huggingface_hub import HfApi\\napi = HfApi()\\nprint(api.whoami())', 'dependencies': ['huggingface-hub']}\n\n"
        "**Run custom Docker command:**\n"
        "{'operation': 'run', 'image': 'nvidia/cuda:12.0-base', 'command': ['nvidia-smi']}\n\n"
        "**List running jobs:**\n"
        "{'operation': 'ps'}\n\n"
        "**Get job logs:**\n"
        "{'operation': 'logs', 'job_id': 'xxx'}\n\n"
        "**Cancel job:**\n"
        "{'operation': 'cancel', 'job_id': 'xxx'}\n\n"
        "**Schedule daily Python job:**\n"
        "{'operation': 'scheduled run', 'script': 'print(\"daily task\")', 'schedule': '@daily'}\n\n"
        "## Important Notes\n"
        "- **CRITICAL: Job files are EPHEMERAL** - ALL files created in HF Jobs (trained models, datasets, outputs, completions etc.) are DELETED when the job completes. You MUST upload any outputs to HF Hub in the script itself (using model.push_to_hub() when training models, dataset.push_to_hub() when creating text based outputs, etc.)."
        "- Always pass full script content - no local files available on server\n"
        "- Use array format for commands: ['/bin/sh', '-lc', 'cmd'] for shell features\n"
        "- hf-transfer is auto-included in uv jobs for faster downloads\n"
        "- **Remember to upload outputs to Hub before job finishes!**"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "run",
                    "ps",
                    "logs",
                    "inspect",
                    "cancel",
                    "scheduled run",
                    "scheduled ps",
                    "scheduled inspect",
                    "scheduled delete",
                    "scheduled suspend",
                    "scheduled resume",
                ],
                "description": (
                    "Operation to execute. Valid values: [run, ps, logs, inspect, cancel, "
                    "scheduled run, scheduled ps, scheduled inspect, scheduled delete, "
                    "scheduled suspend, scheduled resume]"
                ),
            },
            # Python/UV specific parameters
            "script": {
                "type": "string",
                "description": (
                    "Python code to execute. Can be inline code or a raw GitHub URL. "
                    "Auto-uses UV image and builds UV command. "
                    "USED with: 'run', 'scheduled run' (triggers Python mode). "
                    "MUTUALLY EXCLUSIVE with 'command'. "
                    "NOT USED with: 'ps', 'logs', 'inspect', 'cancel', 'scheduled ps/inspect/delete/suspend/resume'."
                ),
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of pip packages to install. Example: ['torch', 'transformers']. "
                    "Only used when 'script' is provided. "
                    "USED with: 'run', 'scheduled run' (optional, only with script). "
                    "NOT USED with: 'ps', 'logs', 'inspect', 'cancel', 'scheduled ps/inspect/delete/suspend/resume'."
                ),
            },
            # Docker specific parameters
            "image": {
                "type": "string",
                "description": (
                    "Docker image to use. Default: UV image if 'script' provided, else 'python:3.12'. "
                    "Can override the default UV image when using 'script'. "
                    "USED with: 'run', 'scheduled run' (optional). "
                    "NOT USED with: 'ps', 'logs', 'inspect', 'cancel', 'scheduled ps/inspect/delete/suspend/resume'."
                ),
            },
            "command": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Command to execute as array. Example: ['python', '-c', 'print(42)']. "
                    "Use this for full Docker control. "
                    "USED with: 'run', 'scheduled run' (triggers Docker mode). "
                    "MUTUALLY EXCLUSIVE with 'script'. "
                    "NOT USED with: 'ps', 'logs', 'inspect', 'cancel', 'scheduled ps/inspect/delete/suspend/resume'."
                ),
            },
            # Hardware and environment
            "hardware_flavor": {
                "type": "string",
                "description": (
                    "Hardware flavor. CPU: cpu-basic, cpu-upgrade, cpu-performance, cpu-xl. "
                    "GPU: t4-small, t4-medium, l4x1, l4x4, a10g-small, a10g-large, a10g-largex2, a10g-largex4, a100-large, h100, h100x8. "
                    "Default: cpu-basic. "
                    "USED with: 'run', 'uv', 'scheduled run', 'scheduled uv' (optional). "
                    "NOT USED with: 'ps', 'logs', 'inspect', 'cancel', 'scheduled ps/inspect/delete/suspend/resume'."
                ),
            },
            "secrets": {
                "type": "object",
                "description": (
                    "Secret environment variables. Format: {'KEY': 'VALUE'}. HF_TOKEN is loaded automatically. "
                    "USED with: 'run', 'uv', 'scheduled run', 'scheduled uv' (optional). "
                    "NOT USED with: 'ps', 'logs', 'inspect', 'cancel', 'scheduled ps/inspect/delete/suspend/resume'."
                ),
            },
            # Job management parameters
            "job_id": {
                "type": "string",
                "description": (
                    "Job ID to operate on. "
                    "REQUIRED for: 'logs', 'inspect', 'cancel'. "
                    "NOT USED with: 'run', 'uv', 'ps', 'scheduled run/uv/ps/inspect/delete/suspend/resume'."
                ),
            },
            # Scheduled job parameters
            "scheduled_job_id": {
                "type": "string",
                "description": (
                    "Scheduled job ID to operate on. "
                    "REQUIRED for: 'scheduled inspect', 'scheduled delete', 'scheduled suspend', 'scheduled resume'. "
                    "NOT USED with: 'run', 'uv', 'ps', 'logs', 'inspect', 'cancel', 'scheduled run', 'scheduled uv', 'scheduled ps'."
                ),
            },
            "schedule": {
                "type": "string",
                "description": (
                    "Cron schedule or preset. Presets: '@hourly', '@daily', '@weekly', '@monthly', '@yearly'. "
                    "Cron example: '0 9 * * 1' (9 AM every Monday). "
                    "REQUIRED for: 'scheduled run', 'scheduled uv'. "
                    "NOT USED with: 'run', 'uv', 'ps', 'logs', 'inspect', 'cancel', 'scheduled ps/inspect/delete/suspend/resume'."
                ),
            },
        },
        "required": ["operation"],
    },
}


async def hf_jobs_handler(arguments: Dict[str, Any]) -> tuple[str, bool]:
    """Handler for agent tool router"""
    try:
        tool = HfJobsTool(namespace=os.environ.get("HF_NAMESPACE", ""))
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        return f"Error executing HF Jobs tool: {str(e)}", False
