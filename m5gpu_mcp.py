"""Stdio MCP server. Only the native cap subprocess runs with sudo."""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, StrictInt


class PowerStatus(BaseModel):
    cap_milliwatts: int = Field(description="Raw driver limit; negative means uncapped.")
    cap_watts: float | None = Field(description="Accepted limit in watts; null means uncapped.")
    capped: bool
    draw_watts: float = Field(ge=0, description="Current filtered GPU draw in watts.")


class GPUControl:
    def __init__(self, binary: Path, max_watts: int):
        self.binary = binary.resolve()
        self.max_watts = max_watts
        self.lock = asyncio.Lock()

    async def run(self, *args: str, privileged: bool = False) -> str:
        command = [str(self.binary), *args]
        if privileged and os.geteuid() != 0:
            command = ["/usr/bin/sudo", "-n", *command]
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except OSError as exc:
            raise RuntimeError(f"Cannot start m5gpu at {self.binary}: {exc}. Run make first.") from exc
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.communicate()
            raise
        if process.returncode:
            detail = stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip()
            hint = " Configure passwordless sudo for the installed m5gpu cap command (see README)." if privileged else ""
            raise RuntimeError(f"m5gpu failed (exit {process.returncode}): {detail}.{hint}")
        return stdout.decode()

    async def status(self) -> PowerStatus:
        return PowerStatus.model_validate_json(await self.run("status", "--json"))

    async def set_cap(self, watts: int) -> PowerStatus:
        # Validate here as well as in MCP: never coerce floats, booleans, or strings.
        if type(watts) is not int or not 1 <= watts <= self.max_watts:
            raise ValueError(f"watts must be an integer from 1 to {self.max_watts}; use remove_gpu_power_limit to uncap")
        async with self.lock:
            await self.run("cap", str(watts), privileged=True)
            return await self.status()

    async def remove_cap(self) -> PowerStatus:
        async with self.lock:
            await self.run("cap", "off", privileged=True)
            return await self.status()


def create_server(binary: Path, max_watts: int = 100) -> FastMCP:
    if max_watts < 1:
        raise ValueError("max_watts must be positive")
    gpu = GPUControl(binary, max_watts)
    server = FastMCP(
        "m5gpu",
        instructions=(
            "Read GPU draw and control its power limit in watts. Limits affect every app "
            "and persist after this server exits. Remove the limit to restore stock behavior. "
            "A cap is a ceiling, not a request to consume that power."
        ),
    )

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def get_gpu_power_status() -> PowerStatus:
        """Read accepted GPU power limit and live filtered draw without running a benchmark."""
        async with gpu.lock:
            return await gpu.status()

    @server.tool(annotations=ToolAnnotations(destructiveHint=False, idempotentHint=True, openWorldHint=False))
    async def set_gpu_power_limit(
        watts: Annotated[StrictInt, Field(ge=1, le=max_watts, description="GPU power ceiling in whole watts.")],
    ) -> PowerStatus:
        """Set a persistent system-wide GPU power ceiling; return accepted limit and draw.

        Requires noninteractive sudo. Zero is forbidden because it parks the GPU.
        """
        return await gpu.set_cap(watts)

    @server.tool(annotations=ToolAnnotations(destructiveHint=False, idempotentHint=True, openWorldHint=False))
    async def remove_gpu_power_limit() -> PowerStatus:
        """Remove the system-wide power limit and restore stock boost; requires noninteractive sudo."""
        return await gpu.remove_cap()

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path(__file__).with_name("m5gpu"))
    parser.add_argument("--max-watts", type=int, default=100, help="Maximum agent-requested cap (default: 100 W)")
    args = parser.parse_args()
    if args.max_watts < 1:
        parser.error("--max-watts must be positive")
    create_server(args.binary, args.max_watts).run(transport="stdio")


if __name__ == "__main__":
    main()
