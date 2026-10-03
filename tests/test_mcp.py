import asyncio
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from m5gpu_mcp import GPUControl, create_server


ROOT = Path(__file__).resolve().parents[1]
STATUS = json.dumps({"cap_milliwatts":40000, "cap_watts":40.0, "capped":True, "draw_watts":12.5})


class ControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_caps_never_execute(self):
        gpu = GPUControl(ROOT / "m5gpu", 100)
        gpu.run = AsyncMock()
        for watts in (0, -1, 101, 1.5, True, "40", float("nan")):
            with self.subTest(watts=watts), self.assertRaises(ValueError):
                await gpu.set_cap(watts)
        gpu.run.assert_not_called()

    async def test_set_reads_accepted_cap(self):
        gpu = GPUControl(ROOT / "m5gpu", 100)
        gpu.run = AsyncMock(side_effect=["GPU capped", STATUS])
        result = await gpu.set_cap(40)
        self.assertEqual(result.cap_watts, 40)
        self.assertEqual(gpu.run.call_args_list[0].args, ("cap", "40"))
        self.assertEqual(gpu.run.call_args_list[0].kwargs, {"privileged": True})
        self.assertEqual(gpu.run.call_args_list[1].args, ("status", "--json"))

    async def test_remove_uses_off_never_zero(self):
        gpu = GPUControl(ROOT / "m5gpu", 100)
        uncapped = json.dumps({"cap_milliwatts":-1000, "cap_watts":None, "capped":False, "draw_watts":0})
        gpu.run = AsyncMock(side_effect=["removed", uncapped])
        result = await gpu.remove_cap()
        self.assertFalse(result.capped)
        self.assertIsNone(result.cap_watts)
        self.assertEqual(gpu.run.call_args_list[0].args, ("cap", "off"))

    async def test_failed_write_does_not_report_success(self):
        gpu = GPUControl(ROOT / "m5gpu", 100)
        gpu.run = AsyncMock(side_effect=RuntimeError("driver rejected cap"))
        with self.assertRaisesRegex(RuntimeError, "driver rejected"):
            await gpu.set_cap(40)
        self.assertEqual(gpu.run.await_count, 1)

    async def test_sudo_noninteractive_and_error(self):
        gpu = GPUControl(ROOT / "m5gpu", 100)
        process = AsyncMock()
        process.returncode = 1
        process.communicate.return_value = (b"", b"sudo: a password is required")
        with patch("m5gpu_mcp.os.geteuid", return_value=501), patch(
            "m5gpu_mcp.asyncio.create_subprocess_exec", return_value=process
        ) as launch:
            with self.assertRaisesRegex(RuntimeError, "passwordless sudo"):
                await gpu.run("cap", "40", privileged=True)
        self.assertEqual(launch.call_args.args, ("/usr/bin/sudo", "-n", str(gpu.binary), "cap", "40"))

    async def test_timeout_reaps_process(self):
        gpu = GPUControl(ROOT / "m5gpu", 100)
        process = AsyncMock()
        process.kill = Mock()
        process.communicate.side_effect = [asyncio.TimeoutError, (b"", b"")]
        with patch("m5gpu_mcp.asyncio.create_subprocess_exec", return_value=process):
            with self.assertRaises(asyncio.TimeoutError):
                await gpu.run("status", "--json")
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.await_count, 2)

    async def test_mcp_schema_and_rejection(self):
        server = create_server(ROOT / "m5gpu", 60)
        tools = {tool.name: tool for tool in await server.list_tools()}
        self.assertEqual(set(tools), {"get_gpu_power_status", "set_gpu_power_limit", "remove_gpu_power_limit"})
        self.assertTrue(tools["get_gpu_power_status"].annotations.readOnlyHint)
        schema = tools["set_gpu_power_limit"].inputSchema["properties"]["watts"]
        self.assertEqual((schema["minimum"], schema["maximum"]), (1, 60))
        for value in (0, -1, 61, 1.5, True, "40"):
            with self.subTest(value=value), self.assertRaises(Exception):
                await server.call_tool("set_gpu_power_limit", {"watts":value})


class NativeTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "m5gpu").exists(), "Run make to test native CLI")
    def test_invalid_caps_refused_before_root_or_driver_access(self):
        for value in ("0", "00", "-0", "garbage", "40foo", "1.5", "999999999999999999999", ""):
            with self.subTest(value=value):
                result = subprocess.run([str(ROOT / "m5gpu"), "cap", value], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("needs root", result.stderr)
                self.assertNotIn("AGXAccelerator", result.stderr)


class StdioTests(unittest.IsolatedAsyncioTestCase):
    async def test_wire_protocol_with_readonly_fake_driver(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "fake-m5gpu"
            binary.write_text("#!/bin/sh\nif [ \"$1\" = status ] && [ \"$2\" = --json ]; then\n"
                              + "printf '%s\\n' '" + STATUS + "'\nelse\nexit 1\nfi\n")
            binary.chmod(0o755)
            params = StdioServerParameters(command=sys.executable, args=[
                str(ROOT / "m5gpu_mcp.py"), "--binary", str(binary),
            ])
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    self.assertEqual(len((await client.list_tools()).tools), 3)
                    result = await client.call_tool("get_gpu_power_status", {})
                    self.assertFalse(result.isError)
                    self.assertEqual(result.structuredContent["cap_watts"], 40)
                    for value in (0, True, 40.5, "40", 101):
                        result = await client.call_tool("set_gpu_power_limit", {"watts": value})
                        self.assertTrue(result.isError)


if __name__ == "__main__":
    unittest.main()
