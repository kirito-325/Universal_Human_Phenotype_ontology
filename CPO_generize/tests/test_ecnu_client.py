from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_api.ecnu_client import api_keys_from_environment


class ApiKeysFromEnvironmentTests(unittest.TestCase):
    def test_disabled_variable_names_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "\n".join(
                    (
                        "ECNU_API_KEY=base-key",
                        "ECNU_API_KEY_SHARD_1=first-key",
                        "ECNU_API_KEY_SHARD_5=monthly-exhausted-key",
                        "ECNU_API_KEY_SHARD_6=another-monthly-exhausted-key",
                        "ECNU_API_KEY_DISABLED_VARIABLES=ECNU_API_KEY_SHARD_5, ECNU_API_KEY_SHARD_6",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                keys = api_keys_from_environment(env_file)

        self.assertEqual(
            keys,
            [
                ("ECNU_API_KEY", "base-key"),
                ("ECNU_API_KEY_SHARD_1", "first-key"),
            ],
        )

    def test_process_environment_can_override_disabled_variables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "\n".join(
                    (
                        "ECNU_API_KEY=base-key",
                        "ECNU_API_KEY_SHARD_1=first-key",
                        "ECNU_API_KEY_DISABLED_VARIABLES=ECNU_API_KEY",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"ECNU_API_KEY_DISABLED_VARIABLES": "ECNU_API_KEY_SHARD_1"},
                clear=True,
            ):
                keys = api_keys_from_environment(env_file)

        self.assertEqual(keys, [("ECNU_API_KEY", "base-key")])


if __name__ == "__main__":
    unittest.main()
