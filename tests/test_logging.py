import logging
import pathlib
import tempfile
import unittest

from src.config import build_log_file_path
from src.logging_utils import configure_file_logger


class TestLoggingConfig(unittest.TestCase):
    def test_build_log_file_path_defaults_to_repo_logs_file(self) -> None:
        path = build_log_file_path({})
        self.assertEqual(path, pathlib.Path("logs/sql-agent.log"))

    def test_build_log_file_path_respects_override(self) -> None:
        path = build_log_file_path({"LOG_FILE": "tmp/app.log"})
        self.assertEqual(path, pathlib.Path("tmp/app.log"))

    def test_configure_file_logger_creates_parent_directory_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = pathlib.Path(temp_dir) / "nested" / "app.log"
            logger = configure_file_logger(log_path)
            logger.info("test message")

            for handler in logger.handlers:
                handler.flush()

            self.assertTrue(log_path.exists())
            self.assertIn("test message", log_path.read_text(encoding="utf-8"))

            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
