import os
import pathlib
import tempfile
import unittest

from src.config import load_environment


class TestLoadEnvironment(unittest.TestCase):
    def test_loads_dotenv_file_into_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = pathlib.Path(temp_dir) / ".env"
            dotenv_path.write_text(
                "OPENAI_API_KEY=test-key\nDATABASE_URI=postgresql://example\n",
                encoding="utf-8",
            )

            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("DATABASE_URI", None)

            loaded = load_environment(dotenv_path)

            self.assertTrue(loaded)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "test-key")
            self.assertEqual(os.environ["DATABASE_URI"], "postgresql://example")


if __name__ == "__main__":
    unittest.main()
