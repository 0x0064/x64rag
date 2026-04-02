import os
from unittest.mock import patch

import pytest

from x64rag.reasoning.cli.config import build_lm_config, load_config
from x64rag.reasoning.cli.constants import ConfigError, load_dotenv


class TestLoadDotenv:
    def test_loads_key_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("FOO_KEY=bar123\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FOO_KEY", None)
            load_dotenv(env_file)
            assert os.environ["FOO_KEY"] == "bar123"

    def test_skips_comments_and_blanks(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nKEY=val\n")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KEY", None)
            load_dotenv(env_file)
            assert os.environ["KEY"] == "val"

    def test_env_var_takes_precedence(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_KEY=from_file\n")
        with patch.dict(os.environ, {"MY_KEY": "from_env"}, clear=False):
            load_dotenv(env_file)
            assert os.environ["MY_KEY"] == "from_env"

    def test_strips_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('QUOTED="hello world"\n')
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QUOTED", None)
            load_dotenv(env_file)
            assert os.environ["QUOTED"] == "hello world"

    def test_missing_file_is_noop(self, tmp_path):
        load_dotenv(tmp_path / "nonexistent")


class TestLoadConfig:
    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="Config not found"):
            load_config(str(tmp_path / "nope.toml"))

    def test_minimal_config_loads(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text('[language_model]\nprovider = "anthropic"\nmodel = "claude-sonnet-4-20250514"\n')
        env = tmp_path / ".env"
        env.write_text("ANTHROPIC_API_KEY=sk-test\n")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            toml = load_config(str(config))

        assert toml["language_model"]["provider"] == "anthropic"

    def test_missing_language_model_raises(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text("[embeddings]\nprovider = 'openai'\n")
        env = tmp_path / ".env"
        env.write_text("")

        toml = load_config(str(config))
        with pytest.raises(ConfigError, match="language_model"):
            build_lm_config(toml)

    def test_unknown_provider_raises(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text('[language_model]\nprovider = "unknown"\n')
        env = tmp_path / ".env"
        env.write_text("")

        toml = load_config(str(config))
        with pytest.raises(ConfigError, match="Unknown language model"):
            build_lm_config(toml)

    def test_env_var_overrides_provider(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text('[language_model]\nprovider = "anthropic"\nmodel = "claude-sonnet-4-20250514"\n')
        env = tmp_path / ".env"
        env.write_text("")

        with patch.dict(os.environ, {"X64RAG_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}, clear=False):
            toml = load_config(str(config))
            lm_config = build_lm_config(toml)
            assert lm_config.client.provider == "openai"

    def test_env_var_overrides_api_key(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text('[language_model]\nprovider = "anthropic"\n')
        env = tmp_path / ".env"
        env.write_text("")

        with patch.dict(os.environ, {"X64RAG_API_KEY": "sk-override"}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            toml = load_config(str(config))
            lm_config = build_lm_config(toml)
            assert lm_config.client.api_key == "sk-override"

    def test_fallback_config(self, tmp_path):
        config = tmp_path / "config.toml"
        config.write_text(
            '[language_model]\nprovider = "anthropic"\n\n'
            '[language_model.fallback]\nprovider = "openai"\nmodel = "gpt-4o"\n'
        )
        env = tmp_path / ".env"
        env.write_text("ANTHROPIC_API_KEY=sk-ant\nOPENAI_API_KEY=sk-oai\n")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            toml = load_config(str(config))
            lm_config = build_lm_config(toml)

        assert lm_config.strategy == "fallback"
        assert lm_config.fallback is not None
        assert lm_config.fallback.provider == "openai"
