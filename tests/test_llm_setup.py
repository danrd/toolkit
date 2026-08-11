"""Tests for llm_kit/llm_setup.py's BaseConfig/LlmConfig."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_kit.llm_setup import (
    BaseConfig,
    LlmConfig,
    _build_cpu_runner,
    _build_gpu_runner,
    _start_llama_cpp_server,
    _start_vllm_server,
    resolve_local_model_path,
)


def test_tokenizer_model_defaults_to_none():
    config = LlmConfig()
    assert config.tokenizer_model is None


def test_tokenizer_model_can_be_set_separately_from_model():
    config = LlmConfig(model="unsloth/Qwen3.6-27B-GGUF", tokenizer_model="Qwen/Qwen3.6-27B")

    assert config.model == "unsloth/Qwen3.6-27B-GGUF"
    assert config.tokenizer_model == "Qwen/Qwen3.6-27B"


def test_resolve_local_model_path_downloads_the_quant_file():
    config = SimpleNamespace(llm=LlmConfig())
    with patch("huggingface_hub.hf_hub_download") as mock_download:
        mock_download.return_value = "/data/pretrained_models/Qwen3.6-27B-Q4_K_M.gguf"
        path = resolve_local_model_path(config)

    mock_download.assert_called_once_with(
        repo_id="unsloth/Qwen3.6-27B-GGUF",
        filename="Qwen3.6-27B-Q4_K_M.gguf",
        local_dir="/data/pretrained_models",
    )
    assert path == "/data/pretrained_models/Qwen3.6-27B-Q4_K_M.gguf"


def test_resolve_local_model_path_honors_pretrained_models_dir_override():
    config = SimpleNamespace(llm=LlmConfig(pretrained_models_dir="/custom/dir"))
    with patch("huggingface_hub.hf_hub_download") as mock_download:
        mock_download.return_value = "/custom/dir/Qwen3.6-27B-Q4_K_M.gguf"
        resolve_local_model_path(config)

    assert mock_download.call_args.kwargs["local_dir"] == "/custom/dir"


def test_resolve_local_model_path_returns_model_as_is_without_quant_file():
    config = SimpleNamespace(llm=LlmConfig(quant_file=""))
    assert resolve_local_model_path(config) == "unsloth/Qwen3.6-27B-GGUF"


def test_resolve_local_model_path_passes_through_an_existing_local_file(tmp_path):
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"")
    config = SimpleNamespace(llm=LlmConfig(model=str(gguf)))

    with patch("huggingface_hub.hf_hub_download") as mock_download:
        path = resolve_local_model_path(config)

    mock_download.assert_not_called()
    assert path == str(gguf)


def _fake_server_config(tmp_path, chat_template_kwargs=None):
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"")
    llm = LlmConfig(model=str(gguf), quant_file="")
    generation = SimpleNamespace(chat_template_kwargs=chat_template_kwargs or {}, max_tokens=64)
    return SimpleNamespace(base=BaseConfig(), llm=llm, generation=generation)


def test_start_llama_cpp_server_passes_chat_template_kwargs_when_set(tmp_path, monkeypatch):
    """llama-cpp-python's server doesn't read chat_template_kwargs from the
    request body (unlike vLLM) - it's a model-load-time CLI flag instead,
    so it has to be on the spawn command, not just in generation_kwargs."""
    monkeypatch.chdir(tmp_path)
    config = _fake_server_config(tmp_path, chat_template_kwargs={"enable_thinking": False})

    with patch("subprocess.Popen") as mock_popen:
        _start_llama_cpp_server(config)

    args = mock_popen.call_args[0][0]
    assert "--chat_template_kwargs" in args
    value = args[args.index("--chat_template_kwargs") + 1]
    assert json.loads(value) == {"enable_thinking": False}


def test_start_llama_cpp_server_omits_chat_template_kwargs_when_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _fake_server_config(tmp_path)

    with patch("subprocess.Popen") as mock_popen:
        _start_llama_cpp_server(config)

    args = mock_popen.call_args[0][0]
    assert "--chat_template_kwargs" not in args


def _fake_vllm_config(**llm_overrides):
    return SimpleNamespace(base=BaseConfig(), llm=LlmConfig(**llm_overrides))


def test_start_vllm_server_installs_and_does_not_stream_output_when_vllm_missing(tmp_path, monkeypatch):
    """vllm isn't importable in this test env, so _start_vllm_server should
    fall into its install branch - a routine success shouldn't flood the
    notebook with pip's output."""
    monkeypatch.chdir(tmp_path)
    config = _fake_vllm_config()

    with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        _start_vllm_server(config)

    assert mock_run.call_args.kwargs.get("capture_output") is True
    assert "--upgrade" not in mock_run.call_args[0][0]
    mock_popen.assert_called_once()


def test_start_vllm_server_skips_install_when_vllm_already_importable(tmp_path, monkeypatch):
    """A forced `--upgrade` on every call is its own hazard on a curated
    environment (Kaggle/Colab) - it can silently pull a vllm release whose
    CUDA wheels don't match the actual GPU/driver stack present. If vllm
    already imports, it should be left alone entirely."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    config = _fake_vllm_config()

    with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
        _start_vllm_server(config)

    mock_run.assert_not_called()
    mock_popen.assert_called_once()


def test_start_vllm_server_raises_with_full_output_when_pip_install_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _fake_vllm_config()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(returncode=1, stdout="resolving...", stderr="conflict")

        with pytest.raises(RuntimeError) as exc_info:
            _start_vllm_server(config)

    message = str(exc_info.value)
    assert "resolving..." in message
    assert "conflict" in message


def test_build_cpu_runner_health_check_failure_reports_timeout_and_log_file(tmp_path, monkeypatch):
    """A health-check failure used to just say "failed health check" - no
    pointer to the timeout that was actually used or to the log file that
    has the real reason (e.g. a large model still loading, or a genuine
    crash), forcing a manual hunt through the working directory to find it."""
    monkeypatch.chdir(tmp_path)
    config = SimpleNamespace(base=SimpleNamespace(device="cpu", server_ready_timeout=5.0),
                              llm=SimpleNamespace(model="fake/model", quant_file=""),
                              generation=SimpleNamespace())
    fake_process = SimpleNamespace(log_file=SimpleNamespace(name="llama_cpp.log"))

    with patch("llm_kit.llm_setup._start_llama_cpp_server", return_value=fake_process), \
         patch("llm_kit.llm_setup._wait_for_server_ready", return_value=False) as mock_wait, \
         patch("llm_kit.llm_setup._terminate_process"), \
         patch("llm_kit.llm_setup.setup_llama_cpp_model", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError) as exc_info:
            _build_cpu_runner(config)

    assert mock_wait.call_args.kwargs["timeout"] == 5.0
    message = str(exc_info.value)
    assert "5.0" in message
    assert "llama_cpp.log" in message


def test_build_cpu_runner_uses_default_server_ready_timeout_when_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = SimpleNamespace(base=SimpleNamespace(device="cpu"),
                              llm=SimpleNamespace(model="fake/model", quant_file=""),
                              generation=SimpleNamespace())
    fake_process = SimpleNamespace(log_file=SimpleNamespace(name="llama_cpp.log"))

    with patch("llm_kit.llm_setup._start_llama_cpp_server", return_value=fake_process), \
         patch("llm_kit.llm_setup._wait_for_server_ready", return_value=False) as mock_wait, \
         patch("llm_kit.llm_setup._terminate_process"), \
         patch("llm_kit.llm_setup.setup_llama_cpp_model", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            _build_cpu_runner(config)

    assert mock_wait.call_args.kwargs["timeout"] == 60.0


def test_build_gpu_runner_health_check_failure_reports_timeout_and_log_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = SimpleNamespace(base=SimpleNamespace(device="gpu", server_ready_timeout=5.0),
                              llm=SimpleNamespace(model="fake/model"),
                              generation=SimpleNamespace())
    fake_process = SimpleNamespace(log_file=SimpleNamespace(name="vllm_server.log"))

    # vllm isn't installed in this test env - the in-process vLLM tier
    # fails naturally with ImportError. setup_hf_model is mocked so the
    # last-resort HF tier doesn't make a real network call for "fake/model".
    with patch("llm_kit.llm_setup._start_vllm_server", return_value=fake_process), \
         patch("llm_kit.llm_setup._wait_for_server_ready", return_value=False) as mock_wait, \
         patch("llm_kit.llm_setup._terminate_process"), \
         patch("llm_kit.llm_setup.setup_hf_model", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError) as exc_info:
            _build_gpu_runner(config)

    assert mock_wait.call_args.kwargs["timeout"] == 5.0
    message = str(exc_info.value)
    assert "5.0" in message
    assert "vllm_server.log" in message


def test_start_vllm_server_passes_tensor_parallel_size_when_set(tmp_path, monkeypatch):
    """Without --tensor-parallel-size, vllm serve loads the whole model
    onto a single GPU regardless of how many are visible - on a
    multi-GPU box with a model too big for one card, that's an OOM, not
    a slow-but-working load."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    config = _fake_vllm_config(tensor_parallel_size=2)

    with patch("subprocess.Popen") as mock_popen:
        _start_vllm_server(config)

    args = mock_popen.call_args[0][0]
    assert "--tensor-parallel-size" in args
    assert args[args.index("--tensor-parallel-size") + 1] == "2"


def test_start_vllm_server_omits_tensor_parallel_size_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    config = _fake_vllm_config()

    with patch("subprocess.Popen") as mock_popen:
        _start_vllm_server(config)

    args = mock_popen.call_args[0][0]
    assert "--tensor-parallel-size" not in args
