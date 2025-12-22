from jp2subs import config, translation


def test_translation_available_local(tmp_path, monkeypatch):
    monkeypatch.delenv("JP2SUBS_LLAMA_BINARY", raising=False)
    monkeypatch.delenv("JP2SUBS_LLAMA_MODEL", raising=False)
    binary = tmp_path / "llama.exe"
    model = tmp_path / "model.gguf"
    binary.write_text("bin", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    cfg = config.AppConfig(
        translation=config.TranslationConfig(
            provider="local", llama_binary=str(binary), llama_model=f'"{model}"'
        )
    )

    ok, reason = translation.is_translation_available(cfg)

    assert ok is True
    assert reason == ""


def test_translation_missing_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("JP2SUBS_LLAMA_BINARY", raising=False)
    monkeypatch.delenv("JP2SUBS_LLAMA_MODEL", raising=False)
    cfg = config.AppConfig(
        translation=config.TranslationConfig(provider="local", llama_binary=str(tmp_path / "missing.exe"))
    )

    ok, reason = translation.is_translation_available(cfg)

    assert ok is False
    assert "binary" in reason.lower()


def test_translation_api_missing_url(monkeypatch):
    monkeypatch.delenv("JP2SUBS_API_URL", raising=False)
    cfg = config.AppConfig(translation=config.TranslationConfig(provider="api"))

    ok, reason = translation.is_translation_available(cfg)

    assert ok is False
    assert "url" in reason.lower()


def test_translation_echo(monkeypatch):
    cfg = config.AppConfig(translation=config.TranslationConfig(provider="echo"))

    ok, reason = translation.is_translation_available(cfg)

    assert ok is True
    assert reason == ""
