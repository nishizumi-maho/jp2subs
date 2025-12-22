"""Translation orchestration for jp2subs."""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from rich.console import Console

from .models import MasterDocument

console = Console()

class TranslationProvider:
    """Base class for pluggable translation providers."""

    def translate_block(
        self,
        prompt: str,
        lines: Sequence[str],
        source_lang: str,
        target_lang: str,
        glossary: Dict[str, str] | None = None,
    ) -> List[str]:
        raise NotImplementedError


@dataclass
class EchoProvider(TranslationProvider):
    """Fallback provider that returns the source text."""

    def translate_block(
        self,
        prompt: str,
        lines: Sequence[str],
        source_lang: str,
        target_lang: str,
        glossary: Dict[str, str] | None = None,
    ) -> List[str]:
        return list(lines)


@dataclass
class LocalLlamaCPPProvider(TranslationProvider):
    binary_path: str
    model_path: str

    def translate_block(
        self,
        prompt: str,
        lines: Sequence[str],
        source_lang: str,
        target_lang: str,
        glossary: Dict[str, str] | None = None,
    ) -> List[str]:  # pragma: no cover - relies on external binary
        glossary_hint = "\n".join(f"{k} -> {v}" for k, v in (glossary or {}).items())
        joined = "\n".join(lines)
        output_format = (
            "Return one translation per input line, each prefixed with its 0-based line number "
            "and a tab character (example: `0\t<translation>`)."
        )

        if _env_truthy(os.getenv("JP2SUBS_LLAMA_CHAT")):
            full_prompt = (
                f"<system>\n{prompt.strip()}\n{output_format}\n</system>\n"
                f"<user>\nGlossary:\n{glossary_hint}\nINPUT:\n{joined}\nOUTPUT:\n</user>"
            )
        else:
            full_prompt = (
                f"{prompt}\n{output_format}\nGlossary:\n{glossary_hint}\nINPUT:\n{joined}\nOUTPUT:"
            ).strip()

        llama_args = shlex.split(os.getenv("JP2SUBS_LLAMA_ARGS", ""))
        cmd = [self.binary_path, "-m", self.model_path, *llama_args, "-p", full_prompt]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output_lines = _parse_llama_output(result.stdout.splitlines(), len(lines))
        return output_lines


@dataclass
class GenericAPIProvider(TranslationProvider):
    api_url: str
    api_key: str | None = None

    def translate_block(
        self,
        prompt: str,
        lines: Sequence[str],
        source_lang: str,
        target_lang: str,
        glossary: Dict[str, str] | None = None,
    ) -> List[str]:  # pragma: no cover - network
        import requests

        payload = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "lines": list(lines),
            "prompt": prompt,
            "glossary": glossary or {},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        return list(data.get("translations", []))


def _parse_llama_output(lines: Sequence[str], expected_len: int) -> List[str]:
    outputs: List[str] = ["" for _ in range(expected_len)]
    for raw_line in lines:
        line = raw_line.strip()
        if not line or "\t" not in line:
            continue
        idx_str, text = line.split("\t", 1)
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        if 0 <= idx < expected_len:
            outputs[idx] = text.strip()
    return outputs


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def translate_document(
    doc: MasterDocument,
    target_langs: Iterable[str],
    mode: str = "llm",
    provider: str = "echo",
    block_size: int = 20,
    glossary: Dict[str, str] | None = None,
    honorifics: str = "keep",
    tics: str = "keep",
) -> MasterDocument:
    provider_impl = _provider_from_name(provider)
    _validate_options(honorifics, tics)

    for lang in target_langs:
        console.log(
            f"Translating to {lang} using mode={mode} provider={provider} honorifics={honorifics} tics={tics}..."
        )
        _translate_lang(doc, lang, provider_impl, block_size, glossary, mode, honorifics, tics)
    return doc


def _translate_lang(
    doc: MasterDocument,
    target_lang: str,
    provider_impl: TranslationProvider,
    block_size: int,
    glossary: Dict[str, str] | None,
    mode: str,
    honorifics: str,
    tics: str,
) -> None:
    doc.ensure_translation_key(target_lang)
    translation_prompt = _build_translate_prompt(target_lang, honorifics, tics, glossary)
    postedit_prompt = _build_postedit_prompt(target_lang, honorifics, tics, glossary)
    expected_mode = mode.lower()

    for start in range(0, len(doc.segments), block_size):
        block = doc.segments[start : start + block_size]
        ids = [str(seg.id) for seg in block]
        source_lines = [f"{seg.id}\t{seg.ja_raw}" for seg in block]

        draft_translations = _run_aligned_block(
            provider_impl,
            translation_prompt,
            source_lines,
            ids,
            glossary,
            source_lang="ja",
            target_lang=target_lang,
        )

        final_translations = draft_translations

        if expected_mode == "draft+postedit":
            postedit_input = [f"{seg.id}\t{seg.ja_raw}\t{draft}" for seg, draft in zip(block, draft_translations)]
            final_translations = _run_aligned_block(
                provider_impl,
                postedit_prompt,
                postedit_input,
                ids,
                glossary,
                source_lang="ja",
                target_lang=target_lang,
            )

        final_translations = [_apply_glossary(text, glossary) for text in final_translations]

        for seg, text in zip(block, final_translations):
            seg.translations[target_lang] = text


def _run_aligned_block(
    provider_impl: TranslationProvider,
    prompt: str,
    lines: List[str],
    expected_ids: List[str],
    glossary: Dict[str, str] | None,
    source_lang: str,
    target_lang: str,
    max_attempts: int = 3,
) -> List[str]:
    attempt = 0
    parsed: Dict[str, str] = {}
    while attempt < max_attempts:
        raw_output = provider_impl.translate_block(prompt, lines, source_lang, target_lang, glossary)
        parsed = _parse_id_aligned_lines(raw_output, expected_ids)
        missing = [id_ for id_ in expected_ids if id_ not in parsed]
        if not missing:
            break
        attempt += 1
    return [parsed.get(id_, "") for id_ in expected_ids]


def _parse_id_aligned_lines(lines: Sequence[str], expected_ids: List[str]) -> Dict[str, str]:
    expected = {str(i) for i in expected_ids}
    parsed: Dict[str, str] = {}
    for line in lines:
        if not line or "\t" not in line:
            continue
        id_part, text = line.split("\t", 1)
        id_clean = id_part.strip()
        if id_clean not in expected or id_clean in parsed:
            continue
        parsed[id_clean] = text.strip()
    return parsed


def _build_translate_prompt(target_lang: str, honorifics: str, tics: str, glossary: Dict[str, str] | None) -> str:
    honorific_rule = "Keep honorific suffixes exactly as-is." if honorifics == "keep" else "Drop honorific suffixes."
    tic_rule = "Preserve interjections and tics verbatim." if tics == "keep" else "Lightly soften tics but do not remove them."
    glossary_rule = "Respect mandatory glossary replacements when applicable." if glossary else ""
    return "\n".join(
        [
            f"You are a professional Japanese-to-{target_lang} subtitle translator for anime.",
            honorific_rule,
            tic_rule,
            "Do not invent or omit meaning; keep repetitions and hesitations.",
            glossary_rule,
            "Input format: ID\tJA_TEXT.",
            "Output format: ID\tTRANSLATION.",
            "Return exactly one line per input line, in the same order, preserving all IDs.",
        ]
    ).strip()


def _build_postedit_prompt(target_lang: str, honorifics: str, tics: str, glossary: Dict[str, str] | None) -> str:
    honorific_rule = "Keep honorific suffixes exactly as-is." if honorifics == "keep" else "Drop honorific suffixes."
    tic_rule = "Preserve interjections and tics verbatim." if tics == "keep" else "Lightly soften tics but do not remove them."
    glossary_rule = "Apply glossary replacements even if the draft missed them." if glossary else ""
    return "\n".join(
        [
            "You are improving a machine-translated draft without drifting from the Japanese.",
            honorific_rule,
            tic_rule,
            glossary_rule,
            "Input format: ID\tJA_TEXT\tDRAFT_TRANSLATION.",
            "Anchor revisions on the Japanese source; fix only fluency and clarity.",
            "Output format: ID\tTRANSLATION.",
            "Return exactly one line per input line, keeping IDs and order unchanged.",
        ]
    ).strip()


def _apply_glossary(text: str, glossary: Dict[str, str] | None) -> str:
    if not glossary:
        return text
    for src, tgt in glossary.items():
        text = text.replace(src, tgt)
    return text


def _validate_options(honorifics: str, tics: str) -> None:
    if honorifics not in {"keep", "drop"}:
        raise ValueError("honorifics must be 'keep' or 'drop'")
    if tics not in {"keep", "light"}:
        raise ValueError("tics must be 'keep' or 'light'")


def _provider_from_name(name: str) -> TranslationProvider:
    name = name.lower()
    if name == "echo":
        return EchoProvider()
    if name == "local":
        binary = os.getenv("JP2SUBS_LLAMA_BINARY", "llama.exe")
        model = os.getenv("JP2SUBS_LLAMA_MODEL", "model.gguf")
        return LocalLlamaCPPProvider(binary_path=binary, model_path=model)
    if name == "api":
        url = os.getenv("JP2SUBS_API_URL", "")
        if not url:
            raise RuntimeError("JP2SUBS_API_URL is required for api provider")
        key = os.getenv("JP2SUBS_API_KEY")
        return GenericAPIProvider(api_url=url, api_key=key)
    return EchoProvider()

