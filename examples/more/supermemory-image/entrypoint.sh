#!/bin/sh
# Launch supermemory-server. The server reads its LLM key from ~/.supermemory/env, but a container
# receives secrets as environment variables -- so mirror whichever provider key is present into that
# file before booting. Extraction fails (documents never reach status=done) without a working key,
# so require one rather than degrade.
set -eu

# The server refuses to boot without SOME provider key and accepts any of four. It does not
# matter which, and on memrank's path it is never called: ingest posts pre-chunked content to
# /v4/memories, which skips the extractor, and the embedder is local. Verified by booting with an
# invalid key -- ingest and retrieve returned the correct document. So the key this asks for is a
# property of the image, not a credential the `supermemory` target spends.
#
# So this writes whichever key is present rather than demanding OpenAI's. Demanding one name made
# supermemory the only target a sweep needed a second provider account to run.
mkdir -p "$HOME/.supermemory"
: > "$HOME/.supermemory/env"
for name in ANTHROPIC_API_KEY OPENAI_API_KEY GEMINI_API_KEY GROQ_API_KEY; do
    eval "value=\${$name:-}"
    if [ -n "$value" ]; then
        printf '%s=%s\n' "$name" "$value" >> "$HOME/.supermemory/env"
    fi
done

if [ ! -s "$HOME/.supermemory/env" ]; then
    echo "error: supermemory needs one of ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY or" >&2
    echo "       GROQ_API_KEY present to start. It is not spent -- see docs/misc/engine-images.md." >&2
    exit 1
fi
chmod 600 "$HOME/.supermemory/env"

exec "$HOME/.local/bin/supermemory-server"
