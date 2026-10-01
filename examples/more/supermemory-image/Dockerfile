# The supermemory-server image the `supermemory` target runs, as a recipe you can build.
#
# THE RECIPE IS PUBLIC; THE BUILT IMAGE IS NOT PUBLISHED. supermemory's local server is a
# closed-source prebuilt binary, so we redistribute nothing: this file fetches it, at YOUR build
# time, from the vendor's own public installer (https://supermemory.ai/install, reached through the
# `supermemory` npm launcher). Building here produces the same container the published
# `supermemory` numbers were measured in, from the vendor's artifact rather than from a copy of it.
# See docs/misc/engine-images.md for what that means for reproducing a result.
#
# WHY A RECIPE AT ALL. supermemory publishes no server image and no plain OSS server -- the open
# repository is Cloudflare Workers plus the hosted SaaS. The only self-hostable artifact that
# speaks the adapter's /v3+/v4 API on port 6767 is the prebuilt "supermemory local" binary.
#
# Version is PINNED to 0.0.3: latest (0.0.6) ships a broken bundle (missing @rivetkit/rivetkit-wasm)
# whose document-processing workflow never finalises, so /v3/documents ingests hang. 0.0.3 matches the
# adapter's engine_version default ("0.0.3-local") and processes documents correctly.
#
# THE INSTALLER IS MUTABLE, and that is the fragility this file cannot remove. The vendor serves it
# from a URL that is not versioned and offers no digest, so two builds of the same
# SUPERMEMORY_VERSION on two days are not guaranteed to be the same bytes. Pin the image you build
# by digest and record it; do not treat "0.0.3" as a reproducible identifier on its own.
#
# Embeddings run locally in-process (bge-base, no key). Document extraction needs a provider key,
# which the entrypoint writes into ~/.supermemory/env from the environment. memrank's own ingest
# path never spends it -- see the target's compose file.
#
#   docker build -t supermemory-server:0.0.3 examples/supermemory-image
#   docker run --rm -p 6767:6767 -e ANTHROPIC_API_KEY=unused-by-supermemory supermemory-server:0.0.3
#   SUPERMEMORY_BASE_URL=http://localhost:6767 uv run memrank submit supermemory demo --on none
FROM node:22-slim

ARG SUPERMEMORY_VERSION=0.0.3
ENV SUPERMEMORY_DATA_DIR=/tmp/supermemory

# supermemory's installer (https://supermemory.ai/install) shells out to curl; node:22-slim omits it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Force curl to HTTP/1.1 with retries: the installer's binary download over HTTP/2 intermittently
# fails with "stream not closed cleanly" (curl error 18). The installer picks up these defaults.
RUN printf 'http1.1\nretry = 5\nretry-delay = 3\nretry-connrefused\n' > /root/.curlrc

# Fetch the pinned linux/amd64 server binary into /root/.supermemory + /root/.local/bin.
# Retry the whole install in case a download still fails mid-stream.
RUN for i in 1 2 3 4 5; do \
        npx -y supermemory@latest local install --version "${SUPERMEMORY_VERSION}" --force && break; \
        echo "install attempt $i failed; retrying in 5s" >&2; sleep 5; \
    done; \
    test -x "$HOME/.local/bin/supermemory-server" \
        || { echo "supermemory-server install failed after retries" >&2; exit 1; }

# The build context is this directory, so the recipe is self-contained: an outsider builds it
# without cloning anything around it.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 6767
ENTRYPOINT ["/entrypoint.sh"]
