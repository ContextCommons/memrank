# Building the `supermemory` engine image

The `supermemory` target names a container that no registry serves publicly. This directory is the
recipe for it, so you can build the same container the published numbers were measured in.

```bash
docker build -t supermemory-server:0.0.3 examples/more/supermemory-image
docker run --rm -p 6767:6767 -e ANTHROPIC_API_KEY=unused-by-supermemory supermemory-server:0.0.3

SUPERMEMORY_BASE_URL=http://localhost:6767 uv run memrank submit supermemory demo --on none
```

The build context is this directory. Nothing outside it is read, so the two files here are the
whole recipe.

## Why a recipe and not an image

supermemory publishes no server image and no plain open-source server: the open repository is
Cloudflare Workers plus the hosted SaaS. The only self-hostable artifact that speaks the adapter's
`/v3`+`/v4` API on port 6767 is the prebuilt `supermemory local` binary, and that binary is
closed-source.

Memrank does not redistribute this image. The `Dockerfile` fetches
the binary at **your** build time from the vendor's own public installer, which redistributes
nothing: what reaches your machine comes from supermemory, exactly as it would if you installed it
yourself.

<a id="two-things-that-will-bite"></a>
## Version and reproducibility limits

**The version is pinned to 0.0.3 on purpose.** 0.0.6 ships a bundle missing
`@rivetkit/rivetkit-wasm`, whose document-processing workflow never finalises, so `/v3/documents`
ingests hang rather than failing. 0.0.3 is also what the adapter's `engine_version` default
(`0.0.3-local`) records, so a receipt from this image names the build that produced it.

**The installer is mutable.** The vendor serves it from an unversioned URL with no published
digest, so two builds of `SUPERMEMORY_VERSION=0.0.3` on two different days are not guaranteed to
be the same bytes. Pin the image you build by digest and record that digest alongside your result;
`0.0.3` on its own does not identify a build. This is a real limit on reproducing a supermemory
number, and it is the reason the target's own provenance records the image digest rather than the
version string.

## The provider key

The server refuses to boot without one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`
or `GROQ_API_KEY`, but the integration does not use that provider key: ingest posts pre-chunked content to
`/v4/memories`, which skips the extractor, and the embedder is local and keyless
(`Xenova/bge-base-en-v1.5`). Pass a placeholder. `memrank targets show supermemory` reports the
target as needing no credential for exactly this reason.
