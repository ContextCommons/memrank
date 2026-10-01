---
status: active
last_reviewed: 2026-09-02
---

# Engine images: what you can obtain

> **Deprecated:** this page describes memrank's older, pre-agent surface, which still ships but will be removed. To evaluate an agent, use `memrank run`; see the [README](../../README.md).

This page lists the container images used by command-line targets and the alternatives when
an image is unavailable. Local retrieval implementations need no engine container. HTTP clients
still require a running service, whether called from Python or the command line.

| Target | Image | Can you obtain it | What you run |
|---|---|---|---|
| `hindsight`, `hindsight:matched` | `ghcr.io/vectorize-io/hindsight:0.6.2` | **Yes** -- the vendor's own published image, anonymous pull | public image; match the recorded digest when reproducing a run |
| `atomicmemory` | `ghcr.io/atomicstrata/atomicmemory-core:latest` | **Yes** -- public package, anonymous pull | public image; match the recorded digest when reproducing a run |
| `supermemory` | built from a recipe; no registry serves it | **You build it** -- [`examples/more/supermemory-image/`](../../examples/more/supermemory-image/README.md) | the same recipe, your own build |
| `mem0` | a patched fork, built privately | **No** | mem0's own server, run by you |

The control arms -- `no-context`, `fixed-context`, `full-context` -- and the in-process memory
baselines `tfidf`, `bm25` and `word-overlap` need no image or API key. They run in-process
after Memrank is installed.

Both anonymous pulls in the table were re-checked on 2026-09-02.

## hindsight

The target uses the vendor's published image without modification. The manifest, compose graph
and image are public. Reproduction also requires matching the recorded image digest,
configuration, dataset and other receipt fields.

## atomicmemory

AtomicStrata publishes this engine image as a public `ghcr.io` package. It supports anonymous
pulls, like the other public image in the table.

The engine source is private. The public image supplies the executable needed for reproduction.

## mem0

**The `mem0` target runs a private fork, and you cannot obtain its image.** Use your own Mem0 server to evaluate a publicly available implementation.

The image is built from a fork of `mem0ai/mem0` at upstream `v2.0.1` carrying **ten unofficial
patches**. That lineage is not hidden: every receipt records it as CycloneDX pedigree -- the
upstream ancestor `pkg:github/mem0ai/mem0@v2.0.1`, the patch count, and the image digest that ran
-- so a reader can see what the number was produced by even though the fork itself is not
published.

What the patches do, from what this repository can establish: they add a **native Voyage embedder**
that upstream has no registry entry for, **pgvector dimension handling** so an embedder swap does
not corrupt the store, and **extraction fallbacks** that change what is retained when fact
extraction returns nothing. These changes affect the fork's supported configurations and behavior. The itemised list of all ten lives in the fork's own repository and is not derivable from
this one; this page will not guess at the remainder.

The target's manifest says it is "mem0 as mem0 ships and evaluates it" -- transcribed components,
the vendor's own retrieval depth. That is true of the *configuration* and not of the *binary*, and
a `mem0` row should be read with that qualifier attached.

**To evaluate mem0 yourself, run mem0 yourself.** memrank's mem0 adapter speaks to any mem0 OSS
server over HTTP:

```bash
# your own mem0 server, from mem0's published instructions
export MEM0_HTTP_URL=http://localhost:8888
```

Then either drive the system directly (see
[`examples/02-your-own-system/`](../../examples/02-your-own-system/README.md) for the shape) or write a target manifest of your own and point `targets.path` at it. What you will be measuring
is upstream mem0, which differs from the private fork.

Publishing the fork is a decision about the fork, not a documentation change, and it has not been
taken.

## supermemory

**The image is not published; the recipe is.** supermemory has no published server image and no
plain open-source server -- the open repository is Cloudflare Workers plus the hosted SaaS. The
only self-hostable artifact that speaks the adapter's API is the prebuilt `supermemory local`
binary, and that binary is **closed-source**. We may not hand it on, so the container we built is
not distributed.

What ships instead is the build: [`examples/more/supermemory-image/`](../../examples/more/supermemory-image/README.md)
is the Dockerfile our own image was built from. It fetches the binary at *your* build time from the
vendor's own public installer, so nothing is redistributed and what lands on your machine comes
from supermemory.

Two limits, stated because they bound what "the same container" means:

- **The version is pinned to `0.0.3`.** 0.0.6 ships a bundle missing `@rivetkit/rivetkit-wasm`
  whose document-processing workflow never finalises, so ingests hang rather than fail. The pin is
  a working-version pin, not a preference.
- **The installer is mutable.** The vendor serves it from an unversioned URL with no published
  digest, so two builds of `0.0.3` on two days are not guaranteed to be the same bytes. Pin the
  image you build by digest and record it; `0.0.3` alone does not identify a build. Version strings alone are insufficient to reproduce the executable.

## What the published package ships

The published package ships the manifest and compose graph for every target whose engine an
outsider can actually obtain. A target whose image resolves only inside our own infrastructure
would name a host that resolves for nobody, which is worse than not offering it -- so `mem0` and
`supermemory` are described here rather than shipped as runnable refs, and the sections above say
what to run in their place.

The adapters for both are public, and so is everything the run loop does with them. Use your own service address or target descriptor to run those integrations.
