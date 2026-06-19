# 0001. Record Architecture Decisions

Date: 2026-06-18

## Status

Accepted

## Context

Vox Studio has several non-negotiable technical constraints: Windows-first native C++,
Qt dynamic LGPL linking, real-time audio constraints, local RVC sidecar integration, and
cloud ElevenLabs integration. These decisions will affect build, packaging, security,
latency, and licensing over the life of the application.

## Decision

We will record meaningful architecture decisions as ADRs in `docs/adr/`. Each ADR will
use a short numbered filename, state its status, explain the context, record the decision,
and summarize consequences.

## Consequences

- Major decisions are reviewable without searching commit history.
- Reversals and superseded decisions can be documented explicitly.
- The ADR directory becomes part of the engineering process from Pass 0 onward.
