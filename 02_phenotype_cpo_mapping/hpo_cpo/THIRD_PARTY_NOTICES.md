# Provenance

This project is a task-specific adaptation of the sibling `deep-life-sci` repository
(LangChain, Inc.). It retains the root/QuickJS/subagent assembly, copies the PubMed/PMC
clients, HTTP/cache helpers and error middleware, and adds cellular evidence and
normalization tools. Source hashes and modifications are documented in UPSTREAM.md.
It does not require the upstream sandbox, UI, or scientific execution stack.

Scientific extraction, mapping, and review contracts are adapted from the user's
`CPO_generize/HPO_link_v6/templates/` and
`CPO_generize/03_MPO_link/mpo_evidence/mpo_evidence_adjudication_system.md`.
The project does not redistribute those projects' literature or ontology snapshots.

Upstream deep-life-sci license:

MIT License

Copyright (c) 2026 LangChain, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
