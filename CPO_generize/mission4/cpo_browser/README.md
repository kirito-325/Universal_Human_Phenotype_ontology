# CPO ontology browser

This local browser keeps HPO and MPO on separate pages while exposing their
original `is_a` trees and paginated self/inherited CPO relations.

Build the read-only SQLite index once:

```bash
python3 -B cpo_browser/build_database.py
```

Start the server:

```bash
python3 -B cpo_browser/server.py
```

Open `http://127.0.0.1:8765/`, or go directly to `/hpo` and `/mpo`.

The default ontology files are the frozen HPO and MPO resources already used
by mission 4. Override them with `--hpo-obo` and `--mpo-obo` when rebuilding.
The generated database lives at `cpo_browser/data/cpo_browser.sqlite3`.
The home and ontology pages read unified term, linked-term, self-relation, and
inherited-relation counts from that database.
