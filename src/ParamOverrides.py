"""
Manual parameter overrides (owner's roadmap step 3, 8 Nov 2026): an administrator sets a value for one model's
setting on Admin > Parameters, and it wins over the tuners' bestParams_<game>.json until someone releases it.

The overrides live in their own file, config/paramOverrides.json (gitignored, written only by the server, params.js),
never in bestParams_<game>.json: the weekend tuners keep tuning and writing their file underneath, a release is the
deletion of one entry, and the tuned value is back on the next run.

    {"version": 1, "games": {"lotto": {
        "overrides": {"markovAlpha": {"value": 0.5, "rows": ["Markov Model"], "tuned": 0.653, "by": "...", "at": "...",
                                      "note": "..."},
                      "sampleTemperatures/Markov Model": {"value": 0.7, "rows": ["Markov Model"], ...}},
        "fresh": {"LSTM Base Model": "2026-11-08T20:00:00Z"}}}}

A key "sampleTemperatures/<row>" sets one entry of the sampleTemperatures map. "fresh" holds a deep-learning row's
request to train from new weights once (set by the server when one of the row's settings is overridden or released):
the predictor honours it on the row's next training and records it in data/models/fresh-<game>.json, so it is done
once, not every day.

Where it applies: Predictor.py's daily predictions (the served configuration) and TrainMetaLearner.py (owner,
8 Nov 2026: the meta-learners are trained on the rows the daily run serves). The weekend tuners score their own file
and the controls read it too - an override is a researcher's experiment on the served rows, not a new tuned value.

    python3 -m src.ParamOverrides     (self-check)
"""
import copy
import json
import os

FILE = os.path.join("config", "paramOverrides.json")
MAP_KEYS = ("sampleTemperatures",)   # the dict-valued settings whose entries an override can set one by one
SEPARATOR = "/"


def _read(path):
    try:
        with open(path, "r") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:   # a damaged file must never cost the day its predictions: say so and serve the tuned values
        print(f"Could not read {path}, no manual overrides applied: {e}")
        return {}


def game_entry(root, game):
    """This game's {"overrides": {...}, "fresh": {...}} from the overrides file (empty when there is none)."""
    entry = ((_read(os.path.join(root, FILE)).get("games") or {}).get(game)) or {}
    return {"overrides": entry.get("overrides") if isinstance(entry.get("overrides"), dict) else {},
            "fresh": entry.get("fresh") if isinstance(entry.get("fresh"), dict) else {}}


def apply(params, overrides):
    """
    (a copy of params with every override in place, [(key, value, tuned value) applied]). The tuned value is what the
    file held, None when the key was not set.
    """
    merged = copy.deepcopy(params) if isinstance(params, dict) else {}
    applied = []
    for key in sorted(overrides or {}):
        record = overrides[key]
        if not isinstance(record, dict) or "value" not in record:
            continue
        value = record["value"]
        if SEPARATOR in key and key.split(SEPARATOR, 1)[0] in MAP_KEYS:
            mapKey, entry = key.split(SEPARATOR, 1)
            table = merged.get(mapKey) if isinstance(merged.get(mapKey), dict) else {}
            tuned = table.get(entry)
            merged[mapKey] = dict(table, **{entry: value})
        else:
            tuned = merged.get(key)
            merged[key] = value
        applied.append((key, value, tuned))
    return merged, applied


def apply_for_game(root, game, params):
    """The served parameters of one game and the overrides applied; prints one line per override."""
    entry = game_entry(root, game)
    merged, applied = apply(params, entry["overrides"])
    for key, value, tuned in applied:
        print(f"Manual override for {game}: {key} = {value!r} (tuned: {tuned!r})")
    return merged, applied


def manual_rows(root, game):
    """{row name: [overridden keys]} - which rows run on a manual setting today (the day file labels them)."""
    rows = {}
    for key, record in game_entry(root, game)["overrides"].items():
        if isinstance(record, dict) and "value" in record:
            for row in record.get("rows") or []:
                rows.setdefault(str(row), []).append(key)
    return {row: sorted(keys) for row, keys in rows.items()}


def label_manual_rows(rows, manual):
    """Marks every row dict that runs on a manual setting (its variant twins too, by "parent") with "manual": [keys]."""
    if not manual:
        return 0
    marked = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        keys = manual.get(row.get("name")) or manual.get(row.get("parent"))
        if keys:
            row["manual"] = list(keys)
            marked += 1
    return marked


# --- fresh retrains ---------------------------------------------------------------------------------------------
def _done_path(root, game):
    return os.path.join(root, "data", "models", f"fresh-{game}.json")


def needs_fresh(root, game, row):
    """The request (its timestamp) when this row should train from new weights now, else None."""
    asked = game_entry(root, game)["fresh"].get(row)
    if not asked:
        return None
    return None if _read(_done_path(root, game)).get(row) == asked else asked


def mark_fresh_done(root, game, row, asked):
    """Records that the row trained from new weights for request `asked` (written after its training succeeded)."""
    path = _done_path(root, game)
    done = _read(path)
    done[row] = asked
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(done, handle, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _self_check():
    import tempfile
    params = {"markovAlpha": 0.65, "useMarkov": True, "sampleTemperatures": {"Markov Model": 1.5}}
    overrides = {"markovAlpha": {"value": 0.5, "rows": ["Markov Model"]}, "useMarkov": {"value": False, "rows": ["Markov Model"]},
                 "sampleTemperatures/Markov Model": {"value": 0.7, "rows": ["Markov Model"]},
                 "sampleTemperatures/LSTM Base Model": {"value": 2, "rows": ["LSTM Base Model"]},
                 "broken": "no value", "newKey": {"value": 3, "rows": []}}
    merged, applied = apply(params, overrides)
    assert merged["markovAlpha"] == 0.5 and merged["useMarkov"] is False and merged["newKey"] == 3
    assert merged["sampleTemperatures"] == {"Markov Model": 0.7, "LSTM Base Model": 2}, merged["sampleTemperatures"]
    assert params["markovAlpha"] == 0.65 and params["sampleTemperatures"] == {"Markov Model": 1.5}, "the tuned dict is untouched"
    assert ("markovAlpha", 0.5, 0.65) in applied and ("sampleTemperatures/Markov Model", 0.7, 1.5) in applied and ("newKey", 3, None) in applied
    assert not any(k == "broken" for k, _, _ in applied), "an entry without a value is ignored"
    assert apply(params, {})[0] == params and apply(None, overrides)[0]["markovAlpha"] == 0.5

    with tempfile.TemporaryDirectory() as root:
        assert game_entry(root, "lotto") == {"overrides": {}, "fresh": {}}, "no file: nothing"
        os.makedirs(os.path.join(root, "config"))
        with open(os.path.join(root, FILE), "w") as handle:
            json.dump({"version": 1, "games": {"lotto": {"overrides": overrides, "fresh": {"LSTM Base Model": "t1"}}}}, handle)
        served, done = apply_for_game(root, "lotto", params)
        assert served["markovAlpha"] == 0.5 and len(done) == 5
        assert apply_for_game(root, "keno", params)[0] == params, "another game: untouched"
        manual = manual_rows(root, "lotto")
        assert manual["Markov Model"] == ["markovAlpha", "sampleTemperatures/Markov Model", "useMarkov"], manual
        rows = [{"name": "Markov Model"}, {"name": "Markov Model · Sampled", "parent": "Markov Model"}, {"name": "Poisson Model"}, "x"]
        assert label_manual_rows(rows, manual) == 2 and rows[1]["manual"] == manual["Markov Model"] and "manual" not in rows[2]
        # a fresh request is honoured once, and again only when asked again
        assert needs_fresh(root, "lotto", "LSTM Base Model") == "t1" and needs_fresh(root, "lotto", "GNN Model") is None
        mark_fresh_done(root, "lotto", "LSTM Base Model", "t1")
        assert needs_fresh(root, "lotto", "LSTM Base Model") is None
        with open(os.path.join(root, FILE), "w") as handle:
            json.dump({"games": {"lotto": {"fresh": {"LSTM Base Model": "t2"}}}}, handle)
        assert needs_fresh(root, "lotto", "LSTM Base Model") == "t2"
        with open(os.path.join(root, FILE), "w") as handle:
            handle.write("{not json")
        assert apply_for_game(root, "lotto", params)[0] == params, "a damaged file: the tuned values, no crash"
    print("ParamOverrides self-check OK")


if __name__ == "__main__":
    _self_check()
