"""Compare prequential performance on real observations or a controlled drift stream."""
import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.io import make_stream_chunks
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
FILES = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
         "/style.css": ("style.css", "text/css")}
FEATURES = ("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g")


def load_stream(dataset):
    rng = np.random.default_rng(42)
    if dataset == "penguins":
        with (ROOT / "demo/palmer_penguins.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        complete = [row for row in rows if all(row[name] not in ("", "NA") for name in FEATURES)]
        ordered = []
        for year in sorted({row["year"] for row in complete}):
            group = [row for row in complete if row["year"] == year]
            ordered.extend(group[index] for index in rng.permutation(len(group)))
        names = sorted({row["species"] for row in complete})
        X = np.array([[float(row[name]) for name in FEATURES] for row in ordered])
        y = np.array([names.index(row["species"]) for row in ordered])
        return X, y, "Palmer Penguins · 2007–2009 by year; fixed seeded order within each year"
    if dataset == "drift":
        X = rng.normal(size=(320, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        y[160:] = 1 - y[160:]
        return X, y, "Synthetic stream · the label rule reverses after sample 160"
    raise ValueError("Choose Palmer Penguins or the synthetic drift stream.")


def evaluate(dataset, model, chunk_size):
    if chunk_size not in {20, 40, 80}:
        raise ValueError("Choose chunk size 20, 40 or 80.")
    if model == "tree":
        estimator = DecisionTreeClassifier(max_depth=5, random_state=42)
    elif model == "forest":
        estimator = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=42)
    else:
        raise ValueError("Choose decision tree or random forest.")
    X, y, description = load_stream(dataset)
    classes = np.unique(y)
    chunks = make_stream_chunks(X, y, chunk_size=chunk_size)
    trainer = StreamTrainer(estimator, classes)
    log = trainer.run_prequential(chunks)
    counts = np.zeros(len(classes), dtype=int)
    baseline_correct, seen, evaluated = 0, 0, 0
    history = []
    for row, (_, labels) in zip(log, chunks, strict=True):
        seen += len(labels)
        if row["evaluated"]:
            evaluated += len(labels)
            baseline_correct += int(np.sum(labels == classes[counts.argmax()]))
        counts += np.array([np.sum(labels == label) for label in classes])
        history.append({
            "chunk": row["chunk"], "seen": seen, "evaluated": evaluated,
            "accuracy": row["accuracy"] if evaluated else None,
            "cumulative": row["cumulative_accuracy"] if evaluated else None,
            "majority": baseline_correct / evaluated if evaluated else None,
            "buffer_bytes": int(row["mem_bytes"]), "elapsed_ms": row["elapsed_s"] * 1000,
        })
    return {"dataset": description, "samples": len(y), "history": history}


def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            status = 200
            if url.path == "/api/experiment":
                kind = "application/json"
                params = parse_qs(url.query)
                try:
                    body = json.dumps(evaluate(
                        params.get("dataset", ["penguins"])[0],
                        params.get("model", ["tree"])[0],
                        int(params.get("chunk", ["40"])[0]),
                    ), allow_nan=False).encode()
                except ValueError as error:
                    status = 400
                    body = json.dumps({"error": str(error)}).encode()
            elif url.path in FILES:
                name, kind = FILES[url.path]
                body = (ROOT / "portfolio" / name).read_bytes()
            else:
                self.send_error(404)
                return
            self.send_response(status)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Open http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8004)
    serve(parser.parse_args().port)
