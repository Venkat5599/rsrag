import argparse
import json
import sys

from rag import service
from rag.evaluation import evaluate, load_cases


def _load_documents(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark adaptive routing and grounded answering")
    parser.add_argument("--documents", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    service.reset()

    indexed = 0
    for document in _load_documents(args.documents):
        indexed += service.index_document(document)

    report = evaluate(service.get_engine(), load_cases(args.cases))
    payload = report.to_dict()
    payload["metrics"]["indexed_chunks"] = float(indexed)

    rendered = json.dumps(payload, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered)

    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
