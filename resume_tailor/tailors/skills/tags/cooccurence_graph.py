import subprocess
import xml.etree.ElementTree as ET
import pickle
from collections import defaultdict
from itertools import combinations
import multiprocessing as mp
from pathlib import Path

def nested_int_defaultdict():
    return defaultdict(int)

def stream_xml_lines(archive_path):
    """Yield each line of Posts.xml from the .7z without extracting to disk."""
    proc = subprocess.Popen(
        ["7z", "x", "-so", str(archive_path)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=1, universal_newlines=True
    )
    for line in proc.stdout:
        yield line

def worker_task(task_q, result_q):
    """Worker: pull a batch from task_q, process, and put partial graph in result_q."""
    while True:
        batch = task_q.get()
        if batch is None:
            # None is our shutdown sentinel
            break

        local = defaultdict(nested_int_defaultdict)
        for line in batch:
            line = line.strip()
            if not line.startswith("<row"):
                continue
            try:
                elem = ET.fromstring(line)
                tags = elem.attrib.get("Tags", "")
                tag_list = [i.strip() for i in tags.split("|") if len(i) > 0]
                if tag_list:
                    for a, b in combinations(sorted(tag_list), 2):
                        local[a][b] += 1
                        local[b][a] += 1
            except ET.ParseError:
                continue

        # send the small partial graph back
        result_q.put(local)

def build_and_save_chunks(
    archive_path: Path,
    lines_per_chunk: int = 20_000_000,
    batch_size: int = 5_000,
    queue_size: int = None,   # how many batches to buffer
    out_folder: Path = Path("cooc_chunks")
):
    """
    1) Stream rows, group into batches, put into task_q (bounded by queue_size).
    2) Worker processes pull and push partial graphs to result_q.
    3) Main loop merges partials as they arrive, and every lines_per_chunk rows:
       • pickle chunk_graph
       • reset row_count and chunk_graph
    """
    if queue_size is None:
        queue_size = mp.cpu_count() * 2

    out_folder.mkdir(exist_ok=True)

    # Queues
    task_q   = mp.Queue(maxsize=queue_size)
    result_q = mp.Queue()

    # Start workers
    workers = []
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=worker_task, args=(task_q, result_q))
        p.start()
        workers.append(p)

    # Main loop: streaming + dispatch + merge
    chunk_idx = 0
    row_count = 0
    chunk_graph = defaultdict(nested_int_defaultdict)
    batch = []

    lines = stream_xml_lines(archive_path)
    for raw in lines:
        raw = raw.strip()
        if not raw.startswith("<row"):
            continue

        # Accumulate batch
        batch.append(raw)
        row_count += 1

        # If batch full, block until there's room in the queue
        if len(batch) >= batch_size:
            task_q.put(batch)  
            batch = []

        # Merge any finished partials
        while not result_q.empty():
            partial = result_q.get()
            for a, nbrs in partial.items():
                out_a = chunk_graph[a]
                for b, cnt in nbrs.items():
                    out_a[b] += cnt

        # On chunk boundary, flush to disk and reset
        if row_count >= lines_per_chunk:
            # dispatch leftover batch
            if batch:
                task_q.put(batch)
                batch = []

            # wait for all queued tasks to finish
            while task_q.qsize() > 0:
                pass  # busy‑wait; or you can call time.sleep(0.1)

            # drain remaining results
            while not result_q.empty():
                partial = result_q.get()
                for a, nbrs in partial.items():
                    out_a = chunk_graph[a]
                    for b, cnt in nbrs.items():
                        out_a[b] += cnt

            # save chunk
            chunk_idx += 1
            path = out_folder / f"cooc_chunk_{chunk_idx}.pkl"
            with open(path, "wb") as f:
                pickle.dump(chunk_graph, f)
            print(f"💾 Saved chunk #{chunk_idx} ({row_count} rows) → {path}")

            # reset
            row_count    = 0
            chunk_graph  = defaultdict(nested_int_defaultdict)

    # final flush after EOF
    if batch:
        task_q.put(batch)
    # signal workers to exit
    for _ in workers:
        task_q.put(None)
    # merge any remaining partials
    while True:
        try:
            partial = result_q.get(timeout=1)
        except:
            break
        for a, nbrs in partial.items():
            out_a = chunk_graph[a]
            for b, cnt in nbrs.items():
                out_a[b] += cnt

    # save last chunk if any
    if row_count > 0:
        chunk_idx += 1
        path = out_folder / f"cooc_chunk_{chunk_idx}.pkl"
        with open(path, "wb") as f:
            pickle.dump(chunk_graph, f)
        print(f"💾 Saved final chunk #{chunk_idx} ({row_count} rows) → {path}")

    # clean up
    for p in workers:
        p.join()

    print("✅ All chunks built and saved.")


def merge_saved_chunks(chunks_folder: Path = Path("cooc_chunks")):
    files = sorted(chunks_folder.glob("cooc_chunk_*.pkl"))
    final = defaultdict(nested_int_defaultdict)
    for idx, file in enumerate(files, start=1):
        with open(file, "rb") as f:
            chunk_graph = pickle.load(f)
        for a, nbrs in chunk_graph.items():
            out_a = final[a]
            for b, cnt in nbrs.items():
                out_a[b] += cnt
        print(f"🔁 Merged chunk {idx}/{len(files)}")
    print("✅ Final graph built.")
    return final