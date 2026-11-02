import gzip
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from bmfm_targets.tokenization.resources import reference_data
from bmfm_targets.tokenization.resources.reference_data import (
    get_L1000_genes,
    get_protein_coding_genes,
)


def test_get_L1000_genes():
    pcg = get_protein_coding_genes()
    l1000 = get_L1000_genes()

    assert len(l1000) == 978
    assert not pd.isna(l1000).any()

    intersect = {g for g in pcg if g in l1000}
    assert len(intersect) == 978


# --- L1000 gene info loading: local file, env var, offline guard, retrying download ---

GENE_INFO_TSV = (
    "pr_gene_id\tpr_gene_symbol\tpr_is_lm\n1\tAAA\t1\n2\tBBB\t0\n3\tCCC\t1\n"
)


@pytest.fixture()
def fake_module_dir(tmp_path, monkeypatch):
    """Point the module's 'next to this file' lookup at an empty temp dir."""
    monkeypatch.setattr(reference_data, "__file__", str(tmp_path / "reference_data.py"))
    monkeypatch.delenv(reference_data.L1000_FILE_ENV, raising=False)
    monkeypatch.delenv(reference_data.L1000_OFFLINE_ENV, raising=False)
    return tmp_path


@pytest.fixture()
def flaky_server():
    """Real local HTTP server: fails `fail_first` requests with 503, then serves gz data."""
    state = {"requests": 0, "fail_first": 1, "user_agents": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"] += 1
            state["user_agents"].append(self.headers.get("User-Agent"))
            if state["requests"] <= state["fail_first"]:
                self.send_response(503)
                self.end_headers()
                return
            body = gzip.compress(GENE_INFO_TSV.encode())
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/gene_info.txt.gz"
    yield state
    server.shutdown()
    server.server_close()


def test_l1000_download_retries_sends_user_agent_and_caches(
    fake_module_dir, flaky_server
):
    genes = reference_data._get_original_L1000_gene_names(url=flaky_server["url"])

    assert list(genes) == ["AAA", "CCC"]  # only pr_is_lm == 1
    assert flaky_server["requests"] == 2  # one 503, then success
    assert all(ua == "bmfm-targets" for ua in flaky_server["user_agents"])

    cached = fake_module_dir / "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt"
    assert cached.exists()
    assert not cached.with_suffix(".txt.tmp").exists()

    # second call must read the cached file and not touch the network
    again = reference_data._get_original_L1000_gene_names(url=flaky_server["url"])
    assert list(again) == ["AAA", "CCC"]
    assert flaky_server["requests"] == 2


def test_l1000_download_raises_after_retries_exhausted(flaky_server):
    flaky_server["fail_first"] = 10**6
    with pytest.raises(Exception, match="503|retries"):
        reference_data._download_with_retries(flaky_server["url"], retries=2, backoff=0)
    assert flaky_server["requests"] == 3  # first attempt + 2 retries


def test_l1000_offline_flag_refuses_to_download(
    fake_module_dir, flaky_server, monkeypatch
):
    monkeypatch.setenv(reference_data.L1000_OFFLINE_ENV, "1")
    with pytest.raises(FileNotFoundError, match="OFFLINE"):
        reference_data._get_original_L1000_gene_names(url=flaky_server["url"])
    assert flaky_server["requests"] == 0


def test_l1000_env_file_is_used_and_must_exist(
    fake_module_dir, flaky_server, monkeypatch
):
    custom = fake_module_dir / "custom_gene_info.csv"
    custom.write_text("pr_gene_symbol,pr_is_lm\nXXX,1\nYYY,0\n")
    monkeypatch.setenv(reference_data.L1000_FILE_ENV, str(custom))
    genes = reference_data._get_original_L1000_gene_names(url=flaky_server["url"])
    assert list(genes) == ["XXX"]
    assert flaky_server["requests"] == 0

    monkeypatch.setenv(reference_data.L1000_FILE_ENV, str(fake_module_dir / "nope"))
    with pytest.raises(FileNotFoundError, match=reference_data.L1000_FILE_ENV):
        reference_data._get_original_L1000_gene_names(url=flaky_server["url"])
