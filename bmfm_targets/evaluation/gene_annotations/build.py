"""
Rebuild `bmfm_targets/evaluation/resources/gene_families.csv` from public sources.

Each gene gets two labels:

- `family` (axis 1): one functional family, assigned greedily in the priority
  order of Supplementary Table S1 from GO branches, TFClass superclasses and
  HGNC symbol prefixes. Genes matching nothing are "Unclassified".
- `expr_regime` (axis 2), first match wins:
  1. `state_dissociation`: van den Brink 2017 core IEGs or GO heat response
     (GO:0009408) or GO response to unfolded protein (GO:0006986), minus HPA
     identity genes;
  2. `state_cell_cycle`: GO cell cycle (GO:0007049), minus HPA identity genes;
  3. `constitutive`: Eisenberg & Levanon 2013 housekeeping genes;
  4. `identity_associated`: HPA "RNA single cell type specificity" is
     "Cell type enriched" or "Group enriched";
  5. `unassigned`.

GO branches contain the root term and all its `is_a` descendants; obsolete terms
are dropped. A gene belongs to a branch if any GAF line annotates it to a term in
the branch (no evidence-code or qualifier filter).

Sources are downloaded to `--cache-dir` and checked against pinned sha256 values.
Rows are sorted by gene, so a rebuild is byte-identical to the packaged CSV.

The gene universe is the 19,027 genes of the packaged table (the genes in the
WCED validation per-gene output). Pass `--universe` to annotate another gene list.

Matching is case-sensitive: source symbols (GAF, HPA, Eisenberg) are upper-cased
and universe symbols are used as given, so the 219 mixed-case symbols
(`C1orf74` and the other `C#orf#` genes) match no source and are
`Unclassified` / `unassigned`.

Usage:
    pip install "bmfm-targets[gene_annotations]"   # rdflib
    python -m bmfm_targets.evaluation.gene_annotations.build --out gene_families.csv
"""

import argparse
import gzip
import hashlib
import re
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

GO_RELEASE = "2025-10-10"
SOURCES = {
    "go_obo": (
        f"https://release.geneontology.org/{GO_RELEASE}/ontology/go-basic.obo",
        "4e3ce039a6ad3d45e2752097625330152cc8ae97d35d824c7fcb0735b45d0fc3",
    ),
    "go_gaf": (
        f"https://release.geneontology.org/{GO_RELEASE}/annotations/goa_human.gaf.gz",
        "37569713d1156138dc569fe6cd358727de4e0f92d9bebd08ef06044af3913d0c",
    ),
    "tfclass": (
        "http://tfclass.bioinf.med.uni-goettingen.de/suppl/tfclass.ttl.gz",
        "248d76024571da17691cc2174ef5bf405392a4335f6fbf492983c8de60c73453",
    ),
    "eisenberg": (
        "https://www.tau.ac.il/~elieis/HKG/HK_genes.txt",
        "df7aa10e635efb5b0df7dd607e5c11c91df2bdf3e321fb97c0c088ea96a60756",
    ),
    "hpa": (
        "https://v25.proteinatlas.org/download/proteinatlas.tsv.zip",
        "ad401f8519ecdee5b67b0ad3bf28175d402d1ea0ad6c542b90fcab2bee39b470",
    ),
}

VAN_DEN_BRINK_CORE = {
    "FOS",
    "FOSB",
    "JUN",
    "JUNB",
    "JUND",
    "EGR1",
    "ATF3",
    "NR4A1",
    "HSPA1A",
    "HSPA1B",
    "DUSP1",
    "ZFP36",
    "IER2",
    "BTG2",
    "SOCS3",
}
GO_HEAT_RESPONSE = "GO:0009408"
GO_UNFOLDED_PROTEIN = "GO:0006986"
GO_CELL_CYCLE = "GO:0007049"
HPA_IDENTITY = ("Cell type enriched", "Group enriched")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, cache_dir: Path) -> Path:
    url, expected = SOURCES[name]
    path = cache_dir / url.rsplit("/", 1)[-1]
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path)
    got = sha256(path)
    if got != expected:
        raise ValueError(
            f"{path}: sha256 {got} != pinned {expected}. The upstream file changed;"
            " pass a local copy instead of relying on the download."
        )
    return path


# --- GO ---------------------------------------------------------------------


def load_go_children(obo_path: Path) -> dict[str, set[str]]:
    """Map each non-obsolete GO term to its direct `is_a` children."""
    children: dict[str, set[str]] = defaultdict(set)
    terms: set[str] = set()

    def flush(term):
        if term and term["id"] and not term["obsolete"]:
            terms.add(term["id"])
            for parent in term["is_a"]:
                children[parent].add(term["id"])

    term = None
    with open(obo_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("["):
                flush(term)
                term = (
                    {"id": None, "is_a": [], "obsolete": False}
                    if line == "[Term]"
                    else None
                )
            elif term is not None:
                if line.startswith("id: "):
                    term["id"] = line[4:].strip()
                elif line.startswith("is_a: "):
                    term["is_a"].append(line[6:].split("!")[0].strip())
                elif line.startswith("is_obsolete: true"):
                    term["obsolete"] = True
    flush(term)
    return {t: children.get(t, set()) & terms for t in terms}


def load_gene2go(gaf_path: Path, universe: set[str]) -> dict[str, set[str]]:
    gene2go: dict[str, set[str]] = defaultdict(set)
    opener = gzip.open if str(gaf_path).endswith(".gz") else open
    with opener(gaf_path, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.split("\t")
            gene = parts[2].upper()
            if gene in universe:
                gene2go[gene].add(parts[4])
    return gene2go


class GOBranches:
    def __init__(self, children: dict[str, set[str]], gene2go: dict[str, set[str]]):
        self.children = children
        self.gene2go = gene2go

    def descendants(self, root: str) -> set[str]:
        out, stack = {root}, [root]
        while stack:
            for child in self.children.get(stack.pop(), ()):
                if child not in out:
                    out.add(child)
                    stack.append(child)
        return out

    def genes(self, root: str) -> set[str]:
        if root not in self.children:
            print(f"warning: {root} not in GO DAG", flush=True)
            return set()
        terms = self.descendants(root)
        return {g for g, ids in self.gene2go.items() if not ids.isdisjoint(terms)}


# --- TFClass ----------------------------------------------------------------


def load_tfclass_superclasses(ttl_path: Path, universe: set[str]) -> dict[str, set]:
    """Map TFClass top-level superclass label -> universe genes under it."""
    from rdflib import RDF, RDFS, Graph, Namespace
    from rdflib.namespace import OWL

    tf = Namespace("http://sybig.de/tfclass#")
    g = Graph()
    opener = gzip.open if str(ttl_path).endswith(".gz") else open
    with opener(ttl_path, "rb") as f:
        g.parse(f, format="turtle")

    def root_of(node):
        frontier = list(g.objects(node, RDFS.subClassOf))
        visited = set()
        while frontier:
            c = frontier.pop()
            if c in visited:
                continue
            visited.add(c)
            parents = list(g.objects(c, RDFS.subClassOf))
            if tf.TF_class in parents:
                return c
            frontier.extend(
                p for p in parents if (p, RDF.type, OWL.Restriction) not in g
            )
        return None

    tf_map: dict[str, set] = defaultdict(set)
    for node in g.subjects(RDFS.label, None):
        label = str(next(g.objects(node, RDFS.label), ""))
        base = label.split("(")[0].strip()
        candidates = [base.upper()] + [
            p.upper() for p in re.split(r"[- ]", base) if len(p) >= 2
        ]
        matched = next((c for c in candidates if c in universe), None)
        if not matched:
            continue
        root = root_of(node)
        if root is None:
            continue
        root_label = str(next(g.objects(root, RDFS.label), ""))
        if root_label:
            tf_map[root_label].add(matched)
    return tf_map


# --- axis 1 -----------------------------------------------------------------


def partition_families(
    universe: set[str], go: GOBranches, tf_map: dict[str, set]
) -> dict[str, str]:
    """Greedy assignment in the Supplementary Table S1 priority order."""
    family: dict[str, str] = {}
    remaining = set(universe)

    def assign(genes, label):
        for gene in set(genes) & remaining:
            family[gene] = label
        remaining.difference_update(genes)

    def prefixed(*prefixes):
        return {g for g in remaining if g.startswith(prefixes)}

    assign(prefixed("MT-"), "Mitochondrial Genome")
    assign(
        {
            g
            for g in go.genes("GO:0005840")
            if g.startswith(("RPL", "RPS", "FAU", "UBA52"))
            and not g.startswith(("MRPL", "MRPS"))
        },
        "Ribosome (Cytosolic)",
    )
    assign(prefixed("MRPL", "MRPS"), "Ribosome (Mitochondrial)")

    mhc = go.genes("GO:0042611") | prefixed("HLA-")
    assign(mhc & {"HLA-A", "HLA-B", "HLA-C", "B2M"}, "MHC Class I (Classical)")
    assign({g for g in mhc if g.startswith("HLA-D")}, "MHC Class II")
    assign(mhc & {"HLA-E", "HLA-F", "HLA-G"}, "MHC Class I (Non-Classical)")
    assign(mhc, "MHC / Antigen Presentation (Other)")

    big3 = {
        "Zinc-coordinating DNA-binding domains": "TF: Zinc-Coordinating",
        "Helix-turn-helix domains": "TF: Helix-Turn-Helix",
        "Basic domains": "TF: Basic Domains",
    }
    for key, label in big3.items():
        assign(tf_map.get(key, set()), label)
    assign(
        set().union(*(v for k, v in tf_map.items() if k not in big3)),
        "TF: Other (Structural)",
    )
    assign(go.genes("GO:0003700"), "TF: Unclassified")

    assign(go.genes("GO:0005929"), "Cilia & Flagella")
    assign(
        go.genes("GO:0005856") | go.genes("GO:0003774") | go.genes("GO:0003779"),
        "Cytoskeleton & Motor",
    )
    assign({g for g in remaining if re.match(r"^(HIST|H2A|H2B|H3C|H4C)", g)}, "Histone")

    assign(go.genes("GO:0003924"), "GTPase / Trafficking")
    assign(go.genes("GO:0016301"), "Kinase")
    assign(go.genes("GO:0016791"), "Phosphatase")
    assign(go.genes("GO:0045202"), "Synapse")
    assign(go.genes("GO:0007155"), "Cell Adhesion")

    assign(go.genes("GO:0005125") | go.genes("GO:0008083"), "Cytokine / Growth Factor")
    assign(
        {g for g in remaining if g.startswith("OR") and len(g) > 2 and g[2].isdigit()},
        "Olfactory Receptor",
    )
    assign(go.genes("GO:0004930"), "GPCR (Non-Olfactory)")
    assign(go.genes("GO:0005216"), "Ion Channel")
    assign(go.genes("GO:0005215") | prefixed("SLC"), "Transporter (SLC/ABC)")
    assign(
        {g for g in remaining if g.startswith("CD") and len(g) > 2 and g[2].isdigit()},
        "Surface Marker (CD)",
    )

    assign(go.genes("GO:0007049"), "Cell Cycle")
    assign(go.genes("GO:0006281"), "DNA Repair")

    assign(go.genes("GO:0004842"), "Ubiquitin Ligase")
    assign(go.genes("GO:0008233"), "Peptidase")
    assign(go.genes("GO:0031012"), "Extracellular Matrix")
    assign(go.genes("GO:0003723"), "RNA Binding")

    assign(go.genes("GO:0016787"), "Enzyme: Hydrolase")
    assign(go.genes("GO:0016740"), "Enzyme: Transferase")
    assign(go.genes("GO:0016491"), "Enzyme: Oxidoreductase")
    assign(
        go.genes("GO:0016829") | go.genes("GO:0016874") | go.genes("GO:0016853"),
        "Enzyme: Other (Lyase/Ligase/Iso)",
    )
    for gene in remaining:
        family[gene] = "Unclassified"
    return family


# --- axis 2 -----------------------------------------------------------------


def load_hpa_specificity(path: Path) -> dict[str, str]:
    col = "RNA single cell type specificity"
    if str(path).endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            with z.open("proteinatlas.tsv") as f:
                table = pd.read_csv(f, sep="\t", low_memory=False)
    else:
        table = pd.read_csv(path, sep="\t", low_memory=False)
    gene_col = "Gene" if "Gene" in table.columns else "Gene name"
    table = table[[gene_col, col]].dropna(subset=[col])
    return dict(zip(table[gene_col].str.upper(), table[col]))


def load_eisenberg(path: Path) -> set[str]:
    genes = set()
    with open(path) as f:
        for line in f:
            tok = line.strip().split()[0].upper() if line.strip() else ""
            if tok and not tok.startswith("#"):
                genes.add(tok)
    return genes


def expression_regimes(
    universe: set[str], go: GOBranches, hpa: dict[str, str], housekeeping: set[str]
) -> dict[str, str]:
    identity = {g for g, v in hpa.items() if v in HPA_IDENTITY} & universe
    dissociation = (
        VAN_DEN_BRINK_CORE
        | (go.genes(GO_HEAT_RESPONSE) & universe)
        | (go.genes(GO_UNFOLDED_PROTEIN) & universe)
    ) - identity
    cell_cycle = (go.genes(GO_CELL_CYCLE) & universe) - identity
    housekeeping = housekeeping & universe

    def regime(gene):
        if gene in dissociation:
            return "state_dissociation"
        if gene in cell_cycle:
            return "state_cell_cycle"
        if gene in housekeeping:
            return "constitutive"
        if hpa.get(gene) in HPA_IDENTITY:
            return "identity_associated"
        return "unassigned"

    return {g: regime(g) for g in universe}


# --- main -------------------------------------------------------------------


def load_universe(path: Path | None) -> set[str]:
    if path is None:
        from bmfm_targets.evaluation.gene_learnability import load_gene_annotations

        return set(load_gene_annotations(normalize_case=False)["gene"])
    if path.suffix == ".csv":
        return set(pd.read_csv(path, keep_default_na=False)["gene"].str.strip())
    return {line.strip() for line in open(path) if line.strip()}


def build(
    universe: set[str],
    cache_dir: Path,
    hpa_path: Path | None = None,
) -> pd.DataFrame:
    go = GOBranches(
        load_go_children(fetch("go_obo", cache_dir)),
        load_gene2go(fetch("go_gaf", cache_dir), universe),
    )
    tf_map = load_tfclass_superclasses(fetch("tfclass", cache_dir), universe)
    family = partition_families(universe, go, tf_map)

    if hpa_path is None:
        hpa_path = fetch("hpa", cache_dir)
    hpa = load_hpa_specificity(hpa_path)
    housekeeping = load_eisenberg(fetch("eisenberg", cache_dir))
    regime = expression_regimes(universe, go, hpa, housekeeping)

    table = pd.DataFrame({"gene": sorted(universe)})
    table["family"] = table["gene"].map(family)
    table["expr_regime"] = table["gene"].map(regime)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "bmfm_targets" / "gene_annotations",
    )
    parser.add_argument(
        "--hpa",
        type=Path,
        default=None,
        help="local proteinatlas.tsv (or .zip) instead of the pinned download",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        help="gene list (.txt, one per line, or .csv with a `gene` column)",
    )
    args = parser.parse_args(argv)
    table = build(load_universe(args.universe), args.cache_dir, args.hpa)
    table.to_csv(args.out, index=False)
    print(f"wrote {len(table)} genes to {args.out}", flush=True)
    print(table["family"].value_counts().to_string(), flush=True)
    print(table["expr_regime"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
