import pytest
from scanpy import read_h5ad

bmfm_core = pytest.importorskip("bmfm_core")

from bmfm_targets.datasets.zheng68k.zheng68k_dataset import Zheng68kDataset
from bmfm_targets.tests import helpers

# additional test_ function args are pytest fixtures defined in conftest.py`


def test_tokenizer_knows_Zheng68k_genes(modular_tokenizer):
    """No unknown genes in Zheng68K index."""
    path = helpers.Zheng68kPaths.root / "h5ad" / "zheng68k.h5ad"
    adata = read_h5ad(path)
    unknown = []
    for token in adata.var_names:
        try:
            modular_tokenizer.get_token_id("[" + token + "]")
        except:
            unknown.append(token)
    assert len(unknown) == 0


def test_tokenizer_knows_Zheng68k_label_columns(
    modular_tokenizer, zheng_dataset_kwargs_after_transform
):
    """No unknown label_columns in Zheng68K index."""
    dataset = Zheng68kDataset(
        **zheng_dataset_kwargs_after_transform,
        split="train",
    )
    for label in dataset.metadata["cell_type_ontology_term_id"].value_counts().index:
        modular_tokenizer.get_token_id("[" + label + "]")


def test_tokenizer_knows_cell_type_annotation_special_tokens(modular_tokenizer):
    special_tokens = [
        "<MOLECULAR_ENTITY>",
        "<MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>",
        "<CELL_TYPE_CLASS>",
        "<PAD>",
        "<EOS>",
        "<DECODER_START>",
        "<SENTINEL_ID_0>",
    ]
    for token in special_tokens:
        modular_tokenizer.get_token_id(token)


def test_tokenizer_raise_unknown_token_in_specific_zheng68k_cta_sample(
    modular_tokenizer,
):
    """Test a specific example found in Zheng68k test set that has exactly one unknown token."""
    sample = "<MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>[MALAT1][RPL13][RPS14][RPS15][RPS18][RPS2][TMSB4X][B2M][EEF1A1][RPL10][RPL11][RPL13A][RPL18A][RPL19][RPL21][RPL23A][RPL32][RPL34][RPL3][RPLP2][RPS12][RPS19][RPS23][RPS25][RPS27A][RPS27][RPS3A][RPS3][RPS4X][RPS6][RPS9][TMSB10][CD52][CORO1A][EEF1D][FTL][GLTSCR2][GNB2L1][GYPC][HLA-A][HLA-B][HLA-C][JUNB][MT-CO1][MYL12A][PFN1][RPL10A][RPL12][RPL14][RPL18][RPL26][RPL27A][RPL27][RPL28][RPL5][RPL6][RPL7][RPL8][RPL9][RPLP1][RPS16][RPS7][RPS8][TPT1][AC006994.2][ACTB][ALDOA][ANAPC11][ANAPC15][ARHGDIB][ARPC2][ATP5G2][BTG1][C1QBP][CCR7][CD2BP2][CD3E][CFL1][CHCHD2][CXCR4][DDX10][DDX5][EIF1][EIF3D][EIF4A1][ERGIC3][FAU][FCGRT][FTH1][FXYD5][GMFG][GPSM3][HINT1][HLA-E][HNRNPA1][HNRNPDL][IFITM2][ITM2B][JUN][LAMTOR4][LDHA][MAN2B2][MT-CO2][MT-CO3][MT-ND1][MT-ND4][MZT2B][NACA][NDUFA4][NDUFB9][NHP2L1][NPM1][OXLD1][P2RY8][PAIP2][PAPOLA][PNRC1][POLR2I][PPP1R7][PRELID1][PSMD4][PTMA][RPL15][RPL24][RPL29][RPL31][RPL35A][RPL35][RPL36A][RPL36][RPL7A][RPLP0][RPS11][RPS13][RPS15A][RPS28][RPS29][RPS5][RPSA][RSL1D1][SH3BGRL][SLC25A6][SOD1][SRP14][STAP1][TMBIM4-1][TRAPPC4][TUBA1B][TXNIP][UBA52][UBC][UBL5][VPS28][WDR83OS][ZNF524][ABL1][AC061992.2][ACN9][ACTG1][AES][ALG5][ANAPC16][ANKRD12][ANXA7][APOO][ARF6][ARHGDIA][ARL6IP4][ARPC1B][ARPC4][ASCC2][ASMTL][ATAD2B][ATP5L][BEX4][BRMS1][BST2][BTF3][C19orf43][C19orf53][C19orf60][C6orf48][C9orf78][CALR][CCNB1IP1][CCND3][CCNDBP1][CD44][CD53][CD79B][CD7][CDCA4][CDK11A][CFP][CHCHD10][CIB1][CIRBP][CLN3-1][CMTM7][CNOT2][COL18A1][COMMD9][COPS5][COPZ1][COX4I1][CPNE1][CRIP1][CTBP1-AS2][CUTC][DCTN2][DDX52][DECR1][DEF6][DENR][DGCR6][DHPS][DHX57][DNAJB1][DNAJC19][DNAJC7][DNPH1][DPP7][DUT][EDEM3][EDF1][EEF1B2][EIF3F][EIF3H][EIF3K][EIF4A3][EIF4H][EMD][EMP3][EPHX2][ERH][ERP29][EVL][FAM118A][FAM213B][FAM35A][FBXO21][FLT3LG][FOS][FUS][G3BP1][GBAS][GIMAP2][GIMAP6][GIMAP7][GPX1][GSTK1][GTF3C5][HCLS1][HIRA][HIST1H1C][HIST1H4C][HNRNPA0][HNRNPA2B1][HNRNPC][HOTAIRM1][HP1BP3][HSD17B10][IL2RG][ILVBL][ISCU][ITGB7][JPX][KLF2][KRT18][LAT][LEMD2][LGALS3BP][LINC00861][LSM1][LSM7][LSP1][LTB][LYPLA1][MANF][MARCKSL1][MED31][MED4][METTL23][MINOS1][MITD1][MPLKIP][MRFAP1][MRPL37][MRPS25][MRPS28][MT-ATP6][MT-CYB][MUM1][MYEOV2][NAA38][NDUFAF1][NDUFS8][NUDT9][OAZ1][ORMDL2][PBXIP1][PEBP1][PFAS][PFDN5][PHPT1][PINX1][PLIN2][PNPLA7][POLE4][PPA2][PPDPF][PPP1R11][PPP1R8][PRDX2][PRKAG2][PRPF31][PRPF40B][PRRC2C][PSMA3][PSMA5][PSMB6][PSMB9][PSMD6][PSME1][PTPRCAP][PTPRC][RAB21][RAB37][RAD23A][RARRES3][RCHY1][RCN2][RGS10][RGS19][RNF126][RNF7][RP11-425I13.3][RP11-544A12.8][RP11-594N15.2][RP11-83A24.2][RP11-861A13.4][RPF1][RPL30][RPL36AL][RPL37][RPL38][RPL4][RPP30][RPS20][RPS24][RPS6KB2][RSBN1L][RTFDC1][S100A4][SAR1B][SDHB][SEPHS2][SERF2][SERP1][SF3B5][SH3BGRL3][SH3YL1][SIGIRR][SIKE1][SLC25A53][SMEK2][SNX3][SRGN][SRPK2][SRSF5][SRSF7][SSR2][STOML2][STRADB][STUB1][TADA3][TAF6][TARBP2][TCEA1][TEX264][TFAP4][TIMM10B][TIMM13][TMEM106C][TMEM134][TMEM14C][TMEM258][TMEM259][TMEM43][TMLHE][TOLLIP-AS1][TOMM20][TP53TG1][TPM3][TRAF3IP3][TRAM1][TRAPPC1][TRIP4][TSC22D3][TUBA4A][TUT1][TYMP][UBALD1][UBB][UBE2L3][UBE2Q1][UBL3][UCP2][UNC119][UQCR11-1][UQCRH][VDAC1][VIM][VKORC1][WBSCR22][XRCC6][XRN2][YBX1][YIF1B][YPEL3][ZCCHC10][ZCCHC9][ZNF236][ZNF773][ZNF93]<CELL_TYPE_CLASS><SENTINEL_ID_0><EOS>"
    seq = "<@TOKENIZER-TYPE=GENE>" + sample

    with pytest.raises(RuntimeError, match="Encountered 1 unknown tokens"):
        modular_tokenizer(
            key_in="sample", sample_dict={"sample": seq}, on_unknown="raise"
        )
