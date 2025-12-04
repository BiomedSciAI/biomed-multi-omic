"""The package consists of modules for DNA sequence tasks."""

from .mpra.mpra_datamodule import DNASeqMPRADataset, DNASeqMPRADataModule
from .splice_site.splice_site_datamodule import (
    DNASeqSpliceSiteDataModule,
    DNASeqSpliceSiteDataset,
)

from .covid.covid_datamodule import DNASeqCovidDataset, DNASeqCovidDataModule
from .drosophila_enhancer.drosophila_enhancer_datamodule import (
    DNASeqDrosophilaEnhancerDataset,
    DNASeqDrosophilaEnhancerDataModule,
)
from .epigenetic_marks.epigenetic_marks_datamodule import (
    DNASeqEpigeneticMarksDataModule,
    DNASeqEpigeneticMarksDataset,
)
from .promoter.promoter_datamodule import (
    DNASeqPromoterDataset,
    DNASeqPromoterDataModule,
)

from .core_promoter.core_promoter_datamodule import (
    DNASeqCorePromoterDataset,
    DNASeqCorePromoterDataModule,
)
from .transcription_factor.transcription_factor_datamodule import (
    DNASeqTranscriptionFactorDataModule,
    DNASeqTranscriptionFactorDataset,
)
from .chromatin_profile.chromatin_profile_datamodule import (
    StreamingDNASeqChromatinProfileDataset,
    StreamingDNASeqChromatinProfileDataModule,
    DNASeqChromatinProfileDataset,
    DNASeqChromatinProfileDataModule,
)

from .snp2trait.snp2trait_datamodule import (
    StreamingDNASeqSnp2TraitDataset,
    StreamingDNASeqSnp2TraitDataModule,
    DNASeqSnp2TraitDataset,
    DNASeqSnp2TraitDataModule,
    StreamingDNASeqSnp2TraitWithSnpPositionDataset,
    StreamingDNASeqSnp2TraitWithSnpPositionDataModule,
)

__all__ = [
    "DNASeqPromoterDataset",
    "DNASeqPromoterDataModule",
    "DNASeqCorePromoterDataset",
    "DNASeqCorePromoterDataModule",
    "DNASeqCorePromoterDataset",
    "DNASeqCorePromoterDataModule",
    "DNASeqMPRADataset",
    "DNASeqMPRADataModule",
    "DNASeqSpliceSiteDataModule",
    "DNASeqSpliceSiteDataset",
    "DNASeqCovidDataset",
    "DNASeqCovidDataModule",
    "DNASeqDrosophilaEnhancerDataset",
    "DNASeqDrosophilaEnhancerDataModule",
    "DNASeqEpigeneticMarksDataset",
    "DNASeqEpigeneticMarksDataModule",
    "DNASeqTranscriptionFactorDataModule",
    "DNASeqTranscriptionFactorDataset",
    "StreamingDNASeqChromatinProfileDataset",
    "StreamingDNASeqChromatinProfileDataModule",
    "DNASeqChromatinProfileDataset",
    "DNASeqChromatinProfileDataModule",
    "StreamingDNASeqSnp2TraitDataset",
    "StreamingDNASeqSnp2TraitDataModule",
    "StreamingDNASeqSnp2TraitWithSnpPositionDataset",
    "StreamingDNASeqSnp2TraitWithSnpPositionDataModule",
    "DNASeqSnp2TraitDataset",
    "DNASeqSnp2TraitDataModule",
]
