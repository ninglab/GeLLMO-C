PROPERTY_FULL_NAMES = {
    "ampa": ["membrane permeability", "permeability through a lipid-infused cellular membrane", "Parallel Artificial Membrane Permeability (PAMPA)"],
    "bbbp": ["BBB permeability", "BBBP", "blood-brain barrier permeability (BBBP)"],
    "carc": ["carcinogenicity", "ability to cause cancer by damaging the genome", "potential to disrupt cellular metabolic processes"],
    "drd2": ["DRD2 inhibition", "Dopamine receptor D2 inhibition probability", "inhibition probability of Dopamine receptor D2 (DRD2)"],
    "erg": ["hERG inhibition", "blocking of Human ether-à-go-go related gene (hERG) potassium channel", "potential to block hERG channel"],
    "hia": ["Intestinal adsorption", "probability to be absorbed in the intestine", "human intestinal adsorption ability"],
    "liver": ["liver injury risk", "drug-induced liver injury (DILI)", "potential to cause liver disease"],
    "mutagenicity": ["Mutagenicity", "Mutagenicity predicted by Ames test", "probability to induce genetic alterations (mutagenicity)"],
    "plogp": ["Penalized octanol-water partition coefficient (penalized logP)", "Penalized logP", 
              "Penalized logP which is logP penalized by synthetic accessibility score and number of large rings"],
    "qed": ["QED", "Quantitative Estimate of Drug-likeness (QED)", "drug-likeness quantified by QED score"],
}

IND_TARGET_COMB = {"bpq": "bbbp+plogp+qed", # optimize CNS drug-likeliness
                   "elq": "erg+liver+qed",  # safety optimization
                   "acep": "ampa+carc+erg+plogp",  # permeability focused optimization      
                   "bdpq": "bbbp+drd2+plogp+qed",  # Dopaminergic CNS drug optimization 
                   "dhmq": "drd2+hia+mutagenicity+qed" # oral antipsychotic drugs
                }

OOD_TARGET_COMB = {"cde": "carc+drd2+erg",  # receptor targeting without blocking herg  
                   "abmp": "ampa+bbbp+mutagenicity+plogp", # membrane permeability + toxicity
                   "bcmq": "bbbp+carc+mutagenicity+qed", # CNS-likeness with carcinogenicity and hERG blocking
                   "bdeq": "bbbp+drd2+erg+qed", # dopaminergic CNS drug with toxicity control
                   "hlmpq": "hia+liver+mutagenicity+plogp+qed" # oral bioavailability with toxicity control
                }

TARGET_TASKS = dict(IND_TARGET_COMB)
TARGET_TASKS.update(OOD_TARGET_COMB) 
# Define property improvement thresholds as the average value across all pairs
PROPERTY_IMPV_THRESHOLDS = {
    "ampa": 0.1,
    "bbbp": 0.1,
    "carc": 0.1,
    "drd2": 0.2,
    "erg": 0.2,
    "hia": 0.1,
    "liver": 0.1,
    "mutagenicity": 0.1,
    "plogp": 1.0,
    "qed": 0.1,
}

# Define property improvement categories
PROPERTY_PERCENT_BUCKETS = {
    "significant": 50,
    "moderate": 25,
    "minor": 10,
}

PROPERTY_IMPV_DIR = {
    "plogp": "higher",
    "qed": "higher",
    "drd2": "higher",
    "bbbp": "higher",
    "hia": "higher",
    "mutagenicity": "lower",
    "carc": "lower",
    "erg": "lower",
    "liver": "lower",
    "ampa": "higher",
}

# Define property test thresholds to include a molecule in the test set
DEFAULT_PROPERTY_TEST_THRESHOLDS_LOWER = {
    "bbbp": 0.6,
    "drd2": 0.1,
    "hia": 0.6,
    "mutagenicity": 0.5,
    "plogp": -1.0,
    "qed": 0.7,
}

# these are 60th percentile of the values across all molecules in training
DEFAULT_PROPERTY_TEST_THRESHOLDS_UPPER = {
    "ampa": 0.9,
    "bbbp": 0.9,
    "carc": 0.2,
    "drd2": 0.2,
    "erg": 0.5,
    "hia": 0.9,
    "liver": 0.5,
    "mutagenicity": 0.2,
    "plogp": 1.5,
    "qed": 0.8,
}

DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_MAX_TOKENS = 1024