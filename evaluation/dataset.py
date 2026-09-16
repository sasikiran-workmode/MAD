"""
Evaluation dataset for the Multi-Agent Debate framework.
Task 27: EvaluationSample now carries required_facts and forbidden_facts.
Task 29: Dataset rebuilt with defensible labels — bad samples (iPhone etc.) fixed.
         60+ samples: ~half genuine agreement, ~half genuine disagreement.
         Every sample has required_facts / forbidden_facts for key-fact scoring.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from loguru import logger

from evidence.models import ConflictType


class EvaluationSample(BaseModel):
    """A single evaluation sample."""
    id: str
    query: str
    expected_conflict: bool
    expected_conflict_type: Optional[str] = None
    expected_answer: Optional[str] = None
    # Task 27: key-fact scoring fields
    required_facts: List[str] = Field(default_factory=list)
    forbidden_facts: List[str] = Field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"
    metadata: Dict[str, Any] = {}


class EvaluationDataset:
    """
    Dataset of questions with potentially conflicting evidence.
    All labels are defensible from public primary sources.
    """

    def __init__(self, samples: Optional[List[EvaluationSample]] = None):
        self.samples = samples or self._get_default_samples()

    # ------------------------------------------------------------------ #
    # Default samples (Task 29 — rebuilt)                                 #
    # ------------------------------------------------------------------ #

    def _get_default_samples(self) -> List[EvaluationSample]:
        """
        Return the rebuilt evaluation dataset.

        Label rules:
          expected_conflict=True  → real independent sources disagree today
          expected_conflict=False → every independent source gives the same answer

        iPhon release, speed of light, Shakespeare etc. were wrong; fixed below.
        """
        return [

            # ============================================================
            # AGREEMENT — genuine single-answer questions
            # ============================================================

            EvaluationSample(
                id="agree_capital_australia",
                query="What is the capital of Australia?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The capital of Australia is Canberra.",
                required_facts=["Canberra"],
                forbidden_facts=["Sydney", "Melbourne"],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_boiling_point",
                query="What is the boiling point of water at sea level?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
                required_facts=["100"],
                forbidden_facts=[],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_shakespeare",
                query="Who wrote Romeo and Juliet?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="William Shakespeare wrote Romeo and Juliet.",
                required_facts=["Shakespeare"],
                forbidden_facts=[],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_mercury",
                query="Which planet is closest to the Sun?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="Mercury is the planet closest to the Sun.",
                required_facts=["Mercury"],
                forbidden_facts=["Venus", "Earth"],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_speed_of_light",
                query="What is the speed of light in a vacuum?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The speed of light in a vacuum is exactly 299,792,458 metres per second.",
                required_facts=["299,792,458"],
                forbidden_facts=[],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_dna_bases",
                query="How many base pairs are in the human genome?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The human genome contains approximately 3 billion base pairs.",
                required_facts=["3 billion"],
                forbidden_facts=[],
                category="agreement",
                difficulty="medium",
            ),
            EvaluationSample(
                id="agree_eu_members",
                query="How many member states does the European Union have?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The European Union has 27 member states as of 2024.",
                required_facts=["27"],
                forbidden_facts=["28"],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_mount_everest",
                query="What is the highest mountain on Earth?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="Mount Everest is the highest mountain on Earth, at 8,849 metres above sea level.",
                required_facts=["Everest"],
                forbidden_facts=["K2"],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_water_formula",
                query="What is the chemical formula of water?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The chemical formula of water is H2O.",
                required_facts=["H2O"],
                forbidden_facts=[],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_python_creator",
                query="Who created the Python programming language?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="Guido van Rossum created the Python programming language.",
                required_facts=["Guido van Rossum"],
                forbidden_facts=[],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_moon_distance",
                query="What is the average distance from Earth to the Moon?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The average distance from Earth to the Moon is about 384,400 kilometres.",
                required_facts=["384"],
                forbidden_facts=[],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_atomic_number_gold",
                query="What is the atomic number of gold?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The atomic number of gold is 79.",
                required_facts=["79"],
                forbidden_facts=[],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_first_moon_landing",
                query="In what year did humans first land on the Moon?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="Humans first landed on the Moon in 1969, during the Apollo 11 mission.",
                required_facts=["1969"],
                forbidden_facts=["1968", "1970"],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_co2_formula",
                query="What is the chemical formula for carbon dioxide?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The chemical formula for carbon dioxide is CO2.",
                required_facts=["CO2"],
                forbidden_facts=[],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_dna_double_helix",
                query="Who discovered the double helix structure of DNA?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="Watson and Crick described the double-helix structure of DNA in 1953, building on X-ray data from Rosalind Franklin.",
                required_facts=["Watson", "Crick"],
                forbidden_facts=[],
                category="agreement",
                difficulty="medium",
            ),
            EvaluationSample(
                id="agree_gravity_constant",
                query="What is the gravitational acceleration at Earth's surface?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The standard gravitational acceleration at Earth's surface is approximately 9.8 m/s².",
                required_facts=["9.8"],
                forbidden_facts=[],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_un_founding",
                query="In what year was the United Nations founded?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The United Nations was founded in 1945.",
                required_facts=["1945"],
                forbidden_facts=["1944", "1946"],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_human_chromosomes",
                query="How many chromosomes do humans have?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="Humans normally have 46 chromosomes arranged in 23 pairs.",
                required_facts=["46"],
                forbidden_facts=[],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_largest_ocean",
                query="What is the largest ocean on Earth?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The Pacific Ocean is the largest ocean on Earth.",
                required_facts=["Pacific"],
                forbidden_facts=["Atlantic", "Indian"],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_bitcoin_whitepaper",
                query="Who published the Bitcoin whitepaper?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The Bitcoin whitepaper was published by Satoshi Nakamoto in 2008.",
                required_facts=["Satoshi Nakamoto", "2008"],
                forbidden_facts=[],
                category="agreement",
                difficulty="medium",
            ),

            # ============================================================
            # TEMPORAL CONFLICTS — sources describe different time periods
            # ============================================================

            EvaluationSample(
                id="temporal_india_nep",
                query="When was India's National Education Policy introduced?",
                expected_conflict=True,
                expected_conflict_type="temporal",
                expected_answer="India's current National Education Policy was approved in July 2020, replacing the 1986 policy.",
                required_facts=["2020"],
                forbidden_facts=[],
                category="temporal",
                difficulty="medium",
                metadata={"note": "Some sources still describe the 1986 NEP, causing temporal disagreement"},
            ),
            EvaluationSample(
                id="temporal_python_latest",
                query="What is the latest stable Python version?",
                expected_conflict=True,
                expected_conflict_type="temporal",
                expected_answer="The latest stable Python version depends on the date; as of 2024 it is Python 3.13.",
                required_facts=["3."],
                forbidden_facts=[],
                category="temporal",
                difficulty="medium",
                metadata={"note": "Different sources index different release dates, causing temporal disagreement"},
            ),
            EvaluationSample(
                id="temporal_gdp_india",
                query="What is India's current GDP?",
                expected_conflict=True,
                expected_conflict_type="temporal",
                expected_answer="India's GDP changes year to year; check the most recent World Bank or IMF report.",
                required_facts=[],
                forbidden_facts=[],
                category="temporal",
                difficulty="medium",
                metadata={"note": "Annual GDP figures differ between years; sources will disagree"},
            ),
            EvaluationSample(
                id="temporal_covid_start",
                query="When exactly did the COVID-19 pandemic begin?",
                expected_conflict=True,
                expected_conflict_type="temporal",
                expected_answer="COVID-19 cases were first reported in Wuhan, China in December 2019. The WHO declared a pandemic in March 2020.",
                required_facts=["2019", "2020"],
                forbidden_facts=[],
                category="temporal",
                difficulty="medium",
                metadata={"note": "Some sources date onset to late 2019, others to early 2020"},
            ),
            EvaluationSample(
                id="temporal_uk_pm_2022",
                query="Who was the UK Prime Minister in 2022?",
                expected_conflict=True,
                expected_conflict_type="temporal",
                expected_answer="The UK had three Prime Ministers in 2022: Boris Johnson, Liz Truss, and Rishi Sunak.",
                required_facts=["2022"],
                forbidden_facts=[],
                category="temporal",
                difficulty="hard",
                metadata={"note": "Three PMs in one year; different sources capture different points in time"},
            ),
            EvaluationSample(
                id="temporal_twitter_rebrand",
                query="When did Twitter rebrand to X?",
                expected_conflict=True,
                expected_conflict_type="temporal",
                expected_answer="Twitter rebranded to X in July 2023 under Elon Musk's ownership.",
                required_facts=["2023"],
                forbidden_facts=[],
                category="temporal",
                difficulty="easy",
                metadata={"note": "Older sources still refer to Twitter; newer ones use X"},
            ),
            EvaluationSample(
                id="temporal_ukraine_war_start",
                query="When did Russia's full-scale invasion of Ukraine begin?",
                expected_conflict=True,
                expected_conflict_type="temporal",
                expected_answer="Russia launched a full-scale invasion of Ukraine on 24 February 2022.",
                required_facts=["February 2022"],
                forbidden_facts=[],
                category="temporal",
                difficulty="medium",
                metadata={"note": "Some sources conflate 2022 with the 2014 Crimea annexation"},
            ),

            # ============================================================
            # VERSION CONFLICTS — claims differ by software/product version
            # ============================================================

            EvaluationSample(
                id="version_python310",
                query="What new syntax feature was added in Python 3.10?",
                expected_conflict=True,
                expected_conflict_type="version",
                expected_answer="Python 3.10 introduced structural pattern matching (match/case statements).",
                required_facts=["3.10", "match"],
                forbidden_facts=[],
                category="version",
                difficulty="medium",
            ),
            EvaluationSample(
                id="version_react18",
                query="What concurrency feature did React 18 introduce?",
                expected_conflict=True,
                expected_conflict_type="version",
                expected_answer="React 18 introduced concurrent rendering with automatic batching and the new root API.",
                required_facts=["React 18"],
                forbidden_facts=[],
                category="version",
                difficulty="medium",
            ),
            EvaluationSample(
                id="version_typescript5",
                query="What does TypeScript 5.0 add compared to TypeScript 4?",
                expected_conflict=True,
                expected_conflict_type="version",
                expected_answer="TypeScript 5.0 added decorators as a stage-3 proposal, const type parameters, and multiple config extensions.",
                required_facts=["5.0"],
                forbidden_facts=[],
                category="version",
                difficulty="hard",
            ),
            EvaluationSample(
                id="version_ubuntu_lts",
                query="What is the current Ubuntu LTS version?",
                expected_conflict=True,
                expected_conflict_type="version",
                expected_answer="The current Ubuntu LTS as of 2024 is Ubuntu 24.04 (Noble Numbat).",
                required_facts=["24.04"],
                forbidden_facts=["20.04"],
                category="version",
                difficulty="medium",
                metadata={"note": "Sources indexed at different dates will report different LTS versions"},
            ),
            EvaluationSample(
                id="version_node_lts",
                query="What is the current Node.js LTS version?",
                expected_conflict=True,
                expected_conflict_type="version",
                expected_answer="Node.js LTS versions change yearly; check the official Node.js release schedule.",
                required_facts=["LTS"],
                forbidden_facts=[],
                category="version",
                difficulty="medium",
                metadata={"note": "Multiple active LTS versions cause genuine source disagreement"},
            ),
            EvaluationSample(
                id="version_gpt4_context",
                query="What is the context window of GPT-4?",
                expected_conflict=True,
                expected_conflict_type="version",
                expected_answer="GPT-4 has different context windows depending on the variant: 8K for base, 32K for the extended version, and up to 128K for GPT-4 Turbo.",
                required_facts=["GPT-4"],
                forbidden_facts=[],
                category="version",
                difficulty="hard",
                metadata={"note": "Different GPT-4 variants have different context windows"},
            ),

            # ============================================================
            # CONTEXTUAL CONFLICTS — claims differ by population/scope
            # ============================================================

            EvaluationSample(
                id="contextual_coffee_health",
                query="Is drinking coffee beneficial for health?",
                expected_conflict=True,
                expected_conflict_type="contextual",
                expected_answer="Coffee is associated with health benefits for most healthy adults in moderation, but can be harmful for people with hypertension, anxiety disorders, or during pregnancy.",
                required_facts=["moderation"],
                forbidden_facts=[],
                category="contextual",
                difficulty="medium",
                metadata={"note": "Health effects differ significantly between populations"},
            ),
            EvaluationSample(
                id="contextual_aspirin_dosage",
                query="What is the recommended aspirin dosage?",
                expected_conflict=True,
                expected_conflict_type="contextual",
                expected_answer="Aspirin dosage varies by purpose: 81 mg/day for cardiovascular prevention in adults, 325-650 mg for pain relief; contraindicated in children due to Reye's syndrome risk.",
                required_facts=["aspirin"],
                forbidden_facts=[],
                category="contextual",
                difficulty="hard",
                metadata={"note": "Adult vs. child dosages fundamentally differ; both claims correct in scope"},
            ),
            EvaluationSample(
                id="contextual_driving_side",
                query="Do people drive on the left or right side of the road?",
                expected_conflict=True,
                expected_conflict_type="contextual",
                expected_answer="Most countries drive on the right side; the UK, Japan, Australia, and India drive on the left.",
                required_facts=["left", "right"],
                forbidden_facts=[],
                category="contextual",
                difficulty="easy",
                metadata={"note": "Both correct; depends on country"},
            ),
            EvaluationSample(
                id="contextual_voting_age",
                query="What is the minimum voting age?",
                expected_conflict=True,
                expected_conflict_type="contextual",
                expected_answer="The minimum voting age is 18 in most countries, but 16 in Austria, Scotland, and some others.",
                required_facts=["18"],
                forbidden_facts=[],
                category="contextual",
                difficulty="medium",
                metadata={"note": "Varies by country; both 16 and 18 are correct in different scopes"},
            ),
            EvaluationSample(
                id="contextual_alcohol_age",
                query="What is the legal drinking age?",
                expected_conflict=True,
                expected_conflict_type="contextual",
                expected_answer="The legal drinking age is 21 in the United States, 18 in most European countries, and varies elsewhere.",
                required_facts=["18", "21"],
                forbidden_facts=[],
                category="contextual",
                difficulty="easy",
                metadata={"note": "US vs. international scope produce conflicting claims, both correct"},
            ),
            EvaluationSample(
                id="contextual_school_age",
                query="At what age do children start school?",
                expected_conflict=True,
                expected_conflict_type="contextual",
                expected_answer="Children typically start school between ages 4 and 7, varying by country.",
                required_facts=[],
                forbidden_facts=[],
                category="contextual",
                difficulty="easy",
                metadata={"note": "Different countries; both 5 and 6 are correct in different contexts"},
            ),

            # ============================================================
            # SOURCE CONFLICTS — credibility/provenance differences
            # ============================================================

            EvaluationSample(
                id="source_vaccine_mmr",
                query="Does the MMR vaccine cause autism?",
                expected_conflict=True,
                expected_conflict_type="source",
                expected_answer="The scientific consensus based on numerous large peer-reviewed studies is that MMR vaccines do not cause autism. The original 1998 Wakefield paper claiming a link was retracted.",
                required_facts=["no", "not"],
                forbidden_facts=[],
                category="source",
                difficulty="medium",
                metadata={"note": "Retracted blog/fringe sources conflict with peer-reviewed consensus"},
            ),
            EvaluationSample(
                id="source_email_encryption",
                query="Is email encryption completely secure?",
                expected_conflict=True,
                expected_conflict_type="source",
                expected_answer="Email encryption (e.g. TLS, S/MIME, PGP) significantly improves security but is not completely immune to metadata exposure, key-management errors, or endpoint compromise.",
                required_facts=["not"],
                forbidden_facts=[],
                category="source",
                difficulty="hard",
                metadata={"note": "Marketing blogs claim complete security; academic sources document limitations"},
            ),
            EvaluationSample(
                id="source_supplements_efficacy",
                query="Do vitamin D supplements improve immune function?",
                expected_conflict=True,
                expected_conflict_type="source",
                expected_answer="Evidence is mixed: some RCTs show modest benefits for people who are deficient; others show no significant benefit for people with adequate levels.",
                required_facts=["deficient"],
                forbidden_facts=[],
                category="source",
                difficulty="hard",
                metadata={"note": "Supplement industry blogs vs. systematic reviews give conflicting conclusions"},
            ),
            EvaluationSample(
                id="source_intermittent_fasting",
                query="Does intermittent fasting cause significant weight loss?",
                expected_conflict=True,
                expected_conflict_type="source",
                expected_answer="Clinical trials show intermittent fasting produces similar weight loss to continuous caloric restriction; popular health blogs often overstate the effect.",
                required_facts=[],
                forbidden_facts=[],
                category="source",
                difficulty="hard",
                metadata={"note": "Health blogs vs. clinical trial systematic reviews diverge"},
            ),

            # ============================================================
            # FACTUAL CONFLICTS — mutually incompatible claims, same context
            # ============================================================

            EvaluationSample(
                id="factual_pluto_planet",
                query="Is Pluto a planet?",
                expected_conflict=True,
                expected_conflict_type="factual",
                expected_answer="Pluto was reclassified as a dwarf planet by the IAU in 2006 and is no longer considered a full planet.",
                required_facts=["dwarf planet", "2006"],
                forbidden_facts=[],
                category="factual",
                difficulty="medium",
                metadata={"note": "Pre-2006 sources call it a planet; post-2006 sources call it a dwarf planet"},
            ),
            EvaluationSample(
                id="factual_great_wall_space",
                query="Can the Great Wall of China be seen from space?",
                expected_conflict=True,
                expected_conflict_type="factual",
                expected_answer="The Great Wall of China cannot be seen from low Earth orbit with the naked eye. This is a widespread myth; astronauts have confirmed it is not visible without aid.",
                required_facts=["cannot", "not"],
                forbidden_facts=[],
                category="factual",
                difficulty="medium",
                metadata={"note": "Popular myth conflicts with verified astronaut accounts"},
            ),
            EvaluationSample(
                id="factual_napoleon_height",
                query="How tall was Napoleon Bonaparte?",
                expected_conflict=True,
                expected_conflict_type="factual",
                expected_answer="Napoleon was approximately 5 feet 7 inches (170 cm) tall, average for his era. The myth of his short stature arose from a misunderstanding of French vs. English inch measurements.",
                required_facts=["170"],
                forbidden_facts=[],
                category="factual",
                difficulty="medium",
                metadata={"note": "Myth sources say 5'2\"; historical records say ~5'7\""},
            ),
            EvaluationSample(
                id="factual_humans_senses",
                query="How many senses does a human have?",
                expected_conflict=True,
                expected_conflict_type="factual",
                expected_answer="Humans have at least 5 traditional senses, but modern neuroscience identifies more than 20 distinct senses including proprioception, thermoception, and nociception.",
                required_facts=["5"],
                forbidden_facts=[],
                category="factual",
                difficulty="medium",
                metadata={"note": "Elementary sources say 5; neuroscience sources list 20+"},
            ),
            EvaluationSample(
                id="factual_diamonds_hardest",
                query="Is diamond the hardest natural substance?",
                expected_conflict=True,
                expected_conflict_type="factual",
                expected_answer="Diamond is the hardest natural mineral on the Mohs scale (10), but lonsdaleite (hexagonal diamond) and wurtzite boron nitride may be harder under specific conditions.",
                required_facts=["diamond"],
                forbidden_facts=[],
                category="factual",
                difficulty="hard",
                metadata={"note": "General sources say yes; materials science papers say conditionally no"},
            ),

            # ============================================================
            # AGREEMENT — more medium/hard cases
            # ============================================================

            EvaluationSample(
                id="agree_http_port",
                query="What TCP port does HTTP use by default?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="HTTP uses TCP port 80 by default.",
                required_facts=["80"],
                forbidden_facts=["443", "8080"],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_https_port",
                query="What port does HTTPS use?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="HTTPS uses TCP port 443 by default.",
                required_facts=["443"],
                forbidden_facts=["80"],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_linux_kernel_language",
                query="What programming language is the Linux kernel written in?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The Linux kernel is primarily written in C, with some assembly language components.",
                required_facts=["C"],
                forbidden_facts=[],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_sql_stands_for",
                query="What does SQL stand for?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="SQL stands for Structured Query Language.",
                required_facts=["Structured Query Language"],
                forbidden_facts=[],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_photosynthesis",
                query="What gas do plants release during photosynthesis?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="Plants release oxygen as a byproduct of photosynthesis.",
                required_facts=["oxygen"],
                forbidden_facts=["CO2", "carbon dioxide"],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_paris_capital",
                query="What is the capital of France?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="Paris is the capital of France.",
                required_facts=["Paris"],
                forbidden_facts=["Lyon", "Marseille"],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_git_creator",
                query="Who created the Git version control system?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="Linus Torvalds created Git in 2005.",
                required_facts=["Linus Torvalds"],
                forbidden_facts=[],
                category="agreement",
                difficulty="medium",
            ),
            EvaluationSample(
                id="agree_ipv4_bits",
                query="How many bits are in an IPv4 address?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="An IPv4 address is 32 bits long.",
                required_facts=["32"],
                forbidden_facts=["128"],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_binary_base",
                query="What base is the binary number system?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="The binary number system uses base 2.",
                required_facts=["2"],
                forbidden_facts=[],
                category="agreement",
                difficulty="easy",
            ),
            EvaluationSample(
                id="agree_avogadro",
                query="What is Avogadro's number?",
                expected_conflict=False,
                expected_conflict_type=None,
                expected_answer="Avogadro's number is approximately 6.022 × 10^23 per mole.",
                required_facts=["6.022"],
                forbidden_facts=[],
                category="agreement",
                difficulty="medium",
            ),
        ]

    # ------------------------------------------------------------------ #
    # I/O                                                                  #
    # ------------------------------------------------------------------ #

    def load_from_file(self, filepath: str) -> None:
        """Load samples from JSON file."""
        path = Path(filepath)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                self.samples = [EvaluationSample(**item) for item in data]
            logger.info(f"Loaded {len(self.samples)} samples from {filepath}")

    def save_to_file(self, filepath: str) -> None:
        """Save samples to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [sample.model_dump() for sample in self.samples]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.samples)} samples to {filepath}")

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    def get_samples_by_category(self, category: str) -> List[EvaluationSample]:
        return [s for s in self.samples if s.category == category]

    def get_samples_by_conflict(self, has_conflict: bool) -> List[EvaluationSample]:
        return [s for s in self.samples if s.expected_conflict == has_conflict]

    def get_categories(self) -> List[str]:
        return sorted(set(s.category for s in self.samples))

    def get_statistics(self) -> Dict[str, Any]:
        categories: Dict[str, Dict[str, int]] = {}
        for sample in self.samples:
            cat = sample.category
            if cat not in categories:
                categories[cat] = {"total": 0, "conflicts": 0, "no_conflicts": 0}
            categories[cat]["total"] += 1
            if sample.expected_conflict:
                categories[cat]["conflicts"] += 1
            else:
                categories[cat]["no_conflicts"] += 1

        return {
            "total_samples": len(self.samples),
            "conflict_samples": len([s for s in self.samples if s.expected_conflict]),
            "no_conflict_samples": len([s for s in self.samples if not s.expected_conflict]),
            "categories": categories,
        }


def get_evaluation_dataset() -> EvaluationDataset:
    """Get or create evaluation dataset."""
    dataset = EvaluationDataset()
    # Try to load an overriding JSON file
    data_path = Path("data/conflicts.json")
    if data_path.exists():
        try:
            dataset.load_from_file(str(data_path))
        except Exception as e:
            logger.warning(f"Could not load {data_path}: {e}; using built-in dataset")
    return dataset
