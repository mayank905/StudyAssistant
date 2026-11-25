"""Quiz, Oral Viva, and Flashcard Spaced Repetition Engine for StudyAssistant.

Features:
1. Grounded MCQ Mock Test Generation & Evaluation:
   - 4 options per question (A, B, C, D)
   - Exactly one single correct answer
   - In-depth citation-backed explanations
   - Weak topics detection and grading rubric

2. Interactive Oral Viva Voce Assessment:
   - Syllabus-grounded open-ended examiner questions
   - Three-dimensional evaluation rubric:
     * Factual Accuracy (0-100)
     * Conceptual Completeness (0-100)
     * Key Technical Terminology (0-100)
   - Missed key points identification
   - Conversational, TTS-ready spoken feedback

3. Flashcards & Spaced Repetition:
   - High-yield cards for definitions, tariff rules, formulas, and warranties
   - Spaced repetition ratings (Easy, Needs Review, Hard)
   - Integration with persistent SQLite database
"""

import json
import random
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from core.db import Database, get_db


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
@dataclass
class MCQQuestion:
    id: str
    subject: str
    chapter: str
    question: str
    options: List[str]  # Exactly 4 options
    correct_index: int  # 0, 1, 2, or 3
    correct_label: str  # 'A', 'B', 'C', or 'D'
    explanation: str
    citation: str
    difficulty: str = "medium"  # 'easy', 'medium', 'hard'
    topic: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionEvaluation:
    question_id: str
    question: str
    options: List[str]
    user_selected_index: Optional[int]
    user_selected_label: Optional[str]
    correct_index: int
    correct_label: str
    is_correct: bool
    explanation: str
    citation: str
    topic: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuizEvaluationResult:
    total_questions: int
    correct_count: int
    score_percentage: float
    grade: str
    time_taken_seconds: int
    breakdown: List[QuestionEvaluation]
    weak_topics: List[str]
    summary_feedback: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "correct_count": self.correct_count,
            "score_percentage": self.score_percentage,
            "grade": self.grade,
            "time_taken_seconds": self.time_taken_seconds,
            "breakdown": [b.to_dict() for b in self.breakdown],
            "weak_topics": self.weak_topics,
            "summary_feedback": self.summary_feedback,
        }


@dataclass
class VivaQuestion:
    id: str
    subject: str
    topic: str
    question: str
    expected_points: List[str]
    key_terms: List[str]
    citation: str
    difficulty: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VivaEvaluation:
    question_id: str
    subject: str
    topic: str
    question: str
    student_response: str
    score: float
    accuracy_score: float
    completeness_score: float
    key_terms_score: float
    feedback: str
    spoken_feedback: str
    correct_points: List[str]
    missed_points: List[str]
    citations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FlashcardItem:
    id: str
    subject: str
    chapter: str
    category: str  # 'definition', 'formula', 'tariff_rule', 'warranty'
    front: str
    back: str
    citation: str
    mastery_level: str = "unseen"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Curated Syllabus Knowledge Bank (Fully grounded in the 4 Insurance PDFs)
# -----------------------------------------------------------------------------
CURATED_MCQ_BANK: List[Dict[str, Any]] = [
    # FIRE INSURANCE
    {
        "id": "fire_mcq_001",
        "subject": "Fire Insurance",
        "chapter": "Standard Fire & Special Perils (SFSP)",
        "topic": "Standard Fire Perils",
        "question": "Under the All India Fire Tariff (AIFT) Standard Fire and Special Perils policy, how many named perils are covered in the standard basic policy?",
        "options": [
            "10 named perils",
            "12 named perils",
            "14 named perils",
            "8 named perils",
        ],
        "correct_index": 1,
        "correct_label": "B",
        "explanation": "The SFSP policy provides cover against 12 specified perils: Fire, Lightning, Explosion/Implosion, Aircraft damage, Riot/Strike/Malicious Damage (RSMD), Storm/Cyclone/Typhoon/Tempest/Hurricane/Tornado/Flood/Inundation (STFI), Impact damage, Subsidence/Landslide/Rockslide, Bursting/overflowing of water tanks, Missile testing, Leakage from automatic sprinkler installations, and Bush fire.",
        "citation": "Fire tariff.pdf, Section 2 - Standard Fire Policy Scope",
        "difficulty": "easy",
    },
    {
        "id": "fire_mcq_002",
        "subject": "Fire Insurance",
        "chapter": "Fire Tariff Rules",
        "topic": "Average Clause & Under-insurance",
        "question": "What is the primary function of the 'Condition of Average' in a Standard Fire and Special Perils policy?",
        "options": [
            "To calculate the average annual premium payable by the insured",
            "To penalize under-insurance by paying claims in pro-rata proportion of Sum Insured to actual value at the time of loss",
            "To split claim liability equally between all co-insurers",
            "To provide average depreciation deduction across building and machinery",
        ],
        "correct_index": 1,
        "correct_label": "B",
        "explanation": "If the property insured shall, at the breaking out of any fire or at the commencement of any destruction of or damage to such property by any other peril hereby insured against be collectively of greater value than the sum insured thereon, then the Insured shall be considered as being his own insurer for the difference, and shall bear a rateable proportion of the loss accordingly.",
        "citation": "Fire tariff.pdf, Condition 10 (Pro-rata Average Clause)",
        "difficulty": "medium",
    },
    {
        "id": "fire_mcq_003",
        "subject": "Fire Insurance",
        "chapter": "Special Clauses",
        "topic": "Reinstatement Value Clause (RVC)",
        "question": "Under a Fire policy with Reinstatement Value Clause (RVC) incorporated, which of the following property categories CANNOT be covered on a reinstatement value basis?",
        "options": [
            "Buildings and civil structures",
            "Industrial machinery and plant",
            "Stocks and merchandise in process / finished goods",
            "Office fixtures and furniture",
        ],
        "correct_index": 2,
        "correct_label": "C",
        "explanation": "Stocks, merchandise, and raw materials cannot be covered on a Reinstatement Value basis. Stocks must always be covered on Market Value (Indemnity) or agreed cost basis, because reinstatement relates to replacing physical capital assets with new ones of identical type without deducting depreciation.",
        "citation": "FIRE_Study_Material.pdf, Special Clauses - Reinstatement Value Clause",
        "difficulty": "medium",
    },
    {
        "id": "fire_mcq_004",
        "subject": "Fire Insurance",
        "chapter": "Tariff Warranties",
        "topic": "Silent Risk Warranty",
        "question": "When an industrial manufacturing unit ceases operations for more than 30 consecutive days, which tariff provision applies?",
        "options": [
            "The policy is cancelled automatically with full refund of premium",
            "Silent Risk rates apply and Silent Risk Warranty is attached to the policy",
            "Coverage automatically doubles without additional premium",
            "Sum insured must be reduced by 50% immediately",
        ],
        "correct_index": 1,
        "correct_label": "B",
        "explanation": "Under AIFT, when a manufacturing facility ceases industrial activity for 30 days or more, Silent Risk rates become applicable. Silent Risk warranty requires electric power to be disconnected (except for lighting/watchman), hazardous goods removed, and no manufacturing undertaken.",
        "citation": "Fire tariff.pdf, General Regulations - Silent Risk Tariff",
        "difficulty": "hard",
    },
    {
        "id": "fire_mcq_005",
        "subject": "Fire Insurance",
        "chapter": "Add-On Covers",
        "topic": "Escalation Clause",
        "question": "What is the maximum percentage escalation allowable under the Escalation Clause in a Standard Fire Policy during the policy period?",
        "options": [
            "Up to 10% of the Sum Insured",
            "Up to 25% of the Sum Insured",
            "Up to 50% of the Sum Insured",
            "Up to 75% of the Sum Insured",
        ],
        "correct_index": 1,
        "correct_label": "B",
        "explanation": "Under the All India Fire Tariff, the Escalation Clause allows an automatic increase in the Sum Insured up to a maximum of 25% to account for inflation during the policy period, with prorata additional premium paid upfront.",
        "citation": "Fire tariff.pdf, Endorsements - Escalation Clause Limit",
        "difficulty": "medium",
    },

    # MOTOR INSURANCE
    {
        "id": "motor_mcq_001",
        "subject": "Motor Insurance",
        "chapter": "Motor Vehicles Act Provisions",
        "topic": "Section 146 Mandatory Insurance",
        "question": "Under Section 146 of the Motor Vehicles Act 1988 (as amended by the Motor Vehicles Amendment Act 2019), what insurance cover is statutory and mandatory before driving a motor vehicle in a public place?",
        "options": [
            "Comprehensive Own Damage and Theft Insurance",
            "Third Party Liability Insurance",
            "Zero Depreciation Bumper-to-Bumper Cover",
            "Personal Accident cover of Rs. 50 Lakhs",
        ],
        "correct_index": 1,
        "correct_label": "B",
        "explanation": "Section 146 of the Motor Vehicles Act 1988 mandates that no person shall use, except as a passenger, or cause or allow any other person to use, a motor vehicle in a public place, unless there is in force in relation to the use of the vehicle a policy of insurance complying with the requirements of Chapter XI (Third Party Liability).",
        "citation": "UPDATED_MOTOR_WRITE_UP_FOR_PE_2026_27_EXAMINATION_ON_27_12_2025.pdf, Section 146 Chapter XI",
        "difficulty": "easy",
    },
    {
        "id": "motor_mcq_002",
        "subject": "Motor Insurance",
        "chapter": "Claim Settlement & Depreciation",
        "topic": "Depreciation Scales for Partial Loss",
        "question": "According to the Indian Motor Tariff depreciation scale, what percentage depreciation applies to rubber parts, nylon/plastic parts, and tyres/tubes in a partial loss claim?",
        "options": [
            "Nil (0%)",
            "30%",
            "50%",
            "Rate based on vehicle age up to 10 years",
        ],
        "correct_index": 2,
        "correct_label": "C",
        "explanation": "Under the Indian Motor Tariff, the depreciation rate for all rubber/nylon/plastic parts, tyres and tubes, batteries and air bags is fixed at 50% regardless of the age of the vehicle.",
        "citation": "UPDATED_MOTOR_WRITE_UP_FOR_PE_2026_27_EXAMINATION_ON_27_12_2025.pdf, Section II - Depreciation Scales",
        "difficulty": "medium",
    },
    {
        "id": "motor_mcq_003",
        "subject": "Motor Insurance",
        "chapter": "Underwriting Guidelines",
        "topic": "No Claim Bonus (NCB) Slabs",
        "question": "What is the maximum No Claim Bonus (NCB) percentage achievable on an own damage premium for private cars under the Indian Motor Tariff?",
        "options": [
            "35% after 3 consecutive claim-free years",
            "45% after 4 consecutive claim-free years",
            "50% after 5 consecutive claim-free years",
            "65% after 6 consecutive claim-free years",
        ],
        "correct_index": 2,
        "correct_label": "C",
        "explanation": "Under GR 27 of the Indian Motor Tariff, NCB follows a graduated scale: 1 year claim-free = 20%, 2 years = 25%, 3 years = 35%, 4 years = 45%, and 5 consecutive claim-free years = 50% maximum.",
        "citation": "UPDATED_MOTOR_WRITE_UP_FOR_PE_2026_27_EXAMINATION_ON_27_12_2025.pdf, GR 27 - No Claim Bonus",
        "difficulty": "easy",
    },
    {
        "id": "motor_mcq_004",
        "subject": "Motor Insurance",
        "chapter": "Statutory Provisions",
        "topic": "Hit and Run Compensation (Solatium Fund)",
        "question": "Following the Motor Vehicles Amendment Act 2019, what are the revised compensation limits under Section 161 for Hit and Run motor accidents?",
        "options": [
            "Rs. 25,000 for death and Rs. 12,500 for grievous hurt",
            "Rs. 1,00,000 for death and Rs. 25,000 for grievous hurt",
            "Rs. 2,00,000 for death and Rs. 50,000 for grievous hurt",
            "Rs. 5,00,000 for death and Rs. 1,00,000 for grievous hurt",
        ],
        "correct_index": 2,
        "correct_label": "C",
        "explanation": "Section 161 of the Motor Vehicles Act (amended 2019) enhanced the compensation for Hit and Run motor accidents from Rs. 25,000 to Rs. 2,00,000 in case of death, and from Rs. 12,500 to Rs. 50,000 in case of grievous hurt.",
        "citation": "UPDATED_MOTOR_WRITE_UP_FOR_PE_2026_27_EXAMINATION_ON_27_12_2025.pdf, Section 161 - Hit & Run Solatium Fund",
        "difficulty": "hard",
    },
    {
        "id": "motor_mcq_005",
        "subject": "Motor Insurance",
        "chapter": "Endorsements & Liability",
        "topic": "Third Party Property Damage (TPPD) Limit",
        "question": "Under a standard Commercial Vehicle Liability policy, what is the default statutory limit of indemnity for Third Party Property Damage (TPPD) under Section 147?",
        "options": [
            "Unlimited liability",
            "Rs. 7.5 Lakhs (which can be restricted to Rs. 6,000 with premium discount)",
            "Rs. 1 Crore mandatory",
            "Rs. 50,000 fixed",
        ],
        "correct_index": 1,
        "correct_label": "B",
        "explanation": "Section 147(2)(b) stipulates a limit of Rs. 7.5 Lakhs in respect of damage to any property of a third party. The Indian Motor Tariff allows insureds to restrict TPPD cover to the earlier statutory limit of Rs. 6,000 to obtain a premium discount under IMT 20.",
        "citation": "UPDATED_MOTOR_WRITE_UP_FOR_PE_2026_27_EXAMINATION_ON_27_12_2025.pdf, Section 147 & IMT 20",
        "difficulty": "medium",
    },

    # ENGINEERING INSURANCE
    {
        "id": "engg_mcq_001",
        "subject": "Engineering Insurance",
        "chapter": "Project Policies",
        "topic": "Contractors All Risks (CAR) Policy",
        "question": "What is the basis of Sum Insured under Section I (Material Damage) of a Contractors All Risks (CAR) policy?",
        "options": [
            "The initial estimate of material procurement cost only",
            "The full estimated completed contract value including materials, wages, freight, customs duties, and construction costs",
            "The depreciated market value of the civil structures",
            "10 times the annual contractor turnover",
        ],
        "correct_index": 1,
        "correct_label": "B",
        "explanation": "Under Section I of the CAR policy, the Sum Insured must represent the full value of the contract works at completion, including materials, wages, freight, customs duties, contractor's profit margin, and any materials supplied by the principal.",
        "citation": "ENGG_Study_Material_Fast_Revision.pdf, CAR Policy - Basis of Sum Insured",
        "difficulty": "medium",
    },
    {
        "id": "engg_mcq_002",
        "subject": "Engineering Insurance",
        "chapter": "Operational Policies",
        "topic": "Machinery Breakdown (MBD) Policy",
        "question": "Which of the following causes of loss is EXPRESSLY COVERED under a standard Machinery Breakdown (MBD) Insurance policy?",
        "options": [
            "Gradual deterioration, wear and tear, and normal cavitation",
            "Electrical breakdown due to short circuit, arcing, or excessive voltage",
            "War, invasion, and nuclear hazards",
            "Loss or damage for which the supplier or manufacturer is responsible by law or contract",
        ],
        "correct_index": 1,
        "correct_label": "B",
        "explanation": "Machinery Breakdown policy covers sudden and unforeseen physical damage to plant and machinery resulting from electrical causes (short-circuiting, arcing, voltage surge) and mechanical causes (centrifugal force, mechanical failure). Normal wear, tear, and manufacturer guarantee faults are strictly excluded.",
        "citation": "ENGG_Study_Material_Fast_Revision.pdf, MBD Policy Coverage & Exclusions",
        "difficulty": "easy",
    },
    {
        "id": "engg_mcq_003",
        "subject": "Engineering Insurance",
        "chapter": "Project Insurance",
        "topic": "Testing and Commissioning Period in EAR",
        "question": "In an Erection All Risks (EAR) policy, what is the maximum standard period permitted for Testing and Commissioning within the basic rate without charging extra premium?",
        "options": [
            "2 weeks",
            "4 weeks (1 month)",
            "3 months",
            "6 months",
        ],
        "correct_index": 1,
        "correct_label": "B",
        "explanation": "Standard EAR tariff policies include up to 4 weeks (1 month) of testing and commissioning period. Any testing beyond 4 weeks requires additional testing premium and prior underwriter agreement.",
        "citation": "ENGG_Study_Material_Fast_Revision.pdf, EAR Policy - Testing Period Guidelines",
        "difficulty": "hard",
    },
    {
        "id": "engg_mcq_004",
        "subject": "Engineering Insurance",
        "chapter": "Operational Policies",
        "topic": "Boiler & Pressure Plant (BPP) Policy",
        "question": "Under a Boiler and Pressure Plant (BPP) Insurance policy, what constitutes the primary insured event?",
        "options": [
            "Flaking of paint on external boiler casing",
            "Explosion or Collapse of the boiler/pressure vessel while working under normal steam/fluid pressure",
            "Loss of steam efficiency over 5 years of operation",
            "Failure to obtain statutory boiler inspector certification",
        ],
        "correct_index": 1,
        "correct_label": "B",
        "explanation": "BPP Insurance indemnifies against sudden and unforeseen damage caused by Explosion or Collapse of boilers and pressure vessels while in the course of ordinary working under pressure.",
        "citation": "ENGG_Study_Material_Fast_Revision.pdf, Boiler & Pressure Plant Policy Rules",
        "difficulty": "medium",
    },
    {
        "id": "engg_mcq_005",
        "subject": "Engineering Insurance",
        "chapter": "Business Interruption",
        "topic": "Advance Loss of Profits (ALOP / DSU)",
        "question": "What triggers a claim under an Advance Loss of Profits (ALOP) / Delay in Start-Up (DSU) policy?",
        "options": [
            "Delay in government environmental clearances",
            "Commercial delay caused by market recession",
            "Financial loss resulting from delay in project completion directly caused by indemnifiable physical damage under the CAR/EAR policy",
            "Bankruptcy or liquidation of the EPC contractor",
        ],
        "correct_index": 2,
        "correct_label": "C",
        "explanation": "ALOP/DSU coverage triggers only when there is a delay in commercial operation date resulting directly from physical loss or damage indemnifiable under Section I of the accompanying CAR or EAR policy.",
        "citation": "ENGG_Study_Material_Fast_Revision.pdf, ALOP / DSU Project Interruption",
        "difficulty": "medium",
    },
]

# -----------------------------------------------------------------------------
# Curated Oral Viva Knowledge Bank
# -----------------------------------------------------------------------------
CURATED_VIVA_BANK: List[Dict[str, Any]] = [
    {
        "id": "viva_fire_001",
        "subject": "Fire Insurance",
        "topic": "Condition of Average & Under-Insurance",
        "question": "Explain the Condition of Average in a Standard Fire Policy. How is a partial loss claim calculated when under-insurance exists?",
        "expected_points": [
            "Definition of under-insurance: when Sum Insured is less than actual value at risk at the time of loss",
            "Insured is treated as their own co-insurer for the uninsured difference",
            "Claim settlement formula: Claim = (Loss Amount * Sum Insured) / Value of Property",
            "Applies pro-rata across all items covered under that item rate",
            "Protects insurer against adverse selection and under-declaration of values",
        ],
        "key_terms": [
            "Condition of Average",
            "Under-insurance",
            "Pro-rata",
            "Sum Insured",
            "Market Value at time of loss",
            "Own insurer for the difference",
        ],
        "citation": "Fire tariff.pdf, Condition 10 (Pro-rata Average Clause)",
        "difficulty": "medium",
    },
    {
        "id": "viva_fire_002",
        "subject": "Fire Insurance",
        "topic": "Reinstatement Value Clause vs Market Value",
        "question": "Compare Reinstatement Value Clause (RVC) with Market Value (Indemnity) basis in Fire Insurance. What property is eligible, and what warranties apply?",
        "expected_points": [
            "RVC pays the cost of replacing property with new property of same kind without deducting depreciation",
            "Market value deduces physical depreciation and wear & tear from replacement cost",
            "RVC applies strictly to Building, Machinery, Plant, and Furniture, NOT to Stocks or merchandise",
            "Insured must reinstate or repair within 12 months (or extension approved by insurer)",
            "Sum insured under RVC must represent full new replacement cost; otherwise average clause applies",
        ],
        "key_terms": [
            "Reinstatement Value Clause",
            "Depreciation deduction",
            "New replacement cost",
            "Stocks excluded",
            "12 months completion timeline",
            "Condition of Average on new value",
        ],
        "citation": "FIRE_Study_Material.pdf, Special Clauses - Reinstatement Value",
        "difficulty": "hard",
    },
    {
        "id": "viva_motor_001",
        "subject": "Motor Insurance",
        "topic": "Section 146 & 147 of Motor Vehicles Act",
        "question": "Discuss the statutory necessity of Third Party insurance under Section 146 of the Motor Vehicles Act 1988, and outline the scope of liabilities covered under Section 147.",
        "expected_points": [
            "Section 146 makes TP insurance mandatory to use a motor vehicle in any public place",
            "Section 147 requires policy to insure against death, bodily injury to third parties, or damage to third party property",
            "Covers liability in respect of death or injury to passengers of public service vehicles and employees under EC Act",
            "Liability for third party death or bodily injury is unlimited in amount",
            "Third party property damage (TPPD) liability has a statutory limit of Rs. 7.5 Lakhs",
        ],
        "key_terms": [
            "Section 146 mandatory cover",
            "Section 147 requirements",
            "Unlimited bodily injury liability",
            "Third Party Property Damage Rs 7.5 Lakhs",
            "Motor Vehicles Amendment Act 2019",
        ],
        "citation": "UPDATED_MOTOR_WRITE_UP_FOR_PE_2026_27_EXAMINATION_ON_27_12_2025.pdf, Chapter XI Sections 146 & 147",
        "difficulty": "medium",
    },
    {
        "id": "viva_motor_002",
        "subject": "Motor Insurance",
        "topic": "Hit and Run Accidents & Solatium Scheme",
        "question": "What is a 'Hit and Run motor accident' under the Motor Vehicles Act? Explain the compensation amounts and the procedure under the Solatium Fund Scheme.",
        "expected_points": [
            "Defined under Section 161 as an accident arising out of use of a motor vehicle where identity of the vehicle cannot be ascertained after reasonable effort",
            "Revised compensation under 2019 Amendment: Rs. 2,00,000 for death, Rs. 50,000 for grievous hurt (earlier Rs. 25,000 and Rs. 12,500)",
            "Fund is administered by the Motor Vehicle Accident Fund / General Insurance Council",
            "Inquiry conducted by Claims Enquiry Officer (Sub-Divisional Magistrate / Tehsildar)",
            "Payment sanctioned by Claims Settlement Commissioner",
        ],
        "key_terms": [
            "Section 161 Hit and Run",
            "Rs 2,00,000 death compensation",
            "Rs 50,000 grievous hurt",
            "Solatium Scheme",
            "Claims Enquiry Officer",
        ],
        "citation": "UPDATED_MOTOR_WRITE_UP_FOR_PE_2026_27_EXAMINATION_ON_27_12_2025.pdf, Section 161",
        "difficulty": "medium",
    },
    {
        "id": "viva_engg_001",
        "subject": "Engineering Insurance",
        "topic": "Contractors All Risks (CAR) vs Erection All Risks (EAR)",
        "question": "Distinguish between Contractors All Risks (CAR) and Erection All Risks (EAR) policies in terms of principal scope, testing coverage, and risk profile.",
        "expected_points": [
            "CAR is designed for civil engineering projects (roads, dams, buildings, bridges) where civil works exceed 50% of contract value",
            "EAR is designed for projects involving erection of mechanical plant, electrical equipment, factories, and power plants",
            "EAR has critical Testing and Commissioning hazard (hot testing / cold testing), which CAR typically does not have",
            "Standard testing period under EAR is 4 weeks (1 month) included in basic rate",
            "Both cover Section I (Material Damage) and Section II (Third Party Liability)",
        ],
        "key_terms": [
            "Civil construction vs Mechanical erection",
            "Testing and commissioning period",
            "4 weeks standard testing",
            "Section I Material Damage",
            "Section II Third Party Liability",
        ],
        "citation": "ENGG_Study_Material_Fast_Revision.pdf, CAR vs EAR Comparison",
        "difficulty": "hard",
    },
]

# -----------------------------------------------------------------------------
# Curated Flashcards Deck
# -----------------------------------------------------------------------------
CURATED_FLASHCARD_BANK: List[Dict[str, Any]] = [
    # Fire Flashcards
    {
        "id": "fc_fire_001",
        "subject": "Fire Insurance",
        "chapter": "Standard Fire Policy",
        "category": "tariff_rule",
        "front": "What are the 12 Named Perils in the Standard Fire and Special Perils (SFSP) Policy?",
        "back": "1. Fire\n2. Lightning\n3. Explosion/Implosion\n4. Aircraft Damage\n5. Riot, Strike, Malicious Damage (RSMD)\n6. Storm, Cyclone, Typhoon, Tempest, Hurricane, Tornado, Flood, Inundation (STFI)\n7. Impact Damage\n8. Subsidence & Landslide/Rockslide\n9. Bursting/Overflowing of Water Tanks/Pipes\n10. Missile Testing Operations\n11. Leakage from Automatic Sprinkler Installations\n12. Bush Fire",
        "citation": "Fire tariff.pdf, Section 2 - Scope of Cover",
    },
    {
        "id": "fc_fire_002",
        "subject": "Fire Insurance",
        "chapter": "Fire Tariff Rules",
        "category": "formula",
        "front": "What is the Formula for Claim Settlement under the Condition of Average?",
        "back": "Claim Amount = (Loss Assessed × Sum Insured) / Actual Value of Property at the Time of Loss.\n\nNote: Applies only when Actual Value > Sum Insured (under-insurance).",
        "citation": "Fire tariff.pdf, Condition 10",
    },
    {
        "id": "fc_fire_003",
        "subject": "Fire Insurance",
        "chapter": "Special Clauses",
        "category": "definition",
        "front": "What is the Reinstatement Value Clause (RVC)?",
        "back": "A special condition providing settlement based on replacement of damaged building or machinery with NEW property of same kind, without deducting physical depreciation. Stocks are strictly excluded.",
        "citation": "FIRE_Study_Material.pdf, Special Clauses",
    },
    {
        "id": "fc_fire_004",
        "subject": "Fire Insurance",
        "chapter": "Tariff Warranties",
        "category": "warranty",
        "front": "What is the Silent Risk Warranty in Fire Insurance?",
        "back": "Applies when manufacturing activity ceases for ≥30 consecutive days. Requires disconnecting power supply (except minimal lighting/security), removing hazardous materials, and ceasing manufacturing operations.",
        "citation": "Fire tariff.pdf, Silent Risk Regulation",
    },
    {
        "id": "fc_fire_005",
        "subject": "Fire Insurance",
        "chapter": "Fire Tariff Deductibles",
        "category": "tariff_rule",
        "front": "What is the standard excess/deductible for AOG (Act of God) perils under Fire Tariff?",
        "back": "5% of the claim amount subject to a minimum of Rs. 10,000 for non-industrial risks, and scaled higher (Rs. 25,000 to Rs. 5,00,000) for industrial risks depending on Sum Insured.",
        "citation": "Fire tariff.pdf, Deductibles Schedule",
    },

    # Motor Flashcards
    {
        "id": "fc_motor_001",
        "subject": "Motor Insurance",
        "chapter": "Motor Vehicles Act",
        "category": "tariff_rule",
        "front": "What is Section 146 of the Motor Vehicles Act 1988?",
        "back": "Statutory mandate making it illegal to drive or use any motor vehicle in a public place without a valid Third Party Liability Insurance policy.",
        "citation": "UPDATED_MOTOR_WRITE_UP_FOR_PE_2026_27_EXAMINATION_ON_27_12_2025.pdf, Section 146",
    },
    {
        "id": "fc_motor_002",
        "subject": "Motor Insurance",
        "chapter": "Depreciation Scales",
        "category": "formula",
        "front": "What are the Fixed Depreciation Rates on Motor Parts in Partial Loss Claims?",
        "back": "• Rubber / Nylon / Plastic parts, Tyres & Tubes, Batteries, Airbags: 50%\n• Fiberglass components: 30%\n• Glass parts: NIL (0%)\n• Other parts (Metallic / Wooden): Age-based scale (0% to 50%).",
        "citation": "UPDATED_MOTOR_WRITE_UP_FOR_PE_2026_27_EXAMINATION_ON_27_12_2025.pdf, Depreciation Scale",
    },
    {
        "id": "fc_motor_003",
        "subject": "Motor Insurance",
        "chapter": "Underwriting Rules",
        "category": "formula",
        "front": "What are the No Claim Bonus (NCB) Slabs under GR 27?",
        "back": "• 1 claim-free year: 20%\n• 2 consecutive claim-free years: 25%\n• 3 consecutive claim-free years: 35%\n• 4 consecutive claim-free years: 45%\n• 5 consecutive claim-free years: 50% (Maximum Cap).",
        "citation": "UPDATED_MOTOR_WRITE_UP_FOR_PE_2026_27_EXAMINATION_ON_27_12_2025.pdf, GR 27",
    },
    {
        "id": "fc_motor_004",
        "subject": "Motor Insurance",
        "chapter": "Statutory Compensation",
        "category": "tariff_rule",
        "front": "What is Section 161 Hit and Run Compensation (2019 Amendment)?",
        "back": "Fixed statutory compensation for victims of hit-and-run accidents where the offending vehicle cannot be identified:\n• Death: Rs. 2,00,000 (increased from Rs. 25,000)\n• Grievous Hurt: Rs. 50,000 (increased from Rs. 12,500).",
        "citation": "UPDATED_MOTOR_WRITE_UP_FOR_PE_2026_27_EXAMINATION_ON_27_12_2025.pdf, Section 161",
    },
    {
        "id": "fc_motor_005",
        "subject": "Motor Insurance",
        "chapter": "IMT Endorsements",
        "category": "definition",
        "front": "What is IMT 21 in Motor Insurance?",
        "back": "Exclusion of Accidents to Unnamed Persons other than Insured and his Paid Driver or Cleaner (limits liability for occupants in private cars / commercial vehicles unless additional premium is paid).",
        "citation": "UPDATED_MOTOR_WRITE_UP_FOR_PE_2026_27_EXAMINATION_ON_27_12_2025.pdf, Endorsements",
    },

    # Engineering Flashcards
    {
        "id": "fc_engg_001",
        "subject": "Engineering Insurance",
        "chapter": "Project Policies",
        "category": "definition",
        "front": "What does Section I of a Contractors All Risks (CAR) Policy cover?",
        "back": "Material Damage to civil construction works (roads, bridges, dams, buildings) against sudden and accidental physical loss/damage from any cause other than specifically excluded.",
        "citation": "ENGG_Study_Material_Fast_Revision.pdf, CAR Section I",
    },
    {
        "id": "fc_engg_002",
        "subject": "Engineering Insurance",
        "chapter": "Project Policies",
        "category": "tariff_rule",
        "front": "What is the standard Testing & Commissioning period allowed in an EAR Policy?",
        "back": "Up to 4 weeks (1 month) of testing and commissioning is included within the basic EAR rate. Extended testing requires additional premium.",
        "citation": "ENGG_Study_Material_Fast_Revision.pdf, EAR Testing Clause",
    },
    {
        "id": "fc_engg_003",
        "subject": "Engineering Insurance",
        "chapter": "Operational Policies",
        "category": "definition",
        "front": "What is the Machinery Breakdown (MBD) Policy coverage?",
        "back": "Covers operational plant & machinery against unforeseen internal mechanical breakdown (centrifugal fracture, jamming) or electrical breakdown (short circuit, arcing, voltage surge). Excludes wear, tear, and corrosion.",
        "citation": "ENGG_Study_Material_Fast_Revision.pdf, MBD Policy Scope",
    },
    {
        "id": "fc_engg_004",
        "subject": "Engineering Insurance",
        "chapter": "Operational Policies",
        "category": "definition",
        "front": "What are the two primary insured events in Boiler and Pressure Plant (BPP) Insurance?",
        "back": "1. Explosion: Sudden tearing apart of vessel by internal steam/gas pressure.\n2. Collapse: Sudden crushing inward by external steam/fluid pressure (vacuum).",
        "citation": "ENGG_Study_Material_Fast_Revision.pdf, BPP Insurance Scope",
    },
    {
        "id": "fc_engg_005",
        "subject": "Engineering Insurance",
        "chapter": "Project Interruption",
        "category": "definition",
        "front": "What is Advance Loss of Profits (ALOP) / Delay in Start-Up (DSU)?",
        "back": "Covers consequential loss of gross profit / standing charges incurred due to delay in project completion directly caused by indemnifiable physical damage under CAR/EAR Section I.",
        "citation": "ENGG_Study_Material_Fast_Revision.pdf, ALOP/DSU Guidelines",
    },
]


# -----------------------------------------------------------------------------
# MCQ Mock Test Generator & Evaluator
# -----------------------------------------------------------------------------
class QuizEngine:
    """Core assessment engine for MCQs, Oral Viva, and Flashcard Spaced Repetition."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_db()

    def generate_mcq_quiz(
        self,
        subject: str = "General",
        chapter: Optional[str] = None,
        num_questions: int = 5,
        difficulty: str = "medium",
        rag_engine: Optional[Any] = None,
    ) -> List[MCQQuestion]:
        """Generate multiple choice questions grounded strictly in the study materials.

        Ensures:
        - Exactly 4 options per question
        - Exactly one single correct answer
        - Full citation and explanation
        """
        # 1. If RAG engine / LLM is passed, attempt LLM generation first
        if rag_engine is not None:
            try:
                llm_questions = self._generate_mcq_via_llm(
                    rag_engine=rag_engine,
                    subject=subject,
                    chapter=chapter,
                    num_questions=num_questions,
                    difficulty=difficulty,
                )
                if llm_questions and len(llm_questions) >= num_questions:
                    return llm_questions[:num_questions]
            except Exception:
                # Graceful fallback to verified curated syllabus bank
                pass

        # 2. Curated grounded question bank selection
        pool = []
        subject_clean = subject.strip().lower()

        for q in CURATED_MCQ_BANK:
            q_sub = q["subject"].lower()
            if subject_clean in ("all", "general", "") or subject_clean in q_sub or q_sub in subject_clean:
                if chapter and chapter.lower() not in q.get("chapter", "").lower():
                    continue
                pool.append(q)

        # Fallback if filter is too narrow
        if not pool:
            pool = CURATED_MCQ_BANK

        # Shuffle and sample requested count
        selected = random.sample(pool, min(num_questions, len(pool)))

        questions: List[MCQQuestion] = []
        for d in selected:
            questions.append(
                MCQQuestion(
                    id=d["id"],
                    subject=d["subject"],
                    chapter=d.get("chapter", ""),
                    question=d["question"],
                    options=list(d["options"]),
                    correct_index=int(d["correct_index"]),
                    correct_label=d["correct_label"],
                    explanation=d["explanation"],
                    citation=d["citation"],
                    difficulty=d.get("difficulty", "medium"),
                    topic=d.get("topic", ""),
                )
            )
        return questions

    def _generate_mcq_via_llm(
        self,
        rag_engine: Any,
        subject: str,
        chapter: Optional[str],
        num_questions: int,
        difficulty: str,
    ) -> List[MCQQuestion]:
        """Query LLM / RAG engine with strict JSON schema instructions."""
        prompt = f"""
You are an authoritative Indian Insurance Institute Examiner generating a mock test.
Subject: {subject}
Chapter/Topic: {chapter or 'General Syllabus'}
Number of Questions: {num_questions}
Difficulty Level: {difficulty}

Instructions:
1. Every question MUST be strictly grounded in the official study materials (Fire Tariff, Motor Tariff, Engineering Insurance).
2. Generate EXACTLY 4 options for each question (Options A, B, C, D).
3. Exactly ONE option must be unambiguously correct.
4. Provide a thorough explanation and cite the specific document and rule/page.
5. Return ONLY a valid JSON array of objects conforming to this schema:
[
  {{
    "id": "unique_id_string",
    "subject": "{subject}",
    "chapter": "chapter name",
    "topic": "topic name",
    "question": "Question text here",
    "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
    "correct_index": 0,
    "correct_label": "A",
    "explanation": "Detailed explanation citing the authoritative source",
    "citation": "Document name and rule/section/page",
    "difficulty": "{difficulty}"
  }}
]
"""
        response_text = ""
        if hasattr(rag_engine, "query"):
            res = rag_engine.query(prompt)
            response_text = res.get("answer", "") if isinstance(res, dict) else str(res)
        elif hasattr(rag_engine, "generate"):
            response_text = rag_engine.generate(prompt)
        elif callable(rag_engine):
            response_text = rag_engine(prompt)

        # Clean JSON markdown code fence if present
        clean_json = re.sub(r"^```(?:json)?", "", response_text.strip(), flags=re.MULTILINE)
        clean_json = re.sub(r"```$", "", clean_json.strip(), flags=re.MULTILINE).strip()

        data = json.loads(clean_json)
        results: List[MCQQuestion] = []
        labels = ["A", "B", "C", "D"]

        for item in data:
            opts = item.get("options", [])
            if len(opts) != 4:
                continue
            idx = int(item.get("correct_index", 0))
            lbl = labels[idx] if 0 <= idx < 4 else "A"
            results.append(
                MCQQuestion(
                    id=item.get("id", str(uuid.uuid4())[:8]),
                    subject=item.get("subject", subject),
                    chapter=item.get("chapter", chapter or ""),
                    question=item.get("question", ""),
                    options=opts,
                    correct_index=idx,
                    correct_label=lbl,
                    explanation=item.get("explanation", ""),
                    citation=item.get("citation", ""),
                    difficulty=item.get("difficulty", difficulty),
                    topic=item.get("topic", ""),
                )
            )
        return results

    def evaluate_quiz_submission(
        self,
        questions: List[MCQQuestion],
        user_answers: Dict[Union[int, str], Union[int, str]],
        time_taken_seconds: int = 0,
        persist: bool = True,
    ) -> QuizEvaluationResult:
        """Evaluate a student's mock test submission and compute analytics."""
        correct_count = 0
        breakdown: List[QuestionEvaluation] = []
        weak_topics: List[str] = []
        labels = ["A", "B", "C", "D"]

        for i, q in enumerate(questions):
            # Answer key might be index int or question id
            ans = user_answers.get(i) if i in user_answers else user_answers.get(q.id)

            selected_idx: Optional[int] = None
            selected_lbl: Optional[str] = None

            if ans is not None:
                if isinstance(ans, int):
                    if 0 <= ans < 4:
                        selected_idx = ans
                        selected_lbl = labels[ans]
                elif isinstance(ans, str):
                    ans_upper = ans.strip().upper()
                    if ans_upper in labels:
                        selected_idx = labels.index(ans_upper)
                        selected_lbl = ans_upper
                    elif ans.isdigit():
                        int_val = int(ans)
                        if 0 <= int_val < 4:
                            selected_idx = int_val
                            selected_lbl = labels[int_val]

            is_correct = selected_idx is not None and selected_idx == q.correct_index
            if is_correct:
                correct_count += 1
            else:
                topic = q.topic or q.chapter or q.subject
                if topic and topic not in weak_topics:
                    weak_topics.append(topic)

            breakdown.append(
                QuestionEvaluation(
                    question_id=q.id,
                    question=q.question,
                    options=q.options,
                    user_selected_index=selected_idx,
                    user_selected_label=selected_lbl,
                    correct_index=q.correct_index,
                    correct_label=q.correct_label,
                    is_correct=is_correct,
                    explanation=q.explanation,
                    citation=q.citation,
                    topic=q.topic or q.chapter,
                )
            )

        total = max(len(questions), 1)
        score_pct = round((correct_count / total) * 100.0, 1)

        # Grading logic
        if score_pct >= 80.0:
            grade = "Distinction (Exam Ready)"
            feedback = "Outstanding performance! You demonstrate thorough mastery of tariff rules and statutory definitions."
        elif score_pct >= 60.0:
            grade = "Pass (Proficient)"
            feedback = "Good work. You have passed the threshold, but review the flagged weak topics to ensure top scores."
        else:
            grade = "Needs Improvement"
            feedback = "Further revision required. Pay close attention to the citations and explanations for questions missed."

        res = QuizEvaluationResult(
            total_questions=len(questions),
            correct_count=correct_count,
            score_percentage=score_pct,
            grade=grade,
            time_taken_seconds=time_taken_seconds,
            breakdown=breakdown,
            weak_topics=weak_topics,
            summary_feedback=feedback,
        )

        if persist and self.db:
            subject = questions[0].subject if questions else "General"
            chapter = questions[0].chapter if questions else ""
            self.db.log_quiz_attempt(
                subject=subject,
                chapter=chapter,
                score=score_pct,
                total_questions=len(questions),
                correct_answers=correct_count,
                time_taken_seconds=time_taken_seconds,
                questions_data=[b.to_dict() for b in breakdown],
                weak_topics=weak_topics,
            )

        return res

    # -------------------------------------------------------------------------
    # Oral Viva Examiner Generation & Evaluation
    # -------------------------------------------------------------------------
    def generate_viva_question(
        self,
        subject: str = "General",
        topic: Optional[str] = None,
        rag_engine: Optional[Any] = None,
    ) -> VivaQuestion:
        """Select or generate a grounded open-ended examiner viva question."""
        if rag_engine is not None:
            try:
                v = self._generate_viva_via_llm(rag_engine, subject, topic)
                if v:
                    return v
            except Exception:
                pass

        pool = []
        subject_clean = subject.strip().lower()

        for q in CURATED_VIVA_BANK:
            q_sub = q["subject"].lower()
            if subject_clean in ("all", "general", "") or subject_clean in q_sub or q_sub in subject_clean:
                if topic and topic.lower() not in q.get("topic", "").lower():
                    continue
                pool.append(q)

        if not pool:
            pool = CURATED_VIVA_BANK

        selected = random.choice(pool)
        return VivaQuestion(
            id=selected["id"],
            subject=selected["subject"],
            topic=selected.get("topic", ""),
            question=selected["question"],
            expected_points=list(selected["expected_points"]),
            key_terms=list(selected["key_terms"]),
            citation=selected["citation"],
            difficulty=selected.get("difficulty", "medium"),
        )

    def _generate_viva_via_llm(
        self, rag_engine: Any, subject: str, topic: Optional[str]
    ) -> Optional[VivaQuestion]:
        """Prompt LLM for a structured viva question."""
        prompt = f"""
You are a senior Oral Viva Examiner for the Insurance Institute of India.
Subject: {subject}
Topic: {topic or 'Core Syllabus'}

Generate an oral examination question testing conceptual understanding.
Output strictly in JSON:
{{
  "id": "viva_{uuid.uuid4().hex[:6]}",
  "subject": "{subject}",
  "topic": "{topic or 'Key Concept'}",
  "question": "Clear spoken question for student",
  "expected_points": ["Point 1", "Point 2", "Point 3"],
  "key_terms": ["Term 1", "Term 2"],
  "citation": "Official study material and page/section",
  "difficulty": "medium"
}}
"""
        response_text = ""
        if hasattr(rag_engine, "query"):
            res = rag_engine.query(prompt)
            response_text = res.get("answer", "") if isinstance(res, dict) else str(res)
        elif hasattr(rag_engine, "generate"):
            response_text = rag_engine.generate(prompt)
        elif callable(rag_engine):
            response_text = rag_engine(prompt)

        clean = re.sub(r"^```(?:json)?", "", response_text.strip(), flags=re.MULTILINE)
        clean = re.sub(r"```$", "", clean.strip(), flags=re.MULTILINE).strip()
        data = json.loads(clean)
        return VivaQuestion(
            id=data.get("id", str(uuid.uuid4())[:8]),
            subject=data.get("subject", subject),
            topic=data.get("topic", topic or ""),
            question=data.get("question", ""),
            expected_points=data.get("expected_points", []),
            key_terms=data.get("key_terms", []),
            citation=data.get("citation", ""),
            difficulty=data.get("difficulty", "medium"),
        )

    def evaluate_viva_response(
        self,
        question: VivaQuestion,
        student_response: str,
        rag_engine: Optional[Any] = None,
        persist: bool = True,
    ) -> VivaEvaluation:
        """Evaluate a student's spoken/text response against rubric."""
        text = student_response.strip().lower()
        if not text:
            eval_res = VivaEvaluation(
                question_id=question.id,
                subject=question.subject,
                topic=question.topic,
                question=question.question,
                student_response="",
                score=0.0,
                accuracy_score=0.0,
                completeness_score=0.0,
                key_terms_score=0.0,
                feedback="No spoken response was provided.",
                spoken_feedback="I did not detect an answer. Please click the microphone and articulate your response.",
                correct_points=[],
                missed_points=list(question.expected_points),
                citations=[question.citation],
            )
            if persist and self.db:
                self.db.log_viva_evaluation(
                    subject=question.subject,
                    topic=question.topic,
                    question=question.question,
                    student_response="",
                    score=0.0,
                    accuracy_score=0.0,
                    completeness_score=0.0,
                    key_terms_score=0.0,
                    feedback=eval_res.feedback,
                    spoken_feedback=eval_res.spoken_feedback,
                    missed_points=eval_res.missed_points,
                    correct_points=[],
                    citations=[question.citation],
                )
            return eval_res

        # 1. Semantic keyword & conceptual pattern scoring
        correct_pts: List[str] = []
        missed_pts: List[str] = []

        # Completeness Check
        for pt in question.expected_points:
            pt_words = [w for w in re.findall(r"\b\w{4,}\b", pt.lower()) if w not in ("under", "which", "shall", "where", "other", "their", "apply", "applies")]
            matches = sum(1 for w in pt_words if w in text)
            if matches >= max(1, len(pt_words) // 3):
                correct_pts.append(pt)
            else:
                missed_pts.append(pt)

        comp_score = (len(correct_pts) / max(len(question.expected_points), 1)) * 100.0

        # Key Terminology Check
        matched_terms = 0
        for term in question.key_terms:
            t_words = [w for w in re.findall(r"\b\w{3,}\b", term.lower())]
            if any(w in text for w in t_words):
                matched_terms += 1
        term_score = (matched_terms / max(len(question.key_terms), 1)) * 100.0

        # Accuracy Check (penalize if words are very sparse or conflicting)
        word_count = len(text.split())
        accuracy_score = min(100.0, (comp_score * 0.6) + (term_score * 0.4))
        if word_count < 10:
            accuracy_score = min(accuracy_score, 40.0)

        # Composite score
        final_score = round((accuracy_score * 0.4) + (comp_score * 0.4) + (term_score * 0.2), 1)

        # Verbal feedback generation (clean text for TTS synthesis)
        if final_score >= 80.0:
            spoken_feedback = (
                f"Excellent answer! You scored {int(final_score)} percent. "
                f"You covered the essential principles and technical terms accurately."
            )
            feedback = f"### 🌟 Excellent Spoken Response ({final_score}%)\nYou clearly articulated the core concepts of **{question.topic}** in accordance with `{question.citation}`."
        elif final_score >= 50.0:
            spoken_feedback = (
                f"Good attempt! You scored {int(final_score)} percent. "
                f"You understood the main concept, but missed some key details such as {missed_pts[0] if missed_pts else 'tariff specifics'}."
            )
            feedback = f"### ⚖️ Satisfactory Response ({final_score}%)\nYou understand the fundamental concept, but your answer omitted key statutory details."
        else:
            spoken_feedback = (
                f"Your answer scored {int(final_score)} percent and requires review. "
                f"Please consult {question.citation} and practice explaining this topic again."
            )
            feedback = f"### ⚠️ Needs Improvement ({final_score}%)\nYour answer did not sufficiently address the expected regulatory criteria."

        eval_res = VivaEvaluation(
            question_id=question.id,
            subject=question.subject,
            topic=question.topic,
            question=question.question,
            student_response=student_response,
            score=final_score,
            accuracy_score=round(accuracy_score, 1),
            completeness_score=round(comp_score, 1),
            key_terms_score=round(term_score, 1),
            feedback=feedback,
            spoken_feedback=spoken_feedback,
            correct_points=correct_pts,
            missed_points=missed_pts,
            citations=[question.citation],
        )

        if persist and self.db:
            self.db.log_viva_evaluation(
                subject=question.subject,
                topic=question.topic,
                question=question.question,
                student_response=student_response,
                score=final_score,
                accuracy_score=round(accuracy_score, 1),
                completeness_score=round(comp_score, 1),
                key_terms_score=round(term_score, 1),
                feedback=feedback,
                spoken_feedback=spoken_feedback,
                missed_points=missed_pts,
                correct_points=correct_pts,
                citations=[question.citation],
            )

        return eval_res

    # -------------------------------------------------------------------------
    # Flashcard Generation & Spaced Repetition Engine
    # -------------------------------------------------------------------------
    def generate_flashcards(
        self,
        subject: str = "General",
        chapter: Optional[str] = None,
        num_cards: int = 10,
        category: Optional[str] = None,
        save_to_db: bool = True,
    ) -> List[FlashcardItem]:
        """Fetch or generate flashcards with definitions, formulas, and tariff rules."""
        pool = []
        subject_clean = subject.strip().lower()

        for c in CURATED_FLASHCARD_BANK:
            c_sub = c["subject"].lower()
            if subject_clean in ("all", "general", "") or subject_clean in c_sub or c_sub in subject_clean:
                if chapter and chapter.lower() not in c.get("chapter", "").lower():
                    continue
                if category and category.lower() != c.get("category", "").lower():
                    continue
                pool.append(c)

        if not pool:
            pool = CURATED_FLASHCARD_BANK

        selected = pool[:num_cards] if len(pool) <= num_cards else random.sample(pool, num_cards)

        cards = [
            FlashcardItem(
                id=c["id"],
                subject=c["subject"],
                chapter=c.get("chapter", ""),
                category=c.get("category", "definition"),
                front=c["front"],
                back=c["back"],
                citation=c.get("citation", ""),
            )
            for c in selected
        ]

        if save_to_db and self.db:
            batch = [
                {
                    "id": c.id,
                    "subject": c.subject,
                    "chapter": c.chapter,
                    "category": c.category,
                    "front": c.front,
                    "back": c.back,
                    "citations": c.citation,
                }
                for c in cards
            ]
            self.db.save_flashcards_batch(batch)

        return cards

    def rate_flashcard(self, card_id: str, rating: str) -> Dict[str, Any]:
        """Update flashcard mastery rating ('easy', 'review', 'hard') in SQLite DB."""
        if not self.db:
            raise RuntimeError("Database instance not available for flashcard rating.")
        return self.db.update_flashcard_rating(card_id, rating)


# Singleton Engine Instance
_default_quiz_engine: Optional[QuizEngine] = None


def get_quiz_engine(db: Optional[Database] = None) -> QuizEngine:
    """Get or create singleton QuizEngine instance."""
    global _default_quiz_engine
    if _default_quiz_engine is None or db is not None:
        _default_quiz_engine = QuizEngine(db=db)
    return _default_quiz_engine
