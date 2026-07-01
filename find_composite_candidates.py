from pathlib import Path
import pandas as pd
import itertools


SCAN_INDEX_PATH = Path("metadata") / "scan_index.csv"
OUTPUT_DIR = Path("metadata") / "composite_candidate_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(SCAN_INDEX_PATH)


def parse_pipe_list(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()

    return set(
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    )


df["label_set"] = df["labels_present"].apply(parse_pipe_list)
df["region_set"] = df["anatomical_region"].apply(parse_pipe_list)

print("Loaded scan index")
print("Number of scans:", len(df))
print(df[["case_id", "dataset", "anatomical_region", "labels_present"]].head())

IGNORE_LABELS = {
    "all_labels_labelmap",
    "vascular_labelmap",
    "background",
}


TOPANEU_NEUROVASCULAR_LABELS = {
    "BA",

    "R-P1P2",
    "L-P1P2",
    "R-P3P4",
    "L-P3P4",

    "R-ICA-C6-C7",
    "L-ICA-C6-C7",
    "R-ICA-C1-C5",
    "L-ICA-C1-C5",

    "R-M1",
    "L-M1",
    "R-M2",
    "L-M2",
    "R-M3",
    "L-M3",

    "R-Pcom",
    "L-Pcom",
    "Acom",

    "R-A1A2",
    "L-A1A2",
    "R-A3",
    "L-A3",
    "3rd-A2",
    "3rd-A3",

    "R-VA",
    "L-VA",

    "R-SCA",
    "L-SCA",
    "R-AICA",
    "L-AICA",
    "R-PICA",
    "L-PICA",

    "R-AChA",
    "L-AChA",

    "R-OA",
    "L-OA",
}


REGION_TARGET_LABELS = {
    # Detailed TopAneu neurovascular region
    "neurovascular_head": TOPANEU_NEUROVASCULAR_LABELS,

    # General head/neck anatomy from TotalSegmentator,
    # plus TopAneu neurovascular labels
    "head_neck": {
        "brain",
        "skull",
        "spinal_cord",
        "thyroid_gland",
        "trachea",

        "common_carotid_artery_left",
        "common_carotid_artery_right",
        "subclavian_artery_left",
        "subclavian_artery_right",

        "vertebrae_C1",
        "vertebrae_C2",
        "vertebrae_C3",
        "vertebrae_C4",
        "vertebrae_C5",
        "vertebrae_C6",
        "vertebrae_C7",

        *TOPANEU_NEUROVASCULAR_LABELS,
    },

    "thorax_chest": {
        "heart",
        "atrial_appendage_left",
        "pulmonary_vein",
        "superior_vena_cava",
        "brachiocephalic_trunk",
        "brachiocephalic_vein_left",
        "brachiocephalic_vein_right",
        "subclavian_artery_left",
        "subclavian_artery_right",

        "aorta",

        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",

        "sternum",
        "costal_cartilages",

        "rib_left_1",
        "rib_left_2",
        "rib_left_3",
        "rib_left_4",
        "rib_left_5",
        "rib_left_6",
        "rib_left_7",
        "rib_left_8",
        "rib_left_9",
        "rib_left_10",
        "rib_left_11",
        "rib_left_12",

        "rib_right_1",
        "rib_right_2",
        "rib_right_3",
        "rib_right_4",
        "rib_right_5",
        "rib_right_6",
        "rib_right_7",
        "rib_right_8",
        "rib_right_9",
        "rib_right_10",
        "rib_right_11",
        "rib_right_12",

        "clavicula_left",
        "clavicula_right",
        "scapula_left",
        "scapula_right",
        "humerus_left",
        "humerus_right",

        "trachea",
        "esophagus",

        "vertebrae_T1",
        "vertebrae_T2",
        "vertebrae_T3",
        "vertebrae_T4",
        "vertebrae_T5",
        "vertebrae_T6",
        "vertebrae_T7",
        "vertebrae_T8",
        "vertebrae_T9",
        "vertebrae_T10",
        "vertebrae_T11",
        "vertebrae_T12",

        "autochthon_left",
        "autochthon_right",
    },

    "abdomen": {
        "liver",
        "spleen",
        "stomach",
        "pancreas",
        "gallbladder",
        "duodenum",
        "small_bowel",
        "colon",

        "kidney_left",
        "kidney_right",
        "kidney_cyst_left",
        "kidney_cyst_right",
        "adrenal_gland_left",
        "adrenal_gland_right",

        "aorta",
        "inferior_vena_cava",
        "portal_vein_and_splenic_vein",

        "vertebrae_L1",
        "vertebrae_L2",
        "vertebrae_L3",

        "autochthon_left",
        "autochthon_right",
        "iliopsoas_left",
        "iliopsoas_right",
    },

    "pelvis_lower_body": {
        "urinary_bladder",
        "prostate",

        "sacrum",
        "hip_left",
        "hip_right",
        "femur_left",
        "femur_right",

        "iliac_artery_left",
        "iliac_artery_right",
        "iliac_vena_left",
        "iliac_vena_right",

        "vertebrae_L4",
        "vertebrae_L5",
        "vertebrae_S1",

        "iliopsoas_left",
        "iliopsoas_right",

        "gluteus_maximus_left",
        "gluteus_maximus_right",
        "gluteus_medius_left",
        "gluteus_medius_right",
        "gluteus_minimus_left",
        "gluteus_minimus_right",
    },
}

def region_score(label_set, target_labels):
    return len(label_set.intersection(target_labels))


for region_name, target_labels in REGION_TARGET_LABELS.items():
    df[f"{region_name}_score"] = df["label_set"].apply(
        lambda labels: len(labels.intersection(target_labels))
    )

    df[f"{region_name}_coverage"] = df[f"{region_name}_score"] / len(target_labels)


score_cols = [
    "neurovascular_head_score",
    "head_neck_score",
    "thorax_chest_score",
    "abdomen_score",
    "pelvis_lower_body_score",
]

df["total_target_score"] = df[score_cols].sum(axis=1)

df_scores = df.sort_values("total_target_score", ascending=False)

df_scores[
    [
        "case_id",
        "dataset",
        "anatomical_region",
        "total_target_score",
        "head_neck_score",
        "thorax_chest_score",
        "abdomen_score",
        "pelvis_lower_body_score",
        "labels_present",
    ]
].head(30).to_csv(
    OUTPUT_DIR / "top_overall_candidate_scans.csv",
    index=False,
)

print("Saved top overall candidates")

for region_name in REGION_TARGET_LABELS.keys():
    score_col = f"{region_name}_score"

    region_df = df[df[score_col] > 0].copy()
    region_df = region_df.sort_values(score_col, ascending=False)

    output_path = OUTPUT_DIR / f"best_{region_name}_scans.csv"

    region_df[
        [
            "case_id",
            "dataset",
            "image_path",
            "label_path",
            "anatomical_region",
            score_col,
            "labels_present",
            "annotation_type",
        ]
    ].head(50).to_csv(output_path, index=False)

    print(f"Saved {region_name} candidates:", output_path)
    

ADJACENT_REGION_PAIRS = [
    ("neurovascular_head", "head_neck"),
    ("head_neck", "thorax_chest"),
    ("thorax_chest", "abdomen"),
    ("abdomen", "pelvis_lower_body"),
]


for region_a, region_b in ADJACENT_REGION_PAIRS:
    score_a = f"{region_a}_score"
    score_b = f"{region_b}_score"

    pair_df = df[
        (df[score_a] > 0) &
        (df[score_b] > 0)
    ].copy()

    pair_df["pair_score"] = pair_df[score_a] + pair_df[score_b]

    pair_df = pair_df.sort_values("pair_score", ascending=False)

    output_path = OUTPUT_DIR / f"bridge_{region_a}_to_{region_b}.csv"

    pair_df[
        [
            "case_id",
            "dataset",
            "image_path",
            "label_path",
            "anatomical_region",
            "pair_score",
            score_a,
            score_b,
            "labels_present",
        ]
    ].head(50).to_csv(output_path, index=False)

    print(f"Saved bridge candidates for {region_a} to {region_b}:", output_path)

ALL_TARGET_LABELS = set()

for labels in REGION_TARGET_LABELS.values():
    ALL_TARGET_LABELS.update(labels)


def get_covered_target_labels(row):
    return row["label_set"].intersection(ALL_TARGET_LABELS)


remaining_labels = set(ALL_TARGET_LABELS)
selected_rows = []

candidate_df = df.copy()

MAX_SCANS_TO_SELECT = 6

for step in range(MAX_SCANS_TO_SELECT):
    best_idx = None
    best_gain = 0

    for idx, row in candidate_df.iterrows():
        covered = get_covered_target_labels(row)
        gain = len(covered.intersection(remaining_labels))

        if gain > best_gain:
            best_gain = gain
            best_idx = idx

    if best_idx is None or best_gain == 0:
        break

    selected_row = candidate_df.loc[best_idx]
    selected_rows.append(selected_row)

    newly_covered = get_covered_target_labels(selected_row)
    remaining_labels -= newly_covered

    candidate_df = candidate_df.drop(index=best_idx)

    print(
        f"Step {step + 1}: selected {selected_row['case_id']} "
        f"from {selected_row['dataset']} | new labels covered: {best_gain}"
    )

selected_df = pd.DataFrame(selected_rows)

selected_df[
    [
        "case_id",
        "dataset",
        "image_path",
        "label_path",
        "anatomical_region",
        "total_target_score",
        "head_neck_score",
        "thorax_chest_score",
        "abdomen_score",
        "pelvis_lower_body_score",
        "labels_present",
    ]
].to_csv(
    OUTPUT_DIR / "greedy_composite_candidate_set.csv",
    index=False,
)

print("Saved greedy composite candidate set")
print("Remaining uncovered target labels:", sorted(remaining_labels))